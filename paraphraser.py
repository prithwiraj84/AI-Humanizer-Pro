"""
Deep Rewrite: sentence-by-sentence paraphrasing with a small local model (no outside AI service).

The model is a T5 paraphraser converted ahead of time to CTranslate2's 8-bit format by
scripts/build_paraphraser.py: about 225 MB on disk and roughly 250 MB of memory once loaded, so it
needs a server with about 1 GB of RAM or more. It's optional. When data/paraphraser/ (or
PARAPHRASER_DIR) holds a model, Deep Rewrite is offered; otherwise it isn't and everything else
works as before.

This module only generates and scores candidates; humanizer.deep_paraphrase() decides which ones are
safe to use.
"""
import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get('PARAPHRASER_DIR') or os.path.join(BASE_DIR, 'data', 'paraphraser')
# About 7 ms per character with 2 CPU threads (generating and checking), so 2,500 characters take
# around 20 seconds; longer texts would keep a small server busy for too long
MAX_CHARS = int(os.environ.get('DEEP_REWRITE_MAX_CHARS', 2500))
CANDIDATES = 3            # sampled rewrites per sentence; the most faithful one is used
MAX_TOKENS = 160          # longer sentences are left to the rule-based layers

_model = None
_load_lock = threading.Lock()
_run_lock = threading.Lock()   # one job at a time: the CPU is the bottleneck
# Deep Rewrite requests allowed in flight: by default one running and one waiting. Each holds one of
# the web server's few threads (gunicorn.conf.py) for up to a minute, so further ones are turned away
# at once rather than queueing behind them and leaving no thread for anything else.
_slots = threading.BoundedSemaphore(int(os.environ.get('DEEP_REWRITE_QUEUE', 2)))


def available():
    return os.path.isfile(os.path.join(MODEL_DIR, 'model.bin')) and os.path.isfile(os.path.join(MODEL_DIR, 'humanizer.json'))


class Model:
    def __init__(self, path):
        import ctranslate2
        import sentencepiece
        with open(os.path.join(path, 'humanizer.json'), encoding='utf-8') as f:
            self.info = json.load(f)
        self.translator = ctranslate2.Translator(
            path, device='cpu', compute_type='int8', inter_threads=1,
            intra_threads=int(os.environ.get('PARAPHRASER_THREADS', 2)))
        self.sp = sentencepiece.SentencePieceProcessor(model_file=os.path.join(path, self.info['tokenizer_file']))

    def source(self, text):
        return self.sp.encode(self.info.get('prefix', '') + text, out_type=str) + ['</s>']

    def target(self, text):
        return self.sp.encode(text, out_type=str)

    def generate(self, sentences):
        """CANDIDATES sampled rewrites for each sentence (an empty list for sentences that are too long)."""
        encoded = [self.source(s) for s in sentences]
        usable = [k for k, tokens in enumerate(encoded) if len(tokens) <= MAX_TOKENS]
        batch = [encoded[k] for k in usable for _ in range(CANDIDATES)]
        results = self.translator.translate_batch(
            batch, beam_size=1, sampling_topk=40, sampling_temperature=0.8, repetition_penalty=1.2,
            no_repeat_ngram_size=3, max_decoding_length=MAX_TOKENS, max_batch_size=24) if batch else []
        outputs = [self.sp.decode(r.hypotheses[0]).strip() for r in results]
        candidates = [[] for _ in sentences]
        for n, k in enumerate(usable):
            candidates[k] = outputs[n * CANDIDATES:(n + 1) * CANDIDATES]
        return candidates

    def fidelity(self, pairs):
        """For each (rewrite, original) pair, how readily the model paraphrases the rewrite back into the
        original: its average log-probability per token. A rewrite that changed the meaning, dropped
        something or added something scores low, because the original no longer follows from it."""
        results = self.translator.score_batch([self.source(r) for r, _ in pairs], [self.target(o) for _, o in pairs],
                                              max_batch_size=24)
        return [sum(r.log_probs) / max(1, len(r.log_probs)) for r in results]


def get_model():
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                _model = Model(MODEL_DIR)
                logger.info('Deep Rewrite model loaded: %s', _model.info.get('name'))
    return _model


def preload():
    """Load the model in the background at startup, so the first Deep Rewrite isn't slow."""
    if not available():
        return

    def load():
        try:
            get_model()
        except Exception as e:
            logger.error('Deep Rewrite model failed to load: %s', type(e).__name__)
    threading.Thread(target=load, daemon=True).start()


def reserve():
    """Take a Deep Rewrite slot without waiting: False if they're all in use. Pair with release()."""
    return _slots.acquire(blocking=False)


def release():
    _slots.release()


def generate(sentences):
    with _run_lock:
        return get_model().generate(sentences)


def fidelity(pairs):
    with _run_lock:
        return get_model().fidelity(pairs)
