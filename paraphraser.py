"""
Deep Rewrite: sentence-by-sentence paraphrasing with small local models (no outside AI service).

Two kinds of model, both converted ahead of time to CTranslate2's 8-bit format by
scripts/build_paraphraser.py:
- data/paraphraser/ (or PARAPHRASER_DIR): a T5 paraphraser, about 225 MB on disk and 250 MB of memory.
  It writes rewrites and scores how faithful every candidate is, so Deep Rewrite needs it.
- data/backtranslate/ (or BACKTRANSLATE_DIR), optional: small OPUS-MT translation models, trained on
  human translations, that take each sentence into another language and back. They are a second
  source of rewrites whose wording comes from human-written text rather than from ChatGPT-style
  paraphrases (which the T5 was trained on). About 150 MB on disk and 200 MB of memory per language.
Without the T5, Deep Rewrite isn't offered and everything else works as before.

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
BACKTRANSLATE_DIR = os.environ.get('BACKTRANSLATE_DIR') or os.path.join(BASE_DIR, 'data', 'backtranslate')


def env_number(name, default, cast=int, minimum=0):
    """A numeric setting from the environment. An empty or malformed value falls back to the default with a
    warning, so a typo in the host's settings never stops the whole app from starting."""
    raw = os.environ.get(name, '').strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        logger.warning('%s=%r is not a number; using the default %s', name, raw, default)
        return default
    if not value >= minimum:        # also rejects NaN
        logger.warning('%s=%r is below %s; using the default %s', name, raw, minimum, default)
        return default
    return value


def usable_cpus():
    """CPUs this process may use: its CPU affinity, capped by a container's CPU quota (cgroup v2 or v1)."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    quota = None
    try:
        with open('/sys/fs/cgroup/cpu.max', encoding='ascii') as f:
            limit, period = f.read().split()[:2]
        if limit != 'max':
            quota = int(limit) / int(period)
    except (OSError, ValueError):
        try:
            with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', encoding='ascii') as f:
                limit = int(f.read())
            with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us', encoding='ascii') as f:
                period = int(f.read())
            if limit > 0 and period > 0:
                quota = limit / period
        except (OSError, ValueError):
            pass
    return min(count, quota) if quota else count


# Measured at 1.5 CPUs: 10 to 20 ms per character (generating, back-translating and checking), so 2,500
# characters take 25 to 50 seconds; longer texts would keep a small server busy for too long
MAX_CHARS = env_number('DEEP_REWRITE_MAX_CHARS', 2500, minimum=1)
CANDIDATES = 3            # sampled T5 rewrites per sentence
MAX_TOKENS = 160          # longer sentences are left to the rule-based layers
# A second thread only helps with two whole CPUs; on less it competes with the web server for the same CPU
THREADS = env_number('PARAPHRASER_THREADS', 2 if usable_cpus() >= 2 else 1, minimum=1)

_model = None
_backtranslator = None
_model_failed = False           # a load that failed once isn't retried on every request
_backtranslate_failed = False
_load_lock = threading.Lock()
_run_lock = threading.Lock()   # one job at a time: the CPU is the bottleneck
# Deep Rewrite requests allowed in flight: by default one running and one waiting. Each holds one of
# the web server's few threads (gunicorn.conf.py) for up to a minute, so further ones are turned away
# at once rather than queueing behind them and leaving no thread for anything else.
_slots = threading.BoundedSemaphore(env_number('DEEP_REWRITE_QUEUE', 2, minimum=1))
_in_flight = 0
_count_lock = threading.Lock()


def available():
    return (not _model_failed and os.path.isfile(os.path.join(MODEL_DIR, 'model.bin'))
            and os.path.isfile(os.path.join(MODEL_DIR, 'humanizer.json')))


def backtranslate_available():
    return not _backtranslate_failed and os.path.isfile(os.path.join(BACKTRANSLATE_DIR, 'humanizer.json'))


def plain_quotes(text):
    """The OPUS-MT vocabularies have no curly quotes; the caller restores the original style afterwards."""
    return text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')


class Model:
    """The T5 paraphraser: writes rewrites and scores how likely one text is as a paraphrase of another."""

    def __init__(self, path):
        import ctranslate2
        import sentencepiece
        with open(os.path.join(path, 'humanizer.json'), encoding='utf-8') as f:
            self.info = json.load(f)
        self.translator = ctranslate2.Translator(path, device='cpu', compute_type='int8', inter_threads=1,
                                                 intra_threads=THREADS)
        self.sp = sentencepiece.SentencePieceProcessor(model_file=os.path.join(path, self.info['tokenizer_file']))

    def source(self, text):
        return self.sp.encode(self.info.get('prefix', '') + text, out_type=str) + ['</s>']

    def target(self, text):
        return self.sp.encode(text, out_type=str)

    def generate(self, sentences, n=CANDIDATES, temperature=0.8, topk=40):
        """n sampled rewrites for each sentence (an empty list for sentences that are too long)."""
        encoded = [self.source(s) for s in sentences]
        usable = [k for k, tokens in enumerate(encoded) if len(tokens) <= MAX_TOKENS]
        batch = [encoded[k] for k in usable for _ in range(n)]
        results = self.translator.translate_batch(
            batch, beam_size=1, sampling_topk=topk, sampling_temperature=temperature, repetition_penalty=1.2,
            no_repeat_ngram_size=3, max_decoding_length=MAX_TOKENS, max_batch_size=24) if batch else []
        outputs = [self.sp.decode(r.hypotheses[0]).strip() for r in results]
        candidates = [[] for _ in sentences]
        for j, k in enumerate(usable):
            candidates[k] = outputs[j * n:(j + 1) * n]
        return candidates

    def score(self, pairs):
        """For each (given, wanted) pair, how readily the model paraphrases `given` into `wanted`:
        (average log-probability per token, the worst single token's, the total). With (rewrite,
        original) this is the round trip: a rewrite that changed the meaning, dropped something or
        added something scores low, because the original no longer follows from it. With (original,
        rewrite) it is the forward check: garbled or off-topic rewrites score low."""
        results = self.translator.score_batch([self.source(g) for g, _ in pairs], [self.target(w) for _, w in pairs],
                                              max_batch_size=24)
        out = []
        for r in results:
            probs = r.log_probs or [0.0]
            out.append((sum(probs) / len(probs), min(probs), sum(probs)))
        return out


class Marian:
    """One OPUS-MT translation direction."""

    def __init__(self, path):
        import ctranslate2
        import sentencepiece
        self.translator = ctranslate2.Translator(path, device='cpu', compute_type='int8', inter_threads=1,
                                                 intra_threads=THREADS)
        self.source = sentencepiece.SentencePieceProcessor(model_file=os.path.join(path, 'source.spm'))
        self.target = sentencepiece.SentencePieceProcessor(model_file=os.path.join(path, 'target.spm'))

    def translate(self, texts, n_best=1):
        """The n_best beam-search translations of each text (beam search keeps names and numbers far
        better than sampling); none for a text too long for the model."""
        encoded = [self.source.encode(plain_quotes(t), out_type=str) + ['</s>'] for t in texts]
        usable = [k for k, tokens in enumerate(encoded) if len(tokens) <= MAX_TOKENS]
        results = self.translator.translate_batch([encoded[k] for k in usable], beam_size=4, num_hypotheses=n_best,
                                                  max_batch_size=32, max_decoding_length=MAX_TOKENS) if usable else []
        out = [[] for _ in texts]
        for k, r in zip(usable, results):
            out[k] = [self.target.decode([p for p in h if p != '<unk>']).strip() for h in r.hypotheses]
        return out


class BackTranslator:
    """English -> pivot language -> English, for each installed pivot (humanizer.json lists them)."""

    def __init__(self, path):
        with open(os.path.join(path, 'humanizer.json'), encoding='utf-8') as f:
            self.info = json.load(f)
        self.pivots = [(p['lang'], Marian(os.path.join(path, p['forward'])), Marian(os.path.join(path, p['back'])))
                       for p in self.info['pivots']]

    def rewrites(self, sentences):
        """Up to 2 candidates per sentence and pivot: the best translation, translated back 2 ways."""
        candidates = [[] for _ in sentences]
        for _, forward, back in self.pivots:
            pivots = forward.translate(sentences, n_best=1)
            flat = [(k, p) for k, options in enumerate(pivots) for p in dict.fromkeys(options) if p]
            returned = back.translate([p for _, p in flat], n_best=2)
            for (k, _), options in zip(flat, returned):
                candidates[k] += options
        return [list(dict.fromkeys(c for c in options if c)) for options in candidates]


def get_model():
    """The T5, loaded on first use. A failed load is remembered, so available() turns Deep Rewrite off
    instead of every request failing the same way."""
    global _model, _model_failed
    if _model is None:
        with _load_lock:
            if _model is None:
                try:
                    _model = Model(MODEL_DIR)
                except Exception as e:
                    _model_failed = True
                    logger.error('Deep Rewrite model failed to load from %s: %s; Deep Rewrite is off',
                                 MODEL_DIR, type(e).__name__)
                    raise
                logger.info('Deep Rewrite model loaded: %s', _model.info.get('name'))
    return _model


def get_backtranslator():
    """The back-translation models, or None if they failed to load (Deep Rewrite then uses the T5 only)."""
    global _backtranslator, _backtranslate_failed
    if _backtranslator is None and not _backtranslate_failed:
        with _load_lock:
            if _backtranslator is None and not _backtranslate_failed:
                try:
                    _backtranslator = BackTranslator(BACKTRANSLATE_DIR)
                except Exception as e:
                    _backtranslate_failed = True
                    logger.error('Back-translation failed to load from %s: %s; Deep Rewrite continues with the T5 only',
                                 BACKTRANSLATE_DIR, type(e).__name__)
                    return None
                logger.info('Back-translation loaded: %s', ', '.join(lang for lang, _, _ in _backtranslator.pivots))
    return _backtranslator


def is_loaded():
    """True once the T5 is in memory (so scoring doesn't trigger a slow first load)."""
    return _model is not None


def preload():
    """Load the models in the background at startup, one after the other, so the first Deep Rewrite isn't slow."""
    if not available():
        return

    def load():
        try:
            get_model()
        except Exception:
            return      # logged by get_model
        if backtranslate_available():
            get_backtranslator()
    threading.Thread(target=load, daemon=True).start()


def reserve():
    """Take a Deep Rewrite slot without waiting: False if they're all in use. Pair with release()."""
    global _in_flight
    if not _slots.acquire(blocking=False):
        return False
    with _count_lock:
        _in_flight += 1
    return True


def release():
    global _in_flight
    with _count_lock:
        _in_flight -= 1
    _slots.release()


def in_flight():
    """Deep Rewrite requests running or waiting now; with more than one, each skips its optional extras."""
    return _in_flight


def generate(sentences, n=CANDIDATES, temperature=0.8, topk=40):
    with _run_lock:
        return get_model().generate(sentences, n, temperature, topk)


def score(pairs):
    with _run_lock:
        return get_model().score(pairs)


def try_score(pairs, wait=0.25):
    """score(), unless another job is using the model: then None after at most `wait` seconds."""
    if not _run_lock.acquire(timeout=wait):
        return None
    try:
        return get_model().score(pairs)
    finally:
        _run_lock.release()


def backtranslate(sentences):
    """Back-translated rewrites of each sentence; none if the models are missing or fail on this input."""
    if not backtranslate_available():
        return [[] for _ in sentences]
    with _run_lock:
        translator = get_backtranslator()
        if translator is None:
            return [[] for _ in sentences]
        try:
            return translator.rewrites(sentences)
        except Exception as e:
            logger.warning('Back-translation failed on this text: %s', type(e).__name__)
            return [[] for _ in sentences]
