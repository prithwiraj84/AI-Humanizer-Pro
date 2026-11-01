"""
Builds data/synonyms.json.gz, the synonym table the humanizer reads at runtime.

NLTK does the heavy lifting here, offline:
  * WordNet supplies synonyms per word and part of speech, in sense-frequency order. Only the
    two most common senses are used, so swaps keep the usual meaning of a word, and a synonym
    must actually be used in that sense (WordNet's SemCor tag counts), which drops pairs like
    "result" -> "event" that share a sense on paper but not in real writing.
  * The Brown corpus supplies word frequencies. Candidates are ranked by them and rare or
    archaic words ("shew", "ameliorate") are dropped, since they make text sound less natural.
  * WordNet usage labels remove vulgar, slang and disparaging words.
lemminflect folds inflected forms ("shows", "showed") into each word's frequency.

At runtime the app loads only this table (a few MB) plus spaCy, which keeps the server inside
small instances such as Render's 512 MB free tier.

Each entry is  "lemma|pos": [lemma_frequency, [[candidate, frequency], ...]]  with pos in n/v/a/r.

Run it whenever the selection rules change (needs: pip install nltk lemminflect):
    python scripts/build_synonyms.py
"""
import gzip
import json
import os
from collections import Counter
from functools import lru_cache

import nltk
from lemminflect import getAllInflections

for corpus in ('wordnet', 'brown'):
    try:
        nltk.data.find(f'corpora/{corpus}.zip')
    except LookupError:
        nltk.download(corpus, quiet=True)

from nltk.corpus import brown, wordnet

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'synonyms.json.gz')

UPOS = {'n': 'NOUN', 'v': 'VERB', 'a': 'ADJ', 'r': 'ADV'}
MIN_WORD_LENGTH = 3
MIN_FREQUENCY = 3        # a candidate must appear at least this often (all forms) in Brown
MIN_SENSE_COUNT = 3      # ...and be tagged with the shared sense at least this often in SemCor
SECOND_SENSE_SHARE = 0.5 # use a word's 2nd sense only if it's tagged at least half as often as its 1st
# Very general words change meaning with context, so they are never swapped in or out
GENERIC_WORDS = {'be', 'bring', 'come', 'do', 'get', 'give', 'go', 'have', 'hold', 'keep', 'lay', 'let',
                 'make', 'put', 'run', 'set', 'stand', 'take', 'turn', 'break', 'thing', 'way'}
TOP_SENSES = 2
MAX_CANDIDATES = 6
EXCLUDED_USAGE_DOMAINS = ('vulgarism.n.01', 'obscenity.n.01', 'disparagement.n.01', 'slang.n.02', 'ethnic_slur.n.01')
EXTRA_BLOCKLIST = {'arse', 'bastard', 'bitch', 'bugger', 'crap', 'damn', 'piss', 'prick', 'screw', 'shit', 'whore'}
# Real synonyms whose connotation changes the message ("strategy" -> "scheme" sounds shady)
AWKWARD_SYNONYMS = {'scheme', 'grapevine', 'plot', 'racket', 'ploy', 'someone', 'somebody', 'soul', 'mortal'}
# Pairs that share a WordNet sense but not everyday usage ("organize your work" is not "form your work")
AWKWARD_PAIRS = {
    'organize': {'form'}, 'organise': {'form'}, 'implement': {'enforce'}, 'provide': {'render', 'furnish'},
    'demonstrate': {'present'}, 'render': {'provide', 'furnish'},
}
# Very common nouns and verbs have many meanings ("improve their results" is not "...their effects"),
# so words used at least this often in Brown are never swapped through WordNet
COMMON_WORD_CAP = {'n': 300, 'v': 600}


def flagged_words():
    flagged = set(EXTRA_BLOCKLIST)
    for name in EXCLUDED_USAGE_DOMAINS:
        try:
            domain = wordnet.synset(name)
        except Exception:
            continue
        for synset in domain.in_usage_domains():
            flagged.update(n.lower() for n in synset.lemma_names())
    return flagged


def main():
    counts = Counter(w.lower() for w in brown.words() if w.isalpha())
    blocked = flagged_words() | GENERIC_WORDS | AWKWARD_SYNONYMS

    @lru_cache(maxsize=None)
    def main_senses(word, pos):
        """The senses a word is mostly used in: its most-tagged sense in SemCor, plus its second
        if that is tagged at least half as often."""
        usage = sorted(((next(l.count() for l in s.lemmas() if l.name() == word), s)
                        for s in wordnet.synsets(word, pos=pos) if word in s.lemma_names()),
                       key=lambda item: -item[0])
        if not usage or usage[0][0] == 0:
            return frozenset()
        keep = [usage[0][1]]
        if len(usage) > 1 and usage[1][0] >= SECOND_SENSE_SHARE * usage[0][0]:
            keep.append(usage[1][1])
        return frozenset(keep)

    @lru_cache(maxsize=None)
    def frequency(word, pos):
        forms = {word}
        for variants in getAllInflections(word, upos=UPOS[pos]).values():
            forms.update(variants)
        return sum(counts[f] for f in forms)

    table = {}
    for pos in UPOS:
        for lemma in sorted(set(wordnet.all_lemma_names(pos=pos))):
            if len(lemma) < MIN_WORD_LENGTH or not lemma.isalpha() or not lemma.islower() or lemma in blocked:
                continue
            if pos in COMMON_WORD_CAP and frequency(lemma, pos) >= COMMON_WORD_CAP[pos]:
                continue
            senses = [s for s in wordnet.synsets(lemma, pos=pos) if lemma in s.lemma_names()][:TOP_SENSES]
            # Only senses the word itself is actually used in ("pipeline" is never used to mean "grapevine")
            usage = {s: next(l.count() for l in s.lemmas() if l.name() == lemma) for s in senses}
            senses = [s for s in senses if usage[s] >= 1]
            if len(senses) == 2 and usage[senses[1]] < SECOND_SENSE_SHARE * usage[senses[0]]:
                senses = senses[:1]
            ranked = {}
            for rank, synset in enumerate(senses):
                for entry in synset.lemmas():
                    name = entry.name()
                    if (name == lemma or name in ranked or not name.isalpha() or not name.islower()
                            or len(name) < MIN_WORD_LENGTH or name in blocked or entry.count() < MIN_SENSE_COUNT):
                        continue
                    # The shared meaning must be one the candidate is mainly used in, too: "organize"
                    # and "form" share a sense, but "form" mostly means "make up", so it isn't offered
                    if synset not in main_senses(name, pos) or name in AWKWARD_PAIRS.get(lemma, ()):
                        continue
                    freq = frequency(name, pos)
                    if freq >= MIN_FREQUENCY:
                        ranked[name] = (rank, -entry.count(), freq)
            if ranked:
                best = sorted(ranked.items(), key=lambda item: item[1][:2])[:MAX_CANDIDATES]
                table[f'{lemma}|{pos}'] = [frequency(lemma, pos), [[name, freq] for name, (_, _, freq) in best]]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with gzip.open(OUT_PATH, 'wt', encoding='utf-8', compresslevel=9) as f:
        json.dump(table, f, separators=(',', ':'), sort_keys=True)
    print(f'Wrote {len(table):,} entries to {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
