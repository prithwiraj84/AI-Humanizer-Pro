"""
Builds data/nominalizations.json.gz: action nouns and the verb they come from
("integration" -> "integrate", "analysis" -> "analyze").

AI text leans on noun-heavy phrasing ("the integration of X has transformed Y"); people more
often use the verb ("integrating X has transformed Y"). The humanizer uses this table to make
that edit. NLTK's WordNet supplies the noun-verb links (derivationally related forms), keeping:
  * nouns whose first or second sense is an act or process, as used in "the N of X"
  * verbs that take a direct object, so "the N of X" can become "N-ing X"
    ("the arrival of the train" can't become "arriving the train")
  * words used often enough in the Brown corpus to read naturally

Run it whenever the selection rules change (needs: pip install nltk, then the "wordnet" and "brown" corpora):
    python scripts/build_nominalizations.py
"""
import gzip
import json
import os
from collections import Counter

from nltk.corpus import brown, wordnet

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'nominalizations.json.gz')
SUFFIXES = ('tion', 'sion', 'ment', 'ance', 'ence', 'ysis', 'al', 'ure', 'age')
ACTION_SENSES = ('noun.act', 'noun.process', 'noun.event')
TRANSITIVE_FRAMES = {8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 31}   # "Somebody ----s something" and kin
MIN_NOUN_COUNT = 3
# Nouns usually followed by the doer, not the thing acted on ("the decision of the court", "the
# participation of citizens"), or mostly used for a thing rather than an act ("location", "constitution")
EXCLUDED = set("""
    absence accordance accompaniment acceleration approval arousal assistance association attendance commitment
    conception constitution contraction contribution conviction decision delegation departure devotion disposition
    encouragement enjoyment entertainment entrance evolution exhibition explosion exposure failure government
    graduation guidance inheritance instruction instrumentation involvement judgment location movement obedience
    observance occupation opposition participation permission possession prostitution qualification relaxation
    rental shrinkage
""".split())
MIN_VERB_COUNT = 5


def main():
    counts = Counter(w.lower() for w in brown.words() if w.isalpha())
    table = {}
    for noun in sorted(set(wordnet.all_lemma_names('n'))):
        if (not noun.isalpha() or not noun.islower() or not noun.endswith(SUFFIXES) or counts[noun] < MIN_NOUN_COUNT
                or noun in EXCLUDED):
            continue
        # Its main sense, or second if that one is an act/process ("improvement" is first "a change for the
        # better", second "the act of improving")
        senses = [s for s in wordnet.synsets(noun, 'n')[:2] if s.lexname() in ACTION_SENSES]
        if not senses:
            continue
        candidates = Counter()
        for lemma in (l for s in senses for l in s.lemmas()):
            if lemma.name() != noun:
                continue
            for related in lemma.derivationally_related_forms():
                verb = related.name()
                if related.synset().pos() != 'v' or not verb.isalpha() or verb[:3] != noun[:3] or verb == noun:
                    continue
                if not TRANSITIVE_FRAMES & set(related.frame_ids()) or counts[verb] + counts[verb + 's'] < MIN_VERB_COUNT:
                    continue
                candidates[verb] += related.count() + 1
        if candidates:
            table[noun] = candidates.most_common(1)[0][0]
    with gzip.open(OUT_PATH, 'wt', encoding='utf-8') as f:
        json.dump(table, f, separators=(',', ':'), sort_keys=True)
    print(f'{len(table):,} nouns saved to {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
