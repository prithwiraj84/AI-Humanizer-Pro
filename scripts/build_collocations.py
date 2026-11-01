"""
Builds data/bigrams.json.gz, the word-pair table the humanizer uses to check that a synonym fits
its neighbours ("land cover" is attested English, "land blanket" is not).

NLTK supplies the Brown corpus (about a million words of edited American English across 15
genres). Every pair of adjacent words seen at least twice is kept, lowercased, with its count.
At runtime the app only loads this table, which keeps memory low on small instances.

Run it whenever the selection rules change (needs: pip install nltk, then the "brown" corpus):
    python scripts/build_collocations.py
"""
import gzip
import json
import os
import re
from collections import Counter

from nltk.corpus import brown

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'bigrams.json.gz')
MIN_COUNT = 2
WORD = re.compile(r"^[a-z][a-z'-]*$")


def main():
    pairs = Counter()
    for sentence in brown.sents():
        words = [w.lower() for w in sentence]
        for first, second in zip(words, words[1:]):
            if WORD.match(first) and WORD.match(second):
                pairs[f'{first} {second}'] += 1
    table = {pair: min(count, 999) for pair, count in pairs.items() if count >= MIN_COUNT}
    with gzip.open(OUT_PATH, 'wt', encoding='utf-8') as f:
        json.dump(table, f, separators=(',', ':'), sort_keys=True)
    print(f'{len(table):,} word pairs saved to {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
