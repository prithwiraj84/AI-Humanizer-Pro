"""
Builds data/person_nouns.json.gz: nouns that name a person ("grandparent", "villager", "monk", "newlywed").

The structure layer splits a trailing "-ing" clause into a sentence of its own ("..., giving credit to families"
-> ". This gave credit to families"). When the sentence's subject is a person, the person did that, not "this",
so the split is skipped. NLTK's WordNet supplies the people, keeping single words:
  * whose first (most common) sense is a person (lexicographer file noun.person)
  * that are ordinary nouns in that sense, not a name ("Kennedy", "Smith")
  * with no artifact among their first three senses ("server", "printer" and "monitor" are as often machines)

Run it whenever the selection rules change (needs: pip install nltk, then the "wordnet" corpus):
    python scripts/build_person_nouns.py
"""
import gzip
import json
import os

from nltk.corpus import wordnet

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'person_nouns.json.gz')


def names_a_person(lemma):
    senses = wordnet.synsets(lemma, pos=wordnet.NOUN)
    if not senses or senses[0].lexname() != 'noun.person':
        return False
    if lemma not in {name.name() for name in senses[0].lemmas()}:
        return False    # only capitalised in that sense: a name
    return not any(sense.lexname() == 'noun.artifact' for sense in senses[:3])


def main():
    people = sorted(lemma for lemma in wordnet.all_lemma_names(pos=wordnet.NOUN)
                    if lemma.isalpha() and lemma.islower() and names_a_person(lemma))
    with gzip.open(OUT_PATH, 'wt', encoding='utf-8') as f:
        json.dump(people, f, separators=(',', ':'))
    print(f'{len(people)} nouns -> {OUT_PATH}')


if __name__ == '__main__':
    main()
