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

It also builds data/noun_senses.json.gz, the nouns whose main WordNet sense is of a kind that some
curated swaps depend on ("encourage the witness" can't become "promote the witness", "reduce the
cream" can't become "cut the cream", and "the cornerstone of the factory" is the stone itself), as
kind: [noun, ...].

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
SENSES_PATH = os.path.join(os.path.dirname(OUT_PATH), 'noun_senses.json.gz')

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
# Identity, culture and belief words: swapping them changes what is said ("cultural events" are not
# "ethnic events", "religious" is not "spiritual"), so they are never swapped in or out
PROTECTED_WORDS = {'cultural', 'ethnic', 'racial', 'religious', 'spiritual', 'national', 'tribal', 'political',
                   'historic', 'historical', 'native', 'foreign', 'domestic', 'civil', 'patriotic', 'sacred', 'holy',
                   'secular', 'sexual', 'brave'}
# Pairs that share a WordNet sense but not everyday usage ("organize your work" is not "form your work")
AWKWARD_PAIRS = {
    'organize': {'form'}, 'organise': {'form'}, 'implement': {'enforce'}, 'provide': {'render', 'furnish'},
    'demonstrate': {'present'}, 'render': {'provide', 'furnish'},
    # "observe a holiday" is not "notice a holiday"; SemCor rarely tags that sense, so it isn't caught above
    'observe': {'notice', 'find', 'detect', 'discover'},
    # Found by reading rewrites of typical AI-written texts: each swap picks a sense the text rarely means
    # ("the results suggest" is not "propose", "presents challenges" is not "shows", "extreme weather" is
    # not "utmost weather", "each entry" in a cache is not "each introduction")
    'suggest': {'propose'}, 'present': {'show', 'demonstrate'}, 'extreme': {'utmost'}, 'learn': {'discover'},
    'regulate': {'govern'}, 'combine': {'compound'}, 'help': {'aid'}, 'note': {'remark'}, 'strengthen': {'fortify'},
    'build': {'construct'}, 'engage': {'pursue'}, 'entire': {'total'}, 'strike': {'impress'}, 'raise': {'lift'},
    'entry': {'debut', 'introduction'}, 'load': {'loading'}, 'manager': {'director', 'coach'},
    'transition': {'passage'}, 'technology': {'engineering'}, 'legacy': {'bequest'},
    'discover': {'observe', 'notice', 'detect', 'learn'},     # "discovering new music" is finding it
    # Found by an adversarial review: the only candidate WordNet offers is another sense ("financial security"
    # is not "fiscal security", "active noise cancellation" is not "fighting", "marked a turning point" is not
    # "tagged", "saving money" is not "salvaging", "the famous bullet trains" are not "noted", "small changes"
    # are not "little changes", "living organisms" are not "populating", photosynthesis is a process, not a
    # "procedure", and "contributes a lot to" is not "lends a lot to")
    'financial': {'fiscal'}, 'fiscal': {'financial'}, 'active': {'fighting'}, 'fighting': {'active'},
    'famous': {'noted'}, 'famed': {'noted'}, 'illustrious': {'noted'}, 'mark': {'tag'}, 'save': {'salvage'},
    'enjoy': {'savor', 'savour', 'bask'}, 'develop': {'evolve'}, 'live': {'populate', 'inhabit'},
    'small': {'little'}, 'little': {'small'}, 'process': {'procedure'}, 'indicate': {'signal', 'point'},
    'contribute': {'lend'}, 'researcher': {'investigator'},
    # Found by a second review: "did not simply give up" is not "did not only give up", "And finally, we discuss"
    # is not "eventually", "a large number of" is not "a big number of", "books were copied by hand" were not
    # "simulated", "sunny" is not "gay", "curious travellers" are not "funny", "launch the product" is not "found"
    # it, "when a node fails" is not "neglects", "randomly assigned" is not "arbitrarily", "connect your bank
    # account" is not "associate" it, "voted to approve" is not "sanction"
    'simply': {'only'}, 'merely': {'only'}, 'finally': {'eventually'}, 'large': {'big'},
    'copy': {'simulate', 'imitate'}, 'particular': {'peculiar', 'special'}, 'study': {'examine', 'analyze', 'analyse'},
    'enough': {'plenty'}, 'secure': {'procure'}, 'meet': {'encounter'}, 'derive': {'deduce'},
    'further': {'foster'}, 'apply': {'utilize', 'employ', 'use'}, 'employ': {'apply', 'use', 'utilize'},
    'form': {'organize', 'organise'}, 'sunny': {'gay', 'cheery'},
    'curious': {'funny', 'peculiar', 'odd', 'queer', 'rum', 'singular'}, 'aged': {'elderly', 'older', 'old', 'senior'},
    'launch': {'found', 'establish'}, 'pull': {'draw'}, 'completely': {'all', 'altogether', 'whole'},
    'fail': {'neglect', 'betray', 'flunk'}, 'notice': {'find', 'discover', 'detect'},
    'randomly': {'arbitrarily', 'haphazardly'}, 'badly': {'poorly', 'ill'}, 'approve': {'sanction'},
    'connect': {'associate', 'relate', 'colligate'},
    # Found by a third review: "every sample they collect" is not "accumulate" or "compile", "otherwise it will
    # melt" is not "thaw", "a particular set" is not "a special set", "The effort paid off" is not "The try", and
    # exercising (a workout) is neither practising a skill nor exerting, nor the other way round
    'collect': {'accumulate', 'compile'}, 'melt': {'thaw'}, 'effort': {'try'}, 'exercise': {'practice', 'exert'},
    'practice': {'exercise'},
}
# Adverbs that complete a phrasal verb ("go away", "put off", "melt away"): their synonyms make another phrasal
# verb ("go off", "put away"), so they are neither swapped nor offered
PARTICLES = {'away', 'forth', 'off', 'out', 'up', 'down', 'back', 'along', 'around', 'over', 'apart', 'aside', 'ahead',
             'forward', 'together', 'behind', 'onward', 'onwards', 'forwards', 'backward', 'backwards'}
# Very common nouns and verbs have many meanings ("improve their results" is not "...their effects"),
# so words used at least this often in Brown are never swapped through WordNet
COMMON_WORD_CAP = {'n': 300, 'v': 600}
# Kinds of noun for data/noun_senses.json.gz: kind -> (how many of a noun's first senses may count, and the test
# a sense must pass). Most kinds count the main sense only: "spirit" is a vital principle before it is a person,
# a "holder" is a device before it is an owner, and "the system" is not a body part; "cream" (the elite before
# the food) and "specialist" (an expert before a doctor) need the second. A kind is a WordNet file ("noun.body")
# or the synsets a sense must come under: a person is one under person.n.01 or a vehicle's operator ("driver"),
# and a body of people is a crowd, a population or the like ("audience", "the next generation"), not an
# institution ("governance"). A group is a body of people that acts as one, an organization, a unit, a club or a
# party ("the leadership", "the membership", "the regiment", "the coalition"), in its first two senses because
# "leadership" and "administration" name an act first; a named one ("the Beatles") is a name, not a kind. An animal
# is one a vet sees, not a microbe ("the bacteria" are studied). A building has a roof and walls, or is a mill, a
# tower, a bridge, a monument, a stadium, a fort, a warehouse or a prison, not any structure or facility ("our
# customer base", "public transportation"). A stone is a gem, a crystal, a rock or another solid with cut faces
# ("the prism", "the glass").
PEOPLE = ('person.n.01', 'operator.n.02')
PEOPLE_GROUPS = ('people.n.01', 'gathering.n.01', 'multitude.n.03', 'citizenry.n.01')
PEOPLE_BODIES = ('body.n.02', 'unit.n.03', 'organization.n.01', 'club.n.02', 'political_party.n.01')
CLINICIANS = ('health_professional.n.01', 'medical_practitioner.n.01', 'therapist.n.01', 'paramedic.n.01',
              'optician.n.01', 'optometrist.n.01', 'clinician.n.01')
MICROBES = ('microorganism.n.01',)
BUILDINGS = ('building.n.01', 'building_complex.n.01', 'tower.n.01', 'bridge.n.01', 'memorial.n.03', 'stadium.n.01',
             'fortification.n.01', 'military_post.n.01', 'storehouse.n.01', 'penal_institution.n.01')
STONES = ('gem.n.02', 'jewel.n.01', 'crystal.n.01', 'mineral.n.01', 'rock.n.01', 'rock.n.02', 'precious_stone.n.01',
          'transparent_gem.n.01', 'polyhedron.n.01', 'glass.n.01')
# ...less the nouns whose kind is WordNet's first sense but rarely the one meant: "a cornerstone of modern
# architecture" is the field, not a building
NOT_OF_KIND = {'building': {'architecture'}}
# ...plus the nouns whose sense of that kind WordNet lacks: "the opposition" is the parties out of power
ALSO_OF_KIND = {'group': {'opposition'}}


def noun_kinds():
    """kind -> (number of senses that may count, test on a synset and the synsets above it)."""
    def under(*names):
        roots = {wordnet.synset(name) for name in names}
        return lambda synset, above: bool((above | {synset}) & roots)

    def in_file(name, also=lambda synset, above: True):
        return lambda synset, above: synset.lexname() == name and also(synset, above)

    def kind_under(*names):
        """A kind of thing under the synsets, not a named one ("the Beatles", "Greenpeace")."""
        test = under(*names)
        return lambda synset, above: not synset.instance_hypernyms() and test(synset, above)

    microbe = under(*MICROBES)
    return {
        'person': (1, in_file('noun.person', under(*PEOPLE))),
        'people': (1, in_file('noun.group', under(*PEOPLE_GROUPS))),
        'group': (2, in_file('noun.group', kind_under(*PEOPLE_BODIES))),
        'clinician': (2, under(*CLINICIANS)),
        'body': (1, in_file('noun.body')),
        'animal': (1, in_file('noun.animal', lambda synset, above: not microbe(synset, above))),
        'food': (2, in_file('noun.food')),
        'building': (1, under(*BUILDINGS)),
        'stone': (2, under(*STONES)),
    }


def build_noun_senses():
    kinds = noun_kinds()
    found = {kind: set() for kind in kinds}
    for noun in sorted(set(wordnet.all_lemma_names(pos='n'))):
        if not noun.isalpha():
            continue
        senses = [s for s in wordnet.synsets(noun, pos='n') if noun in {n.lower() for n in s.lemma_names()}]
        for rank, synset in enumerate(senses[:max(depth for depth, _ in kinds.values())]):
            above = set(synset.closure(lambda s: s.hypernyms() + s.instance_hypernyms()))
            for kind, (depth, test) in kinds.items():
                if rank < depth and test(synset, above):
                    found[kind].add(noun)
    for kind, nouns in NOT_OF_KIND.items():
        found[kind] -= nouns
    for kind, nouns in ALSO_OF_KIND.items():
        found[kind] |= nouns
    with gzip.open(SENSES_PATH, 'wt', encoding='utf-8', compresslevel=9) as f:
        json.dump({kind: sorted(nouns) for kind, nouns in found.items()}, f, separators=(',', ':'), sort_keys=True)
    print(f'Wrote {sum(map(len, found.values())):,} nouns to {SENSES_PATH} ({os.path.getsize(SENSES_PATH) / 1024:.0f} KB)')


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
    blocked = flagged_words() | GENERIC_WORDS | AWKWARD_SYNONYMS | PROTECTED_WORDS

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
            if pos == 'r' and lemma in PARTICLES:
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
                            or len(name) < MIN_WORD_LENGTH or name in blocked or entry.count() < MIN_SENSE_COUNT
                            or (pos == 'r' and name in PARTICLES)):
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
    build_noun_senses()


if __name__ == '__main__':
    main()
