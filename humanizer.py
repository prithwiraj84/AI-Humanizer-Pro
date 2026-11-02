"""
Humanizer engine: rewrites text in a chosen tone and measures what changed.

AI-written text is recognisable by its stock phrases, its predictable word choices and its even
rhythm: sentences of similar length that all open the same way. Each layer below works on one
of those, using spaCy (tags, lemmas, entities, dependency parse) and data built offline with
NLTK (WordNet synonyms and Brown corpus word pairs), so the server needs no language model.

Pipeline (perform_nuclear_chaos):
  1. Phrases       - AI-style transitions, fillers and stock phrases ("Furthermore,", "plays a
                     crucial role in", "not only X but also Y") become plainer wording.
  2. Contractions  - Casual uses them, Standard some of them; Professional and Academic expand them.
  3. Words         - AI-favourite words get curated replacements; other content words get WordNet
                     synonyms of the same part of speech and sense, re-inflected ("demonstrates" ->
                     "shows"), and only if they fit their neighbours in real English (Brown pairs).
  4. Structure     - passive sentences turn active, adverbial phrases and clauses move between the
                     start and the end of their sentence, and safe split points are found.
  5. Rhythm        - long sentences are split and short neighbours joined until sentence lengths
                     vary the way they do in human writing; repeated sentence openings are avoided.
  6. Punctuation   - fewer em dashes; spacing, capitals and "a"/"an" are tidied.
Names, places, organisations, dates, numbers, acronyms, quotes and citations are never changed,
and headings and list items only get word-level edits.
If spaCy isn't installed, only the safe curated swaps of steps 1-3 run, so the app still works.
"""
import difflib
import gzip
import html
import json
import logging
import os
import random
import re
import threading
from collections import Counter

logger = logging.getLogger(__name__)

TONES = ('standard', 'professional', 'casual', 'academic')

# --- SYNONYM TABLE (built offline with NLTK by scripts/build_synonyms.py) ---
# "lemma|pos" -> [lemma_frequency, [[synonym, frequency], ...]], best synonyms first
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
with gzip.open(os.path.join(DATA_DIR, 'synonyms.json.gz'), 'rt', encoding='utf-8') as _f:
    SYNONYMS = json.load(_f)

# --- WORD PAIRS (built offline with NLTK from the Brown corpus by scripts/build_collocations.py) ---
# "first second" -> how often the two words appear side by side; ~9 MB in memory
with gzip.open(os.path.join(DATA_DIR, 'bigrams.json.gz'), 'rt', encoding='utf-8') as _f:
    BIGRAMS = json.load(_f)

def pair_count(first, second):
    return BIGRAMS.get(f'{first} {second}', 0) if first and second else 0

# --- ACTION NOUNS (built offline with NLTK from WordNet by scripts/build_nominalizations.py) ---
# "integration" -> "integrate": nouns that read better as the verb ("integrating X")
with gzip.open(os.path.join(DATA_DIR, 'nominalizations.json.gz'), 'rt', encoding='utf-8') as _f:
    NOMINALIZATIONS = json.load(_f)

# --- SPACY (optional at runtime) ---
try:
    import spacy
    from lemminflect import getInflection
    NLP = spacy.load('en_core_web_sm')
    NLP('Warm up the pipeline.')
    getInflection('warm', tag='VBZ')  # loads lemminflect's tables before request threads use them
except Exception as e:
    logger.warning("spaCy engine unavailable (%s); falling back to basic word swaps", e)
    NLP = None

# spaCy pipelines aren't guaranteed thread-safe, and gunicorn serves this app with threads
_NLP_LOCK = threading.Lock()

# ==========================================
#   SHARED HELPERS
# ==========================================

def pick(options, tone):
    return random.choice(options.get(tone) or options['*'])

def match_case(source, word):
    if len(source) > 1 and source.isupper():
        return word.upper()
    if source[0].isupper():
        return word[0].upper() + word[1:]
    return word

def phrase_regex(phrases):
    # Any of the phrases, across any whitespace, with straight or curly apostrophes
    parts = [r"\s+".join(re.escape(word) for word in phrase.split()).replace("'", "['’]")
             for phrase in sorted(phrases, key=len, reverse=True)]
    return "(?:" + "|".join(parts) + ")"

# Start of a sentence: start of a line, or whitespace after . ! ? (optionally after a closing quote)
SENT_START = r"(?P<pre>^[ \t]*|(?<=[.!?])\s+|(?<=[.!?][\"”’)\]])\s+)"

# ==========================================
#   STEP 1: PHRASE RULES
# ==========================================

# (sentence openers, chance of rewriting, replacements by tone)
TRANSITIONS = [
    (('furthermore', 'moreover', 'additionally', 'in addition', "what's more"), 1.0, {
        'standard': ['Also,', 'Plus,', 'On top of that,', 'Beyond that,'],
        'casual': ['Also,', 'Plus,', 'On top of that,'],
        'professional': ['Also,', 'Beyond that,', 'In addition,'],
        'academic': ['In addition,', 'Further,', 'Beyond this,'],
    }),
    (('in conclusion', 'to conclude', 'in summary', 'to summarize', 'to sum up'), 1.0, {
        'standard': ['Overall,', 'In short,', 'All in all,'],
        'casual': ['All in all,', 'So,', 'In short,'],
        'professional': ['Overall,', 'In short,'],
        'academic': ['Overall,', 'In sum,'],
    }),
    (('consequently', 'therefore', 'thus', 'hence'), 0.6, {
        'standard': ['So', 'As a result,', 'Because of this,'],
        'casual': ['So', 'Because of that,'],
        'professional': ['As a result,', 'Because of this,'],
        'academic': ['As a result,', 'Accordingly,'],
    }),
    (('however', 'nevertheless', 'nonetheless'), 0.4, {
        'standard': ['Still,', 'That said,', 'But'],
        'casual': ['But', 'Still,', 'That said,'],
        'professional': ['That said,', 'Still,'],
        'academic': ['Nevertheless,', 'Even so,'],
    }),
    (('on the other hand',), 0.8, {
        'standard': ['Then again,', 'By contrast,'],
        'casual': ['Then again,', 'But'],
        'professional': ['By contrast,', 'Then again,'],
        'academic': ['By contrast,', 'Conversely,'],
    }),
    (('ultimately', 'in essence', 'essentially'), 0.8, {
        'standard': ['In the end,', 'All told,'],
        'casual': ['In the end,', 'At the end of the day,', 'Basically,'],
        'professional': ['In the end,', 'All told,'],
        'academic': ['Ultimately,', 'In the end,'],
    }),
    (('first and foremost', 'firstly'), 1.0, {'*': ['First,']}),
    (('secondly',), 1.0, {'*': ['Second,']}),
    (('thirdly',), 1.0, {'*': ['Third,']}),
    (('last but not least', 'lastly'), 1.0, {'*': ['Finally,', 'Lastly,']}),
]
TRANSITION_RULES = [(re.compile(SENT_START + "(?:" + phrase_regex(words) + r")\s*,\s*", re.I | re.M), chance, options)
                    for words, chance, options in TRANSITIONS]

# Empty openers that add nothing; dropped, except Professional/Academic keep a short signpost
FILLERS = phrase_regex([
    "it is important to note that", "it's important to note that", "it is important to remember that",
    "it's important to remember that", "it is worth noting that", "it's worth noting that",
    "it is worth mentioning that", "it's worth mentioning that", "it should be noted that",
    "it is crucial to note that", "it is essential to note that", "it is interesting to note that",
    "it goes without saying that", "needless to say", "it's no secret that", "it is no secret that",
])
FILLER_AT_START = re.compile(SENT_START + FILLERS + r",?\s+(?P<first>\w)", re.I | re.M)
FILLER_INSIDE = re.compile(r"\s+" + FILLERS + r",?(?=\s)", re.I)
FILLER_SIGNPOSTS = {'professional': 'Note that ', 'academic': 'Notably, '}

# Wordy or overused phrases, rewritten anywhere in a sentence
PHRASES = [
    (("in order to",), {'*': ['to']}),
    (("due to the fact that", "owing to the fact that", "in light of the fact that"),
     {'*': ['because'], 'academic': ['because', 'since']}),
    (("at this point in time", "at the present time"),
     {'*': ['now'], 'professional': ['currently'], 'academic': ['currently']}),
    (("in today's fast-paced world", "in today's digital age", "in today's world", "in the modern world"),
     {'*': ['today', 'these days'], 'casual': ['these days', 'nowadays'], 'professional': ['today'], 'academic': ['today']}),
    (("a wide range of", "a wide variety of", "a broad range of", "a diverse range of"),
     {'standard': ['many', 'a variety of'], 'casual': ['all kinds of', 'lots of'],
      'professional': ['a variety of', 'many'], 'academic': ['a variety of', 'diverse']}),
    (("a plethora of", "a myriad of", "a multitude of", "an array of"),
     {'standard': ['plenty of', 'many'], 'casual': ['tons of', 'lots of'],
      'professional': ['many', 'numerous'], 'academic': ['numerous', 'many']}),
    (("a significant amount of", "a considerable amount of", "a substantial amount of"),
     {'standard': ['a lot of', 'much'], 'casual': ['a lot of', 'loads of'],
      'professional': ['considerable', 'substantial'], 'academic': ['considerable', 'substantial']}),
    (("the vast majority of",), {'*': ['most']}),
    (("each and every",), {'*': ['every']}),
    (("is a testament to", "stands as a testament to"),
     {'*': ['says a lot about'], 'professional': ['speaks to'], 'academic': ['attests to']}),
    (("are a testament to", "stand as a testament to"),
     {'*': ['say a lot about'], 'professional': ['speak to'], 'academic': ['attest to']}),
    (("a testament to",), {'*': ['proof of', 'a sign of'], 'academic': ['evidence of']}),
    (("cutting-edge", "state-of-the-art"), {'*': ['modern', 'advanced'], 'casual': ['modern', 'up-to-date'], 'academic': ['advanced']}),
    (("ever-evolving", "ever-changing"), {'*': ['changing'], 'academic': ['evolving']}),
    (("in the realm of",), {'*': ['in']}),
    (("navigate the complexities of",), {'*': ['deal with', 'handle']}),
    (("navigates the complexities of",), {'*': ['deals with', 'handles']}),
    (("navigating the complexities of",), {'*': ['dealing with', 'handling']}),
    (("unlock the full potential of", "unlock the potential of"), {'*': ['make the most of']}),
    (("unlocks the full potential of", "unlocks the potential of"), {'*': ['makes the most of']}),
    (("unlocking the full potential of", "unlocking the potential of"), {'*': ['making the most of']}),
    (("with regard to", "with regards to", "in regard to", "in regards to"),
     {'*': ['about', 'on'], 'professional': ['regarding'], 'academic': ['regarding', 'concerning']}),
    (("in the event that",), {'*': ['if']}),
    (("prior to",), {'*': ['before']}),
    (("a crucial role", "a pivotal role", "a vital role", "an essential role", "a significant role", "an integral role",
      "an instrumental role", "a critical role"),
     {'standard': ['a big part', 'a key role', 'a major part'], 'casual': ['a big part', 'a huge part'],
      'professional': ['a key role', 'a major role'], 'academic': ['a central role', 'a major role']}),
    (("we are pleased to announce that", "we are excited to announce that", "we are thrilled to announce that",
      "we're pleased to announce that", "we're excited to announce that", "we're thrilled to announce that"),
     {'standard': ["we're happy to share that", 'good news:'], 'casual': ["we're happy to say that", 'great news:'],
      'professional': ['we are glad to report that', 'we are happy to share that'], 'academic': ['we report that']}),
    (("serves as",), {'*': ['acts as', 'works as'], 'academic': ['functions as', 'acts as']}),
    (("serve as",), {'*': ['act as', 'work as'], 'academic': ['function as', 'act as']}),
    (("served as",), {'*': ['acted as', 'worked as'], 'academic': ['functioned as', 'acted as']}),
    (("embark on", "embark upon"), {'*': ['start', 'begin'], 'casual': ['start', 'kick off']}),
    (("embarks on", "embarks upon"), {'*': ['starts', 'begins']}),
    (("embarked on", "embarked upon"), {'*': ['started', 'began']}),
    (("embarking on", "embarking upon"), {'*': ['starting', 'beginning']}),
    (("in a timely manner",), {'*': ['promptly', 'on time'], 'casual': ['quickly', 'on time']}),
    (("a key aspect of", "an important aspect of", "a crucial aspect of", "a vital aspect of"),
     {'*': ['a key part of', 'an important part of'], 'casual': ['a big part of'],
      'academic': ['a central part of', 'an important part of']}),
    (("it is evident that", "it is clear that", "it is apparent that", "it's clear that", "it's evident that"),
     {'*': ['clearly,'], 'casual': ['clearly,', 'obviously,']}),
    (("has the potential to", "have the potential to"), {'*': ['could', 'can'], 'academic': ['could', 'may']}),
    (("is able to", "are able to"), {'*': ['can']}),
    (("it is essential to", "it is crucial to", "it is vital to", "it is imperative to"),
     {'standard': ['it pays to', "it's key to"], 'casual': ["it's key to", 'you need to'],
      'professional': ['it helps to', 'it is wise to'], 'academic': ['it is necessary to']}),
    (("more important than ever", "more crucial than ever", "more vital than ever"),
     {'*': ['especially important', 'more important than it used to be'], 'casual': ['a bigger deal than ever'],
      'professional': ['especially important'], 'academic': ['increasingly important']}),
    (("a deep dive into",), {'*': ['a close look at']}),
    (("is designed to",), {'*': ['is meant to', 'is built to'], 'academic': ['is intended to']}),
    (("are designed to",), {'*': ['are meant to', 'are built to'], 'academic': ['are intended to']}),
    (("was designed to",), {'*': ['was meant to', 'was built to'], 'academic': ['was intended to']}),
    (("were designed to",), {'*': ['were meant to', 'were built to'], 'academic': ['were intended to']}),
]
# An optional "a"/"an" in front is captured so it can agree with the replacement
PHRASE_RULES = [(re.compile(r"(?P<art>\b(?:a|an)\s+)?\b(?P<phrase>" + phrase_regex(words) + r")\b", re.I), options)
                for words, options in PHRASES]

def rewrite_phrase(match, options, tone):
    new = match_case(match.group('phrase'), pick(options, tone))
    article = match.group('art')
    if not article:
        return new
    wanted = 'an' if starts_with_vowel_sound(new) else 'a'
    return match_case(article.strip(), wanted) + article[len(article.rstrip()):] + new

# "is not only fast but also cheap" -> "is fast and cheap" (also "not just/merely/simply"). Lowercase "not"
# only: a sentence that opens with "Not only ..." uses inverted word order and is left alone.
NOT_ONLY = re.compile(r"\bnot (?:only|just|merely|simply)\s+(?P<a>[^,.;:!?\"“”()]{1,80}?)\s+but(?:\s+also)?\s+")

def pick_fresh(options, tone, used):
    """Like pick(), but avoids repeating a replacement already used in this text ("Also, ... Also, ...")."""
    choices = options.get(tone) or options['*']
    fresh = [c for c in choices if c not in used] or choices
    choice = random.choice(fresh)
    used.add(choice)
    return choice

def apply_phrase_rules(text, tone):
    text = NOT_ONLY.sub(lambda m: m.group('a') + ' ' + random.choice(['and', 'and also']) + ' ', text)

    used = set()
    for pattern, chance, options in TRANSITION_RULES:
        text = pattern.sub(lambda m: m.group('pre') + pick_fresh(options, tone, used) + ' '
                           if random.random() < chance else m.group(0), text)

    signpost = FILLER_SIGNPOSTS.get(tone)
    text = FILLER_AT_START.sub(
        lambda m: m.group('pre') + (signpost + m.group('first') if signpost else m.group('first').upper()), text)
    text = FILLER_INSIDE.sub('', text)

    for pattern, options in PHRASE_RULES:
        text = pattern.sub(lambda m: rewrite_phrase(m, options, tone), text)
    return text

# ==========================================
#   STEP 2: CONTRACTIONS
# ==========================================

# Negations can be contracted anywhere; other contractions are only ever expanded
CONTRACTIONS = {
    "do not": "don't", "does not": "doesn't", "did not": "didn't",
    "is not": "isn't", "are not": "aren't", "was not": "wasn't", "were not": "weren't",
    "cannot": "can't", "could not": "couldn't", "will not": "won't", "would not": "wouldn't",
    "should not": "shouldn't", "has not": "hasn't", "have not": "haven't", "had not": "hadn't",
}
EXPANSIONS = {short: full for full, short in CONTRACTIONS.items()}
EXPANSIONS.update({"i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are"})

# Contracted only when another word follows ("it's done", but never "I know what it is.")
PRONOUN_CONTRACTIONS = {
    "it is": "it's", "that is": "that's", "there is": "there's", "what is": "what's",
    "we are": "we're", "they are": "they're", "you are": "you're", "i am": "i'm",
    "we will": "we'll", "you will": "you'll", "it will": "it'll", "they will": "they'll",
}

CONTRACT_PATTERN = re.compile(r"\b" + phrase_regex(CONTRACTIONS) + r"\b", re.I)
PRONOUN_PATTERN = re.compile(r"\b" + phrase_regex(PRONOUN_CONTRACTIONS) + r"\b(?=\s+[a-z])", re.I)
EXPAND_PATTERN = re.compile(r"\b" + phrase_regex(EXPANSIONS) + r"\b", re.I)

# People contract in everyday writing; AI text rarely does. Standard mixes both.
CONTRACT_CHANCE = {'casual': 1.0, 'standard': 0.5}

def apply_tone_contractions(text, tone):
    def replace_from(table, chance=1.0):
        def replace(match):
            if random.random() >= chance:
                return match.group(0)
            key = re.sub(r'\s+', ' ', match.group(0).lower()).replace('’', "'")
            return match_case(match.group(0), table[key])
        return replace

    if tone in CONTRACT_CHANCE:
        chance = CONTRACT_CHANCE[tone]
        text = CONTRACT_PATTERN.sub(replace_from(CONTRACTIONS, chance), text)
        return PRONOUN_PATTERN.sub(replace_from(PRONOUN_CONTRACTIONS, chance), text)
    return EXPAND_PATTERN.sub(replace_from(EXPANSIONS), text)

# ==========================================
#   STEP 3: WORD-LEVEL REWRITING
# ==========================================

UPOS_TO_WORDNET = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}
UNINFLECTED_TAGS = {'VB', 'VBP', 'NN', 'JJ', 'RB'}
SKIP_TAGS = {'JJR', 'JJS', 'RBR', 'RBS'}  # comparatives re-inflect badly ("beautifuler")

# How much rarer than the original a synonym may be (share of its corpus frequency).
# Academic may reach for rarer, more formal words; Chaos Mode loosens the limit for everyone.
SYNONYM_FLOOR = {'standard': 0.3, 'casual': 0.5, 'professional': 0.3, 'academic': 0.0}
CHAOS_FLOOR = 0.1

# Share of eligible content words that get a WordNet synonym. Predictable word choice is one of the
# strongest signs of AI text, and the word-pair check below keeps the swaps idiomatic.
WORD_RATE = {'standard': 0.5, 'casual': 0.55, 'professional': 0.35, 'academic': 0.35}
CHAOS_WORD_RATE = 0.8

# Nouns carry most of a sentence's meaning and WordNet's noun synonyms often miss the context
# ("updated the details" is not "updated the points"), so only Chaos Mode swaps them, less often,
# and only when the new word is attested next to a neighbour in real English
NOUN_WEIGHT = {False: 0.0, True: 0.45}

# Curated adverb swaps that only read well in some positions ("manage time effectively" -> "well",
# but not "effectively reduces" -> "well reduces"; "truly matters" but not "truly unique" -> "really unique")
AFTER_VERB_ONLY = {'effectively|ADV', 'tirelessly|ADV'}
VERB_HEAD_ONLY = {'truly|ADV'}
# ...and verb swaps that only work with a direct object ("reduce costs" -> "cut costs", but
# "provide that ..." or "establish whether ..." can't take "give" or "build")
OBJECT_ONLY = {'reduce|VERB', 'pose|VERB', 'provide|VERB', 'organize|VERB', 'implement|VERB', 'establish|VERB',
               'maintain|VERB', 'identify|VERB', 'examine|VERB', 'analyze|VERB', 'analyse|VERB', 'investigate|VERB',
               'acquire|VERB', 'emphasize|VERB', 'highlight|VERB'}
# Adjectives swapped only in front of a noun ("a significant change" -> "a major change", but
# "statistically significant" and "is significant for" keep their technical sense)
BEFORE_NOUN_ONLY = {'significant|ADJ', 'substantial|ADJ', 'various|ADJ', 'fundamental|ADJ', 'appropriate|ADJ',
                    'consistent|ADJ', 'traditional|ADJ', 'frequent|ADJ', 'effective|ADJ'}
POSITION_DEPENDENT = AFTER_VERB_ONLY | VERB_HEAD_ONLY | OBJECT_ONLY | BEFORE_NOUN_ONLY
# Plain-English swaps Academic doesn't make (its readers expect the technical word)
ACADEMIC_KEEPS = {'significant|ADJ', 'substantial|ADJ', 'fundamental|ADJ', 'effective|ADJ', 'traditional|ADJ',
                  'analyze|VERB', 'analyse|VERB', 'investigate|VERB', 'examine|VERB', 'identify|VERB'}

# Names are never touched. Inside dates, times, amounts and other number phrases only the curated
# swaps apply ("approximately three additional days" -> "about three extra days").
NAME_ENTITIES = {'PERSON', 'ORG', 'GPE', 'LOC', 'NORP', 'FAC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE'}

# Words that make text read as AI-written, keyed "lemma|POS". Always replaced.
AI_WORDS = {
    'utilize|VERB': {'*': ['use']},
    'utilise|VERB': {'*': ['use']},
    'leverage|VERB': {'*': ['use'], 'casual': ['use', 'tap into']},
    'delve|VERB': {'standard': ['dig', 'look'], 'casual': ['dig'], 'professional': ['look'], 'academic': ['look']},
    'facilitate|VERB': {'standard': ['help', 'support'], 'casual': ['help'], 'professional': ['support', 'enable'],
                        'academic': ['enable', 'support']},
    'foster|VERB': {'*': ['encourage', 'support'], 'casual': ['boost', 'encourage'], 'academic': ['promote', 'encourage']},
    'bolster|VERB': {'*': ['strengthen', 'boost'], 'casual': ['boost']},
    'showcase|VERB': {'*': ['show', 'highlight']},
    'underscore|VERB': {'*': ['highlight', 'stress']},
    'elucidate|VERB': {'*': ['explain', 'clarify']},
    'commence|VERB': {'*': ['start', 'begin']},
    'endeavor|VERB': {'*': ['try']},
    'endeavour|VERB': {'*': ['try']},
    'ascertain|VERB': {'*': ['determine', 'find out'], 'casual': ['find out', 'figure out']},
    'enhance|VERB': {'standard': ['improve', 'boost'], 'casual': ['boost', 'improve'], 'professional': ['improve'],
                     'academic': ['improve', 'strengthen']},
    'streamline|VERB': {'*': ['simplify']},
    'harness|VERB': {'*': ['use', 'tap'], 'academic': ['use', 'exploit']},
    'crucial|ADJ': {'standard': ['key', 'vital'], 'casual': ['key', 'big'], 'professional': ['key', 'essential'],
                    'academic': ['essential', 'central']},
    'pivotal|ADJ': {'standard': ['key', 'central'], 'casual': ['key', 'big'], 'professional': ['key', 'central'],
                    'academic': ['central']},
    'paramount|ADJ': {'*': ['vital', 'essential'], 'casual': ['huge', 'vital']},
    'robust|ADJ': {'standard': ['strong', 'solid'], 'casual': ['solid', 'strong'], 'professional': ['strong', 'reliable'],
                   'academic': ['strong', 'sound']},
    'seamless|ADJ': {'*': ['smooth']},
    'comprehensive|ADJ': {'standard': ['thorough', 'complete', 'full'], 'casual': ['full', 'complete'],
                          'professional': ['thorough', 'complete'], 'academic': ['thorough', 'extensive']},
    'meticulous|ADJ': {'*': ['careful', 'thorough']},
    'intricate|ADJ': {'*': ['complex', 'detailed']},
    'multifaceted|ADJ': {'*': ['complex', 'varied']},
    'myriad|ADJ': {'*': ['many', 'countless']},
    'vibrant|ADJ': {'*': ['lively', 'bright']},
    'innovative|ADJ': {'standard': ['new', 'creative'], 'casual': ['new', 'fresh'], 'professional': ['creative', 'new'],
                       'academic': ['novel', 'new']},
    'groundbreaking|ADJ': {'*': ['new', 'original'], 'academic': ['novel']},
    'noteworthy|ADJ': {'*': ['notable', 'striking']},
    'invaluable|ADJ': {'*': ['very useful', 'priceless'], 'professional': ['very valuable'], 'academic': ['highly valuable']},
    'seamlessly|ADV': {'*': ['smoothly']},
    'meticulously|ADV': {'*': ['carefully']},
    'significantly|ADV': {'standard': ['greatly', 'considerably'], 'casual': ['greatly', 'hugely'],
                          'professional': ['considerably', 'greatly'], 'academic': ['considerably', 'substantially']},
    'additionally|ADV': {'*': ['also']},
    'realm|NOUN': {'*': ['area', 'field', 'world']},
    'tapestry|NOUN': {'*': ['mix', 'blend']},
    'endeavor|NOUN': {'*': ['effort', 'attempt']},
    'endeavour|NOUN': {'*': ['effort', 'attempt']},
    'utilization|NOUN': {'*': ['use']},
    'utilisation|NOUN': {'*': ['use']},
    'paradigm|NOUN': {'*': ['model', 'pattern']},
    'holistic|ADJ': {'*': ['complete'], 'academic': ['integrated']},
    'transformative|ADJ': {'*': ['major', 'powerful'], 'academic': ['far-reaching']},
    'ensure|VERB': {'*': ['make sure'], 'professional': ['make sure', 'ensure'], 'academic': ['ensure']},
    'empower|VERB': {'*': ['help', 'enable'], 'casual': ['help'], 'academic': ['enable']},
    'elevate|VERB': {'*': ['raise', 'improve'], 'casual': ['boost', 'lift']},
    'garner|VERB': {'*': ['gain', 'earn'], 'casual': ['get', 'earn']},
    'spearhead|VERB': {'*': ['lead']},
    'underpin|VERB': {'*': ['support'], 'academic': ['support', 'underlie']},
    'encompass|VERB': {'*': ['include', 'cover']},
    'boast|VERB': {'*': ['have', 'offer']},
    'optimize|VERB': {'*': ['improve', 'fine-tune'], 'academic': ['optimize']},
    'optimise|VERB': {'*': ['improve', 'fine-tune'], 'academic': ['optimise']},
    'revolutionize|VERB': {'*': ['transform', 'change'], 'casual': ['shake up', 'change']},
    'resonate|VERB': {'*': ['connect', 'click'], 'professional': ['connect'], 'academic': ['connect']},
    'cultivate|VERB': {'*': ['build', 'develop'], 'academic': ['develop']},
    'intricacy|NOUN': {'*': ['detail', 'complexity']},
    'nuance|NOUN': {'*': ['subtlety', 'detail']},
    'nuanced|ADJ': {'*': ['subtle', 'detailed']},
    'unprecedented|ADJ': {'*': ['record', 'unheard-of'], 'academic': ['unprecedented']},
    'commendable|ADJ': {'*': ['admirable', 'impressive']},
    'pressing|ADJ': {'*': ['urgent', 'serious']},
    'decisive|ADJ': {'*': ['firm', 'strong'], 'academic': ['firm', 'resolute']},
    'viable|ADJ': {'*': ['workable', 'practical'], 'academic': ['feasible', 'practical']},
    'valuable|ADJ': {'*': ['useful', 'helpful'], 'academic': ['useful', 'important']},
    'utmost|ADJ': {'*': ['greatest', 'highest']},
    'tirelessly|ADV': {'*': ['hard', 'nonstop'], 'professional': ['hard'], 'academic': ['persistently']},
    'effectively|ADV': {'*': ['well'], 'academic': ['effectively', 'well']},
    'truly|ADV': {'*': ['really'], 'professional': ['genuinely'], 'academic': ['genuinely']},
}

# Formal words with plain everyday equivalents, the kind of edit a copy editor makes.
# Every swap here reads correctly in almost any sentence, so they can be applied freely.
PLAIN_WORDS = {
    'individual|NOUN': ['person'],
    'numerous|ADJ': ['many'],
    'approximately|ADV': ['about', 'around'],
    'sufficient|ADJ': ['enough'],
    'require|VERB': ['need'],
    'indicate|VERB': ['show', 'suggest'],
    'demonstrate|VERB': ['show'],
    'assist|VERB': ['help'],
    'obtain|VERB': ['get'],
    'purchase|VERB': ['buy'],
    'attempt|VERB': ['try'],
    'inquire|VERB': ['ask'],
    'possess|VERB': ['have'],
    'reside|VERB': ['live'],
    'modify|VERB': ['change'],
    'eliminate|VERB': ['remove', 'cut'],
    'terminate|VERB': ['end'],
    'anticipate|VERB': ['expect'],
    'additional|ADJ': ['extra'],
    'initial|ADJ': ['first'],
    'beneficial|ADJ': ['helpful', 'useful'],
    'optimal|ADJ': ['ideal'],
    'frequently|ADV': ['often'],
    'subsequently|ADV': ['later'],
    'currently|ADV': ['now'],
    'consequently|ADV': ['so'],
    'illustrate|VERB': ['show'],
    'exhibit|VERB': ['show'],
    'necessitate|VERB': ['need'],
    'comprise|VERB': ['include'],
    'retain|VERB': ['keep'],
    'cease|VERB': ['stop'],
    'adequate|ADJ': ['enough'],
    'assistance|NOUN': ['help'],
    'methodology|NOUN': ['method'],
    'objective|NOUN': ['goal', 'aim'],
    'component|NOUN': ['part'],
    'nevertheless|ADV': ['still'],
    'nonetheless|ADV': ['still'],
    'reduce|VERB': ['cut', 'lower'],
    'pose|VERB': ['present', 'raise'],
    'provide|VERB': ['give', 'offer'],
    'organize|VERB': ['arrange', 'plan'],
    'implement|VERB': ['apply', 'use'],
    'establish|VERB': ['build', 'set up'],
    'maintain|VERB': ['keep'],
    'identify|VERB': ['spot', 'find'],
    'examine|VERB': ['look at', 'study'],
    'analyze|VERB': ['study', 'look at'],
    'analyse|VERB': ['study', 'look at'],
    'investigate|VERB': ['look into', 'study'],
    'acquire|VERB': ['get', 'gain'],
    'emphasize|VERB': ['stress', 'highlight'],
    'highlight|VERB': ['point out', 'show'],
    'enable|VERB': ['allow'],
    'significant|ADJ': ['major', 'big'],
    'substantial|ADJ': ['large', 'major'],
    'various|ADJ': ['different', 'several'],
    'fundamental|ADJ': ['basic', 'core'],
    'appropriate|ADJ': ['suitable', 'fitting'],
    'consistent|ADJ': ['steady', 'regular'],
    'traditional|ADJ': ['conventional', 'standard'],
    'frequent|ADJ': ['common', 'regular'],
    'effective|ADJ': ['useful', 'working'],
    'immediately|ADV': ['right away', 'at once'],
    'primarily|ADV': ['mainly', 'mostly'],
    'particularly|ADV': ['especially'],
    'typically|ADV': ['usually'],
    'rapidly|ADV': ['quickly', 'fast'],
    'continuously|ADV': ['constantly'],
}
# Chance of making a plain-English swap; Academic keeps its formal vocabulary
PLAIN_CHANCE = {'casual': 1.0, 'standard': 0.85, 'professional': 0.5, 'academic': 0.0}
IRREGULAR_PLURALS = {'person': 'people'}

def neighbours(token):
    """The lowercase words either side of a token (None next to punctuation or at the edge)."""
    doc = token.doc
    before = doc[token.i - 1].lower_ if token.i > 0 and doc[token.i - 1].is_alpha else None
    after = doc[token.i + 1].lower_ if token.i + 1 < len(doc) and doc[token.i + 1].is_alpha else None
    return before, after

def context_score(before, word, after):
    words = word.lower().split()
    return pair_count(before, words[0]) + pair_count(words[-1], after)

def choose_synonym(token, entry, tone, floor, avoid, need_evidence):
    """(lemma, inflected word) for a WordNet synonym in the tone's style that fits the token's neighbours,
    or (None, None). If the original is an established pairing in the Brown corpus ("rely on", seen
    at least twice), the synonym must be attested there too; nouns always need that evidence."""
    lemma_frequency, candidates = entry
    pool = [word for word, frequency in candidates if frequency >= floor * lemma_frequency and word not in avoid]
    if tone == 'casual':
        order = sorted(pool[:4], key=len) + pool[4:]
    elif tone == 'academic':
        order = sorted(pool, key=len, reverse=True)
    elif tone == 'professional':
        order = pool
    else:
        order = random.sample(pool[:3], len(pool[:3])) + pool[3:]
    before, after = neighbours(token)
    attested = context_score(before, token.lower_, after) >= 2
    # A noun right after its modifier ("remote sensing") must keep a known pairing with it
    modified = need_evidence and token.i > 0 and token.doc[token.i - 1].head == token and \
        token.doc[token.i - 1].dep_ in ('amod', 'compound', 'nmod')
    for lemma in order:
        word = inflect(lemma, token.tag_)
        if not word or word.lower() == token.lower_:
            continue
        if modified and not pair_count(before, word.lower().split()[0]):
            continue
        if (attested or need_evidence) and not context_score(before, word, after):
            continue
        return lemma, word
    return None, None

def inflect(lemma, tag):
    # Multi-word replacements ("find out") inflect their first word only
    head, space, rest = lemma.partition(' ')
    if tag == 'NNS' and head in IRREGULAR_PLURALS:
        head = IRREGULAR_PLURALS[head]
    elif tag not in UNINFLECTED_TAGS:
        forms = getInflection(head, tag=tag)
        if not forms:
            return None
        head = forms[0]
    return head + space + rest

def starts_with_vowel_sound(word):
    w = word.lower()
    if w.startswith(('hour', 'honest', 'honor', 'honour', 'heir')):
        return True
    if w.startswith(('uni', 'use', 'usu', 'uti', 'ubiq', 'eu', 'ewe', 'one', 'once')):
        return False
    return w[:1] in 'aeiou'

def can_rewrite(token):
    doc = token.doc
    in_hyphenated_word = ((token.i > 0 and doc[token.i - 1].text == '-' and not doc[token.i - 1].whitespace_)
                          or (token.i + 1 < len(doc) and doc[token.i + 1].text == '-' and not token.whitespace_))
    in_compound_term = token.dep_ == 'compound' or any(c.dep_ == 'compound' for c in token.children)
    return (token.pos_ in UPOS_TO_WORDNET and token.is_alpha and len(token.text) >= 4
            and not token.is_stop and token.ent_type_ not in NAME_ENTITIES
            and not (token.is_upper and len(token.text) > 1)          # acronyms
            and not in_compound_term                                  # terms like "data pipeline"
            and token.tag_ not in SKIP_TAGS and not in_hyphenated_word)

def takes_preposition(token):
    """Verbs and adjectives bound to a preposition ("rely on", "aware of", "carry out"). Their WordNet
    synonyms rarely take the same one ("trust on"), so only the curated swaps may change them."""
    return token.pos_ in ('VERB', 'ADJ') and any(
        child.dep_ in ('prep', 'prt') and 0 < child.i - token.i <= 2 for child in token.children)

def curated_fits(token, key):
    if key in AFTER_VERB_ONLY:
        return token.head.pos_ in ('VERB', 'AUX') and token.i > token.head.i
    if key in VERB_HEAD_ONLY:
        return token.head.pos_ in ('VERB', 'AUX')
    if key in OBJECT_ONLY:
        return any(c.dep_ == 'dobj' for c in token.children) and not takes_preposition(token)
    if key in BEFORE_NOUN_ONLY:
        before = token.doc[token.i - 1].lower_ if token.i > 0 else ''
        return token.dep_ == 'amod' and token.head.i > token.i and not before.endswith('ly')
    return True

def fixed_frame(token):
    """Words whose synonyms rarely fit the same construction: "rely on"-style pairs and verbs
    with a to-infinitive ("struggle to manage" is not "fight to manage")."""
    return takes_preposition(token) or (token.pos_ == 'VERB' and any(c.dep_ == 'xcomp' for c in token.children))

def plan_word_edits(doc, tone, deep_mode):
    """Layer 3: token index -> replacement text, for curated and WordNet swaps."""
    rate = CHAOS_WORD_RATE if deep_mode else WORD_RATE[tone]
    floor = CHAOS_FLOOR if deep_mode else SYNONYM_FLOOR[tone]
    plain_chance = 1.0 if deep_mode and tone != 'academic' else PLAIN_CHANCE[tone]
    edits = {}
    for sent in doc.sents:
        avoid = {t.lower_ for t in sent} | {t.lemma_.lower() for t in sent}
        for token in sent:
            if not can_rewrite(token):
                continue
            lemma = token.lemma_.lower()
            key = f'{lemma}|{token.pos_}'
            word = None
            if key in AI_WORDS and curated_fits(token, key):
                replacement = pick(AI_WORDS[key], tone)
            elif (key in PLAIN_WORDS and curated_fits(token, key) and random.random() < plain_chance
                  and not (tone == 'academic' and key in ACADEMIC_KEEPS)):
                replacement = random.choice(PLAIN_WORDS[key])
            elif token.ent_type_ or fixed_frame(token) or key in AI_WORDS or key in PLAIN_WORDS:
                continue  # number phrases, fixed constructions and curated words out of place stay as they are
            elif random.random() < rate * (NOUN_WEIGHT[deep_mode] if token.pos_ == 'NOUN' else 1.0):
                entry = SYNONYMS.get(f'{lemma}|{UPOS_TO_WORDNET[token.pos_]}')
                if not entry:
                    continue
                replacement, word = choose_synonym(token, entry, tone, floor, avoid, token.pos_ == 'NOUN')
            else:
                continue
            word = word or (inflect(replacement, token.tag_) if replacement else None)
            if word and word.lower() != token.lower_:
                edits[token.i] = match_case(token.text, word)
                avoid.add(replacement)

    # "a" / "an" must agree with a swapped word that follows it
    for i in list(edits):
        if i == 0 or not edits[i] or (i - 1) in edits or doc[i - 1].lower_ not in ('a', 'an'):
            continue
        wanted = 'an' if starts_with_vowel_sound(edits[i]) else 'a'
        if doc[i - 1].lower_ != wanted:
            edits[i - 1] = match_case(doc[i - 1].text, wanted)
    return edits

# ==========================================
#   STEP 4: SENTENCE STRUCTURE
# ==========================================
# A sentence is a list of pieces, [token index or None for inserted words, text], so phrases can
# be moved, split off and joined while keeping track of where each word came from.

NO_SPACE_BEFORE = {',', '.', ';', ':', '!', '?', ')', ']', '}', '%', '”', "n't", "n’t"}
NO_SPACE_AFTER = {'(', '[', '{', '“', '$'}

def render(pieces, doc):
    """Words that are still neighbours keep their original spacing; moved or inserted ones get
    standard English spacing."""
    out, previous = [], None
    for index, text in pieces:
        if not text:
            continue
        if previous is not None:
            prev_index, prev_text = previous
            if prev_index is not None and index is not None and index == prev_index + 1:
                out.append(doc[prev_index].whitespace_)
            elif not (text in NO_SPACE_BEFORE or text.startswith(("'", "’")) or prev_text in NO_SPACE_AFTER):
                out.append(' ')
        out.append(text)
        previous = (index, text)
    return ''.join(out)

def keeps_capital(token):
    return (token.pos_ == 'PROPN' or token.ent_type_ in NAME_ENTITIES or token.text == 'I'
            or (token.is_upper and len(token.text) > 1) or any(c.isupper() for c in token.text[1:]))

def first_piece(pieces):
    return next((p for p in pieces if p[1]), None)

def capitalize_first(pieces):
    piece = first_piece(pieces)
    if piece:
        piece[1] = piece[1][0].upper() + piece[1][1:]

def lowercase_first(pieces, doc):
    piece = first_piece(pieces)
    if piece and (piece[0] is None or not keeps_capital(doc[piece[0]])):
        piece[1] = piece[1][0].lower() + piece[1][1:]

def take(pieces, start, end):
    """Copies of the pieces for tokens start..end (inclusive)."""
    return [list(p) for p in pieces if p[0] is not None and start <= p[0] <= end]

class Sentence:
    """One sentence on its way through the structure and rhythm layers."""

    def __init__(self, pieces, doc, span=None, origin=None):
        self.pieces, self.doc, self.span = pieces, doc, span
        self.splits = []        # (piece position of the "," or ";", kind) while the sentence is in its original order
        self.reordered = False
        self.origin = origin    # halves of one split share it, so they aren't simply joined back
        self.must_split = False  # set by the coverage pass when a split is the only safe change

    def words(self):
        return sum(1 for _, text in self.pieces if text and text[0].isalnum())

    def opener(self):
        piece = first_piece(self.pieces)
        return piece[1].lower() if piece else ''

    def text(self):
        return render(self.pieces, self.doc)

def restructurable(sent):
    """Plain statements only: ends in a full stop, no quotes, brackets, colons, dashes or links."""
    texts = {t.text for t in sent}
    return (sent[-1].text == '.' and 6 <= len(sent) <= 45 and sent.root.pos_ in ('VERB', 'AUX')
            and not texts & {'"', '“', '”', '(', ')', '[', ']', ':', '—', '–'}
            and not any(t.like_url or t.like_email for t in sent))

# --- Moving adverbial phrases and clauses between the start and the end of a sentence ---

TIME_NOUNS = {'year', 'month', 'week', 'day', 'decade', 'century', 'time', 'period', 'past', 'future', 'era',
              'morning', 'evening', 'night', 'quarter', 'season', 'hour', 'minute', 'moment', 'age', 'today', 'recent'}
ALWAYS_MOVABLE_PREPS = {'during', 'after', 'before', 'since', 'throughout', 'despite', 'following', 'amid', 'until'}
SETTING_PREPS = {'in', 'on', 'at', 'over', 'within', 'across', 'for'}     # movable when they give a time or place
UNMOVABLE_PREPS = {'of', 'as', 'like', 'than', 'to'}
MOVABLE_MARKS = {'because', 'since', 'although', 'though', 'when', 'whenever', 'while', 'after', 'before', 'if',
                 'unless', 'once', 'whereas', 'until'}
CONCESSIVE = {'although', 'though', 'whereas', 'while', 'despite', 'even'}   # keep a comma when moved to the end
DISCOURSE_PHRASES = {
    'in addition', 'in short', 'in conclusion', 'in sum', 'in particular', 'in fact', 'in general', 'in contrast',
    'in the end', 'in other words', 'on top of that', 'beyond that', 'beyond this', 'for example', 'for instance',
    'of course', 'by contrast', 'as a result', 'because of this', 'because of that', 'on the other hand',
    'at the same time', 'all in all', 'after all', 'above all', 'at the end of the day', 'in turn', 'to date', 'so far',
    'in practice', 'in theory', 'in reality', 'in total', 'in brief', 'in summary', 'at first', 'at last', 'at times',
    'by default', 'on average', 'on balance', 'for now', 'for this reason', 'to this end', 'in this case',
    'in that case', 'in this way', 'with this in mind', 'as such', 'by comparison', 'in comparison',
}
PRONOUN_SUBJECTS = {'it', 'they', 'he', 'she', 'this', 'these', 'those', 'that', 'them', 'its', 'their'}
# Verbs whose "to ..." is part of their meaning, not a purpose ("ask someone to help", "want to go")
TO_COMPLEMENT_VERBS = {
    'ask', 'tell', 'allow', 'enable', 'encourage', 'help', 'want', 'expect', 'require', 'force', 'persuade', 'invite',
    'urge', 'remind', 'teach', 'cause', 'get', 'lead', 'permit', 'advise', 'instruct', 'order', 'warn', 'need',
    'try', 'plan', 'hope', 'decide', 'seem', 'tend', 'begin', 'start', 'continue', 'fail', 'manage', 'struggle',
    'attempt', 'aim', 'intend', 'agree', 'refuse', 'promise', 'prefer', 'like', 'love', 'hate', 'choose', 'learn',
    'forget', 'remember', 'offer', 'wish', 'appear', 'happen', 'have', 'use', 'go', 'used', 'train', 'prepare',
}

def adjunct_span(child):
    span = child.doc[child.left_edge.i: child.right_edge.i + 1]
    if span.text.lower() in DISCOURSE_PHRASES or not 2 <= len(span) <= 14:
        return None
    first = span[0]
    if child.dep_ == 'advcl' and first.dep_ in ('mark', 'advmod') and first.head == child and first.lower_ in MOVABLE_MARKS:
        # "because"/"although" are tagged as markers, "when"/"while" as adverbs of the clause's verb
        return span
    if child.dep_ in ('advcl', 'xcomp'):
        # A purpose clause: "... every year to mark the end of British rule". spaCy often labels it
        # xcomp; it is only a purpose clause if the verb doesn't take "to ..." itself ("want to go",
        # "asked the team to review") and already has its object
        purpose = first.lower_ == 'to' and first.dep_ == 'aux' and first.head == child and child.tag_ == 'VB'
        if purpose and child.dep_ == 'xcomp':
            verb = child.head
            purpose = (verb.lemma_.lower() not in TO_COMPLEMENT_VERBS
                       and any(c.dep_ == 'dobj' and c.i < child.i for c in verb.children))
        return span if purpose else None
    return span if child.dep_ == 'prep' else None

def front_trailing_adjunct(sent, pieces):
    """"Many studies lack data despite recent advances." -> "Despite recent advances, many studies lack data.\""""
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    child = next((c for c in reversed(list(root.rights)) if not c.is_punct), None)
    if child is None or child.right_edge.i != stop - 1:
        return None
    span = adjunct_span(child)
    if span is None:
        return None
    if child.dep_ == 'prep' and child.lower_ not in ALWAYS_MOVABLE_PREPS and not (
            child.lower_ in SETTING_PREPS and any(t.ent_type_ in ('DATE', 'TIME', 'GPE', 'LOC', 'EVENT')
                                                  or t.lemma_.lower() in TIME_NOUNS for t in span)):
        return None
    if span.start - sent.start < 4 or (span.start <= root.i + 2 and takes_preposition(root)):
        return None
    moved = take(pieces, span.start, span.end - 1)
    main = take(pieces, sent.start, span.start - 1)
    while main and main[-1][1] in (',', ''):
        main.pop()      # "X, although Y." -> "Although Y, X."
    capitalize_first(moved)
    lowercase_first(main, doc)
    return moved + [[None, ',']] + main + take(pieces, stop, stop)

def back_fronted_adjunct(sent, pieces):
    """"When a user asks, the system checks the cache." -> "The system checks the cache when a user asks.\""""
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    child = next((c for c in root.lefts if c.left_edge.i == sent.start), None)
    if child is None or (child.dep_ == 'prep' and child.lower_ in UNMOVABLE_PREPS):
        return None
    span = adjunct_span(child)
    if span is None or span.end >= stop or doc[span.end].text != ',' or stop - (span.end + 1) < 4:
        return None
    if child.dep_ == 'prep' and len(span) < 3:
        return None     # short signposts ("In practice,", "At first,") read oddly at the end
    subject = next((c for c in root.children if c.dep_ in ('nsubj', 'nsubjpass', 'expl')), None)
    if subject is None or subject.i <= span.end or subject.lower_ in PRONOUN_SUBJECTS:
        return None     # "If it's cached, it is returned" can't become "It is returned if it's cached"
    if any(t.text in (',', ';') or t.lower_ in PRONOUN_SUBJECTS for t in doc[span.end + 1: stop]):
        return None     # a pronoun would come before the noun it stands for ("stores it ... if the data ...")
    moved = take(pieces, span.start, span.end - 1)
    main = take(pieces, span.end + 1, stop - 1)
    lowercase_first(moved, doc)
    capitalize_first(main)
    joiner = [[None, ',']] if doc[span.start].lower_ in CONCESSIVE else []
    return main + joiner + moved + take(pieces, stop, stop)

# --- Passive to active: "The cache is updated by a worker." -> "A worker updates the cache." ---

SUBJECT_FORMS = {'me': 'I', 'him': 'he', 'her': 'she', 'us': 'we', 'them': 'they'}
OBJECT_FORMS = {'i': 'me', 'he': 'him', 'she': 'her', 'we': 'us', 'they': 'them'}
PLURAL_PRONOUNS = {'they', 'them', 'we', 'us', 'you', 'i', 'me'}

def is_time_phrase(token):
    return any(t.ent_type_ in ('DATE', 'TIME') or t.lemma_.lower() in TIME_NOUNS for t in token.subtree)

def leading_signpost(sent, start):
    """Tokens before `start` if they are a discourse opener plus comma ("However,", "In practice,"), else None."""
    prefix = sent.doc[sent.start:start]
    if not len(prefix):
        return prefix
    if prefix[-1].text != ',' or len(prefix) > 5 or prefix[0].head != sent.root or prefix[0].dep_ not in ('advmod', 'prep', 'cc'):
        return None
    return prefix

def passive_to_active(sent, pieces):
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    if root.tag_ != 'VBN' or next((p[1] for p in pieces if p[0] == root.i), None) != root.text:
        return None     # not passive, or the verb was already swapped for a synonym
    kids = list(root.children)
    auxpass = [c for c in kids if c.dep_ == 'auxpass']
    subject = next((c for c in kids if c.dep_ == 'nsubjpass'), None)
    agent = next((c for c in kids if c.dep_ == 'agent' and c.i > root.i), None)
    if (len(auxpass) != 1 or auxpass[0].lower_ not in ('is', 'are', 'was', 'were') or subject is None or agent is None
            or any(c.dep_ in ('aux', 'neg') for c in kids) or subject.lower_ in ('which', 'that', 'who', 'what')):
        return None
    doer = next((c for c in agent.children if c.dep_ == 'pobj'), None)
    prefix = leading_signpost(sent, subject.left_edge.i)
    between = doc[auxpass[0].i + 1: root.i]     # adverbs inside the verb: "is often limited"
    if (doer is None or prefix is None or doer.left_edge.i != agent.i + 1 or agent.right_edge.i - agent.i > 10
            or subject.right_edge.i + 1 != auxpass[0].i or any(t.dep_ != 'advmod' for t in between)
            or root.i - subject.left_edge.i > 14):
        return None
    if auxpass[0].lower_ in ('was', 'were'):
        tag = 'VBD'
    else:
        plural = (doer.tag_ in ('NNS', 'NNPS') or doer.lower_ in PLURAL_PRONOUNS
                  or any(c.dep_ == 'conj' for c in doer.children))
        tag = 'VBP' if plural else 'VBZ'
    forms = getInflection(root.lemma_, tag=tag)
    if not forms:
        return None
    # A time phrase the parser hung on the doer ("by a worker every five minutes") goes to the end
    doer_end = agent.right_edge.i
    trailing = next((c for c in doer.rights if c.dep_ in ('appos', 'npadvmod') and is_time_phrase(c)), None)
    if trailing is not None:
        doer_end = trailing.left_edge.i - 1
    doers = take(pieces, agent.i + 1, doer_end)
    receivers = take(pieces, subject.left_edge.i, subject.right_edge.i)
    rest = [p for p in take(pieces, root.i + 1, stop - 1) if not agent.i <= p[0] <= doer_end]
    if len(doers) == 1 and doers[0][1].lower() in SUBJECT_FORMS:
        doers[0][1] = SUBJECT_FORMS[doers[0][1].lower()]
    if len(receivers) == 1 and receivers[0][1].lower() in OBJECT_FORMS:
        receivers[0][1] = OBJECT_FORMS[receivers[0][1].lower()]
    lowercase_first(receivers, doc)
    head = take(pieces, prefix.start, prefix.end - 1) if len(prefix) else []
    if head:
        lowercase_first(doers, doc)
    else:
        capitalize_first(doers)
    adverbs = take(pieces, between.start, between.end - 1) if len(between) else []
    return head + doers + adverbs + [[None, forms[0]]] + receivers + rest + take(pieces, stop, stop)

# --- "Task managers allow you to plan your day." -> "With task managers, you can plan your day." ---

ENABLING_VERBS = {'allow', 'enable', 'let'}

def enabling_to_with(sent, pieces):
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    if root.lemma_.lower() not in ENABLING_VERBS or root.tag_ not in ('VBZ', 'VBP', 'VBD'):
        return None
    kids = list(root.children)
    subject = next((c for c in kids if c.dep_ == 'nsubj' and c.i < root.i), None)
    action = next((c for c in kids if c.dep_ in ('xcomp', 'ccomp') and c.i > root.i), None)
    # The person is the verb's object ("enable users to work") or, as spaCy often parses "allow you
    # to work", the subject of the action
    person = next((c for c in kids if c.dep_ == 'dobj' and c.i > root.i), None)
    if person is None and action is not None:
        person = next((c for c in action.children if c.dep_ == 'nsubj' and c.i < action.i), None)
    if subject is None or person is None or action is None or any(c.dep_ in ('aux', 'neg') for c in kids):
        return None
    if subject.lower_ in PRONOUN_SUBJECTS or subject.left_edge.i != sent.start or subject.pos_ == 'PRON':
        return None
    to = doc[action.i - 1]
    if person.right_edge.i + 1 != (to.i if to.lower_ == 'to' else action.i) or action.tag_ != 'VB':
        return None
    adverbs = [c for c in kids if c.dep_ == 'advmod' and subject.right_edge.i < c.i < root.i]
    if root.i - subject.right_edge.i - 1 != len(adverbs):
        return None
    actor = take(pieces, person.left_edge.i, person.right_edge.i)
    if len(actor) == 1 and actor[0][1].lower() in SUBJECT_FORMS:
        actor[0][1] = SUBJECT_FORMS[actor[0][1].lower()]
    means = take(pieces, subject.left_edge.i, subject.right_edge.i)
    lowercase_first(means, doc)
    adverb_pieces = [list(p) for p in pieces if p[0] in {a.i for a in adverbs}]
    tense = [[None, 'could']] if root.tag_ == 'VBD' else [[None, 'can']]
    return ([[None, 'With']] + means + [[None, ',']] + actor + tense + adverb_pieces
            + take(pieces, action.i, stop - 1) + take(pieces, stop, stop))

# --- "The integration of X has transformed Y." -> "Integrating X has transformed Y." ---
# AI text is noun-heavy; people tend to use the verb. Word positions are kept (the noun's piece
# becomes the -ing form, "the" and "of" become empty), so this combines with the other transforms.

NOMINAL_RATE = {'standard': 0.7, 'casual': 0.8, 'professional': 0.5, 'academic': 0.3}
GERUND_PREPOSITIONS = {'in', 'for', 'by', 'through', 'after', 'before', 'without', 'on', 'about'}
GERUND_VERBS = {'require', 'involve', 'include', 'need', 'avoid', 'consider', 'recommend', 'suggest', 'mean', 'support'}
GROUP_NOUNS = {'people', 'person', 'user', 'team', 'board', 'committee', 'government', 'company', 'staff', 'customer',
               'member', 'student', 'citizen', 'community', 'public', 'author', 'researcher', 'participant', 'employee'}

def denominalize(sent, pieces):
    doc = sent.doc
    for token in sent:
        verb = NOMINALIZATIONS.get(token.lower_)
        if verb is None or token.tag_ != 'NN' or token.i < 1 or token.i + 2 >= sent.end:
            continue
        det, of = doc[token.i - 1], doc[token.i + 1]
        if det.lower_ != 'the' or det.head != token or of.lower_ != 'of' or of.head != token:
            continue
        if any(c.dep_ not in ('det', 'prep') for c in token.children) or token.right_edge != of.right_edge:
            continue    # "the rapid integration of", "the integration of X with Y"
        thing = next((c for c in of.children if c.dep_ == 'pobj'), None)
        if thing is None or thing.ent_type_ in ('PERSON', 'ORG', 'NORP') or thing.lemma_.lower() in GROUP_NOUNS or thing.pos_ == 'PRON':
            continue    # "the education of children" is fine; "the approval of the board" has the board as the doer
        role = token.dep_
        if not (role in ('nsubj', 'nsubjpass')
                or (role == 'pobj' and token.head.lower_ in GERUND_PREPOSITIONS)
                or (role == 'dobj' and token.head.lemma_.lower() in GERUND_VERBS)):
            continue
        forms = getInflection(verb, tag='VBG')
        if not forms:
            continue
        new = [list(p) for p in pieces]
        for piece in new:
            if piece[0] in (det.i, of.i):
                piece[1] = ''
            elif piece[0] == token.i:
                piece[1] = forms[0][0].upper() + forms[0][1:] if det.i == sent.start else forms[0]
        return new
    return None

# --- "Fortunately, there are many tools that can help." -> "Fortunately, many tools can help." ---

def existential_to_direct(sent, pieces):
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    if root.lemma_ != 'be' or any(c.dep_ == 'neg' for c in root.children):
        return None
    there = next((c for c in root.children if c.dep_ == 'expl' and c.lower_ == 'there'), None)
    thing = next((c for c in root.children if c.dep_ == 'attr' and c.i > root.i), None)
    if there is None or thing is None or there.i + 1 != root.i or root.i + 1 != thing.left_edge.i:
        return None
    clause = next((t for t in thing.subtree if t.dep_ == 'relcl'), None)
    if clause is None or clause.right_edge.i != stop - 1:
        return None
    relative = clause.left_edge
    if relative.lower_ not in ('that', 'who', 'which') or relative.dep_ not in ('nsubj', 'nsubjpass'):
        return None
    prefix = leading_signpost(sent, there.i)
    if prefix is None:
        return None
    head = take(pieces, prefix.start, prefix.end - 1) if len(prefix) else []
    noun_phrase = take(pieces, thing.left_edge.i, relative.i - 1)
    if head:
        lowercase_first(noun_phrase, doc)
    else:
        capitalize_first(noun_phrase)
    return head + noun_phrase + take(pieces, relative.i + 1, stop - 1) + take(pieces, stop, stop)

# --- "It's important to choose a policy." -> "Choosing a policy is important." ---

EVALUATIVE_ADJECTIVES = {'important', 'essential', 'crucial', 'necessary', 'vital', 'helpful', 'useful', 'wise', 'easy',
                         'hard', 'difficult', 'possible', 'impossible', 'common', 'smart', 'critical', 'beneficial',
                         'worthwhile', 'advisable', 'best', 'better', 'good', 'key'}

def extraposed_to_gerund(sent, pieces):
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    if root.lemma_ != 'be' or any(c.dep_ in ('neg', 'aux') for c in root.children):
        return None
    it = next((c for c in root.children if c.dep_ in ('nsubj', 'expl') and c.lower_ == 'it'), None)
    adjective = next((c for c in root.children if c.dep_ == 'acomp' and c.lower_ in EVALUATIVE_ADJECTIVES), None)
    if it is None or adjective is None or it.i + 1 != root.i or adjective.left_edge.i != root.i + 1:
        return None
    action = next((c for c in list(adjective.children) + list(root.children) if c.dep_ == 'xcomp' and c.tag_ == 'VB'), None)
    if action is None or doc[action.i - 1].lower_ != 'to' or action.i - 1 != adjective.i + 1 or action.right_edge.i != stop - 1:
        return None
    prefix = leading_signpost(sent, it.i)
    if prefix is None:
        return None
    current = next((p[1] for p in pieces if p[0] == action.i), action.text)
    gerund = inflect(current.lower(), 'VBG')
    if not gerund:
        return None
    verb = 'was' if root.lower_ == 'was' else 'is'
    head = take(pieces, prefix.start, prefix.end - 1) if len(prefix) else []
    subject = [[action.i, gerund]] + take(pieces, action.i + 1, stop - 1)
    if head:
        lowercase_first(subject, doc)
    else:
        capitalize_first(subject)
    return (head + subject + [[None, verb]] + take(pieces, adjective.left_edge.i, adjective.i)
            + take(pieces, stop, stop))

# --- "unity, peace, and progress" -> "peace, unity, and progress" ---
# Plain nouns or adjectives only: names, numbers and verbs (steps in a procedure) keep their order.

LIST_RATE = {'standard': 0.5, 'casual': 0.6, 'professional': 0.35, 'academic': 0.25}

def list_items(token):
    """The items of a list headed by `token` as (start, end) token ranges, or None."""
    doc = token.doc
    conjuncts = [token] + sorted(token.conjuncts, key=lambda t: t.i)
    if token.dep_ == 'conj' or len(conjuncts) < 3 or any(c.i < token.i for c in token.conjuncts):
        return None
    if token.pos_ not in ('NOUN', 'ADJ') or any(c.pos_ != token.pos_ for c in conjuncts):
        return None
    items = []
    for c in conjuncts:
        if any(t.dep_ not in ('conj', 'cc', 'punct') for t in c.rights) or c.like_num or c.ent_type_:
            return None     # items with trailing modifiers ("tools for X"), numbers or names
        start = min([c.i] + [t.left_edge.i for t in c.lefts if t.dep_ not in ('cc', 'punct')])
        if c.i - start > 3:
            return None
        items.append((start, c.i))
    for (_, end), (start, _) in zip(items, items[1:]):
        if any(t.lower_ not in (',', 'and', 'or') for t in doc[end + 1: start]):
            return None
    if items[0][0] > 0 and doc[items[0][0] - 1].lower_ in ('a', 'an'):
        return None     # a shared "a"/"an" would have to agree with whichever item comes first
    return items

def reorder_list(sent, pieces):
    doc = sent.doc
    for token in sent:
        items = list_items(token)
        if not items:
            continue
        k, m = random.sample(range(len(items)), 2)
        order = list(range(len(items)))
        order[k], order[m] = order[m], order[k]
        groups = [take(pieces, start, end) for start, end in items]
        region = range(items[0][0], items[-1][1] + 1)
        item_tokens = {i for start, end in items for i in range(start, end + 1)}
        new, placed = [], 0
        for piece in pieces:
            if piece[0] is not None and piece[0] in region and piece[0] in item_tokens:
                # The first token of each item slot receives a whole item; the item's other tokens are skipped
                slot = next((n for n, (start, _) in enumerate(items) if piece[0] == start), None)
                if slot is not None:
                    new.extend(groups[order[slot]])
                    placed += 1
                continue
            new.append(list(piece))
        if placed != len(items):
            return None
        if items[0][0] == sent.start:
            moved_first = groups[order[0]]
            for group in groups:
                lowercase_first(group, doc)
            capitalize_first(moved_first)
        return new
    return None

# --- "People across the country celebrate ..." -> "Across the country, people celebrate ..." ---

SPREAD_PREPOSITIONS = {'across', 'around', 'throughout'}

def front_subject_place(sent, pieces):
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    subject = next((c for c in root.children if c.dep_ == 'nsubj' and c.i < root.i), None)
    if subject is None or subject.left_edge.i != sent.start or subject.pos_ == 'PRON':
        return None
    place = next((c for c in subject.rights if c.dep_ == 'prep' and c.lower_ in SPREAD_PREPOSITIONS), None)
    if place is None or place.right_edge.i != subject.right_edge.i or subject.right_edge.i + 1 != root.i:
        return None
    moved = take(pieces, place.i, place.right_edge.i)
    main = take(pieces, sent.start, place.i - 1) + take(pieces, root.i, stop - 1)
    capitalize_first(moved)
    lowercase_first(main, doc)
    return moved + [[None, ',']] + main + take(pieces, stop, stop)

# --- Coverage: no longer sentence is left exactly as the AI wrote it ---

COVERAGE_MIN_WORDS = 8
REORDERING = (existential_to_direct, extraposed_to_gerund, passive_to_active, front_trailing_adjunct,
              back_fronted_adjunct, enabling_to_with, front_subject_place)

def untouched(pieces, doc):
    indices = [p[0] for p in pieces]
    return None not in indices and indices == sorted(indices) and all(p[1] == doc[p[0]].text for p in pieces)

def force_word_edit(sent, pieces, tone, deep_mode):
    """One safe word swap: the same rules as plan_word_edits, without the dice."""
    doc = sent.doc
    floor = CHAOS_FLOOR if deep_mode else SYNONYM_FLOOR[tone]
    avoid = {t.lower_ for t in sent} | {t.lemma_.lower() for t in sent}
    candidates = [t for t in sent if can_rewrite(t) and not t.ent_type_ and not fixed_frame(t)]
    random.shuffle(candidates)
    for token in candidates:
        key = f'{token.lemma_.lower()}|{token.pos_}'
        word = None
        if key in AI_WORDS or key in PLAIN_WORDS:
            if not curated_fits(token, key) or (tone == 'academic' and key in ACADEMIC_KEEPS):
                continue
            choice = pick(AI_WORDS[key], tone) if key in AI_WORDS else random.choice(PLAIN_WORDS[key])
            word = inflect(choice, token.tag_)
        elif token.pos_ != 'NOUN' or deep_mode:
            entry = SYNONYMS.get(f'{token.lemma_.lower()}|{UPOS_TO_WORDNET[token.pos_]}')
            if entry:
                _, word = choose_synonym(token, entry, tone, floor, avoid, token.pos_ == 'NOUN')
        if not word or word.lower() == token.lower_:
            continue
        new = [list(p) for p in pieces]
        for piece in new:
            if piece[0] == token.i:
                piece[1] = match_case(token.text, word)
            elif piece[0] == token.i - 1 and doc[token.i - 1].lower_ in ('a', 'an'):
                piece[1] = match_case(piece[1], 'an' if starts_with_vowel_sound(word) else 'a')
        return new
    return None

def ensure_change(sentence, tone, deep_mode):
    """Give an untouched sentence one change: a structural one where possible, else a word swap."""
    sent, pieces = sentence.span, sentence.pieces
    options = [force_word_edit]
    if restructurable(sent):
        options += [denominalize, reorder_list, *REORDERING]
    random.shuffle(options)
    for option in options:
        new = (option(sent, pieces, tone, deep_mode) if option is force_word_edit else option(sent, pieces))
        if new:
            sentence.pieces = new
            sentence.reordered = option in REORDERING
            return
    # Nothing else fits: split it at a safe point, if it has one (done in the rhythm layer)
    sentence.must_split = True

# Chance of changing a sentence's structure, by tone (Academic keeps more of its passive voice)
PASSIVE_RATE = {'standard': 0.7, 'casual': 0.85, 'professional': 0.55, 'academic': 0.2}
MOVE_RATE = {'standard': 0.4, 'casual': 0.45, 'professional': 0.3, 'academic': 0.25}
# The constructions AI text is known for are rewritten more often than plain reordering
SIGNATURE_RATE = {'standard': 0.8, 'casual': 0.9, 'professional': 0.6, 'academic': 0.4}
CHAOS_STRUCTURE_BOOST = 0.3

def restructure(sent, pieces, tone, deep_mode, after_move):
    boost = CHAOS_STRUCTURE_BOOST if deep_mode else 0.0
    if random.random() < SIGNATURE_RATE[tone] + boost:
        new = existential_to_direct(sent, pieces) or extraposed_to_gerund(sent, pieces)
        if new:
            return new
    if random.random() < PASSIVE_RATE[tone] + boost:
        new = passive_to_active(sent, pieces)
        if new:
            return new
    if not after_move and random.random() < MOVE_RATE[tone] + boost:
        moves = [front_trailing_adjunct, back_fronted_adjunct, enabling_to_with, front_subject_place]
        random.shuffle(moves)
        for move in moves:
            new = move(sent, pieces)
            if new:
                return new
    return None

# --- Where a sentence could become two ---

def split_points(sent, pieces):
    """", and"/", but" before a clause with its own subject, "; " before a clause, ", which ..." at the end."""
    doc, stop = sent.doc, sent[-1].i
    position = {p[0]: k for k, p in enumerate(pieces) if p[0] is not None}
    points = []
    for token in sent:
        i = token.i
        if i - sent.start < 6 or stop - i < 6:
            continue
        before = doc[i - 1]
        if token.dep_ == 'cc' and token.lower_ in ('and', 'but') and before.text == ',' and token.head.dep_ == 'ROOT':
            clause = next((c for c in token.head.children if c.dep_ == 'conj' and c.i > i), None)
            if clause is not None and clause.pos_ in ('VERB', 'AUX') and any(
                    c.dep_ in ('nsubj', 'nsubjpass', 'expl') for c in clause.children):
                points.append((position[i - 1], token.lower_))
        elif token.text == ';':
            if any(t.dep_ in ('nsubj', 'nsubjpass', 'expl') and t.head.i > i for t in doc[i + 1: stop]):
                points.append((position[i], ';'))
        elif (token.lower_ == 'which' and before.text == ',' and token.dep_ in ('nsubj', 'nsubjpass')
              and token.head.dep_ == 'relcl' and token.head.right_edge.i == stop - 1):
            verb = token.head
            plural = verb.tag_ == 'VBP' or any(c.lower_ in ('are', 'were', 'have') for c in verb.children)
            points.append((position[i - 1], 'which-plural' if plural else 'which'))
        elif (token.lower_ in ('as', 'because') and token.dep_ == 'mark' and before.text == ','
              and token.head.dep_ == 'advcl' and token.head.head.dep_ == 'ROOT' and token.head.right_edge.i == stop - 1
              and any(c.dep_ in ('nsubj', 'nsubjpass', 'expl') for c in token.head.children)):
            # "..., as the country gained freedom" -> ". The country gained freedom"
            points.append((position[i - 1], token.lower_))
        elif (token.lower_ == 'who' and token.dep_ == 'nsubj' and token.head.dep_ == 'relcl'
              and token.head.right_edge.i == stop - 1 and stop - i >= 5
              and (token.head.head.tag_ in ('NNS', 'NNPS') or token.head.head.lower_ in ('others', 'people'))):
            # "... and many others who fought bravely" -> "... and many others. They fought bravely"
            points.append((position[i - 1] if before.text == ',' else position[i], 'who'))
    return points

def split_sentence(sentence, point, tone):
    position, kind = point
    left = [list(p) for p in sentence.pieces[:position]] + [[None, '.']]
    rest = [list(p) for p in sentence.pieces[position + 1:]]
    if kind == 'and':
        rest = rest[1:]
    elif kind == 'but':
        rest = [[None, 'However,' if tone == 'academic' else 'But']] + rest[1:]
    elif kind == 'which':
        rest = [[None, random.choice(['This', 'That'])]] + rest[1:]
    elif kind == 'which-plural':
        rest = [[None, 'They']] + rest[1:]
    elif kind == 'as':
        rest = rest[1:]
    elif kind == 'because':
        opener = "That's because" if tone in ('standard', 'casual') else 'This is because'
        rest = [[None, opener]] + rest[1:]
    elif kind == 'who':
        if rest and rest[0][1].lower() == 'who':
            rest = rest[1:]
        rest = [[None, 'They']] + rest
    capitalize_first(rest)
    origin = object()
    return [Sentence(left, sentence.doc, origin=origin), Sentence(rest, sentence.doc, origin=origin)]

# ==========================================
#   STEP 5: RHYTHM
# ==========================================
# People mix short and long sentences; AI text keeps them evenly sized. The spread of sentence
# lengths (standard deviation / mean) is pushed toward a human-like level by splitting long
# sentences at safe points and joining short neighbours.

RHYTHM_TARGET = {'standard': 0.55, 'casual': 0.6, 'professional': 0.45, 'academic': 0.45}
RHYTHM_OPS = {'standard': 0.45, 'casual': 0.5, 'professional': 0.3, 'academic': 0.3}   # changes per sentence
CHAOS_RHYTHM_TARGET, CHAOS_RHYTHM_OPS = 0.65, 0.7
MERGE_MAX_WORDS = 14

# Openers that become the joining word when two sentences merge ("X. However, Y." -> "X, but Y.")
JOINING_OPENERS = {'also': 'and', 'plus': 'and', 'however': 'but', 'but': 'but', 'still': 'but', 'yet': 'but', 'so': 'so'}

def spread(lengths):
    mean = sum(lengths) / len(lengths)
    return (sum((x - mean) ** 2 for x in lengths) / len(lengths)) ** 0.5 / mean if mean else 0.0

def mergeable(a, b):
    if a.origin is not None and a.origin is b.origin:
        return False
    for s in (a, b):
        last = s.pieces[-1][1] if s.pieces else ''
        if last != '.' or s.words() > MERGE_MAX_WORDS or any(t in (';', ':', '"', '“', '”', '(') for _, t in s.pieces):
            return False
    return True

def merge_sentences(a, b, tone):
    left = [list(p) for p in a.pieces[:-1]]
    right = [list(p) for p in b.pieces]
    conjunction = 'and'
    head = first_piece(right)
    # "We added caching. This cut latency." -> "We added caching, which cut latency."
    if head and head[0] is not None and head[1].lower() in ('this', 'that'):
        token = a.doc[head[0]]
        following = a.doc[token.i + 1] if token.i + 1 < len(a.doc) else None
        if token.dep_ == 'nsubj' and following is not None and following.pos_ in ('VERB', 'AUX'):
            return Sentence(left + [[None, ','], [None, 'which']] + right[right.index(head) + 1:], a.doc)
    opener = head[1].lower().rstrip(',') if head else ''
    if opener in JOINING_OPENERS:
        conjunction = JOINING_OPENERS[opener]
        right = right[right.index(head) + 1:]
        if right and right[0][1] == ',':
            right = right[1:]
    lowercase_first(right, a.doc)
    joiner = [[None, ';']] if tone == 'academic' and conjunction == 'and' else [[None, ','], [None, conjunction]]
    return Sentence(left + joiner + right, a.doc)

def vary_rhythm(sentences, tone, deep_mode):
    for k in range(len(sentences) - 1, -1, -1):
        s = sentences[k]
        if s.must_split and s.splits:
            sentences[k:k + 1] = split_sentence(s, random.choice(s.splits), tone)
    target = CHAOS_RHYTHM_TARGET if deep_mode else RHYTHM_TARGET[tone]
    budget = max(1, round(len(sentences) * (CHAOS_RHYTHM_OPS if deep_mode else RHYTHM_OPS[tone])))
    for _ in range(budget):
        lengths = [s.words() for s in sentences]
        current = spread(lengths) if len(lengths) > 1 else 0.0
        if current >= target and random.random() < 0.7:
            break
        options = []
        for k, s in enumerate(sentences):
            for point in s.splits:
                left = sum(1 for _, t in s.pieces[:point[0]] if t and t[0].isalnum())
                options.append((spread(lengths[:k] + [left, lengths[k] - left] + lengths[k + 1:]) - current, 'split', k, point))
        for k in range(len(sentences) - 1):
            if mergeable(sentences[k], sentences[k + 1]):
                merged = lengths[:k] + [lengths[k] + lengths[k + 1]] + lengths[k + 2:]
                options.append(((spread(merged) if len(merged) > 1 else 0.0) - current, 'merge', k, None))
        if not options:
            break
        random.shuffle(options)
        gain, operation, k, point = max(options, key=lambda option: option[0])
        if gain <= 0.01:
            break
        if operation == 'split':
            sentences[k:k + 1] = split_sentence(sentences[k], point, tone)
        else:
            sentences[k:k + 2] = [merge_sentences(sentences[k], sentences[k + 1], tone)]
    return sentences

def vary_openings(sentences):
    """Two sentences in a row that open with the same word: restructure the second if possible."""
    for k in range(1, len(sentences)):
        s = sentences[k]
        if s.opener() != sentences[k - 1].opener() or s.span is None or s.reordered:
            continue
        for transform in (front_trailing_adjunct, passive_to_active, back_fronted_adjunct, enabling_to_with,
                          front_subject_place):
            new = transform(s.span, s.pieces)
            if new:
                s.pieces, s.reordered, s.splits = new, True, []
                break

# ==========================================
#   STEP 6: PUNCTUATION
# ==========================================

EM_DASH_PAIR = re.compile(r'\s*—\s*([^—\n]{1,80}?)\s*—\s*')
EM_DASH_ONE = re.compile(r'\s*—\s*(?=[a-z])')

def soften_dashes(text, tone):
    """AI text leans on em dashes; most become parentheses (or commas, in Casual)."""
    if '—' not in text or random.random() < 0.3:
        return text
    aside = (lambda m: ', ' + m.group(1) + ', ') if tone == 'casual' else (lambda m: ' (' + m.group(1) + ') ')
    return EM_DASH_ONE.sub(', ', EM_DASH_PAIR.sub(aside, text))

def tidy(text):
    # Only spacing the layers can introduce; the writer's own spacing is left alone
    text = re.sub(r'[ \t]+([,.;:!?)])(?=\s|$)', r'\1', text)
    return re.sub(r'\([ \t]+', '(', text)

# ==========================================
#   PUTTING THE LAYERS TOGETHER
# ==========================================

BULLET = re.compile(r'^\s*(?:[-*•▪◦]|\d+[.)]|[A-Za-z][.)])\s')
PROSE_END = re.compile(r'[.!?]["”’)\]]?\s*$')
LINE_BREAKS = re.compile(r'(\s*\n\s*)')

def is_prose(block):
    """Running text, as opposed to a heading or list item (those only get word-level edits)."""
    return not BULLET.match(block) and bool(PROSE_END.search(block)) and len(block.split()) >= 6

def rewrite_block(doc, tone, deep_mode, prose):
    edits = plan_word_edits(doc, tone, deep_mode)
    if not prose:
        return ''.join(edits.get(t.i, t.text) + t.whitespace_ for t in doc)
    sentences = []
    for sent in doc.sents:
        pieces = [[t.i, edits.get(t.i, t.text)] for t in sent if not t.is_space]
        if not pieces:
            continue
        sentence = Sentence(pieces, doc, sent)
        boost = CHAOS_STRUCTURE_BOOST if deep_mode else 0.0
        if restructurable(sent):
            if random.random() < NOMINAL_RATE[tone] + boost:
                pieces = denominalize(sent, pieces) or pieces
            if random.random() < LIST_RATE[tone] + boost:
                pieces = reorder_list(sent, pieces) or pieces
            sentence.pieces = pieces
            new = restructure(sent, pieces, tone, deep_mode, after_move=bool(sentences and sentences[-1].reordered))
            if new:
                sentence.pieces, sentence.reordered = new, True
        if untouched(sentence.pieces, doc) and sentence.words() >= COVERAGE_MIN_WORDS:
            ensure_change(sentence, tone, deep_mode)
        if restructurable(sent) and not sentence.reordered:
            sentence.splits = split_points(sent, sentence.pieces)
        sentences.append(sentence)
    sentences = vary_rhythm(sentences, tone, deep_mode)
    vary_openings(sentences)
    return ' '.join(soften_dashes(s.text(), tone) for s in sentences)

def rewrite_with_spacy(text, tone, deep_mode):
    parts = LINE_BREAKS.split(text)      # paragraphs and lines at even positions, the breaks between them at odd ones
    jobs = []
    for k in range(0, len(parts), 2):
        block = parts[k]
        core = block.strip()
        if core:
            jobs.append((k, block[:len(block) - len(block.lstrip())], core, block[len(block.rstrip()):]))
    with _NLP_LOCK:
        docs = list(NLP.pipe([core for _, _, core, _ in jobs]))
    for (k, lead, core, trail), doc in zip(jobs, docs):
        parts[k] = lead + tidy(rewrite_block(doc, tone, deep_mode, is_prose(core))) + trail
    return ''.join(parts)

# A word with optional surrounding punctuation, e.g. '"Hello,' -> ('"', 'Hello', ',')
WORD_PATTERN = re.compile(r'^(\W*)([A-Za-z]+)(\W*)$')

# Curated words that are also common as another part of speech ("an attempt", "a myriad of",
# "individual results", "the objective view"); without tagging, the fallback can't tell which is meant
AMBIGUOUS_WORDS = frozenset('''
    leverage foster underscore harness myriad individual purchase attempt objective exhibit showcase
    assist cease component initial streamline bolster endeavor endeavour comprehensive
'''.split())

def single_part_of_speech(word):
    """True if the word can only be one part of speech (so a curated swap can't change its meaning)."""
    return word not in AMBIGUOUS_WORDS and sum(f'{word}|{p}' in SYNONYMS for p in 'nvar') <= 1

def rewrite_basic(text, tone, deep_mode):
    """Fallback without spaCy: only the curated swaps, for words that can't be read as another part of
    speech. Without tagging, WordNet synonyms pick the wrong sense ("land cover" -> "land blanket",
    "despite" -> "scorn"), so they are left out; the phrase rules and contractions still apply."""
    plain_chance = 1.0 if deep_mode and tone != 'academic' else PLAIN_CHANCE[tone]
    tokens = re.split(r'(\s+)', text)
    for i, token in enumerate(tokens):
        match = WORD_PATTERN.match(token)
        if not match:
            continue
        lead, core, trail = match.groups()
        lower = core.lower()
        if not single_part_of_speech(lower):
            continue
        ai_key = next((f'{lower}|{p}' for p in UPOS_TO_WORDNET if f'{lower}|{p}' in AI_WORDS), None)
        plain_key = next((f'{lower}|{p}' for p in UPOS_TO_WORDNET if f'{lower}|{p}' in PLAIN_WORDS), None)
        if (ai_key or plain_key) in POSITION_DEPENDENT:
            continue    # needs the parse to know it's in the right position
        if ai_key:
            replacement = pick(AI_WORDS[ai_key], tone)
        elif plain_key and random.random() < plain_chance:
            replacement = random.choice(PLAIN_WORDS[plain_key])
        else:
            continue
        tokens[i] = lead + match_case(core, replacement) + trail
    return ''.join(tokens)

# ==========================================
#   ENTRY POINTS
# ==========================================

def perform_nuclear_chaos(text, tone, deep_mode):
    text = apply_phrase_rules(text, tone)
    text = apply_tone_contractions(text, tone)
    if NLP is not None:
        return rewrite_with_spacy(text, tone, deep_mode)
    return rewrite_basic(text, tone, deep_mode)

def comparable(word):
    """A word without case or surrounding punctuation, so "Indian," and "Indian." are the same word."""
    return re.sub(r'^\W+|\W+$', '', word.lower())

def calculate_diff_metrics(original, humanized):
    """Compare the rewrite with the original, word by word. Each output word is one of:
      added      - not in the original (shown in red)
      moved      - an original word in a new place, i.e. part of a restructured sentence (yellow underline)
      unchanged  - same word in the same place; the longest such run is also "longest" (blue)
    Case and surrounding punctuation are ignored, so splitting a sentence doesn't mark its words as new."""
    original_words, new_words = original.split(), humanized.split()
    before, after = [comparable(w) for w in original_words], [comparable(w) for w in new_words]

    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    match = matcher.find_longest_match(0, len(before), 0, len(after))
    longest_unchanged = " ".join(original_words[match.a: match.a + match.size])
    structural_change_score = (1 - matcher.ratio()) * 100

    kept_before, kept_after, changes_count = set(), set(), 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            kept_before.update(range(i1, i2))
            kept_after.update(range(j1, j2))
        else:
            changes_count += max(i2 - i1, j2 - j1)

    # Original words that left their place, available to be recognised elsewhere as "moved"
    leftover = Counter(word for i, word in enumerate(before) if i not in kept_before and word)
    kinds = []
    for j, word in enumerate(after):
        if j in kept_after:
            kinds.append('unchanged longest' if match.b <= j < match.b + match.size else 'unchanged')
        elif word and leftover[word] > 0:
            leftover[word] -= 1
            kinds.append('moved')
        else:
            kinds.append('added')

    # Sentences of the output that contain moved words were restructured
    restructured, sentence = set(), 0
    for word, kind in zip(new_words, kinds):
        if kind == 'moved':
            restructured.add(sentence)
        if re.search(r'[.!?]["”’)\]]?$', word):
            sentence += 1

    # Rebuild the output with its original whitespace, escaping the text for safe HTML. Neighbouring
    # words of the same kind share one span, so a highlighted run includes the spaces inside it.
    diff_parts, open_kind, spacing, word_index = [], None, '', 0
    for token in re.split(r'(\s+)', humanized):
        if not token:
            continue
        if token.isspace():
            spacing += token
            continue
        kind = kinds[word_index]
        word_index += 1
        if kind != open_kind:
            if open_kind is not None:
                diff_parts.append('</span>')
            diff_parts.append(spacing + f'<span class="{kind}">')
            open_kind = kind
        else:
            diff_parts.append(spacing)
        diff_parts.append(html.escape(token))
        spacing = ''
    if open_kind is not None:
        diff_parts.append('</span>')
    diff_parts.append(spacing)

    return {
        "diff_html": "".join(diff_parts),
        "longest_unchanged": longest_unchanged if longest_unchanged else "None",
        "structural_score": round(structural_change_score, 1),
        "changed_words_count": changes_count,
        "new_words_count": kinds.count('added'),
        "moved_words_count": kinds.count('moved'),
        "restructured_sentences": len(restructured),
    }
