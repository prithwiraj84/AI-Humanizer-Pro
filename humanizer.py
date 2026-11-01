"""
Humanizer engine: rewrites text in a chosen tone and measures what changed.

Pipeline (perform_nuclear_chaos):
  1. Phrase rules  - AI-style transitions, fillers and wordy phrases ("Furthermore,",
                     "It is important to note that", "in order to") become plainer wording.
  2. Contractions  - Casual adds them; Professional and Academic expand them.
  3. spaCy pass    - using part-of-speech tags, lemmas, named entities and the parse:
                     * AI-favourite words ("utilize", "delve", "robust") get plain replacements
                     * other content words get WordNet synonyms of the same part of speech and
                       sense, re-inflected to match ("demonstrates" -> "shows")
                     * names, places, organisations, dates, numbers and acronyms are never touched
                     * "a"/"an" is corrected after a swap
                     * Chaos Mode and Casual split long compound sentences in two
If spaCy isn't installed, step 3 falls back to plain word swaps so the app still runs.
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

logger = logging.getLogger(__name__)

TONES = ('standard', 'professional', 'casual', 'academic')

# --- SYNONYM TABLE (built offline with NLTK by scripts/build_synonyms.py) ---
# "lemma|pos" -> [lemma_frequency, [[synonym, frequency], ...]], best synonyms first
with gzip.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'synonyms.json.gz'),
               'rt', encoding='utf-8') as _f:
    SYNONYMS = json.load(_f)

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

def apply_phrase_rules(text, tone):
    for pattern, chance, options in TRANSITION_RULES:
        text = pattern.sub(lambda m: m.group('pre') + pick(options, tone) + ' ' if random.random() < chance else m.group(0), text)

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

CONTRACT_PATTERN = re.compile(r"\b" + phrase_regex(CONTRACTIONS) + r"\b", re.I)
EXPAND_PATTERN = re.compile(r"\b" + phrase_regex(EXPANSIONS) + r"\b", re.I)

def apply_tone_contractions(text, tone):
    if tone == 'casual':
        pattern, table = CONTRACT_PATTERN, CONTRACTIONS
    elif tone in ('professional', 'academic'):
        pattern, table = EXPAND_PATTERN, EXPANSIONS
    else:
        return text

    def replace(match):
        key = re.sub(r'\s+', ' ', match.group(0).lower()).replace('’', "'")
        return match_case(match.group(0), table[key])

    return pattern.sub(replace, text)

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

# Nouns carry most of a sentence's meaning and WordNet's noun synonyms often miss the context
# ("updated the details" is not "updated the points"), so only Chaos Mode swaps them, and half as often
CHAOS_NOUN_WEIGHT = 0.5

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
}
# Chance of making a plain-English swap; Academic keeps its formal vocabulary
PLAIN_CHANCE = {'casual': 1.0, 'standard': 0.7, 'professional': 0.4, 'academic': 0.0}
IRREGULAR_PLURALS = {'person': 'people'}

def pick_synonym(entry, tone, floor, avoid):
    lemma_frequency, candidates = entry
    pool = [word for word, frequency in candidates if frequency >= floor * lemma_frequency and word not in avoid]
    if not pool:
        return None
    if tone == 'casual':
        return min(pool[:4], key=len)
    if tone == 'academic':
        return max(pool, key=len)
    if tone == 'professional':
        return pool[0]
    return random.choice(pool[:3])

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

def plan_sentence_split(sent, edits, capitalize):
    """Turn "X, and Y" into "X. Y" when both halves are long, complete clauses."""
    if len(sent) < 22:
        return
    for token in sent:
        if token.dep_ != 'cc' or token.lower_ not in ('and', 'but') or token.head.dep_ != 'ROOT':
            continue
        if token.i == sent.start or token.doc[token.i - 1].text != ',':
            continue
        clause = next((c for c in token.head.children if c.dep_ == 'conj' and c.i > token.i), None)
        if clause is None or clause.pos_ not in ('VERB', 'AUX'):
            continue
        if not any(c.dep_ in ('nsubj', 'nsubjpass', 'expl') for c in clause.children):
            continue
        if token.i - sent.start < 8 or sent.end - token.i < 8:
            continue
        edits[token.i - 1] = '.'
        if token.lower_ == 'and':
            edits[token.i] = ''
            capitalize.add(token.i + 1)
        else:
            edits[token.i] = 'But'
        return

def rewrite_with_spacy(text, tone, deep_mode):
    with _NLP_LOCK:
        doc = NLP(text)

    intensity = 0.4 if deep_mode else 0.1
    floor = CHAOS_FLOOR if deep_mode else SYNONYM_FLOOR[tone]
    plain_chance = (1.0 if deep_mode and tone != 'academic' else PLAIN_CHANCE[tone])
    edits = {}         # token index -> new text ('' drops the token and its trailing space)
    capitalize = set()

    for sent in doc.sents:
        avoid = {t.lower_ for t in sent} | {t.lemma_.lower() for t in sent}
        for token in sent:
            if not can_rewrite(token):
                continue
            lemma = token.lemma_.lower()
            key = f'{lemma}|{token.pos_}'
            weight = (CHAOS_NOUN_WEIGHT if deep_mode else 0.0) if token.pos_ == 'NOUN' else 1.0
            if key in AI_WORDS:
                replacement = pick(AI_WORDS[key], tone)
            elif key in PLAIN_WORDS and random.random() < plain_chance:
                replacement = random.choice(PLAIN_WORDS[key])
            elif token.ent_type_ or takes_preposition(token):
                continue  # number phrases and "rely on"-style pairs only get the curated swaps above
            elif random.random() < intensity * weight:
                entry = SYNONYMS.get(f'{lemma}|{UPOS_TO_WORDNET[token.pos_]}')
                replacement = pick_synonym(entry, tone, floor, avoid) if entry else None
            else:
                continue
            word = inflect(replacement, token.tag_) if replacement else None
            if word and word.lower() != token.lower_:
                edits[token.i] = match_case(token.text, word)
                avoid.add(replacement)
        if deep_mode or tone == 'casual':
            plan_sentence_split(sent, edits, capitalize)

    # "a" / "an" must agree with a swapped word that follows it
    for i in list(edits):
        if i == 0 or not edits[i] or (i - 1) in edits or doc[i - 1].lower_ not in ('a', 'an'):
            continue
        wanted = 'an' if starts_with_vowel_sound(edits[i]) else 'a'
        if doc[i - 1].lower_ != wanted:
            edits[i - 1] = match_case(doc[i - 1].text, wanted)

    out = []
    for token in doc:
        new = edits.get(token.i, token.text)
        if token.i in capitalize and new:
            new = new[0].upper() + new[1:]
        out.append(new + ('' if new == '' else token.whitespace_))
    return ''.join(out)

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

def calculate_diff_metrics(original, humanized):
    original_words = original.split()
    new_words = humanized.split()

    matcher = difflib.SequenceMatcher(None, original_words, new_words, autojunk=False)
    match = matcher.find_longest_match(0, len(original_words), 0, len(new_words))
    longest_unchanged = " ".join(original_words[match.a: match.a + match.size])
    similarity = matcher.ratio()
    structural_change_score = (1 - similarity) * 100

    added_indexes = set()
    changes_count = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        added_indexes.update(range(j1, j2))
        changes_count += max(i2 - i1, j2 - j1)

    # Rebuild the output with its original whitespace, escaping the text for safe HTML
    diff_parts = []
    word_index = 0
    for token in re.split(r'(\s+)', humanized):
        if not token:
            continue
        if token.isspace():
            diff_parts.append(token)
            continue
        css_class = "added" if word_index in added_indexes else "unchanged"
        diff_parts.append(f'<span class="{css_class}">{html.escape(token)}</span>')
        word_index += 1

    return {
        "diff_html": "".join(diff_parts),
        "longest_unchanged": longest_unchanged if longest_unchanged else "None",
        "structural_score": round(structural_change_score, 1),
        "changed_words_count": changes_count
    }
