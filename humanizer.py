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
import bisect
import difflib
import functools
import gzip
import html
import json
import logging
import os
import random
import re
import threading
import time
import unicodedata
from collections import Counter

import paraphraser

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
# spaCy's parser sometimes ends a sentence after an abbreviation ("2 tbsp. | of butter", "built ca. | 1890") or
# with no punctuation at all ("asked me for directions | and I answered"). Each piece would then be rewritten as a
# sentence of its own, so a word may only start a sentence after a line break, after closing punctuation or a
# symbol (an emoji can end a sentence in a post), and - after a full stop - when it isn't a lower-case word or a
# number following an abbreviation.
ABBREVIATIONS = {'ca', 'approx', 'appr', 'tbsp', 'tsp', 'oz', 'lb', 'lbs', 'no', 'nos', 'vol', 'fig', 'figs', 'pp',
                 'vs', 'est', 'ed', 'eds', 'incl', 'excl', 'dept', 'govt', 'min', 'max', 'sec', 'hr', 'hrs', 'wk',
                 'mo', 'yr', 'yrs', 'sq', 'ft', 'st', 'ave', 'rd', 'blvd', 'inc', 'ltd', 'co', 'corp', 'jr', 'sr',
                 'dr', 'mr', 'mrs', 'ms', 'prof', 'mt', 'etc', 'cf', 'al', 'op', 'cit', 'ibid', 'nr', 'pt', 'pts'}
SENTENCE_CLOSERS = set('"”’\')]}»')
# Inline code ("`git add .`") is one piece: no sentence starts inside it, so it is never split and joined again
INLINE_CODE = re.compile(r"(?<!\w)`[^`\n]+`(?!\w)")

def may_start_sentence(token, j):
    """`j` is the index of the nearest token before this one that isn't a closing mark (-1 if there is none)."""
    doc, before = token.doc, token.doc[token.i - 1]
    if '\n' in before.whitespace_ or (before.is_space and '\n' in before.text):
        return True
    if j < 0:
        return True
    prev = doc[j]
    if prev.is_space and '\n' in prev.text:
        return True
    if prev.text[-1].isalnum() or prev.text[-1] in ',-–—(/&':
        return False
    if prev.text.endswith('.'):
        if token.text[:1].islower() and not any(c.isupper() for c in token.text[1:]):
            return False   # "2 tbsp. of butter", "Baker Ave. in 2019", 'She said "Stop." then left'
        if token.text[:1].isdigit():
            word = doc[j - 1] if prev.text == '.' and j > 0 else prev
            w = word.text.rstrip('.').lower()
            if word.text[:1].islower() and (w in ABBREVIATIONS
                                            or (len(w) <= 6 and w.isalpha() and not getAllLemmas(w))):
                return False   # "built ca. 1890", "weighs approx. 3 kg"
    return True

def whole_sentences(doc):
    """Pipeline step before the parser: marks the words that can't start a sentence (the parser keeps to it). The
    nearest word before each token that isn't a closing mark is carried along, so a long run of quotes or brackets
    costs one pass, not a walk back from every token; and the marks are written in one go, since spaCy looks over the
    whole text each time one token's mark is set."""
    in_code = set()
    if '`' in doc.text:
        for m in INLINE_CODE.finditer(doc.text):
            span = doc.char_span(m.start(), m.end(), alignment_mode='expand')
            if span is not None:
                in_code.update(t.i for t in span if t.idx > m.start())
    blocked, last = [], -1
    for token in doc:
        if token.i and not token.is_space and (token.i in in_code or not may_start_sentence(token, last)):
            blocked.append(token.i)
        if token.text not in SENTENCE_CLOSERS:
            last = token.i
    if blocked:
        starts = doc.to_array(SPACY_SENT_START)
        starts[blocked] = NOT_A_SENTENCE_START
        doc.from_array([SPACY_SENT_START], starts)
    return doc

try:
    import numpy
    import spacy
    from spacy.attrs import SENT_START as SPACY_SENT_START
    from spacy.language import Language
    from lemminflect import getAllLemmas, getInflection
    NOT_A_SENTENCE_START = numpy.iinfo(numpy.uint64).max     # how spaCy's arrays hold the mark "False" (-1)
    NLP = spacy.load('en_core_web_sm')
    if not Language.has_factory('whole_sentences'):
        Language.component('whole_sentences', func=whole_sentences)
    NLP.add_pipe('whole_sentences', before='parser')
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

APOSTROPHE_CLITIC = re.compile(r"['’](?:s|re|ve|ll|d|m)", re.I)

def apostrophe_attached(index, text, doc):
    """True for a piece written against the word before it although it starts with an apostrophe: a contraction,
    the possessive ("it's", "people's", "students'") and a closing quote mark. An opening or elision mark isn't
    ("'em", "'Tis", "'90s", where spaCy splits the mark off the word)."""
    if APOSTROPHE_CLITIC.fullmatch(text):
        return True
    return text in ("'", "’") and not (index is not None and not doc[index].whitespace_ and index + 1 < len(doc)
                                       and doc[index + 1].text[:1].isalnum())

# ==========================================
#   TEXT HYGIENE
# ==========================================
# Text pasted from chat tools often carries invisible characters (zero-width spaces, soft hyphens,
# direction marks), odd spaces (non-breaking, narrow) and the occasional look-alike letter from another
# alphabet. They break word counts, spell-checkers, search and screen readers, and Turnitin reports hidden
# and replaced characters to instructors as deliberate manipulation. Input is cleaned before rewriting,
# so the output never contains them.

INVISIBLE = re.compile('[­᠎​‌‎‏‪-‮⁠-⁤⁦-⁩﻿]')
ODD_SPACES = re.compile('[  -   　]')
JOINER_IN_WORD = re.compile(r'(?<=\w)‍|‍(?=\w)')     # zero-width joiners stay only inside emoji
LOOKALIKES = str.maketrans({
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x', 'і': 'i', 'ј': 'j', 'ѕ': 's', 'һ': 'h', 'ԁ': 'd',
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T', 'Х': 'X',
    'У': 'Y', 'І': 'I', 'Ј': 'J', 'Ѕ': 'S', 'ο': 'o', 'ν': 'v', 'ρ': 'p', 'ι': 'i', 'κ': 'k', 'α': 'a',
    'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Ζ': 'Z', 'Η': 'H', 'Ι': 'I', 'Κ': 'K', 'Μ': 'M', 'Ν': 'N', 'Ο': 'O', 'Ρ': 'P',
    'Τ': 'T', 'Υ': 'Y', 'Χ': 'X',
})
LATIN_LETTER = re.compile('[A-Za-z]')
NON_LATIN_LETTER = re.compile('[Ͱ-Ͽἀ-῿Ѐ-ԯ]')      # Greek and Cyrillic
WORD_RUN = re.compile(r'\w+')
# Line and page breaks of other systems become ordinary line breaks (a lone "\r" from old Mac text, form
# feeds from PDFs); any other control character is dropped
OTHER_BREAKS = re.compile('\r(?!\n)|[\x0b\x0c\x1c-\x1e\x85  ]')
CONTROL_CHARACTERS = re.compile('[\x00-\x08\x0e-\x1b\x1f\x7f-\x84\x86-\x9f]')

def fix_lookalikes(m):
    """A Latin word with stray Greek or Cyrillic look-alikes ("Аpple") is fixed. A Greek or Cyrillic word
    with a stray Latin letter ("Мiсто", typed on a mixed keyboard) stays as typed: converting its look-alikes
    would make it more mixed, not less."""
    word = m.group(0)
    fixed = word.translate(LOOKALIKES)
    if fixed != word and LATIN_LETTER.search(word) and not NON_LATIN_LETTER.search(fixed):
        return fixed
    return word

def clean_text(text):
    text = unicodedata.normalize('NFC', text)
    text = CONTROL_CHARACTERS.sub('', OTHER_BREAKS.sub('\n', text))
    text = INVISIBLE.sub('', JOINER_IN_WORD.sub('', text))
    text = ODD_SPACES.sub(' ', text)
    return WORD_RUN.sub(fix_lookalikes, text)

# --- Quotations are someone's exact words, so no layer changes them ---
# Double quotes, and single quotes as British writers use them. A straight quote opens a quotation only after a
# space, bracket or dash (so the inch mark of 6.1" is not one), and a straight single quote only before a
# letter. A quotation may be hard-wrapped onto the next line, and one left open runs to the end of its paragraph.
_QUOTED_CHARACTER = r'(?:[^{0}\n]|\n(?![ \t]*\n){1})'

# The mark of an elision ("open 'til midnight", "Let 'em know", "‘Tis the season") opens no quotation, unless
# one that starts with the elision closes after its own punctuation within 300 characters ("'Tis the season,'
# she said")
ELISIONS = r'(?:[Tt]is|[Tt]was|[Tt]ill?|[Ee]m|[Cc]ause|[Cc]os|[Bb]out|[Rr]ound|[Nn]|[Nn]uff|[Cc]ept|[Nn]eath|[Ss]pose)\b'

def single_quoted_character(mark):
    """A character of a quotation closed by `mark`. A mark followed by a letter is an apostrophe ("‘We don’t
    know’"), and so is a plural's before a lower-case word when the closing mark follows on the line with no
    other quotation opening on the way ("‘The teachers’ union won’t accept it’, she said"; an elision or a decade
    opens none: "’til", "the ’90s"). Another plural's mark on the way is an apostrophe too and closes nothing
    ("‘The teachers’ and nurses’ unions won’t accept it,’"). So two scare quotes ("'experts' say so, and their
    'solutions' ...") are two quotations, and so is one before a plural's possessive ("the 'experts' warn that
    the workers' pay will fall"): the mark after "experts" closes the quotation. The closing mark is looked for
    within 500 characters."""
    return _QUOTED_CHARACTER.format(
        mark, r'|{0}(?=\w)|(?<=s){0}(?=[ \t]+[a-z](?:[^{0}‘\n]|(?<=\w){0}(?=\w)|(?<!\w)[{0}‘](?=\d|{1})|(?<=s){0}'
              r'(?=[ \t]+[a-z])){{0,500}}(?:(?<!s){0}|{0}(?![ \t]+[a-z]))(?!\w))'.format(mark, ELISIONS))

STRAIGHT_OPENING = r"(?<![^\s(\[{—–])'"
# Triple quotes ('"""The views are stunning,""" she wrote'), guillemets («...», ‹...›) and German low-high quotes
# („...“) are quotations too, and so is a straight double quote right after a mark of emphasis (* _ ~), a comma, a
# colon, a semicolon or "=" when a letter or digit follows ('*"We utilize ..."*', 'She wrote,"We ..."', 'alt="A view"'):
# a closing quote after a comma has a space after it ('"Yes," she said'), so it opens nothing.
TRIPLE_QUOTED_CHARACTER = r'(?:[^{0}\n]|{0}(?!{0}{0})|\n(?![ \t]*\n))'
QUOTED = re.compile(
    r'"""' + TRIPLE_QUOTED_CHARACTER.format('"') + r'*(?:"""|$)'
    + r"|'''" + TRIPLE_QUOTED_CHARACTER.format("'") + r"*(?:'''|$)"
    + r'|"(?<=[*_~,:;=]")(?=[A-Za-z0-9‘\'])' + _QUOTED_CHARACTER.format('"', '') + r'*(?:"|$)'
    + r'|«' + _QUOTED_CHARACTER.format('»', '') + r'*(?:»|$)'
    + r'|‹' + _QUOTED_CHARACTER.format('›', '') + r'*(?:›|$)'
    + r'|„' + _QUOTED_CHARACTER.format('“”', '') + r'*(?:[“”]|$)'
    + r'|(?<![^\s(\[{—–])"(?=\S)' + _QUOTED_CHARACTER.format('"', '') + r'*(?:"|$)'
    + r'|“' + _QUOTED_CHARACTER.format('”', '') + r'*(?:”|$)'
    + r'|‘(?=[^\d\s])(?!' + ELISIONS + ')' + single_quoted_character('’') + r'*(?:’|$)'
    + '|' + STRAIGHT_OPENING + r'(?=[A-Za-z])(?!' + ELISIONS + ')' + single_quoted_character("'") + r"*(?:'(?!\w)|$)"
    + r'|‘(?=' + ELISIONS + ')' + single_quoted_character('’') + r'{1,300}(?<=[,.!?…—–-])’'
    + '|' + STRAIGHT_OPENING + '(?=' + ELISIONS + ')' + single_quoted_character("'") + r"{1,300}(?<=[,.!?…—–-])'(?!\w)",
    re.M)

# Inline code and fenced code blocks are left as written too ("npm run optimize" is a command, not prose). A fence
# of three or more backticks opens on a line of its own ("```bash"), or after text when only its language follows on
# that line ("Here is the code: ```python"), and closes at the start of a line with as many (a "````" block may show
# a "```" one) or at the end of a code line ("npm run optimize```"). A backtick in prose ("Wrap code in ``` fences",
# or typed for an apostrophe: "We don`t use it") opens nothing. A fence that opens after text only begins where its
# run of backticks does, so a long run of them is read once, not once for every backtick in it.
# A fence of three or more tildes ("~~~") works the same way on a line of its own, unless it underlines the line of text
# above it, as a heading may be underlined ("Results\n~~~~~~~"). After a blank line, lines indented by four spaces or a
# tab are a code block when every one of them looks like code: it doesn't end like a sentence ("npm run optimize"), or
# it holds "=", "()", "{", "}" or "--". An indented paragraph of prose ends like one and is rewritten.
INDENTED_CODE_LINE = r'(?: {4}|\t)(?=[^\n]*(?:=|\(\)|[{}]|--)|[^\n]*[^.!?"”’\s][ \t]*$)[^\n]*'
CODE_SPANS = re.compile(r'(?:^[ \t]*|(?=[ \t`])(?:(?<=[^\s`])[ \t]*|(?<=`)[ \t]+)(?=`{3,}[\w+#-]*[ \t]*\n))(`{3,})[^`\n]*\n.*?'
                        r'(?:^[ \t]*\1|(?<=\S)\1(?=[ \t]*$)|\Z)'
                        r'|^(?<![^\s:]\n)[ \t]*(~{3,})[^~\n]*\n.*?(?:^[ \t]*\2|\Z)'
                        r'|(?:\A|^(?<=\n\n))' + INDENTED_CODE_LINE + r'(?:\n' + INDENTED_CODE_LINE
                        + r')*(?=\n[ \t]*(?:\n|\Z)|\Z)|' + INLINE_CODE.pattern, re.S | re.M)
# So are "'Tis" and "'Twas", and with them the order of their sentence: the parser takes them for a verb, and a phrase
# moved on that reading garbles the sentence ("To use our holiday deals, 'Tis the season")
TIS_TWAS = re.compile(r"(?<!\w)['’‘][Tt](?:is|was)\b")
# And so is markup: a tag with its attribute values ('<img alt="A robust bridge">'), a comment, and the content of a
# script, style, pre or code element. A tag has a letter right after its "<", so "x < y" in prose is none. The content
# of an element ends at the next tag of its kind, and a comment at the next comment, so an unclosed one is read once,
# not once for every later one.
HTML_SPANS = re.compile(r'<(script|style|pre|code)\b[^>]*>(?:(?!</?\1\b).)*</\1\s*>|<!--(?:(?!<!--).)*?-->'
                        r'|</?[A-Za-z][\w:-]*(?:\s[^<>]*)?/?>', re.S | re.I)

@functools.lru_cache(maxsize=32)
def quoted_ranges(text):
    """Where the text is kept as written (quotations, code, "'Tis", markup), as (start, end) pairs."""
    return tuple(sorted((m.start(), m.end()) for pattern in (QUOTED, CODE_SPANS, TIS_TWAS, HTML_SPANS)
                        for m in pattern.finditer(text)))

@functools.lru_cache(maxsize=32)
def quote_reach(text):
    """The starts of the text's quoted ranges in order, and for each the furthest end among it and the ranges before it:
    a stretch from a to b overlaps a quotation when the furthest end of the ranges starting before b lies after a. One
    binary search then answers, not a pass over every quotation of a text full of them."""
    starts, reach, furthest = [], [], -1
    for start, end in quoted_ranges(text):
        furthest = max(furthest, end)
        starts.append(start)
        reach.append(furthest)
    return starts, reach

def overlaps_quote(text, start, end):
    starts, reach = quote_reach(text)
    k = bisect.bisect_left(starts, end)
    return k > 0 and reach[k - 1] > start

def in_quote(m):
    """True if a regex match overlaps a quotation in the text it was found in."""
    return overlaps_quote(m.string, m.start(), m.end())

def guarded(fn):
    """A substitution callback that leaves matches inside quotations alone."""
    return lambda m: m.group(0) if in_quote(m) else fn(m)

def quoted_tokens(doc):
    """Indices of the tokens inside quotations, worked out once per parsed text."""
    if 'quoted' not in doc.user_data:
        text = doc.text
        doc.user_data['quoted'] = frozenset(t.i for t in doc if overlaps_quote(text, t.idx, t.idx + 1))
    return doc.user_data['quoted']

def has_quoted(doc, indices):
    quoted = quoted_tokens(doc)
    return bool(quoted) and any(i in quoted for i in indices if i is not None)

# "a 8% rise" -> "an 8% rise", "an 13% rise" -> "a 13% rise": numbers that are read with a vowel sound
# start with 8, or are 11 or 18 (also as 11,000 / 18 million, and years like 1886)
ARTICLE_BEFORE_NUMBER = re.compile(r'\b([Aa]n?)(\s+)(\d[\d,]*)(?=[\s%.\-]|$)')
OPENS_SENTENCE = re.compile(r'(?:[.!?]["”’)\]]*\s+|\n[ \t]*(?:(?:[-*•▪◦]|\d+[.)])[ \t]+)?)\Z')

def number_takes_an(number):
    digits = number.replace(',', '')
    return (digits.startswith('8') or (digits[:2] in ('11', '18') and len(digits) in (2, 5, 8))
            or (len(number) == 4 and number[:2] in ('11', '18')))

def fix_number_articles(text, source=''):
    """Make "a"/"an" agree with a number after it, where the rewriting put them together. Pairs the writer
    typed are left as they are (a label like "vitamin A 800 IU", or "a" in another language)."""
    written = {m.group(0) for m in ARTICLE_BEFORE_NUMBER.finditer(source)} if source else set()

    def fix(m):
        if m.group(0) in written:
            return m.group(0)
        if m.group(1)[0] == 'A':
            head = m.string[:m.start()].rstrip('"“‘([')
            if head.strip() and not OPENS_SENTENCE.search(head[-40:]):
                return m.group(0)     # a capital "A" inside a sentence is a label ("Plan A 18 months")
        elif re.search(r'\d[ \t]+\Z', m.string[max(0, m.start() - 20):m.start()]):
            return m.group(0)         # "de 8 a 11": a range in Spanish or Portuguese
        wanted = 'an' if number_takes_an(m.group(3)) else 'a'
        return (wanted if m.group(1).islower() else wanted.capitalize()) + m.group(2) + m.group(3)
    return ARTICLE_BEFORE_NUMBER.sub(fix, text)

# ==========================================
#   STEP 1: PHRASE RULES
# ==========================================

# (sentence openers, chance of rewriting, replacements by tone)
TRANSITIONS = [
    # The replacements avoid "In addition," and "Nevertheless,": they are on the same lists of stock openers
    (('furthermore', 'moreover', 'additionally', 'in addition', "what's more"), 1.0, {
        'standard': ['Also,', 'Plus,', 'On top of that,', 'Beyond that,'],
        'casual': ['Also,', 'Plus,', 'On top of that,'],
        'professional': ['Also,', 'Beyond that,'],
        'academic': ['Further,', 'Beyond this,', 'Also,'],
    }),
    (('in conclusion', 'to conclude', 'in summary', 'to summarize', 'to sum up', 'overall', 'all in all'), 1.0, {
        'standard': ['In short,', 'All told,'],
        'casual': ['So,', 'In short,'],
        'professional': ['In short,', 'All told,'],
        'academic': ['In sum,', 'Taken together,'],
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
        'academic': ['Even so,', 'Yet'],
    }),
    (('on the other hand',), 0.8, {
        'standard': ['Then again,', 'By contrast,'],
        'casual': ['Then again,', 'But'],
        'professional': ['By contrast,', 'Then again,'],
        'academic': ['By contrast,', 'Conversely,'],
    }),
    # "Finally," would make a summing-up a step in a sequence, and the long idioms are clichés themselves
    (('ultimately',), 1.0, {'*': ['In the end,']}),
    # "Essentially," sums up rather than concludes, so it isn't "In the end,"
    (('in essence', 'essentially'), 0.8, {
        'standard': ['Basically,', 'In short,'], 'casual': ['Basically,'], 'professional': ['In short,'],
        'academic': ['In essence,', 'At heart,'],
    }),
    (('first and foremost', 'firstly', 'first of all'), 1.0, {'*': ['First,']}),
    (('secondly',), 1.0, {'*': ['Second,']}),
    (('thirdly',), 1.0, {'*': ['Third,']}),
    (('last but not least', 'lastly'), 1.0, {'*': ['Finally,', 'Lastly,']}),
]
# Stock openers with a plainer one of the same kind: contrast stays contrast ("In contrast," -> "By contrast,"), time
# stays time, a share of cases says how often ("In rare cases," -> "Rarely,"), a last step stays a step. None of them
# is a stock opener to avoid at any cost, so once its replacements are used up in a text, the next one stays as
# written rather than repeat one ("Rarely, ... Rarely, ...").
FIRST_PLAIN_OPENER = len(TRANSITIONS)
TRANSITIONS += [
    (('in contrast',), 1.0, {'*': ['By contrast,'], 'casual': ['By contrast,', 'But']}),
    (('in recent years',), 1.0, {'*': ['Over the past few years,', 'In the past few years,']}),
    (('in rare cases', 'in rare instances'), 1.0, {'*': ['Rarely,']}),
    (('in some cases', 'in certain cases', 'in some instances'), 1.0, {'*': ['Sometimes,']}),
    (('in many cases',), 1.0, {'*': ['Often,']}),
    (('in most cases',), 1.0, {'*': ['Usually,']}),
    (('all things considered', 'everything considered', 'taking everything into account'), 1.0, {'*': ['On balance,']}),
    (('that being said', 'having said that', 'with that being said', 'with that said'), 1.0,
     {'*': ['Even so,', 'Still,'], 'academic': ['Even so,']}),
    (('as a matter of fact',), 1.0, {'*': ['In fact,']}),
    (('similarly', 'in a similar vein', 'along the same lines'), 0.8,
     {'*': ['Likewise,', 'In the same way,'], 'academic': ['Likewise,']}),
    (('subsequently',), 1.0, {'*': ['Later,', 'After that,'], 'academic': ['Later,']}),
    (('initially',), 1.0, {'*': ['At first,']}),
    (('currently',), 1.0, {'*': ['At the moment,', 'Right now,'], 'professional': ['At present,', 'At the moment,'],
                           'academic': ['At present,']}),
    # Both are fine; the one the writer used becomes the other
    (('for example', 'for instance'), 0.5, {'*': ['For instance,', 'For example,']}),
    (('finally',), 1.0, {'*': ['Lastly,', 'To finish,']}),
    (('truth be told', 'to tell the truth'), 1.0, {'*': ['In truth,', 'Honestly,'], 'casual': ['Honestly,', 'To be honest,'],
                                                   'professional': ['In truth,'], 'academic': ['In truth,']}),
    (('to get started',), 1.0, {'*': ['To start,', 'To begin,']}),
    (('most importantly', 'above all else'), 1.0, {'*': ['Above all,']}),
    (('in a nutshell',), 1.0, {'*': ['In short,']}),
    (('going forward', 'moving forward'), 1.0, {'*': ['From now on,']}),
    (('generally speaking', 'broadly speaking'), 1.0, {'*': ['In general,', 'Generally,']}),
    (('undoubtedly', 'without a doubt', 'unquestionably'), 1.0, {'*': ['Certainly,']}),
    (('in light of this', 'in light of that', 'in view of this'), 1.0, {'*': ['Given this,']}),
    # The other side of the same point, not a comparison of two things ("By contrast,")
    (('on the flip side',), 1.0, {'*': ['Then again,']}),
    (('by the same token',), 1.0, {'*': ['Likewise,']}),
    (('as mentioned earlier', 'as mentioned above', 'as previously mentioned', 'as noted earlier', 'as noted above',
      'as discussed earlier', 'as stated earlier'), 1.0, {'*': ['As noted,']}),
    (('to achieve this', 'to accomplish this'), 1.0, {'*': ['To do this,']}),
    (('to illustrate',), 1.0, {'*': ['For example,', 'For instance,']}),
    (('by doing so',), 1.0, {'*': ['This way,', 'That way,']}),
]

# Some openers only have their plain equivalent in one use:
# - a share of cases is a frequency, but before an example the opener introduces it ("In some cases, such as those
#   involving children, the court ..."), and after cases that are things ("The suitcases arrived damaged. In some
#   cases, the zips ...", "The court heard 40 cases. In some cases, ...") it means some of those, not "Sometimes"
OPENER_EXAMPLE_FOLLOWS = re.compile(r"(?:such\s+as|like|including|especially|particularly|notably|for\s+(?:example|"
                                    r"instance)|e\.g\.)(?!\w)", re.I)
OPENER_CASES_NEARBY = re.compile(r"cases?\b", re.I)

def opener_share_of_cases(m):
    return (not OPENER_EXAMPLE_FOLLOWS.match(m.string, m.end())
            and not OPENER_CASES_NEARBY.search(m.string, max(0, m.start() - 200), m.start()))

# - "Finally," before an instruction is the last step ("Finally, stir in the parmesan" -> "Lastly, stir in ..."); before
#   a statement it may mean "at last" ("Finally, the rain stopped", "Finally, light reached the cave"). A word that is
#   also a noun or an adjective is an instruction only with its object after it ("Finally, water the plants", "Finally,
#   clean the brushes"; but "Finally, dry land.", "Finally, open water appeared").
OPENER_NEXT_WORDS = re.compile(r"(?P<verb>[a-z]+)(?:[ \t]+(?P<next>[A-Za-z]+))?")

def opener_instruction_follows(m):
    found = OPENER_NEXT_WORDS.match(m.string, m.end())
    if not found or not base_verb(found.group('verb')):
        return False
    after = PHRASE_FUNCTION_WORD.match((found.group('next') or '').lower())
    return not word_classes(found.group('verb')) & {'NOUN', 'ADJ'} or bool(after and after.group('determiner'))

# - "Going forward," and "Moving forward," mean "from now on" before a plan made by people or an organisation, not in a
#   story or about a machine ("Moving forward, the troops will cross the river", "Going forward, the car needs more
#   torque"): "we", "I", "you" or a group of people plans, and a plan word follows
OPENER_PLAN_FOLLOWS = re.compile(r"[^.!?\n]*?\b(?:will|['’]ll|plans?|intends?|aims?|needs?|must|should|going\s+to|wants?)\b", re.I)
OPENER_PLANNERS = re.compile(r"[ \t]*(?:(?:we|I|you|our|everyone|both[ \t]+sides|management|leadership)\b|(?:(?:the|our|your|"
                             r"their|all|both|many|most|its)[ \t]+)?(?P<who>[a-z]+)\b)")
PLANNING_BODIES = {'company', 'team', 'business', 'organisation', 'organization', 'council', 'government', 'board',
                   'firm', 'department', 'school', 'group', 'club', 'charity', 'industry', 'sector', 'agency',
                   'ministry', 'university', 'leaders', 'regulators', 'policymakers', 'stakeholders', 'sides'}

def opener_plan_follows(m):
    planner = OPENER_PLANNERS.match(m.string, m.end())
    who = planner.group('who') if planner else None
    if not planner or (who is not None and who not in ALLOWED_PEOPLE and who not in PLANNING_BODIES):
        return False
    return bool(OPENER_PLAN_FOLLOWS.match(m.string, m.end()))

# - "To get started," is the first step of a task for a person ("To get started, you will need ...", "To get started,
#   download the app"), not a machine being started ("To get started, the mower needs fresh fuel")
OPENER_TASK_SUBJECT = re.compile(r"[ \t]*(?:you|we|I|let['’]s|let[ \t]+us)\b")

def opener_task_follows(m):
    return bool(OPENER_TASK_SUBJECT.match(m.string, m.end())) or opener_instruction_follows(m)

# - "To illustrate," introduces an example, unless it is someone's purpose in a story ("To illustrate, she sketched a
#   map"): a pronoun or a past tense right after it keeps the opener
OPENER_SIX_WORDS = re.compile(r"[ \t]*([A-Za-z'’]+)(?:[ \t,]+([A-Za-z'’]+))?(?:[ \t,]+([A-Za-z'’]+))?(?:[ \t,]+([A-Za-z'’]+))?"
                              r"(?:[ \t,]+([A-Za-z'’]+))?(?:[ \t,]+([A-Za-z'’]+))?")
OPENER_PRONOUNS = {'i', 'you', 'we', 'they', 'he', 'she', 'it'}

def simple_past_form(word):
    """True if a word in lower case is the simple past of a verb ("sketched", "drew"); False without lemminflect."""
    return 'getInflection' in globals() and any(word in getInflection(lemma, tag='VBD') and word != lemma
                                                 for lemma in getAllLemmas(word).get('VERB', ()))

def opener_introduces_example(m):
    found = OPENER_SIX_WORDS.match(m.string, m.end())
    words = [w.lower() for w in found.groups() if w] if found else []
    return bool(words) and words[0] not in OPENER_PRONOUNS and not any(simple_past_form(w) for w in words)

# - "By doing so," is "This way," when it gives what the step makes possible ("By doing so, you can cut your bills"),
#   not what someone did ("By doing so, he broke the terms of his parole")
OPENER_BENEFIT_FOLLOWS = re.compile(r"[ \t]*you\b|[^.!?\n]*?\b(?:can|could|will|would|may|might)\b|[^.!?\n]*?['’]ll\b", re.I)

def opener_benefit_follows(m):
    return bool(OPENER_BENEFIT_FOLLOWS.match(m.string, m.end()))

# - "On the flip side," near a record or a coin may be the other side itself ("On the flip side, the record has three
#   bonus tracks")
OPENER_TWO_SIDED = re.compile(r"\b(?:vinyl|[AB]-sides?|LPs?|cassettes?|coins?|(?:the|this|that|its|his|her|their)[ \t]+"
                              r"(?:record|album)s?\b(?![ \t]+(?:high|low|number|level|year|label|time|sales)\b))", re.I)

def opener_flip_side_figurative(m):
    return not OPENER_TWO_SIDED.search(m.string, max(0, m.start() - 200), min(len(m.string), m.end() + 200))

OPENER_CONDITIONS = {'in rare cases': opener_share_of_cases, 'in some cases': opener_share_of_cases,
                     'in many cases': opener_share_of_cases, 'in most cases': opener_share_of_cases,
                     'finally': opener_instruction_follows, 'going forward': opener_plan_follows,
                     'to get started': opener_task_follows, 'to illustrate': opener_introduces_example,
                     'by doing so': opener_benefit_follows, 'on the flip side': opener_flip_side_figurative}

def phrase_keywords(phrases):
    """The longest word of each phrase, in lower case: a text without any of them can't contain the phrases, so their
    pattern needn't be run over it."""
    return tuple({max(re.findall(r"[a-z]+", phrase.lower()), key=len) for phrase in phrases})

# The comma and the spaces after it stay on the opener's line ("Moreover,\n" must not swallow the line break)
TRANSITION_RULES = [(re.compile(SENT_START + "(?:" + phrase_regex(words) + r")[ \t]*,[ \t]*", re.I | re.M), chance, options,
                     OPENER_CONDITIONS.get(words[0]), phrase_keywords(words))
                    for words, chance, options in TRANSITIONS]
# Chaos Mode rewrites every optional opener ("However," 0.4 -> 1.0), and leaves most additive and summing-up openers
# out altogether: swapping "Moreover," for "Also," keeps just as many formulaic openers. Contrast and cause
# ("However", "Therefore") carry meaning and are only swapped.
CHAOS_OPENER_BOOST = 1.0
DROPPABLE_TRANSITIONS = {0, 1}
DROP_CHANCE = 0.8
# Words of a replacement opener that may be used nearby without echoing it
OPENER_LINK_WORDS = {'that', 'this', 'with', 'from', 'over', 'because', 'same', 'what'}

def opener_chance(chance, deep_mode):
    return chance + (1 - chance) * CHAOS_OPENER_BOOST if deep_mode else chance

WORD_FOLLOWS = re.compile(r'[^\S\n]*[^\w\s]*\w')
# "Also," and "Plus," aren't used where the sentence already says it ("also", "too", "as well")
ALSO_OPENERS = ('Also,', 'Plus,')
SAYS_ALSO = re.compile(r'\b(?:also|too|as\s+well)\b', re.I)

def capitalized(word):
    """The word with a capital first letter, unless it has capitals inside ("iPhone", "mRNA", "pH")."""
    return word if any(c.isupper() for c in word[1:]) else word[:1].upper() + word[1:]

# Where an opener or a filler is left out, the next word starts the sentence and takes a capital, unless it is code, a
# web address, an e-mail address or a file or identifier name ("It is important to note that npm_config_cache must be
# set" -> "npm_config_cache must be set", never "Npm_config_cache"; "https://...", "support@...", "package.json")
KEEPS_ITS_CASE = re.compile(r"`|(?:[a-z][\w+.-]*://|www\.)|[^\s@]+@|\w*[_/]|\w+\.\w", re.I)

def sentence_capital(text, start, word):
    """`word`, which starts at `start` in `text`, as the first word of a sentence."""
    return word if KEEPS_ITS_CASE.match(text, start) or text[start - 1:start] == '`' else capitalized(word)

# Empty openers that add nothing; dropped, except Professional/Academic keep a short signpost
FILLERS = phrase_regex([
    "it is important to note that", "it's important to note that", "it is important to remember that",
    "it's important to remember that", "it is worth noting that", "it's worth noting that",
    "it is worth mentioning that", "it's worth mentioning that", "it should be noted that",
    "it is crucial to note that", "it is essential to note that", "it is interesting to note that",
    "it goes without saying that", "needless to say", "it's no secret that", "it is no secret that",
    "it is important to acknowledge that", "it's important to acknowledge that", "it is important to mention that",
    "it's important to mention that", "it is important to recognize that", "it's important to recognize that",
    "it is important to understand that", "it's important to understand that", "it should be emphasized that",
    "it should be mentioned that", "it should be stressed that", "it is worth highlighting that",
    "it's worth highlighting that", "it is crucial to recognize that", "it is essential to recognize that",
    "it is crucial to understand that", "it is essential to understand that", "it bears mentioning that",
    "it's important to bear in mind that", "it is important to bear in mind that", "it's worth remembering that",
    "it is worth remembering that", "it is important to keep in mind that", "it's important to keep in mind that",
    "it is also worth noting that", "it's also worth noting that", "it is also important to note that",
    "it's also important to note that", "it should also be noted that", "it is also worth mentioning that",
    "it's also worth mentioning that", "it is worth pointing out that", "it's worth pointing out that",
    "it is important to point out that", "it's important to point out that",
])

# "In today's fast-paced digital landscape" and its many relatives are filler for "today"; nouns that carry
# content ("today's job market", "today's economy") are left alone, and so are "the modern era" and "the
# current era", which name a span of history ("the first player in the modern era to ..."), and an age or
# world "of" something ("in the digital age of streaming")
TODAY_FILLER = re.compile(
    r"(?:\bin today['’]s\s+(?:(?:fast-paced|rapidly\s+(?:evolving|changing)|ever-(?:changing|evolving)|constantly\s+"
    r"(?:evolving|changing)|modern|digital|interconnected|globali[sz]ed|competitive|dynamic|tech-driven|"
    r"increasingly\s+(?:digital|connected|interconnected|complex))\s+){0,3}(?:world|landscape|age|era|society|times)\b"
    r"|\bin\s+(?:this|the)\s+(?:digital\s+age|day\s+and\s+age)\b"
    r"|\bin\s+an\s+increasingly\s+(?:digital|connected|interconnected)\s+world\b)(?!\s+of\b)", re.I)
TIME_WORDS = re.compile(r"\b(?:today|nowadays|these days)\b", re.I)
# A filler starts a sentence, or a list item or quoted line after its marker ("- It is important to note that ...")
FILLER_AT_START = re.compile(r"(?P<pre>^[ \t]*(?:(?:[-*•▪◦>]|\d+[.)]|[A-Za-z][.)])[ \t]+)?|(?<=[.!?])\s+|"
                             r"(?<=[.!?][\"”’)\]])\s+)" + FILLERS + r",?\s+(?P<first>\w+)", re.I | re.M)
# (a match begins where a run of spaces begins: tried inside the run as well, a long run would be read again from
# every one of its spaces)
FILLER_INSIDE = re.compile(r"(?<!\s)\s+" + FILLERS + r",?(?=\s)", re.I)
# Inside a sentence the filler goes, unless what is doubted or denied is the filler itself: "We doubt it is worth noting
# that the fee rose" doesn't say "We doubt the fee rose"
FILLER_DOUBTED = re.compile(r"\b(?:doubts?|doubted|den(?:y|ies|ied)|disagree[sd]?|disputes?|disputed|questions?|"
                            r"questioned|(?:nobody|no[ \t]+one|never)[ \t]+said|n['’]t[ \t]+say)(?:[ \t]+that)?[ \t]*\Z", re.I)
FILLER_SIGNPOSTS = {'professional': 'Note that ', 'academic': 'Notably, '}

# "plays a crucial role in" and friends: the whole frame is the cliché, so it is replaced as a unit, in a
# form that agrees with the subject and keeps the tense ("play" also follows modals: "can play ...")
ROLE_ADJECTIVES = ('crucial', 'pivotal', 'vital', 'key', 'significant', 'critical', 'central', 'essential',
                   'integral', 'instrumental', 'important', 'major', 'fundamental', 'decisive', 'big', 'huge')

def role_frames(verb):
    return tuple(f"{verb} {'an' if a[0] in 'aeiou' else 'a'} {a} role in" for a in ROLE_ADJECTIVES)

# A vessel or vehicle after "embark on" is boarded, not begun ("embarked on the night ferry")
NOT_A_VEHICLE = (r"(?!\s+(?:(?:a|an|the|this|that|their|his|her|our|its|my|your)\s+)?(?:[\w-]+\s+)?"
                 r"(?:ships?|ferry|ferries|boats?|cruises?|planes?|flights?|trains?|bus|buses|vessels?|aircraft|"
                 r"yachts?|liners?|submarines?|shuttles?|spacecraft)\b)")

# Someone who "served as mayor" held the office, while "acted as mayor" stood in for the mayor: before an office,
# a job or a title ("served as Surveyor of the King's Works", "served as the country's first female prime
# minister") the phrase stays. The words in front of the office belong to its name; a preposition, conjunction,
# determiner or helping verb ends the name ("served as a reminder that every member ...", "a meeting place where
# the mayor ...").
NOT_AN_OFFICE = (r"(?!\s+(?:(?:a|an|the|its|his|her|their|our)\s+)?(?:(?-i:[A-Z])|(?:(?!(?:of|for|to|in|on|at|by|from|"
                 r"with|about|into|onto|upon|over|under|above|below|after|before|between|among|during|through|within|"
                 r"without|across|along|around|behind|beside|beyond|near|like|unlike|against|despite|toward|towards|"
                 r"via|per|than|as|that|which|who|whom|whose|where|when|while|if|because|and|or|but|nor|so|yet|a|an|"
                 r"the|this|these|those|every|each|some|any|no|its|his|her|their|our|my|your|is|are|was|were|be|been|"
                 r"has|have|had|will|would|can|could|may|might|must|should|shall)\b)[\w'’-]+\s+){0,5}(?:mayor|"
                 r"president|chair\w*|director|governor|minister|secretary|ambassador|ceo|captain|head|judge|editor|"
                 r"treasurer|senator|council+or|advis[eo]r|coach|manager|nurse|soldier|officer|chief|commander|"
                 r"leader|deputy|dean|principal|professor|trustee|clerk|chancellor|vice|consul|bishop|pastor|priest|"
                 r"chaplain|surgeon|doctor|teacher|lecturer|spokes\w+|speaker|commissioner|attorney|sheriff|member|"
                 r"representative|delegate)\b))")

# Fixed phrases that also have a literal or a technical sense keep the writer's words where the words around them show
# it. After the phrase (a lookahead, the third item of an entry):
# - "around the clock" says when only where the phrase ends ("staffed around the clock, seven days a week"), not before
#   a noun ("spaced evenly around the clock face")
CLOCK_PHRASE_ENDS = (r"(?=[ \t]*(?:[.,;:!?)\]\"”’\n]|$)|\s+(?:and|or|but|to|in|on|at|for|with|so|while|until|since|during|"
                     r"as|by|from|when|if|because|throughout|every|seven|now|again)\b)")
# - "in the field of" before a discipline is "in" ("in the field of medicine" -> "in medicine"), but not before a crop or
#   other land ("grazed in the field of clover") or "study" ("a career in the field of study you enjoy")
FIELD_OF_STUDY = (r"(?=\s+(?:(?:computer|data|political|social|natural|life|health|environmental|renewable|artificial|"
                  r"machine|public|mental|clinical|applied|modern|digital|biomedical|software|civil|mechanical|electrical|"
                  r"chemical|cognitive|behaviou?ral|veterinary|space|quantum|financial|international|criminal|human)\s+)?"
                  r"(?:medicine|law|engineering|sciences?|research|education|health|healthcare|nursing|finance|marketing|"
                  r"accounting|banking|architecture|design|energy|technology|computing|ai|intelligence|learning|robotics|"
                  r"genetics|cybersecurity|agriculture|journalism|mathematics|maths?|dentistry|pharmacy|policy|logistics|"
                  r"history|literature|philosophy|art|music|[a-z]+(?:ology|ics|istry|onomy|graphy))\b)")
# - "sheds light on" is "helps explain" only before what can be explained ("sheds light on how bees find their way"),
#   not before what is shown or lit ("sheds light on the lives of women", "The lamp sheds light on the desk")
EXPLAINABLE_AFTER = (r"(?=\s+(?:how|why|what|whether|when|where|who|the\s+(?:causes?|reasons?|origins?|roots?|role|"
                     r"mechanisms?|factors?|forces|links?|relationship|effects?|impact|decline|rise|success|failure|"
                     r"spread))\b)")
# - a good or bad effect on an amount may be its direction ("temperature has a negative effect on viscosity", "a
#   positive effect on crime rates"), not harm or benefit (Academic keeps all of them, see keeps_fixed_phrase)
NOT_AN_AMOUNT_AFTER = (r"(?!\s+(?:(?:the|its|their|our|this|these)\s+)?(?:[\w-]+\s+){0,2}(?:rates?|levels?|viscosity|"
                       r"temperatures?|pressures?|concentrations|concentration\s+of|density|speed|frequency|values?|risk|"
                       r"incidence|prevalence|prices?|volume|output|demand|supply|counts?|numbers?|crime|mortality|scores?|"
                       r"yields?)\b)")
# - "the majority of the court" is the judges who formed the majority ("Writing for the majority of the court")
NOT_A_BENCH_AFTER = r"(?!\s+(?:the\s+)?(?:court|bench|justices|judges|panel|tribunal|jury)\b)"
# - "make a decision" before a technical term ("make a decision tree") or a clause about the decision itself, and before
#   "regarding", where "decided regarding the merger" doesn't read ("a decision regarding" may become "a decision on")
NOT_DECISION_TERM_AFTER = (r"(?!\s+(?:that|which|who|whom|whose|trees?|matrix|matrices|rules?|boundary|boundaries|process|"
                           r"support|tables?|makers?|models?|regarding|concerning)\b)")
# - a count phrase before a singular verb is a singular subject ("A growing number of studies has questioned ...") that
#   "more and more" or "many" can't take
NOT_SINGULAR_VERB_AFTER = r"(?!(?:\s+[\w'’-]+){1,2}\s+(?:has|is|was|does)\b)"
# - the power of a stone, a sword or a spell is its magic, not its potential ("unleash the power of the ancient stone")
NOT_AN_ARTEFACT_AFTER = (r"(?!\s+(?:(?:the|this|that|an?|its|their|his|her)\s+)?(?:[\w-]+\s+)?(?:stones?|swords?|rings?|"
                         r"amulets?|crystals?|gems?|orbs?|staff|wands?|talismans?|relics?|artifacts?|artefacts?|spells?|"
                         r"dragons?|gods?)\b)")
# - "one of the ... who has" agrees with "one": without it the plural would take a singular verb ("She is one of the most
#   gifted players who has ever lived" can't become "among the most gifted players who has ever lived")
NOT_ONE_WHO_SINGULAR = (r"(?![^.!?;\n]*?\b(?:who|that|which)(?:[ \t]+(?:also|still|now|always|never|often|[a-z]+ly))?[ \t]+"
                        r"(?:has|is|was|does|[a-z]+s)\b)")
# Before the phrase (the fourth item: what the words before it must not end with):
# - after a mishap "in an attempt to" says when it happened, not what for ("The climber died in an attempt to reach the
#   summit" isn't "died to reach")
MISHAP_BEFORE = (r"\b(?:died|dies|die|dying|killed|injured|hurt|wounded|drowned|perished|crashed|fell|fallen|lost|"
                 r"arrested|caught|shot|stabbed|trapped|stranded|hospitali[sz]ed|detained|captured|disappeared|vanished|"
                 r"missing|burned|burnt)(?:[ \t]+\w+ly)?[ \t]+")
# - a story "set in the near future" has a setting, not a time ("The novel is set in the near future")
SETTING_BEFORE = r"\b(?:set|sets|takes?[ \t]+place|took[ \t]+place|taking[ \t]+place|unfolds?|unfolding)[ \t]+"
# - planets "in conjunction with" each other are in the sky together ("Jupiter is in conjunction with Saturn")
CONJUNCTION_BE_BEFORE = r"\b(?:is|are|was|were|be|been|being|am)[ \t]+"
# - an element of a substance is a chemical element ("Carbon is an essential element of all known life")
CHEMISTRY_BEFORE = (r"\b(?:carbon|oxygen|hydrogen|nitrogen|iron|calcium|sodium|potassium|magnesium|zinc|copper|sulphur|"
                    r"sulfur|phosphorus|silicon|chlorine|iodine|helium|uranium|lithium|minerals?|chemicals?|atoms?)\b"
                    r"[^.!?\n]*")
# - a rock or a shell has a real impact ("The asteroid had an impact on the far side of the moon")
PROJECTILE_BEFORE = r"\b(?:asteroids?|meteor\w*|comets?|bullets?|missiles?|projectiles?|bombs?|shells?|debris)\b[^.!?\n]*"

# Wordy or overused phrases, rewritten anywhere in a sentence
PHRASES = [
    # "play" also follows modals and "to" ("can play", "to play"), so its replacements are verbs whose base form
    # is also the plural; "played" also follows "has", so its replacements are the same in both uses
    (role_frames('plays'), {'standard': ['is central to', 'matters a great deal for', 'contributes a lot to'],
                            'casual': ['is a big part of', 'matters a lot for', 'contributes a lot to'],
                            'professional': ['is central to', 'contributes greatly to'],
                            'academic': ['is central to', 'contributes substantially to']}),
    (role_frames('play'), {'standard': ['matter a great deal for', 'contribute a lot to'],
                           'casual': ['matter a lot for', 'contribute a lot to'],
                           'professional': ['contribute greatly to', 'matter a great deal for'],
                           'academic': ['contribute substantially to']}),
    (role_frames('played'), {'standard': ['mattered a great deal for', 'contributed a lot to'],
                             'casual': ['mattered a lot for', 'contributed a lot to'],
                             'professional': ['contributed greatly to'], 'academic': ['contributed substantially to']}),
    (role_frames('playing'), {'*': ['contributing a lot to'], 'professional': ['contributing greatly to'],
                              'academic': ['contributing substantially to']}),
    (("a role to play",), {'*': ['a part to play']}),
    (("an integral part of",), {'*': ['a core part of', 'a big part of'], 'professional': ['a core part of', 'a key part of'],
                                 'academic': ['a central part of', 'an essential part of']}),
    (("for years to come", "for generations to come", "for decades to come"),
     {'*': ['well into the future', 'for a long time yet'], 'academic': ['well into the future']}),
    (("paves the way for",), {'*': ['opens the door to', 'clears the way for']}),
    (("pave the way for",), {'*': ['open the door to', 'clear the way for']}),
    (("paved the way for",), {'*': ['opened the door to', 'cleared the way for']}),
    (("paving the way for",), {'*': ['opening the door to', 'clearing the way for']}),
    (("has emerged as",), {'*': ['has become'], 'academic': ['has become', 'has developed into']}),
    (("have emerged as",), {'*': ['have become'], 'academic': ['have become', 'have developed into']}),
    (("a broad spectrum of", "a wide spectrum of"), {'*': ['many', 'a wide mix of'], 'academic': ['many', 'diverse']}),
    (("is key to",), {'*': ['goes a long way toward', 'is central to'], 'casual': ['goes a long way toward'],
                      'academic': ['is central to']}),
    (("are key to",), {'*': ['go a long way toward', 'are central to'], 'casual': ['go a long way toward'],
                       'academic': ['are central to']}),
    (("has a significant impact on", "has a profound impact on", "has a major impact on", "has a huge impact on",
      "has a tremendous impact on"), {'*': ['greatly affects', 'strongly affects'], 'academic': ['substantially affects']}),
    (("have a significant impact on", "have a profound impact on", "have a major impact on", "have a huge impact on",
      "have a tremendous impact on"), {'*': ['greatly affect', 'strongly affect'], 'academic': ['substantially affect']}),
    (("had a significant impact on", "had a profound impact on", "had a major impact on", "had a huge impact on",
      "had a tremendous impact on"), {'*': ['greatly affected', 'strongly affected'], 'academic': ['substantially affected']}),
    # A factor stays a factor: "a key factor in lung cancer" is a cause, not "a big part of" it
    (("a key factor in", "a crucial factor in", "a critical factor in", "a vital factor in"),
     {'*': ['a big factor in', 'a major factor in'], 'academic': ['a major factor in']}),
    (("bridge the gap",), {'*': ['close the gap']}),
    (("bridges the gap",), {'*': ['closes the gap']}),
    (("bridging the gap",), {'*': ['closing the gap']}),
    (("bridged the gap",), {'*': ['closed the gap']}),
    # Only the figurative "dive deep into" ("dive into the pool" is literal)
    (("dive deep into", "dive deeper into"), {'*': ['look closely at']}),
    (("dives deep into", "dives deeper into"), {'*': ['looks closely at']}),
    (("diving deep into", "diving deeper into"), {'*': ['looking closely at']}),
    (("a game-changer", "a game changer"), {'*': ['a turning point', 'a big shift'], 'professional': ['a turning point', 'a major shift'],
                                            'academic': ['a turning point']}),
    (("game-changing",), {'*': ['far-reaching', 'major'], 'academic': ['far-reaching']}),
    # "cause" goes with harm ("helped cause peace" reads wrongly), so "lead to"; the passive "was brought
    # about (by)" is a special rule below
    (("bring about",), {'*': ['lead to']}),
    (("brings about",), {'*': ['leads to']}),
    (("bringing about",), {'*': ['leading to']}),
    (("harness the power of",), {'*': ['use', 'make use of']}),
    (("harnesses the power of",), {'*': ['uses', 'makes use of']}),
    (("harnessing the power of",), {'*': ['using', 'making use of']}),
    (("navigate the challenges of",), {'*': ['deal with the challenges of', 'handle the challenges of']}),
    (("navigating the challenges of",), {'*': ['dealing with the challenges of', 'handling the challenges of']}),
    (("a deep understanding of",), {'*': ['a solid grasp of', 'a good grasp of'], 'academic': ['a thorough understanding of']}),
    (("in order to",), {'*': ['to']}),
    (("due to the fact that", "owing to the fact that", "in light of the fact that"),
     {'*': ['because'], 'academic': ['because', 'since']}),
    (("at this point in time", "at the present time"),
     {'*': ['now'], 'professional': ['currently'], 'academic': ['currently']}),
    # "in the modern world today" says "today" twice
    (("in today's fast-paced world", "in today's digital age", "in today's world", "in the modern world",
      "in the modern world today", "in today's modern world"),
     {'*': ['today', 'these days'], 'casual': ['these days', 'nowadays'], 'professional': ['today'], 'academic': ['today']}),
    (("a wide range of", "a wide variety of", "a broad range of", "a diverse range of", "a wide array of", "a vast array of",
      "a broad array of", "a diverse array of"),
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
    (("unlock the full potential of", "unlock the potential of", "unlock the power of", "unleash the power of",
      "unleash the full potential of", "unleash the potential of"), {'*': ['make the most of']}, NOT_AN_ARTEFACT_AFTER),
    (("unlocks the full potential of", "unlocks the potential of", "unlocks the power of", "unleashes the power of",
      "unleashes the full potential of", "unleashes the potential of"), {'*': ['makes the most of']}, NOT_AN_ARTEFACT_AFTER),
    (("unlocking the full potential of", "unlocking the potential of", "unlocking the power of", "unleashing the power of",
      "unleashing the full potential of", "unleashing the potential of"), {'*': ['making the most of']},
     NOT_AN_ARTEFACT_AFTER),
    (("with regard to", "with regards to", "in regard to", "in regards to"),
     {'*': ['about', 'on'], 'professional': ['regarding'], 'academic': ['regarding', 'concerning']}),
    (("in the event that",), {'*': ['if']}),
    (("prior to",), {'*': ['before']}),
    (("a crucial role", "a pivotal role", "a vital role", "an essential role", "a significant role", "an integral role",
      "an instrumental role", "a critical role"),
     {'standard': ['a big part', 'a key role', 'a major part'], 'casual': ['a big part', 'a huge part'],
      'professional': ['a key role', 'a major role'], 'academic': ['a central role', 'a major role']}),
    (("serves as",), {'*': ['acts as', 'works as'], 'academic': ['functions as', 'acts as']}, NOT_AN_OFFICE),
    (("serve as",), {'*': ['act as', 'work as'], 'academic': ['function as', 'act as']}, NOT_AN_OFFICE),
    (("served as",), {'*': ['acted as', 'worked as'], 'academic': ['functioned as', 'acted as']}, NOT_AN_OFFICE),
    # Only the figurative "embark on a journey"; "embarked on the ferry" is boarding it
    (("embark on", "embark upon"), {'*': ['start', 'begin'], 'casual': ['start', 'kick off']}, NOT_A_VEHICLE),
    (("embarks on", "embarks upon"), {'*': ['starts', 'begins']}, NOT_A_VEHICLE),
    (("embarked on", "embarked upon"), {'*': ['started', 'began']}, NOT_A_VEHICLE),
    (("embarking on", "embarking upon"), {'*': ['starting', 'beginning']}, NOT_A_VEHICLE),
    (("in a timely manner",), {'*': ['promptly', 'on time'], 'casual': ['quickly', 'on time']}),
    (("a key aspect of", "an important aspect of", "a crucial aspect of", "a vital aspect of", "a key component of",
      "an essential component of", "a crucial component of", "a vital component of", "an important component of",
      "an integral component of"),
     {'*': ['a key part of', 'an important part of'], 'casual': ['a big part of'],
      'academic': ['a central part of', 'an important part of']}),
    (("a key element of", "an essential element of", "a crucial element of", "a vital element of", "an important element of"),
     {'*': ['a key part of', 'an important part of'], 'casual': ['a big part of'],
      'academic': ['a central part of', 'an important part of']}, '', CHEMISTRY_BEFORE),
    (("is able to", "are able to"), {'*': ['can']}),
    # The superlative stays; only the stock "one of the most" frame goes
    (("is one of the most",), {'*': ['is among the most']}, NOT_ONE_WHO_SINGULAR),
    (("was one of the most",), {'*': ['was among the most']}, NOT_ONE_WHO_SINGULAR),
    (("remains one of the most",), {'*': ['remains among the most']}, NOT_ONE_WHO_SINGULAR),
    (("is the process by which", "is the process through which"),
     {'*': ['is how'], 'academic': ['is the means by which']}),
    # Both say the two go together, and neither that one causes the other
    (("is associated with",), {'*': ['is linked to']}),
    (("are associated with",), {'*': ['are linked to']}),
    (("was associated with",), {'*': ['was linked to']}),
    (("were associated with",), {'*': ['were linked to']}),
    (("more important than ever", "more crucial than ever", "more vital than ever"),
     {'*': ['especially important', 'more important than it used to be'], 'casual': ['a bigger deal than ever'],
      'professional': ['especially important'], 'academic': ['increasingly important']}),
    (("a deep dive into",), {'*': ['a close look at']}),
    # "meant to" is weaker than "designed to" (a purpose, not how it was made)
    (("is designed to",), {'*': ['is built to'], 'academic': ['is intended to']}),
    (("are designed to",), {'*': ['are built to'], 'academic': ['are intended to']}),
    (("was designed to",), {'*': ['was built to'], 'academic': ['was intended to']}),
    (("were designed to",), {'*': ['were built to'], 'academic': ['were intended to']}),
    (("as a result of",), {'*': ['because of']}),
    (("in search of",), {'*': ['looking for'], 'academic': ['in search of']}),
    (("on a global scale",), {'*': ['worldwide', 'globally']}),
    (("has its headquarters in",), {'*': ['is based in']}),
    (("have their headquarters in",), {'*': ['are based in']}),
    (("had its headquarters in",), {'*': ['was based in']}),
    (("one third of",), {'*': ['a third of']}),
    (("one quarter of",), {'*': ['a quarter of']}),
    # (the opener "In recent years," is a transition above, which looks out for repeats)
    (("in recent years",), {'*': ['over the past few years', 'in the past few years']}, r"(?![ \t]*,)"),
    (("in recent decades",), {'*': ['over the past few decades']}),
    (("keep in mind that", "bear in mind that"), {'*': ['remember that']}),
    (("keeping in mind that", "bearing in mind that"), {'*': ['remembering that']}),
    (("a great deal of",), {'*': ['a lot of'], 'professional': ['a great deal of'], 'academic': ['a great deal of']}),
    (("around the clock", "round the clock"), {'*': ['day and night', '24 hours a day'], 'casual': ['day and night', '24/7']},
     CLOCK_PHRASE_ENDS),
    (("on a single charge",), {'*': ['on one charge']}),
    (("surprised to discover",), {'*': ['surprised to find', 'surprised to learn']}),
    (("shocked to discover",), {'*': ['shocked to find', 'shocked to learn']}),
    (("amazed to discover",), {'*': ['amazed to find']}),
    (("delighted to discover",), {'*': ['delighted to find']}),
    # A count stays a count, so "of" follows the same rules as for "most" (see BARE_QUANTITIES)
    (("a growing number of", "an increasing number of", "a rising number of"), {'*': ['more and more']},
     NOT_SINGULAR_VERB_AFTER),
    (("a large number of", "a great number of", "a huge number of", "a significant number of",
      "a considerable number of", "a substantial number of"), {'*': ['many'], 'casual': ['lots of', 'many']},
     NOT_SINGULAR_VERB_AFTER),
    (("a small number of",), {'*': ['a few']}, NOT_SINGULAR_VERB_AFTER),
    (("the majority of",), {'*': ['most']}, NOT_A_BENCH_AFTER),
    (("on a daily basis",), {'*': ['daily', 'every day']}),
    (("on a weekly basis",), {'*': ['weekly', 'every week']}),
    (("on a monthly basis",), {'*': ['monthly', 'every month']}),
    (("on a yearly basis", "on an annual basis"), {'*': ['every year', 'yearly'], 'academic': ['annually']}),
    (("on a regular basis",), {'*': ['regularly']}),
    (("in the near future", "in the not-too-distant future", "in the not too distant future"), {'*': ['soon']}, '',
     SETTING_BEFORE),
    (("in the field of",), {'*': ['in']}, FIELD_OF_STUDY),
    (("for the purpose of",), {'*': ['for']}),
    (("in an effort to", "in an attempt to", "in a bid to"), {'*': ['to']}, '', MISHAP_BEFORE),
    (("in order for",), {'*': ['for']}),
    (("so as to",), {'*': ['to']}),
    (("in the absence of",), {'*': ['without']}),
    (("despite the fact that", "in spite of the fact that"), {'*': ['although', 'even though']}),
    (("with the exception of",), {'*': ['except for', 'apart from']}),
    (("in close proximity to",), {'*': ['close to', 'near']}),
    (("in conjunction with",), {'*': ['together with']}, '', CONJUNCTION_BE_BEFORE),
    (("a high degree of",), {'*': ['great'], 'professional': ['considerable'], 'academic': ['considerable']}),
    (("is considered to be",), {'*': ['is considered']}),
    (("are considered to be",), {'*': ['are considered']}),
    (("was considered to be",), {'*': ['was considered']}),
    (("were considered to be",), {'*': ['were considered']}),
    (("sheds light on",), {'*': ['helps explain']}, EXPLAINABLE_AFTER),
    (("shedding light on",), {'*': ['helping to explain']}, EXPLAINABLE_AFTER),
    (("take into account", "take into consideration"), {'*': ['consider']}),
    (("takes into account", "takes into consideration"), {'*': ['considers']}),
    (("took into account", "took into consideration", "taken into account", "taken into consideration"),
     {'*': ['considered']}),
    (("taking into account", "taking into consideration"), {'*': ['considering']}),
    # Not before a clause about the decision itself ("made a decision that changed everything") or a term ("make a
    # decision tree")
    (("make a decision",), {'*': ['decide']}, NOT_DECISION_TERM_AFTER),
    (("makes a decision",), {'*': ['decides']}, NOT_DECISION_TERM_AFTER),
    (("made a decision",), {'*': ['decided']}, NOT_DECISION_TERM_AFTER),
    (("making a decision",), {'*': ['deciding']}, NOT_DECISION_TERM_AFTER),
    (("make the decision to",), {'*': ['decide to']}),
    (("makes the decision to",), {'*': ['decides to']}),
    (("made the decision to",), {'*': ['decided to']}),
    (("making the decision to",), {'*': ['deciding to']}),
    (("make an effort to",), {'*': ['try to']}),
    (("makes an effort to",), {'*': ['tries to']}),
    (("made an effort to",), {'*': ['tried to']}),
    (("making an effort to",), {'*': ['trying to']}),
    # An effect without a direction is "affect"; a good one "benefit", a bad one "harm"
    (("has an impact on", "has an effect on"), {'*': ['affects']}, '', PROJECTILE_BEFORE),
    (("have an impact on", "have an effect on"), {'*': ['affect']}, '', PROJECTILE_BEFORE),
    (("had an impact on", "had an effect on"), {'*': ['affected']}, '', PROJECTILE_BEFORE),
    (("having an impact on", "having an effect on"), {'*': ['affecting']}, '', PROJECTILE_BEFORE),
    (("has a positive impact on", "has a positive effect on", "has a beneficial effect on"), {'*': ['benefits']},
     NOT_AN_AMOUNT_AFTER),
    (("have a positive impact on", "have a positive effect on", "have a beneficial effect on"), {'*': ['benefit']},
     NOT_AN_AMOUNT_AFTER),
    (("had a positive impact on", "had a positive effect on", "had a beneficial effect on"), {'*': ['benefited']},
     NOT_AN_AMOUNT_AFTER),
    (("having a positive impact on", "having a positive effect on", "having a beneficial effect on"), {'*': ['benefiting']},
     NOT_AN_AMOUNT_AFTER),
    (("has a negative impact on", "has a negative effect on", "has a detrimental effect on", "has an adverse effect on"),
     {'*': ['harms', 'hurts'], 'casual': ['hurts']}, NOT_AN_AMOUNT_AFTER),
    (("have a negative impact on", "have a negative effect on", "have a detrimental effect on", "have an adverse effect on"),
     {'*': ['harm', 'hurt'], 'casual': ['hurt']}, NOT_AN_AMOUNT_AFTER),
    (("had a negative impact on", "had a negative effect on", "had a detrimental effect on", "had an adverse effect on"),
     {'*': ['harmed', 'hurt'], 'casual': ['hurt']}, NOT_AN_AMOUNT_AFTER),
    (("having a negative impact on", "having a negative effect on", "having a detrimental effect on",
      "having an adverse effect on"), {'*': ['harming', 'hurting'], 'casual': ['hurting']}, NOT_AN_AMOUNT_AFTER),
]
# An optional "a"/"an" in front is captured so it can agree with the replacement; a third item of an entry is a
# lookahead the text after the phrase must pass, and a fourth what the words before it must not end with. Each rule
# keeps the keywords that tell where it can match.
PHRASE_RULES = [(re.compile(r"(?P<art>\b(?:a|an)\s+)?\b(?P<phrase>" + phrase_regex(words) + r")\b" + (extra[0] if extra else ''),
                            re.I),
                 options, phrase_keywords(words), re.compile('(?:' + extra[1] + r')\Z', re.I) if len(extra) > 1 else None)
                for words, options, *extra in PHRASES]
# "cannot be overstated" keeps a modal, so it agrees with any subject ("These risks can hardly be overstated");
# before a "by" agent only the passive reads ("... can hardly be overstated by anyone who ...")
_OVERSTATED = (r"(?P<art>\b(?:a|an)\s+)?\b(?P<phrase>"
               + phrase_regex(("cannot be overstated", "can't be overstated", "can not be overstated")) + r")\b")
PHRASE_RULES += [
    (re.compile(_OVERSTATED + r"(?!\s+by\b)", re.I),
     {'*': ['would be hard to overstate', 'can hardly be overstated'],
      'academic': ['would be difficult to overstate', 'can scarcely be overstated']}, ('overstated',), None),
    (re.compile(_OVERSTATED + r"(?=\s+by\b)", re.I), {'*': ['can hardly be overstated'], 'academic': ['can scarcely be overstated']},
     ('overstated',), None),
]

# --- Phrases whose rewrite depends on the words around them ---
SENTENCE_OPENING = re.compile(r'(?:\n[ \t]*(?:(?:[-*•▪◦]|\d+[.)])[ \t]+)?|[.!?]["”’)\]]*[ \t]*)\Z')
CLAUSE_OPENING = re.compile(r'(?:[;:,(—–]["”’)\]]*[ \t]*|\b(?:that|and|but|so|yet)[ \t]+)\Z', re.I)

def text_before(m, width=40):
    return m.string[max(0, m.start() - width):m.start()]

def opens_sentence(m):
    before = text_before(m)
    return (m.start() <= 40 and not before.strip()) or bool(SENTENCE_OPENING.search(before))

def opens_clause(m):
    return opens_sentence(m) or bool(CLAUSE_OPENING.search(text_before(m)))

def base_verb(word):
    """True if the word is the base form of a verb ("note", "act"); False when lemminflect isn't loaded."""
    return 'getAllLemmas' in globals() and word in getAllLemmas(word).get('VERB', ())

def word_classes(word):
    """The parts of speech a word in lower case can have ({'NOUN', 'VERB'} for "uses"); none for a word lemminflect
    doesn't know, or when it isn't loaded."""
    return set(getAllLemmas(word)) if 'getAllLemmas' in globals() and word.islower() else set()

# Function words: determiners and pronouns, which start a noun phrase, and prepositions and conjunctions, which
# don't
PHRASE_FUNCTION_WORD = re.compile(r"(?:(?P<determiner>the|a|an|this|that|these|those|my|your|his|her|its|our|their|"
                                  r"it|them|him|me|us|you|we|they|he|she|everything|everyone|everybody|something|"
                                  r"someone|nothing|anything|all|some|any|each|every|both|no|what|whatever)|of|in|on|"
                                  r"at|to|for|with|from|by|into|onto|over|under|about|after|before|since|until|during|"
                                  r"through|across|within|without|between|among|against|despite|near|like|per|via|"
                                  r"than|as|and|or|but|nor|so|yet|if|because|while|when|where|who|whom|whose|"
                                  r"which)\Z")
# The words on the line after a phrase
WORDS_AFTER_PHRASE = re.compile(r"(?:[ \t]+[A-Za-z]+){1,4}")

def _context_rule(words, after=''):
    return re.compile(r"\b(?P<phrase>" + phrase_regex(words) + r")\b" + after, re.I)

ANNOUNCE_OPTIONS = {'standard': ["we're happy to share that", 'good news:'], 'casual': ["we're happy to say that", 'great news:'],
                    'professional': ['we are glad to report that', 'we are happy to share that'], 'academic': ['we report that']}
EVIDENT_OPTIONS = {'*': ['clearly,'], 'casual': ['clearly,', 'obviously,']}
ESSENTIAL_OPTIONS = {'standard': ['it pays to', "it's key to"], 'casual': ["it's key to", 'you need to'],
                     'professional': ['it helps to', 'it is wise to'], 'academic': ['it is necessary to']}
POTENTIAL_OPTIONS = {'*': ['could', 'can'], 'academic': ['could', 'may']}

FRAME_COMMA_FOLLOWS = re.compile(r"[ \t]*,")

def without_doubled_comma(m, new):
    """A replacement that ends in a comma loses it before the writer's own ("It is clear that, in the long run, ..." ->
    "Clearly, in the long run, ...", not "Clearly,, in ...")."""
    return new[:-1] if new.endswith(',') and FRAME_COMMA_FOLLOWS.match(m.string, m.end()) else new

def announce(m, tone):
    """"Good news:" only reads where a sentence starts ("Today, good news: ..." doesn't), and not before the writer's
    comma ("We are pleased to announce that, after months of work, ...")."""
    options = ANNOUNCE_OPTIONS[tone]
    if not opens_sentence(m) or FRAME_COMMA_FOLLOWS.match(m.string, m.end()):
        options = [o for o in options if not o.endswith(':')]
    return match_case(m.group('phrase'), random.choice(options))

def evident(m, tone):
    """"Clearly," has to start its clause: "We think it is clear that ..." can't become "We think clearly, ...",
    which reads as how they think."""
    return match_case(m.group('phrase'), without_doubled_comma(m, pick(EVIDENT_OPTIONS, tone))) if opens_clause(m) else m.group(0)

# After a word that is also a noun, the end of the clause or a preposition shows it is the noun ("it is essential
# to progress.", "it is vital to progress in medicine")
NOUN_READING = re.compile(r"\s*(?:[.,;:!?)]|$)|\s+(?:of|in|on|for|and|or|with|across|between)\b")

def essential_to(m, tone):
    """Only before a verb: in "it is crucial to the success of the project" the "to" is a preposition, and
    "it pays to the success" says something else; nor before a noun that is also a verb ("it is essential to
    progress" isn't "it pays to progress")."""
    word = m.group('next').lower()
    if not base_verb(word) or ('NOUN' in getAllLemmas(word) and NOUN_READING.match(m.string, m.end('next'))):
        return m.group(0)
    return match_case(m.group('phrase'), pick(ESSENTIAL_OPTIONS, tone))

AUX_BEFORE = re.compile(r"\b(?:do|does|did|not|n['’]t|will|would|shall|should|can|could|may|might|must|to)\s+\Z", re.I)
QUESTION_START = re.compile(r"[\s\"“‘(]*(?:(?:why|how|what|where|when|who|which)\s+)?"
                            r"(?:do|does|did|will|wo|would|can|ca|could|should|shall|may|might|must)(?:n['’]t)?\b", re.I)

# A helping verb or "to" with adverbs between it and "have" ("may well have", "will soon have", "did not even have", "to
# one day have"), or anywhere earlier in the sentence ("will, for the first time, have"): "have" is then no present
# tense that "can" could replace
AUX_AND_ADVERBS_BEFORE = re.compile(r"\b(?P<aux>will|would|could|may|might|should|must|can|shall|to|do|does|did)(?:n['’]t|"
                                    r"[ \t]+not)?(?:[ \t]+(?:\w+ly|soon|also|still|now|then|never|always|already|even|"
                                    r"just|only|often|well|ever|someday|one[ \t]+day|some[ \t]+day))+[ \t]+\Z", re.I)
AUX_EARLIER_IN_SENTENCE = re.compile(r"\b(?:will|would|could|may|might|should|must|can|cannot|shall|to|do|does|did|"
                                     r"won['’]t)\b[^.!?\n]*\Z", re.I)

def potential(m, tone):
    """"has the potential to" -> "could". After a modal the modal says it ("could have the potential to" ->
    "could"), "does not have the potential to" -> "cannot"; after any other helping verb, "to", or in a
    question ("Does it have the potential to ...?") no modal fits, so the phrase stays, and so does a plain "have"
    after a helping verb or "to" earlier in the sentence ("may well have the potential to")."""
    if m.group('modal'):
        return m.group('modal')
    if m.group('neg'):
        return match_case(m.group('neg'), 'could not' if m.group('neg').lower().startswith('did') else 'cannot')
    sentence_start = max(m.string.rfind(c, 0, m.start()) for c in '.!?\n') + 1
    if (AUX_BEFORE.search(text_before(m, 16))
            or (m.group('have').lower() == 'have' and AUX_EARLIER_IN_SENTENCE.search(text_before(m, 40)))
            or QUESTION_START.match(m.string, sentence_start, m.start())):
        return m.group(0)
    return match_case(m.group('have'), pick(POTENTIAL_OPTIONS, tone))

PASSIVE_BEFORE = re.compile(r"\b(?:is|are|was|were|be|been|being)\s+(?:\w+ly\s+)?\Z", re.I)
# "led to" needs an object. Where none follows "brought about" (the end of the clause, a preposition or
# conjunction, a helping verb, a time or place), the phrase is a passive with no doer: "the reforms brought about
# in the 1990s were popular", "the changes the war brought about were huge", "the changes brought about that year".
BROUGHT_NO_OBJECT = re.compile(r"\s*(?:[.,;:!?)\]]|$)|\s+(?:\w+ly\s+)?(?:in|on|at|during|through|throughout|over|"
                               r"under|since|after|before|from|within|across|among|between|via|with|without|for|to|"
                               r"into|against|despite|following|until|till|by|as|when|while|where|because|and|or|but|"
                               r"is|are|was|were|be|been|has|have|had|will|would|can|could|may|might|must|shall|"
                               r"should|here|there|now|then|today|yesterday|(?:last|this|that|next|each|every)\s+"
                               r"(?:year|month|week|day|decade|century|summer|winter|spring|autumn|season|term|time)|"
                               r"(?:[\w-]+\s+){1,2}(?:ago|later|earlier))\b", re.I | re.M)

def verb_comes_next(m):
    """True if the next word after the match (after any adverb) can only be a verb, so the match has no object: in
    "the changes brought about helped the economy" and "the reforms brought about remain popular" the phrase
    describes the noun before it, and the verb belongs to that noun. A participle with a noun after it starts the
    object instead ("brought about improved conditions", "sweeping changes")."""
    found = WORDS_AFTER_PHRASE.match(m.string, m.end())
    words = found.group(0).split() if found else []
    while len(words) > 1 and 'ADV' in word_classes(words[0]):
        words.pop(0)                    # "mainly", "quickly", "still"
    classes = word_classes(words[0]) if words else set()
    if not classes or classes & {'NOUN', 'ADJ', 'PROPN'}:
        return False
    if (words[0].endswith(('ed', 'en', 'ing')) and len(words) > 1 and not PHRASE_FUNCTION_WORD.match(words[1])
            and (word_classes(words[1]) & {'NOUN', 'ADJ'} or (words[1].islower() and not word_classes(words[1])))):
        return False
    return True

def brought_about(m, tone):
    """"was brought about by" and "the changes brought about (mainly) by" are passive: "caused"; the active form
    is "led to". A passive with no doer stays, because "cause" goes with harm ("peace was brought about in 1998"
    isn't "peace was caused") and "led to" needs an object ("the reforms brought about in the 1990s were
    popular")."""
    if re.match(r'\s+(?:\w+ly\s+)?by\b', m.string[m.end():]):
        return match_case(m.group('phrase'), 'caused')
    if PASSIVE_BEFORE.search(text_before(m, 30)) or BROUGHT_NO_OBJECT.match(m.string, m.end()) or verb_comes_next(m):
        return m.group(0)
    return match_case(m.group('phrase'), 'led to')

# Frames that put a comment or a hedge in a clause of their own become an adverb at the start of the clause they
# introduce, as "it is clear that" becomes "clearly,": "It can be argued that the policy failed" -> "Arguably, the
# policy failed", "It is not surprising that ..." -> "Unsurprisingly, ...", "It is possible that ..." -> "Perhaps ...",
# "It is estimated that ..." -> "Estimates suggest that ...". Inside a clause ("We think it is possible that ...") the
# frame stays.
ARGUABLY_OPTIONS = {'*': ['arguably,']}
UNSURPRISINGLY_OPTIONS = {'*': ['unsurprisingly,'], 'casual': ['unsurprisingly,', 'no wonder']}
UNDENIABLY_OPTIONS = {'*': ['undeniably,']}
CERTAINLY_OPTIONS = {'*': ['certainly,']}
PERHAPS_OPTIONS = {'*': ['perhaps']}
ESTIMATES_OPTIONS = {'*': ['estimates suggest that']}

def frame_to_adverb(options, fits=None):
    """The rewrite for one frame; `fits` is a further test of the match the frame must pass."""
    def rewrite(m, tone):
        if not opens_clause(m) or (fits and not fits(m)):
            return m.group(0)
        return match_case(m.group('phrase'), without_doubled_comma(m, pick(options, tone)))
    return rewrite

# "Perhaps" says something may be or may have been so: it fits "It is possible that" before a modal or a past ("It is
# possible that the results were affected by the weather"), but not before a general present, where the frame says
# what can happen ("It is possible that two users share one account, so the system allows it")
POSSIBLE_EVENT_FOLLOWS = re.compile(r"[^.!?;\n]{0,160}?\b(?:will|would|could|might|may|can|must|should|shall|had|has|have|"
                                    r"was|were|did|been|[a-z]{3,}ed)\b", re.I)

def possible_event_follows(m):
    return bool(POSSIBLE_EVENT_FOLLOWS.match(m.string, m.end()))

# "In addition to its parks, the city ..." -> "Besides its parks, ...", "As well as its parks, ..."; Casual "On top of"
# only before an -ing ("On top of being cheaper, ..."), since before a thing it reads as a place ("On top of the tower,
# the castle has a moat"). Inside a sentence only "as well as" reads, and only after a noun ("free parking as well as
# discounts", not "free parking besides discounts"): after a verb it compares how well ("The new pump works in addition
# to the old one" isn't "works as well as the old one"), and after an amount it isn't idiomatic ("costs $10 in addition
# to tax").
IN_ADDITION_OPTIONS = {'standard': ['besides', 'as well as'], 'casual': ['besides', 'as well as', 'on top of'],
                       'professional': ['as well as', 'besides'], 'academic': ['as well as', 'besides']}
ONE_WORD_AFTER = re.compile(r"[ \t]+([A-Za-z]+)")
WORDS_BEFORE_ADDITION = re.compile(r"[\w'’-]+")
ADDITION_SUBJECT_PRONOUNS = {'i', 'you', 'we', 'they', 'he', 'she', 'it', 'who', 'that', 'which'}
ADVERBS_AFTER_SUBJECT = {'also', 'still', 'now', 'only', 'often', 'usually', 'always', 'even', 'just', 'already', 'never',
                         'sometimes', 'can', 'could', 'will', 'would', 'may', 'might', 'must', 'should', 'does', 'do',
                         'did'}

def doing_follows(m):
    found = ONE_WORD_AFTER.match(m.string, m.end())
    word = found.group(1) if found else ''
    return word.endswith('ing') and 'VERB' in word_classes(word) and not base_verb(word)

def in_addition_to(m, tone):
    if opens_sentence(m):
        options = [o for o in IN_ADDITION_OPTIONS[tone] if o != 'on top of' or doing_follows(m)]
        return match_case(m.group('phrase'), random.choice(options))
    words = WORDS_BEFORE_ADDITION.findall(text_before(m, 50))
    if not words or any(c.isdigit() for c in words[-1]):
        return m.group(0)
    if 'VERB' in word_classes(words[-1]):
        k = len(words) - 2
        while k >= 0 and (words[k].lower().endswith('ly') or words[k].lower() in ADVERBS_AFTER_SUBJECT):
            k -= 1
        subject = words[k] if k >= 0 else ''
        classes = word_classes(subject.lower())
        if (subject.lower() in ADDITION_SUBJECT_PRONOUNS or subject[:1].isupper() or k < len(words) - 2
                or ('NOUN' in classes and 'ADJ' not in classes)):
            return m.group(0)
    return match_case(m.group('phrase'), 'as well as')

# "had the ability to" after a form of "have" is "been able to" ("has had the ability to" -> "has been able to")
HAVE_BEFORE_HAD = re.compile(r"\b(?:has|have|had|having|['’]ve|['’]d)[ \t]+\Z", re.I)

def has_ability_frame(m, tone):
    """"has the ability to" -> "can", "had the ability to" -> "could"; after a modal or "to" it is "be able to" ("will
    have the ability to" -> "will be able to", also with an adverb between: "will soon have the ability to" -> "will
    soon be able to"), after a form of "have" "been able to" ("has had the ability to" -> "has been able to"), and "does
    not have the ability to" -> "cannot". After another helping verb or in a question ("Does it have the ability to
    ...?") the phrase stays."""
    if m.group('modal'):
        modal = m.group('modal')
        return modal if modal.lower() == 'can' else modal + ' be able to'
    if m.group('neg'):
        return match_case(m.group('neg'), 'could not' if m.group('neg').lower().startswith('did') else 'cannot')
    have = m.group('have').lower()
    if have == 'had' and HAVE_BEFORE_HAD.search(text_before(m, 12)):
        return 'been able to'
    with_adverbs = AUX_AND_ADVERBS_BEFORE.search(text_before(m, 40))
    if with_adverbs and have == 'have':
        return m.group(0) if with_adverbs.group('aux').lower() in ('can', 'do', 'does', 'did') else 'be able to'
    if have == 'have' and AUX_EARLIER_IN_SENTENCE.search(text_before(m, 40)):
        return m.group(0)
    sentence_start = max(m.string.rfind(c, 0, m.start()) for c in '.!?\n') + 1
    if AUX_BEFORE.search(text_before(m, 16)) or QUESTION_START.match(m.string, sentence_start, m.start()):
        return m.group(0)
    return match_case(m.group('have'), {'had': 'could', 'having': 'being able to'}.get(have, 'can'))

# "is also one of the most popular" -> "is also among the most popular", "were one of the largest" -> "were among the
# largest": the stock frame goes and the superlative stays (without an adverb, "is/was/remains one of the most" are fixed
# phrases above). Only before a superlative: a word in -est that comes from a describing word ("largest" -> "large",
# "best" -> "good"), so "one of the forest paths" and "one of the honest few" stay.
ONE_OF_THE = re.compile(r"\b(?:is|was|are|were|remains|remain|remained)(?:[ \t]+(?:also|still|now|arguably|probably|easily|"
                        r"perhaps|undoubtedly|certainly|surely|clearly))?[ \t]+(?P<frame>one[ \t]+of[ \t]+the)"
                        r"(?=[ \t]+(?P<degree>most|least|[a-z]+est)\b)" + NOT_ONE_WHO_SINGULAR, re.I)

def one_of_the_among(m, tone):
    degree = m.group('degree').lower()
    if degree not in ('most', 'least'):
        lemmas = getAllLemmas(degree) if 'getAllLemmas' in globals() else {}
        describing = lemmas.get('ADJ', ()) + lemmas.get('ADV', ())
        if not describing or degree in describing:
            return m.group(0)
    return m.string[m.start():m.start('frame')] + 'among the'

CONTEXT_RULES = [
    (_context_rule(("we are pleased to announce that", "we are excited to announce that", "we are thrilled to announce that",
                    "we're pleased to announce that", "we're excited to announce that", "we're thrilled to announce that")),
     announce),
    (_context_rule(("it is evident that", "it is clear that", "it is apparent that", "it's clear that", "it's evident that")),
     evident),
    (_context_rule(("it is essential to", "it is crucial to", "it is vital to", "it is imperative to"),
                   r"(?=\s+(?:\w+ly\s+)?(?P<next>[A-Za-z]+))"), essential_to),
    (re.compile(r"\b(?:(?P<modal>could|may|might)\s+have|(?P<neg>(?:do|does|did)\s+not|(?:do|does|did)n['’]t)\s+have"
                r"|(?P<have>has|have))\s+the\s+potential\s+to\b", re.I), potential),
    (_context_rule(("brought about",)), brought_about),
    (_context_rule(("it can be argued that", "it could be argued that")), frame_to_adverb(ARGUABLY_OPTIONS)),
    (_context_rule(("it is not surprising that", "it's not surprising that", "it is no surprise that", "it's no surprise that",
                    "it comes as no surprise that", "it is hardly surprising that", "it's hardly surprising that",
                    "it is unsurprising that")), frame_to_adverb(UNSURPRISINGLY_OPTIONS)),
    (_context_rule(("there is no denying that", "it is undeniable that", "it cannot be denied that")),
     frame_to_adverb(UNDENIABLY_OPTIONS)),
    (_context_rule(("there is no doubt that", "there's no doubt that")), frame_to_adverb(CERTAINLY_OPTIONS)),
    (_context_rule(("it is possible that", "it's possible that")), frame_to_adverb(PERHAPS_OPTIONS, possible_event_follows)),
    (_context_rule(("it is estimated that", "it has been estimated that")), frame_to_adverb(ESTIMATES_OPTIONS)),
    (re.compile(r"\b(?:(?P<modal>will|would|could|may|might|should|must|can|to)\s+have|(?P<neg>(?:do|does|did)\s+not|"
                r"(?:do|does|did)n['’]t)\s+have|(?P<have>has|have|had|having))\s+the\s+(?:ability|capacity)\s+to\b", re.I),
     has_ability_frame),
    (_context_rule(("in addition to",)), in_addition_to),
    (ONE_OF_THE, one_of_the_among),
]
CONTEXT_OPTIONS = [ANNOUNCE_OPTIONS, EVIDENT_OPTIONS, ESSENTIAL_OPTIONS, POTENTIAL_OPTIONS,
                   {'*': ['caused', 'led to', 'could not']}, ARGUABLY_OPTIONS, UNSURPRISINGLY_OPTIONS, UNDENIABLY_OPTIONS,
                   PERHAPS_OPTIONS, ESTIMATES_OPTIONS, IN_ADDITION_OPTIONS,
                   {'*': ['be able to', 'being able to', 'been able to', 'the groundwork for', 'due to', 'set to',
                          'likely to', 'all over the world', 'look at', 'looks at', 'looked at', 'looking at', 'look into',
                          'looks into', 'looked into', 'looking into']}]

# A quantity phrase keeps its "of" before a determiner or a pronoun: "the vast majority of them" is "most of them",
# never "most them", and "most of what we eat". "many" only takes it before a plural ("many of these", not "many of
# the population"), "much" only before a singular, and the adjectives not at all ("considerable the budget"); there,
# and before a number ("a wide range of 50 products" isn't "many of 50 products"), the writer's phrase stays. Before
# a word in lower case "US" and "IT" are names, not "us" and "it" ("most US adults"), and a capital "A" is no article
# ("most A-list celebrities"). "More and more" takes "of" as "most" does ("more and more of them"), "a few" as "many"
# does ("a few of these").
BARE_QUANTITIES = {'most', 'much', 'many', 'considerable', 'substantial', 'numerous', 'diverse', 'more and more', 'a few',
                   'great'}
DETERMINER_AFTER = re.compile(r"\s+(?:(?P<number>[$€£]?\d)|(?P<plural>these|those|them|(?-i:us|Us|US(?![ \t]+[a-z]))|"
                              r"you|whom)\b|(?:the|this|that|which|what|whatever|whose|(?-i:an?)|my|your|his|her|its|our|"
                              r"their|(?-i:it|It|IT(?![ \t]+[a-z]))|all|both|each|some|any)\b|[\w-]+['’]s?(?=\s)|"
                              r"(?P<name>(?-i:[A-Z])))", re.I)

# A name after the phrase describes what is counted when a noun follows it ("most British voters", "many Microsoft
# products"). When its verb follows, the name is what is counted: a place or a body as a whole keeps "of" ("most of
# Europe now uses the euro"), people don't ("most Americans support the plan"), and the verb tells which ("uses"
# goes with one, "support" with many). Where the words after the name don't tell, the writer's phrase stays. The name
# stops at the end of its sentence ("the vast majority of Americans. It ..."), but not after "U.S." or "St."
NAME_AFTER_QUANTITY = re.compile(r"\s+(?P<name>[A-Z][\w'’&.-]*(?:(?<![a-z][a-z]\.)[ \t]+(?:(?:and|or|&)[ \t]+)?"
                                 r"(?:[A-Z][\w'’&.-]*|"
                                 r"\d[\d,.]*))*)(?P<rest>[^\n]{0,60})")
QUANTITY_HELPING_VERBS = {'is', 'are', 'was', 'were', 'has', 'have', 'had', 'does', 'do', 'did', 'will', 'would',
                          'can', 'could', 'may', 'might', 'must', 'shall', 'should'}

def quantity_takes_of(match):
    """For a quantity phrase before a capital: True if "of" stays ("most of Europe", "most of New York's budget"),
    False if it goes ("most Americans", "most British voters", "most UK households"), None if the words after the
    name don't tell. In a title or in capitals ("The Vast Majority Of People") a capital marks no name."""
    if any(word[0].isupper() for word in match.group('phrase').split()[1:]):
        return False
    found = NAME_AFTER_QUANTITY.match(match.string, match.end())
    if not found or 'getAllLemmas' not in globals():
        return None
    name, words = found.group('name'), re.findall(r"[^\W\d_]+|\d+|[^\w\s]", found.group('rest'))
    if re.search(r"['’]s?\Z", name):
        return True                     # a possessive: "most of New York's budget"
    if re.search(r"[a-z][a-z]\.\Z", name):
        name, words = name[:-1], []     # the name ends its sentence: no verb of its own follows ("of Americans.")
    elif found.group('rest')[:1] not in ('', ' ', '\t', ',', ';', ':', '!', '?', ')', '.'):
        return None                     # "I/O operations": the capital starts no plain name
    while (len(words) > 1 and not PHRASE_FUNCTION_WORD.match(words[0]) and 'ADV' in word_classes(words[0])
           and (words[1] in QUANTITY_HELPING_VERBS or 'VERB' in word_classes(words[1]))):
        words.pop(0)                    # an adverb before the verb: "Europe now uses", "Europe still has"
    word, after = (words + ['', ''])[:2]
    classes, after_classes, function = word_classes(word), word_classes(after), PHRASE_FUNCTION_WORD.match(after)
    if word in QUANTITY_HELPING_VERBS or not word.islower() or PHRASE_FUNCTION_WORD.match(word):
        verb = word if word in QUANTITY_HELPING_VERBS else ''      # "Europe is", "Europe, Asia and ...", "Europe."
    elif word.endswith('ing') or not classes & {'VERB', 'AUX'}:
        return False                    # a noun or an adjective: "British voters", "Apple apps", "NHS funding"
    elif 'NOUN' not in classes:
        verb = word
    # A word that is a noun and a verb ("uses", "tools", "staff", "cut") is the name's verb when its object follows
    # ("Europe uses the euro", "Europe cut its emissions"). Without -s it is otherwise a noun the name describes,
    # unless it may be a past tense ("IT staff", "Microsoft support"; people take the plain verb either way:
    # "Millennials rent"). With -s it is the verb before a bare object ("Europe uses coal") or when it is no plural
    # noun ("Africa remains"), and a noun when a verb follows ("Google tools changed").
    elif (function and function.group('determiner')) or after[:1].isupper() or after.isdigit():
        verb = word
    elif not word.endswith('s'):
        return None if word in getInflection(word, tag='VBD') else False
    elif (after_classes and after_classes <= {'NOUN', 'ADJ'}) or set(getAllLemmas(word)['NOUN']) == {word}:
        verb = word
    elif not function and after_classes & {'VERB', 'AUX'} and 'NOUN' not in after_classes:
        return False
    else:
        return None                     # "Europe lies in ...", "Microsoft products in stock"
    if verb in ('is', 'was', 'has', 'does') or (verb.endswith('s') and verb not in QUANTITY_HELPING_VERBS
                                                 and not base_verb(verb)):
        return True                     # "Europe uses", "Texas is"
    if verb in ('are', 'were', 'have', 'do') or (verb not in QUANTITY_HELPING_VERBS and base_verb(verb)
                                                 and verb not in getInflection(verb, tag='VBD')):
        return False                    # "Americans support"; "cut" may be the past, which tells nothing
    # No verb that tells: a name for people ends like "Americans", "Democrats", "Muslims", "Brits", "Hispanics";
    # one for a place or a body may end in -s too ("Texas", "Wales", "Congress") or not at all ("Europe"), and so
    # may a people with no plural ("Swiss", "Chinese", "British"): there the writer's phrase stays
    last = name.split()[-1]
    if re.search(r'(?:an|er|at|im|it|ic|ist|on|ot|al)s\Z', last):
        return False
    return None if re.search(r'[^s]s\Z|ss\Z|(?:ese|ish|ch)\Z', last) and not last.isupper() else True

# A phrase in Title Case inside an ordinary line is part of a title or a name ("Proust's In Search of Lost Time runs to
# seven volumes", "Bill Haley's Rock Around the Clock"); a heading in Title Case all through ("The Vast Majority Of
# People Agree") is rewritten like any other line. After "such", "quite", "rather" or "what" a phrase that starts with
# "a" is graded ("Quite a small number of patients" isn't "Quite a few patients", nor "Such a large number of fans"
# "Such many fans"). Academic keeps a good or bad effect, which in research may give the direction of a change.
TITLE_TEST_WORD = re.compile(r"\b[a-z]{4,}\b")
DEGREE_BEFORE_ARTICLE = re.compile(r"\b(?:such|quite|rather|what)[ \t]+\Z", re.I)
SIGNED_EFFECT = re.compile(r"\b(?:positive|negative|beneficial|detrimental|adverse)\s+(?:impact|effect)\b", re.I)

def keeps_fixed_phrase(match, tone, other_sense_before):
    """True where the fixed phrase stays as written: a title, a graded "a", a signed effect in Academic, or words before
    it (`other_sense_before`, the fourth item of its entry) that show another sense."""
    phrase, text = match.group('phrase'), match.string
    words = phrase.split()
    if any(word[:1].isupper() for word in words[1:]) and not phrase.isupper():
        line_start, line_end = text.rfind('\n', 0, match.start()) + 1, text.find('\n', match.end())
        rest = text[line_start:match.start()] + ' ' + text[match.end():line_end if line_end != -1 else len(text)]
        if any(not PHRASE_FUNCTION_WORD.match(word) for word in TITLE_TEST_WORD.findall(rest)):
            return True
    if words[0].lower() in ('a', 'an') and DEGREE_BEFORE_ARTICLE.search(text, max(0, match.start() - 12), match.start()):
        return True
    if tone == 'academic' and SIGNED_EFFECT.search(phrase):
        return True
    return bool(other_sense_before and other_sense_before.search(text, max(0, match.start() - 80), match.start()))

# The words after a phrase decide which replacements fit:
# - "many", "numerous", "a few" and "several" count things, so before a mass noun ("a wide range of accommodation", "a
#   wide variety of furniture") they don't fit, while "a variety of" and "lots of" do;
# - "most", "many", "lots of" and the other plain quantities make the counted noun the subject, so where the writer made
#   the verb agree with the phrase ("A wide range of options is available", "The majority of residents has voted", "A
#   plethora of tools exists"), or "there is" stands before it ("There is a wide range of options"), they would leave a
#   plural with a singular verb ("Many options is available");
# - a role in doing something is something one contributes to, not something that "matters for" it ("played a key role
#   in unlocking the secrets" -> "contributed a lot to unlocking ...", never "mattered a great deal for unlocking").
# Where no replacement fits, the writer's phrase stays.
COUNT_ONLY_QUANTIFIERS = {'many', 'numerous', 'a few', 'several'}
PLURAL_QUANTIFIERS = {'most', 'many', 'numerous', 'a few', 'several', 'plenty of', 'lots of', 'tons of', 'loads of',
                      'all kinds of', 'more and more', 'diverse'}
PLURAL_ONLY_NOUNS = {'people', 'children', 'men', 'women', 'police', 'cattle', 'feet', 'teeth', 'mice', 'geese', 'clothes',
                     'savings', 'earnings', 'belongings', 'surroundings', 'outskirts', 'premises', 'remains', 'contents'}
SINGULAR_VERB_FORMS = {'is', 'was', 'has', 'does', "isn't", "wasn't", "hasn't", "doesn't"}
PLURAL_VERB_FORMS = {'are', 'were', 'have', 'do', "aren't", "weren't", "haven't", "don't", 'can', 'could', 'will',
                     'would', 'may', 'might', 'must', 'shall', 'should'}
# The words of the clause after the phrase, with its commas: the verb may come after a list of examples between commas
# ("A wide range of options, including loans and grants, is available"), and a list that runs to the end of the clause
# has no verb of its own ("a wide range of infections, such as chest infections and dental abscesses.")
QUANTITY_CLAUSE_WORDS = re.compile(r"[A-Za-z'’-]+|[,.;:!?()\n]")
EXAMPLES_START = {'such', 'including', 'like', 'especially', 'notably', 'particularly', 'mainly', 'mostly', 'from'}
THERE_IS_BEFORE = re.compile(r"\b(?:there[ \t]+(?:is|was|has[ \t]+been|remains|exists|seems[ \t]+to[ \t]+be)|there['’]s|"
                             r"(?:is|was)[ \t]+there)[ \t]+\Z", re.I)

def quantity_nouns(words):
    """Where the nouns a quantity phrase counts are among the words after it, past determiners and describing words, as
    (the first one that is only a noun, the last one): "of these new products" -> both "products", "of accommodation
    options" -> "accommodation" and "options", "of local products" -> both "products". A plural noun ends the run
    ("factors plays a role"). None if something else comes first."""
    for k, word in enumerate(words[:5]):
        low = word.lower()
        determiner = PHRASE_FUNCTION_WORD.match(low)
        classes = word_classes(low)
        if ((determiner and determiner.group('determiner')) or word.endswith('-') or ('-' in low and not classes)
                or (classes and 'NOUN' not in classes)):
            continue        # "these", "high-", "high-quality", "new", "very", "highly", "existing"
        if not word.islower():
            return None
        first, last = None, k
        while True:
            if first is None and 'ADJ' not in word_classes(words[last].lower()):
                first = last
            following = words[last + 1].lower() if last + 1 < len(words) else ''
            if (plural_noun(words[last].lower()) or not following.isalpha() or PHRASE_FUNCTION_WORD.match(following)
                    or 'NOUN' not in word_classes(following)):
                return (last if first is None else first), last
            last += 1
    return None

def plural_noun(word):
    """True if a noun in lower case can be a plural ("options", "courses", "people", "clothes"); a word lemminflect
    doesn't know counts by its -s. "data" and "media" are read as mass nouns ("most data is")."""
    if word in PLURAL_ONLY_NOUNS:
        return True
    if word in ('data', 'media') or not word.endswith('s') or word.endswith(('ss', 'us', 'is')):
        return False
    lemmas = getAllLemmas(word).get('NOUN', ()) if 'getAllLemmas' in globals() else ()
    return not lemmas or any(lemma != word for lemma in lemmas)

def singular_verb_after(words, head):
    """True if the first verb after the counted noun is singular ("options is", "tools exists", "factors plays a
    role", "courses that suits every budget"). A word that is also a plural noun counts as the verb right after the
    noun or after "that", "which" or "who"; elsewhere it is passed over, and so is a list of examples after a comma."""
    examples = False
    for k in range(head + 1, len(words)):
        word = words[k].lower().replace('’', "'")
        if word == ',':
            examples = k + 1 < len(words) and words[k + 1].lower() in EXAMPLES_START
            continue
        if examples:
            continue
        if word in SINGULAR_VERB_FORMS:
            return True
        if word in PLURAL_VERB_FORMS:
            return False
        if word.endswith('s') and 'getAllLemmas' in globals() and any(
                word in getInflection(lemma, tag='VBZ') for lemma in getAllLemmas(word).get('VERB', ())):
            if k == head + 1 or words[k - 1].lower() in ('that', 'which', 'who') or 'NOUN' not in word_classes(word):
                return True
    return False

def fitting_phrase_options(match, choices):
    """The replacements in `choices` that fit the words after the phrase (see above)."""
    following = ONE_WORD_AFTER.match(match.string, match.end())
    if following and following.group(1).lower().endswith('ing'):
        choices = [c for c in choices if not c.startswith('matter')]
    if not match.group('phrase').lower().endswith('of') or not any(c in PLURAL_QUANTIFIERS for c in choices):
        return choices
    words = []
    for found in QUANTITY_CLAUSE_WORDS.finditer(match.string, match.end(), match.end() + 160):
        if (found.group(0) != ',' and not found.group(0)[0].isalpha()) or len(words) == 14:
            break
        words.append(found.group(0))
    nouns = quantity_nouns(words)
    if nouns is None:
        return choices
    first, head = nouns
    if not plural_noun(words[first].lower()):
        choices = [c for c in choices if c not in COUNT_ONLY_QUANTIFIERS]
    if plural_noun(words[head].lower()) and (
            singular_verb_after(words, head) or THERE_IS_BEFORE.search(match.string, max(0, match.start() - 30), match.start())):
        choices = [c for c in choices if c not in PLURAL_QUANTIFIERS]
    return choices

def rewrite_phrase(match, options, tone, other_sense_before=None):
    if keeps_fixed_phrase(match, tone, other_sense_before):
        return match.group(0)
    choices = fitting_phrase_options(match, options.get(tone) or options['*'])
    if not choices:
        return match.group(0)
    new = match_case(match.group('phrase'), random.choice(choices))
    after = DETERMINER_AFTER.match(match.string, match.end())
    of = match.group('phrase').split()[-1]
    if after and new.lower() in BARE_QUANTITIES and of.lower() == 'of':
        needs_of = quantity_takes_of(match) if after.group('name') else True
        if needs_of is None:
            return match.group(0)
        if needs_of:
            word, plural = new.lower(), bool(after.group('plural'))
            takes_of = (word in ('most', 'more and more') or (word == 'much' and not plural)
                        or (word in ('many', 'a few') and plural))
            if after.group('number') or not takes_of:
                return match.group(0)
            new += ' ' + of
    article = match.group('art')
    if not article:
        return new
    wanted = 'an' if starts_with_vowel_sound(new) else 'a'
    return match_case(article.strip(), wanted) + article[len(article.rstrip()):] + new

# --- Plainer wording used at a rate ---
# Each rule below has a chance by tone, (outside Chaos Mode, in Chaos Mode); a tone it leaves out keeps the writer's
# wording (Academic keeps "more than", "such as", "per", "that" and "allows").
PHRASE_RATE_ALWAYS = {tone: (1.0, 1.0) for tone in TONES}
PHRASE_RATE_MOSTLY = {tone: (0.8, 1.0) for tone in TONES}

def phrase_rate(rates, tone, deep_mode):
    return rates.get(tone, (0.0, 0.0))[1 if deep_mode else 0]

def word_before_match(m, width=30):
    found = re.search(r"([A-Za-z'’-]+)[ \t]+\Z", text_before(m, width))
    return found.group(1) if found else ''

# "Ahead of the king's visit," -> "Before the king's visit,". Before an event or a date "ahead of" says when; before
# anything else it is a place in a race or a ranking ("well ahead of its rivals", "ahead of the curve", "ahead of last
# year's sales"), and after "be", "stay", "finish" or a margin it says where something stands ("We are well ahead of
# the deadline", "two points ahead of the final"): there it stays. So does "hours ahead of the final": "before" there
# could be moved away from its "hours". After a verb of motion or a distance it is a place too ("The boats raced ahead
# of the storm", "a police escort rode ahead of the funeral", "a few steps ahead of the tour"), and so is it after
# "think" or "plan"; "the spring" may be a part of a machine ("Place the washer ahead of the spring").
OCCASION_NOUNS = {
    'visit', 'visits', 'meeting', 'meetings', 'summit', 'election', 'elections', 'vote', 'referendum', 'launch', 'release',
    'debut', 'premiere', 'opening', 'ceremony', 'wedding', 'funeral', 'festival', 'holiday', 'holidays', 'trip', 'tour',
    'match', 'final', 'finals', 'semi-final', 'semi-finals', 'semifinal', 'race', 'tournament', 'championship',
    'championships', 'conference', 'talks', 'negotiations', 'hearing', 'trial', 'deadline', 'exam', 'exams', 'interview',
    'debate', 'speech', 'announcement', 'weekend', 'arrival', 'inauguration', 'anniversary', 'celebration',
    'celebrations', 'olympics', 'derby', 'rematch', 'fixture', 'season', 'auction', 'storm', 'hurricane', 'verdict',
    'ruling', 'christmas', 'easter', 'thanksgiving', 'ramadan', 'diwali', 'eid', 'hanukkah', 'monday', 'tuesday',
    'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'january', 'february', 'march', 'april', 'june', 'july',
    'august', 'september', 'october', 'november', 'december', 'spring', 'summer', 'autumn', 'winter'}
OCCASION_WORD = re.compile(r"[ \t]+([^\s,.;:!?()\"“”]+)")
OCCASION_ENDS = {'and', 'but', 'or', 'in', 'on', 'at', 'to', 'for', 'with', 'from', 'by', 'after', 'when', 'as', 'while',
                 'where', 'which', 'who', 'that', 'this', 'next', 'last', 'than', 'because', 'so'}
AHEAD_STANDING_BEFORE = re.compile(r"(?:\b(?:is|are|was|were|be|been|being|am|stay\w*|keep\w*|kept|remain\w*|get|gets|got|"
                             r"getting|finish\w*|came|come|comes|placed|rank\w*|lead|leads|led|moved?|moves|moving|"
                             r"climb\w*|pull\w*|edged?|edges|sit|sits|sat|lie|lies|lay|stand|stands|stood|ran|run|runs)"
                             r"(?:[ \t]+(?:well|far|way|just|still|already|slightly|comfortably|clearly|so|much|even|now))?|"
                             r"\b(?:far|way|well|points?|places?|lengths?|laps?|goals?|shots?|strokes?|games?|votes?|"
                             r"seats?|seconds?|minutes?|hours?|days?|weeks?|months?|years?|moments?|shortly|long|just)|"
                             r"['’](?:s|re))\s+\Z", re.I)
AHEAD_MOTION_BEFORE = re.compile(r"\b(?:rac(?:e|es|ed|ing)|driv(?:e|es|en|ing)|drove|walk\w*|rode|rid(?:e|es|ing)|marched|"
                                 r"marching|sail\w*|fl(?:y|ew|ies|ying|own)|went|go|goes|going|gone|travel\w*|headed|"
                                 r"heading|speed\w*|sped|rush\w*|hurr(?:y|ied|ies|ying)|stream\w*|surg(?:e|es|ed|ing)|"
                                 r"charg(?:e|es|ed|ing)|strode|strid\w+|cycl\w+|steam\w*|flee\w*|fled|escap\w+|swept|"
                                 r"sweep\w*|think\w*|thought|look\w*|plan\w*|stay\w*)\b(?:[ \t]+[\w'’-]+){0,3}[ \t]+\Z", re.I)
AHEAD_DISTANCE_BEFORE = re.compile(r"\b(?:steps?|paces?|yards?|metres?|meters?|miles?|feet|foot|km|kilometres?|kilometers?|"
                                   r"inches|blocks?|lengths?)[ \t]+\Z", re.I)
AHEAD_OF_A_SPRING = re.compile(r"[ \t]+the[ \t]+spring\b(?=[ \t]*(?:[,.;:!?)\n]|$)|[ \t]+(?:and|or|to|in|on|at|for|with|from|by|"
                               r"into|onto)\b)", re.I)

def ahead_of_event(m, tone):
    before = text_before(m, 40)
    if (AHEAD_STANDING_BEFORE.search(before[-30:]) or AHEAD_MOTION_BEFORE.search(before)
            or AHEAD_DISTANCE_BEFORE.search(before) or AHEAD_OF_A_SPRING.match(m.string, m.end())):
        return m.group(0)
    words, at = [], m.end()
    while len(words) < 6:
        found = OCCASION_WORD.match(m.string, at)
        if not found or (words and found.group(1).lower() in OCCASION_ENDS):
            break
        words.append(found.group(1))
        at = found.end()
    ended = at == len(m.string) or m.string[at] in ',.;:!?)\n' or (len(words) < 6 and bool(OCCASION_WORD.match(m.string, at)))
    last = re.sub(r"['’]s?\Z", '', words[-1]).lower() if words else ''
    if not ended or last not in OCCASION_NOUNS:
        return m.group(0)
    return match_case(m.group(0), 'before')

# "tend to" + a verb states what usually happens: "Students who slept longer tended to earn higher grades" -> "...
# generally earned higher grades". The verb takes the tense and person of "tend"; after "do" it stays plain ("don't
# tend to notice" -> "don't usually notice", "Do older dogs tend to sleep more?" -> "Do older dogs usually sleep
# more?"). Left alone: "tend to" + a noun ("nurses tend to the wounded", "tend to people who ...", "tended to mother and
# baby"), an opinion ("I tend to agree": an inclination, not a habit), a modal in the clause ("may also tend to"),
# "tend to be" after "do" or in a question ("Do older dogs tend to be calmer?" can't take "are"), "tended to be" (whose
# verb would need a subject the phrase can't see), a "tend" after another adverb of frequency ("They often tend to
# forget"), and "tends"/"tended" before a second verb they govern ("tends to improve mood and reduce stress": only the
# first verb would take the tense).
TEND_TO = re.compile(r"\b(?P<tend>tend|tends|tended)\s+to\s+(?P<verb>[a-z]+)\b")
TEND_NOUN_ENDS = re.compile(r"[ \t]*(?:[^\w \t]|$)|[ \t]+(?:who|whom|that|which|and|or)\b", re.M)
TEND_MODAL_IN_CLAUSE = re.compile(r"\b(?:(?:can|could|will|would|shall|should|may|might|must)(?:n['’]t)?|(?:ca|wo)n['’]t)\b"
                                  r"(?:[ \t]+[\w'’-]+){0,4}[ \t]+\Z", re.I)
TEND_DO_IN_CLAUSE = re.compile(r"\b(?:do|does|did)(?:n['’]t)?\b(?:[ \t]+[\w'’-]+){0,4}[ \t]+\Z", re.I)
TEND_COORDINATED = re.compile(r"(?:,|\b(?:and|or|but)\b)[ \t]+(?:[a-z]+ly[ \t]+)?(?P<verb>[a-z]+)\b")
COORDINATION_STOP = re.compile(r"[.;:!?\n]")

def clause_before_match(m):
    """The words of the clause before a match, from the last stop (. ; : ! ? or a line break)."""
    return m.string[max(m.string.rfind(c, 0, m.start()) for c in '.;:!?\n') + 1:m.start()]

def coordinated_verb_follows(m, end):
    """True if a second plain verb is joined on after `end` in the clause ("... and reduce stress", ", cause ...")."""
    stop = COORDINATION_STOP.search(m.string, end)
    rest = m.string[end:stop.start() if stop else len(m.string)]
    return any(base_verb(c.group('verb')) and not PHRASE_FUNCTION_WORD.match(c.group('verb'))
               for c in TEND_COORDINATED.finditer(rest))
TEND_CARED_FOR = {'people', 'staff', 'man', 'men', 'house', 'garden', 'water', 'stock'}
TEND_HABIT_OPTIONS = {'standard': ['usually', 'generally'], 'casual': ['usually'], 'professional': ['generally', 'typically'],
                 'academic': ['generally', 'typically']}
TEND_HABIT_ECHOES = {'usually': ('usually', 'typically'), 'typically': ('usually', 'typically'), 'generally': ('generally',)}
TEND_OPINION_VERBS = {'agree', 'disagree', 'think', 'believe', 'feel', 'prefer', 'favour', 'favor', 'doubt', 'suspect', 'assume',
                 'side', 'lean', 'suppose', 'guess', 'imagine', 'trust', 'view', 'see', 'regard'}
TEND_HOW_OFTEN = re.compile(r"\b(?:usually|generally|typically|often|always|never|rarely|seldom|sometimes|frequently|commonly|"
                       r"normally|mostly|occasionally)[ \t]+\Z", re.I)
TEND_DO_SUPPORT = re.compile(r"\b(?:do|does|did)(?:[ \t]+not|n['’]t)[ \t]+\Z", re.I)

def tend_to_habit(m, tone):
    tend, verb, before = m.group('tend'), m.group('verb'), text_before(m, 20)
    if not base_verb(verb) or verb in TEND_OPINION_VERBS or TEND_HOW_OFTEN.search(before):
        return m.group(0)
    if verb in TEND_CARED_FOR or ('NOUN' in word_classes(verb) and TEND_NOUN_ENDS.match(m.string, m.end('verb'))):
        return m.group(0)     # "nurses tend to people in need": what is cared for
    clause = clause_before_match(m)
    if TEND_MODAL_IN_CLAUSE.search(clause) or (verb == 'be' and (TEND_DO_IN_CLAUSE.search(clause)
                                                                or ends_in_question(m.string, m.end()))):
        return m.group(0)
    if tend != 'tend' and coordinated_verb_follows(m, m.end('verb')):
        return m.group(0)
    # A finding in the past is a tendency, not a habit ("tended to report higher stress" -> "generally reported"); an
    # adverb used nearby isn't repeated, in case the two sentences are joined ("typically" may become "usually" later)
    nearby = m.string[max(0, m.start() - 160):m.end() + 160].lower()
    choices = [c for c in (['generally'] if tend == 'tended' else TEND_HABIT_OPTIONS[tone])
               if not any(w in nearby for w in TEND_HABIT_ECHOES[c])]
    if not choices:
        return m.group(0)
    adverb = random.choice(choices)
    if TEND_DO_SUPPORT.search(before):
        return adverb + ' ' + verb if tend == 'tend' and verb != 'be' else m.group(0)
    if AUX_BEFORE.search(before):
        return m.group(0)
    if verb == 'be':
        if tend == 'tended' or re.search(r"\bI[ \t]+\Z", before):
            return m.group(0)
        return ('is ' if tend == 'tends' else 'are ') + adverb
    if tend == 'tend':
        return adverb + ' ' + verb
    forms = getInflection(verb, tag='VBZ' if tend == 'tends' else 'VBD')
    return adverb + ' ' + forms[0] if forms else m.group(0)

# "allows users to set limits" -> "lets users set limits", "allowing you to focus" -> "letting you focus". Only when the
# one allowed is a person or a group: with other words "to" may belong to the noun ("allow access to work files",
# "allow time to dry"). A passive ("are not allowed to enter") has no object and keeps its wording. A bare "allow"
# becomes "let" only after a modal, "to" or "do" ("would allow us to" -> "would let us"): "Flexible hours let parents
# ..." reads as the past as easily as the present. For the same reason a past "allowed" is left to the structure layer,
# which sees whether its subject is singular ("The new law let workers strike"): after a plural one "let" reads as now
# ("The new rules let tenants keep pets"). A verb that is also a noun before another plain verb would join the group's
# name ("lets farmers market produce" reads as a farmers' market), and "allow them to be right" concedes rather than
# permits: both keep "allow".
ALLOW_TO = re.compile(r"\b(?P<allow>allow|allows|allowing)\s+(?P<who>[a-z]+(?:[ \t]+[a-z]+){0,2}?)\s+to\s+(?P<verb>[a-z]+)\b")
LET_FORMS = {'allow': 'let', 'allows': 'lets', 'allowing': 'letting'}
ALLOWED_PEOPLE = {
    'you', 'us', 'them', 'him', 'her', 'me', 'everyone', 'anyone', 'everybody', 'anybody', 'people', 'users', 'customers',
    'students', 'employees', 'staff', 'workers', 'teams', 'companies', 'businesses', 'developers', 'readers', 'visitors',
    'members', 'parents', 'patients', 'children', 'kids', 'teachers', 'researchers', 'organizations', 'organisations',
    'governments', 'professionals', 'families', 'travelers', 'travellers', 'players', 'drivers', 'consumers', 'buyers',
    'clients', 'managers', 'owners', 'scientists', 'doctors', 'nurses', 'homeowners', 'tenants', 'landlords', 'investors',
    'shoppers', 'learners', 'runners', 'cyclists', 'citizens', 'residents', 'guests', 'viewers', 'listeners', 'creators',
    'artists', 'writers', 'farmers', 'firms', 'schools', 'banks', 'retailers', 'brands', 'partners', 'participants',
    'subscribers', 'beginners', 'adults', 'women', 'men', 'individuals', 'communities', 'councils', 'hospitals',
    'universities', 'engineers', 'designers', 'officials', 'administrators', 'admins', 'editors', 'pupils', 'athletes',
    'fans', 'volunteers', 'colleagues'}

ALLOWED_TO_CONCEDE = {'right', 'wrong', 'correct', 'mistaken'}

def allow_to_let(m, tone):
    words, before = m.group('who').split(), text_before(m, 20)
    if words[-1] not in ALLOWED_PEOPLE or not base_verb(m.group('verb')) or PASSIVE_BEFORE.search(before):
        return m.group(0)
    if m.group('allow') == 'allow' and not AUX_BEFORE.search(before):
        return m.group(0)
    following = ONE_WORD_AFTER.match(m.string, m.end())
    after = following.group(1) if following else ''
    if m.group('verb') == 'be' and after in ALLOWED_TO_CONCEDE:
        return m.group(0)
    if 'NOUN' in word_classes(m.group('verb')) and base_verb(after) and not PHRASE_FUNCTION_WORD.match(after):
        return m.group(0)
    for word in words[:-1]:
        leading = PHRASE_FUNCTION_WORD.match(word)
        if not (leading and leading.group('determiner')) and 'ADJ' not in word_classes(word):
            return m.group(0)
    return LET_FORMS[m.group('allow')] + m.string[m.end('allow'):m.end('who')] + ' ' + m.group('verb')

# "is expected to begin next spring" -> "is due to begin next spring": for a scheduled start or end with its time,
# "expected" and "due" say the same. "is expected to rise" -> "is likely to rise": a forecast of an amount. Other verbs
# keep "expected", which may be a duty ("Students are expected to attend every lecture"), and so do people and groups of
# them, one or many ("Guests are expected to arrive by noon", "Each student is expected to finish by Friday", "The
# audience is expected to rise for the anthem"), and "widely expected", whose adverb belongs to it. After "in", "on",
# "at" or "by" a time must follow: "end in a draw", "close at a record high" and "finish in the top four" give a result,
# not a schedule. A name may be a person, so it is only ever "set to" ("Smith is set to start on Saturday").
EXPECTED_TO = re.compile(r"\bexpected(?=\s+to\s+(?P<verb>begin|start|open|close|end|finish|resume|reopen|launch|commence|"
                         r"take\s+place|be\s+completed|be\s+finished|rise|fall|grow|increase|decrease|decline|drop|"
                         r"climb|double|triple|peak|slow|ease|shrink|soar|surge|rebound)\b)")
EXPECTED_SCHEDULED_VERBS = {'begin', 'start', 'open', 'close', 'end', 'finish', 'resume', 'reopen', 'launch', 'commence',
                   'take place', 'be completed', 'be finished'}
EXPECTED_PERSON = re.compile(r"\b(?P<who>[A-Za-z]+)[ \t]+(?:is|are|was|were|isn['’]t|aren['’]t|wasn['’]t|weren['’]t)"
                             r"(?:[ \t]+not)?(?:[ \t]+(?:also|now|still|currently))?[ \t]+\Z")
SCHEDULE_TIME_WORDS = (r"(?:the[ \t]+)?(?:early|late|mid|middle|end|start|beginning|first|second|third|last|next|coming|spring|"
                       r"summer|autumn|fall|winter|morning|afternoon|evening|night|week|weekend|month|year|new[ \t]+year|"
                       r"noon|midnight|dawn|dusk|january|february|march|april|may|june|july|august|september|october|"
                       r"november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday|christmas|easter|"
                       r"tomorrow|today|tonight)\b")
EXPECTED_SCHEDULE_TIME = re.compile(r"\s+(?:(?:in|on|by)\s+(?:\d|" + SCHEDULE_TIME_WORDS + r")|at\s+(?:\d{1,2}(?:[:.]\d\d)?"
                                    r"[ \t]*(?:[ap]\.?m\b\.?|o['’]clock\b)|" + SCHEDULE_TIME_WORDS + r")|(?:next|this|later|"
                                    r"early|soon|before|after|within|around|from|during|tomorrow|tonight|today|shortly|"
                                    r"until)\b)", re.I)
BE_BEFORE_EXPECTED = re.compile(r"\b(?:is|are|was|were|be|been|isn['’]t|aren['’]t|wasn['’]t|weren['’]t|not)[ \t]+"
                                r"(?:(?:also|now|still|currently)[ \t]+)?\Z", re.I)
EXPECTED_SCHEDULED_OPTIONS = {'*': ['due', 'set']}
PEOPLE_GROUP_WORDS = {'i', 'he', 'she', 'we', 'they', 'who', 'you', 'everyone', 'everybody', 'anyone', 'person', 'people', 'staff',
                 'audience', 'crowd', 'congregation', 'class', 'jury', 'team', 'squad', 'crew', 'household', 'family'}

def word_names_people(word):
    """True if a word in lower case names a person or people ("students", "manager", "audience")."""
    lemmas = getAllLemmas(word).get('NOUN', ()) if 'getAllLemmas' in globals() else ()
    return (word in PEOPLE_GROUP_WORDS or word in ALLOWED_PEOPLE or word in PERSON_WORDS
            or any(lemma in PERSON_WORDS for lemma in lemmas))

def expected_to_due(m, tone):
    before = text_before(m, 40)
    if not BE_BEFORE_EXPECTED.search(before):
        return m.group(0)
    person = EXPECTED_PERSON.search(before)
    if person and word_names_people(person.group('who').lower()):
        return m.group(0)
    if re.sub(r'\s+', ' ', m.group('verb')) not in EXPECTED_SCHEDULED_VERBS:
        return 'likely'
    if not EXPECTED_SCHEDULE_TIME.match(m.string, m.end('verb')):
        return m.group(0)
    return 'set' if person and person.group('who')[:1].isupper() else pick(EXPECTED_SCHEDULED_OPTIONS, tone)

# "marked a turning point" -> "proved a turning point", "marks a milestone" -> "is a milestone", "has marked a
# turning point" -> "has been a turning point": the stock verb goes, the claim stays. Only where what marks it is the
# turning point itself: an event, a time or a document ("The launch marked ...", "This year marks ...", "The 2015
# agreement marked ...", "..., which marked ..."). A person or a body that marks one does something ("The coach marked
# a turning point by changing the formation", "The teacher marked a milestone on the chart"), and a marker in the
# landscape makes it literal ("A small cairn marks a turning point in the path"); the passive ("was marked by") stays
# too.
MARKED_MILESTONE = re.compile(r"\b(?P<have>(?:has|have|had)[ \t]+)?(?P<verb>marks|marked)(?=[ \t]+an?[ \t]+(?:(?:major|"
                              r"significant|key|crucial|pivotal|important|historic|decisive|real|true|genuine|clear|"
                              r"defining)[ \t]+)?(?:turning[ \t]+point|milestone|watershed)\b)")
MILESTONE_EVENTS = {
    'launch', 'election', 'vote', 'treaty', 'war', 'battle', 'year', 'years', 'decade', 'day', 'moment', 'event', 'deal',
    'law', 'act', 'reform', 'reforms', 'merger', 'crisis', 'victory', 'defeat', 'win', 'loss', 'release', 'debut',
    'change', 'move', 'step', 'shift', 'success', 'failure', 'rise', 'fall', 'collapse', 'era', 'period', 'century',
    'trip', 'visit', 'summit', 'match', 'game', 'season', 'final', 'anniversary', 'arrival', 'approval', 'renewal',
    'proposal', 'trial', 'survival', 'revival', 'festival', 'refusal', 'withdrawal', 'departure', 'closure',
    'partnership', 'championship', 'friendship', 'breakthrough', 'discovery', 'recovery', 'surgery', 'strike', 'coup',
    'ban', 'cut', 'cuts', 'vaccine', 'speech', 'report', 'ruling', 'verdict', 'budget', 'plan', 'policy', 'race',
    'storm', 'flood', 'fire', 'pandemic', 'recession', 'boom', 'bill', 'order', 'rule', 'rules', 'scheme'}
MILESTONE_EVENT_ENDING = re.compile(r"[a-z]{2,}(?:tion|sion|ment|ance|ence|ing|ery)s?\Z")
MILESTONE_AGENTS = {'nation', 'institution', 'organisation', 'organization', 'foundation', 'commission', 'administration',
                    'association', 'federation', 'corporation', 'delegation', 'generation', 'population', 'station',
                    'coalition', 'opposition', 'government', 'management', 'parliament', 'department', 'audience',
                    'residence', 'sibling', 'darling', 'building'}
MILESTONE_CLAUSE_WORDS = re.compile(r"[A-Za-z]+|\d+")
MILESTONE_RELATIVE = re.compile(r"\s*(?:which|who|whom|whose|that|where|when)\b", re.I)

def marked_by_event(m):
    """True if what marks the turning point is an event, a time or a document: a word of that kind among the last five
    of its clause ("The opening of the first railway marked", "The 1969 moon landing marked"; a relative clause between
    them is passed over: "The Industrial Revolution, which began in Britain, marked"). "this" or "it", and "which" after
    a comma, stand for something said before; a "which" or "that" with no comma stands for the noun right before it,
    which must be an event too ("the decision that marked", but not "The buoy that marks a turning point")."""
    parts = [p for p in re.split(r"[,;:()]", re.split(r"[.!?\n]", text_before(m, 100))[-1]) if p.strip()]
    words = [w.lower() for w in MILESTONE_CLAUSE_WORDS.findall(parts[-1])] if parts else []
    if words and words[-1] in ('this', 'it', 'these', 'those'):
        return True
    if words and words[-1] in ('which', 'that'):
        if len(words) == 1:
            return True                 # "..., which marked a turning point"
        words = words[:-1]
    if len(parts) > 1 and MILESTONE_RELATIVE.match(parts[-1]):
        words = [w.lower() for w in MILESTONE_CLAUSE_WORDS.findall(parts[-2])]
    words = words[-5:]
    if not words or words[-1] == 'who':
        return False
    return not any(w in MILESTONE_AGENTS for w in words) and any(
        w.isdigit() or w in MILESTONE_EVENTS or MILESTONE_EVENT_ENDING.match(w) for w in words)

def marked_milestone(m, tone):
    if not marked_by_event(m):
        return m.group(0)
    if m.group('have'):
        return m.group('have') + 'been'
    if PASSIVE_BEFORE.search(text_before(m, 20)):
        return m.group(0)
    return 'is' if m.group('verb') == 'marks' else 'proved'

# "Her research laid the foundation for modern cancer treatments" -> "... laid the groundwork for ...". Only for an idea,
# a field or an institution: the main noun of what follows is abstract by its ending or a listed idea ("growth",
# "modern cancer care", "the Reformation"). A building or a thing keeps its foundation ("laid the foundations for a
# cottage by the lake", "the porch", "the new library"), and so does anything laid by builders.
LAID_FOUNDATION = re.compile(r"\b(?P<verb>lay|lays|laid|laying)[ \t]+the[ \t]+foundations?[ \t]+(?:for|of)\b")
FOUNDATION_BUILT_THINGS = {'house', 'houses', 'home', 'homes', 'building', 'buildings', 'church', 'cathedral', 'chapel', 'temple',
                'mosque', 'synagogue', 'tower', 'library', 'school', 'hospital', 'bridge', 'hall', 'museum', 'theatre',
                'theater', 'stadium', 'wall', 'walls', 'extension', 'garage', 'shed', 'barn', 'factory', 'mill', 'station',
                'office', 'offices', 'block', 'apartment', 'apartments', 'structure', 'monument', 'castle', 'palace',
                'warehouse', 'plant', 'terminal', 'dam', 'road', 'railway', 'university', 'facility', 'residence',
                'mansion', 'basement', 'pavement', 'settlement', 'compartment', 'pharmacy', 'township', 'plantation',
                'entrance', 'pavilion', 'observatory', 'laboratory', 'embassy', 'junction', 'fortification',
                'fortifications'}
FOUNDATION_BUILDERS = re.compile(r"\b(?:workers|builders|masons|bricklayers|crews?|labou?rers|construction|contractors?)\b", re.I)
FOUNDATION_OBJECT_WORDS = re.compile(r"[A-Za-z'’-]+|[^\sA-Za-z'’-]")
FOUNDATION_ABSTRACT_ENDING = re.compile(r"[a-z]{3,}(?:tion|sion|ism|ogy|ics|ness|ship|ity|ance|ence|ment|cy)s?\Z")
FOUNDATION_IDEAS = {
    'growth', 'success', 'peace', 'research', 'science', 'future', 'industry', 'theory', 'theories', 'field', 'work',
    'reform', 'reforms', 'recovery', 'progress', 'change', 'changes', 'career', 'study', 'studies', 'trade', 'care',
    'medicine', 'law', 'laws', 'art', 'thought', 'ideas', 'idea', 'economy', 'era', 'age', 'treatments', 'computing',
    'internet', 'web', 'methods', 'method', 'approach', 'approaches', 'union', 'league', 'trust', 'system', 'systems',
    'framework', 'rules', 'state', 'nation', 'faith', 'culture', 'cultures', 'health', 'wealth', 'strength',
    'knowledge', 'learning', 'understanding', 'thinking', 'engineering', 'teaching', 'nursing', 'banking', 'writing',
    'design', 'discipline', 'profession', 'tradition', 'society', 'empire', 'republic', 'alliance', 'treaty', 'rights',
    'renaissance', 'enlightenment', 'literature', 'music', 'genre', 'language', 'mathematics', 'algebra', 'geometry',
    'calculus', 'chemistry', 'astronomy', 'evolution', 'company', 'firm', 'brand', 'dynasty', 'kingdom', 'party',
    'life', 'family', 'deal', 'victory', 'win', 'breakthrough', 'breakthroughs', 'discoveries', 'discovery', 'advances',
    'advance', 'lifestyle', 'habit', 'habits', 'skill', 'skills', 'mind', 'world'}
FOUNDATION_HEAD_STOPS = {'himself', 'herself', 'themselves', 'itself', 'myself', 'ourselves', 'yourself', 'last', 'next', 'is',
                         'are', 'was', 'were', 'will', 'would', 'has', 'have', 'had', 'can', 'could', 'did', 'does'}
FOUNDATION_LEADING_WORDS = {'the', 'a', 'an', 'its', 'their', 'his', 'her', 'our', 'my', 'your', 'this', 'that', 'these',
                            'those', 'all', 'much', 'many', 'some', 'every', 'each'}
# A span or a kind of something counts what follows "of" ("decades of trade", "new kinds of treatment")
FOUNDATION_SPANS = {'decades', 'years', 'centuries', 'generations', 'kinds', 'sorts', 'types', 'forms', 'waves'}

def foundation_head(m):
    """The main noun of what a foundation is laid for: the last word before a function word or punctuation ("for modern
    cancer care in Europe" -> "care", "of the company's success" -> "success", "for decades of trade" -> "trade")."""
    head = ''
    for word in FOUNDATION_OBJECT_WORDS.findall(m.string, m.end(), m.end() + 100):
        low = word.lower()
        if head in FOUNDATION_SPANS and low == 'of':
            head = ''
            continue
        if not word[0].isalpha() or (head and (PHRASE_FUNCTION_WORD.match(low) or low in FOUNDATION_HEAD_STOPS)):
            break
        if head or low not in FOUNDATION_LEADING_WORDS:
            head = re.sub(r"['’]s?\Z", '', low)
    return head

def laid_foundation(m, tone):
    head = foundation_head(m)
    if (head in FOUNDATION_BUILT_THINGS or not (head in FOUNDATION_IDEAS or FOUNDATION_ABSTRACT_ENDING.match(head))
            or FOUNDATION_BUILDERS.search(text_before(m, 60))):
        return m.group(0)
    return m.group('verb') + ' the groundwork for'

# "This study examined whether ..." -> "This study looked at whether ..." (Professional and Academic: "assessed
# whether"); the same for "explore". "investigate" is a formal inquiry ("Police are investigating whether the fire was
# arson"), which "look into" keeps in Standard and Casual and the other tones leave as written. "test" would fit a
# hypothesis but not a question of fact ("examined whether the rules had been followed").
STUDY_WHETHER = re.compile(r"\b(?P<stem>examin|investigat|explor)(?P<ending>es|ed|ing|e)(?=\s+whether\b)", re.I)
WHETHER_OPTIONS = {'standard': ['look at', 'assess'], 'casual': ['look at'], 'professional': ['assess'], 'academic': ['assess']}
INVESTIGATE_OPTIONS = {'standard': ['look into'], 'casual': ['look into']}

def study_whether(m, tone):
    options = INVESTIGATE_OPTIONS if m.group('stem').lower() == 'investigat' else WHETHER_OPTIONS
    if tone not in options or 'getInflection' not in globals():
        return m.group(0)
    verb, _, particle = random.choice(options[tone]).partition(' ')
    tag = {'es': 'VBZ', 'ed': 'VBD', 'ing': 'VBG'}.get(m.group('ending').lower())
    if tag:
        forms = getInflection(verb, tag=tag)
        if not forms:
            return m.group(0)
        verb = forms[0]
    return match_case(m.group(0), verb + (' ' + particle if particle else ''))

# "The authors caution that ..." -> "warn that ...". The verb follows its subject (a noun, a pronoun, a name), a modal or
# an adverb; the noun follows a determiner, a preposition, an adjective or a verb ("a note of caution that", "urged
# caution against") and stays. A plain "caution" is the verb only after a plural subject ("Experts caution that"):
# after a singular noun or an adjective it is the noun ("a general caution against", "a safety caution", "His final
# caution that"); "cautions" is the verb after a noun or a name that isn't an adjective ("The report cautions that").
CAUTION_THAT = re.compile(r"\b(?P<verb>caution|cautions|cautioned|cautioning)(?=[ \t]+(?:that|against)\b)")
WARN_FORMS = {'caution': 'warn', 'cautions': 'warns', 'cautioned': 'warned', 'cautioning': 'warning'}
CAUTION_SUBJECTS = {'we', 'they', 'he', 'she', 'i', 'you', 'it', 'who', 'which', 'that', 'also', 'further', 'may', 'might',
                    'would', 'should', 'must', 'will', 'can', 'could', 'often', 'now', 'still'}
TAKES_CAUTION = {'urge', 'urged', 'urges', 'urging', 'express', 'expressed', 'expresses', 'expressing', 'exercise',
                 'exercised', 'exercises', 'exercising', 'advise', 'advised', 'advises', 'advising', 'counsel', 'counselled',
                 'counseled', 'use', 'used', 'uses', 'show', 'showed', 'shown', 'shows', 'sound', 'sounded', 'sounds',
                 'strike', 'struck', 'strikes', 'recommend', 'recommended', 'recommends', 'add', 'added', 'adds', 'note',
                 'noted', 'notes', 'voice', 'voiced', 'voices', 'raise', 'raised', 'raises', 'offer', 'offered', 'offers',
                 'issue', 'issued', 'issues', 'preach', 'preached', 'preaches'}

def caution_that(m, tone):
    before = word_before_match(m)
    lower = before.lower()
    if lower in CAUTION_SUBJECTS or lower.endswith('ly'):
        return WARN_FORMS[m.group('verb')]
    classes = word_classes(lower)
    if (lower in TAKES_CAUTION or PHRASE_FUNCTION_WORD.match(lower) or lower in ('great', 'extreme', 'more', 'less', 'much')
            or not (classes or before[:1].isupper())):
        return m.group(0)
    verb = m.group('verb')
    plural_subject = lower.endswith('s') and not lower.endswith(('ss', "'s", '’s')) and 'NOUN' in classes and 'ADJ' not in classes
    if (verb == 'caution' and not plural_subject) or (verb == 'cautions' and 'ADJ' in classes):
        return m.group(0)
    return WARN_FORMS[verb] if 'NOUN' in classes or not classes else m.group(0)

# "go far beyond" -> "go well beyond" (not after a word that grades or denies the "far": "so", "too", "as", "how", "not",
# "very", "isn't"), and not where it opens a sentence about a place ("Far beyond the mountains lies a hidden kingdom")
FAR_BEYOND = re.compile(r"\bfar(?=[ \t]+beyond\b)", re.I)
FAR_GRADED = re.compile(r"(?:\b(?:so|too|as|how|not|very|quite|rather|that|this|pretty)|n['’]t)[ \t]+\Z", re.I)

def far_beyond(m, tone):
    if opens_sentence(m) or FAR_GRADED.search(text_before(m, 10)):
        return m.group(0)
    return match_case(m.group(0), 'well')

# "The museum is located in the old town" -> "The museum is in the old town", "was situated" -> "was". A past "was
# located" often means "was found" ("The missing hiker was located in a ravine") and stays.
PLACED_IN = re.compile(r"\b(?:(?P<be>is|are)[ \t]+(?:located|situated)|(?P<was>was|were)[ \t]+situated)[ \t]+(?=(?:in|on|at|near|"
                       r"inside|outside|along|beside|between|within|just|right|next[ \t]+to|close[ \t]+to)\b)")

def placed_in(m, tone):
    return (m.group('be') or m.group('was')) + ' '

# "More than 40 countries signed" -> "Over 40 countries signed", "with more than five years of experience" -> "with
# over five years of experience": before a count of things the two say the same. Only where the count opens its clause
# or follows a preposition: after a verb "over" and "under" read as prepositions, so the count would no longer be the
# verb's object ("comprised less than 3% of all responses" must stay a share for the word layer to keep "comprised"),
# and a measure stays too ("rose more than 4%", "took more than a decade"), as a later move could take it to the front
# ("Over 4% after the announcement, shares rose"). Not after a word that grades the comparison ("no more than", "much
# more than", "a p-value of less than"), nor where the sentence already says "over" or "under". "Under" before people
# reads as their age ("Teams with fewer than 11 players" isn't "Teams with under 11 players"), so it isn't used there.
COMPARED_SPELLED_COUNT = (r"(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|"
                 r"seventy|eighty|ninety|a[ \t]+hundred|hundreds|a[ \t]+thousand|thousands|a[ \t]+million|millions|"
                 r"a[ \t]+billion|billions|a[ \t]+dozen|dozens|half|a[ \t]+third|a[ \t]+quarter)")
COMPARED_AMOUNT = re.compile(r"\b(?P<cmp>more|less|fewer)[ \t]+than[ \t]+(?=(?P<amount>(?:[$€£]?\d[\d,.]*%?|" + COMPARED_SPELLED_COUNT +
                             r"(?![\w-]))(?:[ \t]+(?:hundred|thousand|million|billion))?))", re.I)
GRADED_COMPARISON = re.compile(r"\b(?:no|not|any|much|far|little|nothing|even|way|lot|bit|rather|slightly|considerably|"
                               r"significantly|substantially|somewhat|marginally|barely|hardly|scarcely|only|just|almost|"
                               r"nearly|still|ever|of|times|or)[ \t]+\Z", re.I)
COMPARED_COUNTED_WORD = re.compile(r"[ \t]+(?P<word>[a-z][a-z-]*)")
COMPARED_MEASURE_WORDS = {'second', 'seconds', 'minute', 'minutes', 'hour', 'hours', 'day', 'days', 'night', 'nights', 'week', 'weeks',
                 'month', 'months', 'year', 'years', 'decade', 'decades', 'century', 'centuries', 'time', 'times', 'percent',
                 'per', 'cent', 'points', 'degrees', 'miles', 'kilometres', 'kilometers', 'km', 'metres', 'meters', 'feet',
                 'inches', 'kilograms', 'kg', 'grams', 'pounds', 'tons', 'tonnes', 'litres', 'liters', 'dollars', 'euros',
                 'pence', 'cents', 'mph'}

OVER_COUNT_AFTER = {'with', 'for', 'from', 'among', 'in', 'to', 'at', 'on', 'across', 'within', 'by'}

def more_than_to_over(m, tone):
    if GRADED_COMPARISON.search(text_before(m, 16)) or not (opens_clause(m) or word_before_match(m).lower() in OVER_COUNT_AFTER):
        return m.group(0)
    counted = COMPARED_COUNTED_WORD.match(m.string, m.end('amount'))
    word = counted.group('word') if counted else ''
    if word in COMPARED_MEASURE_WORDS:
        then = COMPARED_COUNTED_WORD.match(m.string, counted.end())
        word = 'of' if then and then.group('word') == 'of' else ''     # "five years of experience" counts years
    if word != 'of' and ('NOUN' not in word_classes(word) or PHRASE_FUNCTION_WORD.match(word)):
        return m.group(0)
    new = 'over' if m.group('cmp').lower() == 'more' else 'under'
    if new == 'under' and word_names_people(word):
        return m.group(0)
    if re.search(r'\b' + new + r'\b', sentence_around(m.string, m.start(), m.end()), re.I):
        return m.group(0)
    return match_case(m.group('cmp'), new) + ' '

# "costs such as rent" -> "costs like rent" (Standard and Casual), but not "such as it is", "such as to" or the old "such
# as wish to join" (those who), nor after "like" in the same clause ("I'd like fruits such as apples")
SUCH_AS = re.compile(r"\bsuch[ \t]+as\b(?![ \t]+(?:it[ \t]+is|it[ \t]+was|they[ \t]+are|to|wish|wishes|want|wants|are|is|was|"
                     r"were|have|has|had|may|might|can|could|will|would|shall|should|do|does|did)\b)")
LIKE_IN_CLAUSE = re.compile(r"\blik(?:e|es|ed|ing)\b", re.I)

def such_as_to_like(m, tone):
    start = max(m.string.rfind(c, 0, m.start()) for c in '.!?\n;,') + 1
    return m.group(0) if LIKE_IN_CLAUSE.search(m.string, start, m.start()) else 'like'

# "150 minutes per week" -> "150 minutes a week", "$4 per month" -> "$4 a month": a rate after a number. The number must
# count the rate's own noun, a few words before "per" with no determiner, pronoun or verb "be" in between ("300,000 new
# homes per year", "two litres of water per day"): without one "per" belongs to a noun ("the cost per person", "In 2019
# the fee per year was raised"), and so it does after a year or a label ("Version 3 limits uploads per hour"). A
# compound unit stays ("9.8 metres per second squared").
PER_TIME = re.compile(r"\bper[ \t]+(?P<unit>second|minute|hour|day|night|week|month|year)\b(?![ \t]+(?:squared|cubed|per)\b)")
PER_COUNT = re.compile(r"[$€£]?\d[\d,.]*(?:%|[a-z]{1,3})?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                       r"fifteen|twenty|thirty|forty|fifty|sixty|hundred|hundreds|thousand|thousands|million|millions|"
                       r"several|few|once|twice|dozen|dozens", re.I)
PER_BREAKS = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'which', 'who', 'its', 'their', 'his', 'her', 'our', 'my',
              'your', 'each', 'every', 'is', 'are', 'was', 'were', 'be', 'been'}
PER_LABELS = {'version', 'section', 'chapter', 'article', 'room', 'level', 'phase', 'step', 'figure', 'table', 'page', 'model',
              'series', 'class', 'grade', 'stage', 'part', 'unit', 'no', 'number', 'route', 'gate', 'platform', 'floor', 'year'}
PER_YEAR_NUMBER = re.compile(r"(?:1[5-9]|20)\d\d")

def per_time_unit(m, tone):
    words = text_before(m, 60).split()
    for k in range(len(words) - 1, max(-1, len(words) - 6), -1):
        word = words[k].strip('()"“”‘’\'')
        if not word or word[-1] in ',.;:!?' or word.lower() in PER_BREAKS:
            return m.group(0)
        if PER_COUNT.fullmatch(word):
            if PER_YEAR_NUMBER.fullmatch(word) or (k and words[k - 1].lower().strip('(') in PER_LABELS):
                return m.group(0)
            return ('an ' if m.group('unit') == 'hour' else 'a ') + m.group('unit')
    return m.group(0)

# "in just six months" -> "in only six months", "With just a laptop" -> "With only a laptop": "just" before an amount
# means "only". "just over", "just under" and "just about" are no amounts and stay, and so do "just one of those" and a
# time of day ("It was just 5 o'clock").
JUST_A_COUNT = re.compile(r"\bjust(?![ \t]+one[ \t]+of[ \t]+those\b)(?=[ \t]+(?:[$€£]?\d|(?:one|two|three|four|five|six|seven|"
                          r"eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|a[ \t]+hundred|hundreds|"
                          r"a[ \t]+thousand|thousands|a[ \t]+few|a[ \t]+handful|a[ \t]+fraction|a[ \t]+couple|half)\b))"
                          r"(?![ \t]+(?:\d{1,2}(?:[:.]\d\d)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
                          r"[ \t]*(?:o['’]clock|[ap]\.?m\b))", re.I)
WITH_JUST = re.compile(r"(?<=\bwith[ \t])just(?=[ \t]+an?[ \t])", re.I)

def just_to_only(m, tone):
    return match_case(m.group(0), 'only')

# "Critics argued that the plan ..." -> "Critics argued the plan ...", "He said that it ..." -> "He said it ...":
# Standard and Casual drop the "that" of a reported clause where a subject follows it (a determiner, a pronoun, a
# name or a number). It stays after a noun ("his claims that the drug is safe" can't lose it), in a passive ("it was
# found that ..."), and before anything that could make "that" a pointing word ("argued that point"). A form that is
# also a noun is the noun after a possessive, a participle or an adjective ("false claims that the vaccine ...", "Fans'
# hopes that ...", "Growing fears that ...").
SAYING_VERBS = (r"said|says|say|think|thinks|thought|believe|believes|believed|hope|hopes|hoped|know|knows|knew|feel|feels|"
                r"felt|realize|realizes|realized|realise|realises|realised|guess|reckon|reckons|reckoned|mean|means|meant")
CLAIMING_VERBS = (r"suggest|suggests|suggested|argue|argues|argued|claim|claims|claimed|found|find|finds|show|shows|showed|"
                  r"shown|notice|notices|noticed|noted|insist|insists|insisted|fear|fears|feared|predict|predicts|"
                  r"predicted|conclude|concludes|concluded|admit|admits|admitted|discover|discovers|discovered|learn|"
                  r"learns|learned|learnt|confirm|confirms|confirmed|expect|expects|announce|announces|announced")
CLAUSE_SUBJECT_AHEAD = (r"(?=[ \t]+(?:(?:the|a|an|this|these|those|its|their|his|her|our|my|your|he|she|they|we|you|it|"
                        r"there|some|many|most|all|no|every|each|both|several)\b|I\b|[A-Z][a-z]|\d))")
SAYING_THAT = re.compile(r"\b(?P<verb>" + SAYING_VERBS + r")[ \t]+that" + CLAUSE_SUBJECT_AHEAD)
CLAIMING_THAT = re.compile(r"\b(?P<verb>" + CLAIMING_VERBS + r")[ \t]+that" + CLAUSE_SUBJECT_AHEAD)
THAT_NOUN_BEFORE = re.compile(r"\b(?:the|a|an|his|her|their|its|our|my|your|this|these|those|of)[ \t]+\Z", re.I)

def drop_reported_that(m, tone):
    before = text_before(m, 24)
    if THAT_NOUN_BEFORE.search(before) or PASSIVE_BEFORE.search(before):
        return m.group(0)
    if 'NOUN' in word_classes(m.group('verb')):
        word = word_before_match(m)
        classes = word_classes(word.lower())
        if (re.search(r"['’]s?\Z", word) or word.lower().endswith(('ed', 'ing'))
                or ('ADJ' in classes and not classes & {'NOUN', 'VERB'})):
            return m.group(0)
    return m.group('verb')

# ...and after an adjective of feeling or certainty ("I am confident that my skills ..." -> "I am confident my skills
# ..."), but not after "so", "too" or "such" ("so certain that ..." gives a result) or in "It is certain that ...".
FEELING_THAT = re.compile(r"\b(?P<verb>confident|sure|certain|convinced|glad|pleased|happy|afraid|worried|proud|delighted|"
                          r"surprised)[ \t]+that" + CLAUSE_SUBJECT_AHEAD)
FEELING_GRADED = re.compile(r"\b(?:so|too|such|it[ \t]+is|it[ \t]+was|it['’]s|is[ \t]+it|was[ \t]+it)[ \t]+\Z", re.I)

def drop_that_after_feeling(m, tone):
    return m.group(0) if FEELING_GRADED.search(text_before(m, 12)) else m.group('verb')

# "so that they could run on the new platform" -> "so they could run ...": before a subject the "that" adds nothing
SO_THAT = re.compile(r"\bso[ \t]+that(?=[ \t]+(?:(?:the|a|an|this|these|those|its|their|his|her|our|my|your|he|she|they|we|"
                     r"you|it|there|people|everyone|everybody)\b|I\b))")

def so_that_to_so(m, tone):
    return m.group(0) if re.search(r"\b(?:even|and|or|not)[ \t]+\Z", text_before(m, 8)) else 'so'

# "communities around the world" -> "communities worldwide", "visitors from around the world" -> "visitors from all over
# the world". After a verb of travel, or a journey a few words back, it is a journey ("sailed around the world", "to fly
# solo around the world", "the trip of a lifetime around the world") and stays, and so does a phrase that goes on
# ("around the world in 80 days", "around the world's oceans").
AROUND_THE_WORLD = re.compile(r"\b(?P<from>from[ \t]+)?(?P<phrase>(?:a?round|across)[ \t]+the[ \t]+(?:world|globe))\b"
                              r"(?![ \t]+(?:of|in)\b)(?!-)", re.I)
JOURNEY_WORDS = {'trip', 'trips', 'tour', 'tours', 'journey', 'journeys', 'voyage', 'voyages', 'cruise', 'cruises', 'flight',
                 'flights', 'race', 'races', 'way', 'halfway', 'miles', 'kilometres', 'kilometers', 'distance', 'travel',
                 'travels', 'sail', 'sails', 'cycle', 'cycles', 'walk', 'walks', 'run', 'runs', 'row', 'rows', 'drive',
                 'drives', 'ride', 'rides', 'swim', 'swims', 'go', 'goes', 'circle', 'circles', 'spin', 'spins', 'echo',
                 'echoes', 'shot', 'shots', 'orbit', 'orbits', 'turn', 'turns', 'trek', 'treks', 'hike', 'hikes'}

AROUND_TRAVEL_BEFORE = re.compile(r"\b(?:fl(?:y|ew|own|ying|ies)|bike[ds]?|biking|cycl\w+|backpack\w*|hitchhik\w+|solo|"
                                  r"travel\w*|sail\w*|walk\w*|ran|run\w*|drive|drove|driv\w+|rode|rid(?:e|es|ing)|swim\w*|"
                                  r"swam|journey\w*|trek\w*|tour\w*|wander\w*|roam\w*|circl\w+|orbit\w*|voyage\w*|cruis\w+|"
                                  r"trips?|went|go|goes|going|gone|laps?|rac(?:e|es|ed|ing)|laid|lay|lays|laying|wrap\w*|"
                                  r"stretch\w*|encircl\w+)\b(?:[ \t]+[\w'’-]+){0,4}[ \t]+\Z", re.I)
WORLD_POSSESSIVE_FOLLOWS = re.compile(r"['’]")

def around_the_world(m, tone):
    if WORLD_POSSESSIVE_FOLLOWS.match(m.string, m.end()):
        return m.group(0)
    if m.group('from'):
        return m.group('from') + match_case(m.group('phrase'), 'all over the world')
    if opens_sentence(m):
        return match_case(m.group('phrase'), 'worldwide') if m.string[m.end():m.end() + 1] == ',' else m.group(0)
    word = word_before_match(m)
    if (not word.islower() or word in JOURNEY_WORDS or 'NOUN' not in word_classes(word)
            or AROUND_TRAVEL_BEFORE.search(text_before(m, 60))):
        return m.group(0)
    return 'worldwide'

# Chaos Mode drops every empty intensifier before a word that reads the same without it ("very busy" -> "busy", "really
# helps" -> "helps"); outside it the writer's emphasis stays. Words whose degree is information keep it ("very rare", "very low", "very few", "very likely"),
# and so do "not very", "the very first", "a very old" (whose article would need to change), "really?" in a question,
# and the second of a doubled one ("really, really happy").
PLAIN_DEGREE_WORDS = {
    'important', 'useful', 'helpful', 'good', 'nice', 'easy', 'difficult', 'hard', 'busy', 'popular', 'interesting',
    'beautiful', 'effective', 'efficient', 'clear', 'simple', 'happy', 'proud', 'excited', 'pleased', 'grateful',
    'thankful', 'careful', 'friendly', 'comfortable', 'convenient', 'powerful', 'impressive', 'successful', 'valuable',
    'attractive', 'exciting', 'enjoyable', 'rewarding', 'challenging', 'competitive', 'stressful', 'frustrating',
    'satisfying', 'satisfied', 'special', 'clean', 'tasty', 'delicious', 'fresh', 'fun', 'welcoming', 'knowledgeable',
    'reliable', 'accurate', 'detailed', 'informative', 'engaging', 'inspiring', 'motivated', 'passionate', 'keen',
    'eager', 'confident', 'determined', 'committed', 'dedicated', 'excellent', 'amazing', 'great', 'stylish', 'smooth',
    'quiet', 'versatile', 'intuitive', 'affordable', 'generous', 'supportive', 'professional', 'responsive', 'pleasant'}
REALLY_VERBS = {'help', 'helps', 'helped', 'like', 'likes', 'liked', 'love', 'loves', 'loved', 'enjoy', 'enjoys', 'enjoyed',
                'appreciate', 'appreciates', 'appreciated', 'value', 'values', 'valued', 'matter', 'matters', 'mattered',
                'need', 'needs', 'want', 'wants'}
EMPTY_INTENSIFIER = re.compile(r"\b(?P<word>very|really)[ \t]+(?P<next>[a-z]+)\b")
INTENSIFIER_GRADED_BEFORE = re.compile(r"(?:\b(?:not|never|no|the|a|an|so|too|how|this|that|as)|n['’]t|\b(?:very|really)[ \t]*,)"
                                       r"[ \t]+\Z", re.I)

def drop_intensifier(m, tone):
    word, following = m.group('word'), m.group('next')
    if following not in PLAIN_DEGREE_WORDS and not (word == 'really' and following in REALLY_VERBS):
        return m.group(0)
    if INTENSIFIER_GRADED_BEFORE.search(text_before(m, 10)) or (word == 'really' and ends_in_question(m.string, m.end())):
        return m.group(0)
    return following

# "may result in disciplinary action" -> "may lead to ...", "resulted in" -> "led to". "result" and "results" are also
# nouns ("the result in Ohio", "the results in Table 2"), so they count as the verb only after a modal, "to" or a
# negation ("result"), or after "which", "that", "it", "this" or an adverb ("results"); "the resulting" is an adjective
# and "in-depth" a word of its own ("The resulting in-depth review"). A sentence that already says "led" or "lead"
# keeps "resulted in" ("I led a campaign that resulted in ..." isn't "I led a campaign that led to ...").
RESULT_IN = re.compile(r"\b(?P<verb>result|results|resulted|resulting)[ \t]+in\b(?!-)")
LED_IN_SENTENCE = re.compile(r"\b(?:led|leads?|leading)\b", re.I)
RESULTING_DETERMINER_BEFORE = re.compile(r"\b(?:the|a|an|this|that|these|those|its|their|his|her|our|any)[ \t]+\Z", re.I)
LEAD_FORMS = {'result': 'lead to', 'results': 'leads to', 'resulted': 'led to', 'resulting': 'leading to'}
RESULT_VERB_BEFORE = {'result': {'may', 'might', 'can', 'could', 'will', 'would', 'should', 'must', 'to', 'not', 'cannot',
                                 'usually', 'often', 'also', 'eventually', 'ultimately', 'typically', 'sometimes', 'rarely',
                                 'generally', 'directly', 'inevitably', 'likely', 'only'},
                      'results': {'which', 'that', 'it', 'this', 'often', 'usually', 'also', 'eventually', 'ultimately',
                                  'typically', 'sometimes', 'rarely', 'generally', 'directly', 'inevitably', 'only'}}

def result_in_to_lead(m, tone):
    verb = m.group('verb')
    if LED_IN_SENTENCE.search(sentence_around(m.string, m.start(), m.end())):
        return m.group(0)
    if verb == 'resulting' and RESULTING_DETERMINER_BEFORE.search(text_before(m, 10)):
        return m.group(0)
    if verb in RESULT_VERB_BEFORE:
        before = word_before_match(m).lower().replace('’', "'")
        if before not in RESULT_VERB_BEFORE[verb] and not (verb == 'result' and before.endswith("n't")):
            return m.group(0)
    return LEAD_FORMS[verb]

# "helps to reduce costs" -> "helps reduce costs": "help" takes the plain verb as well; not the noun ("of help to the
# team", "gives help to people who ...", "the help to find work"). A word after "to" that is also a noun is a verb
# only with its object after it ("help to fund a new roof"). A plain "help" is the verb only after what can go before
# a verb: a modal, "to", "do", "not", a pronoun, an adverb or a plural subject ("Volunteers help to clean the beach");
# after an adjective or another verb it is the noun ("financial help to pay the rent", "We hired help to clean").
HELP_TO = re.compile(r"\b(?P<help>help|helps|helped|helping)[ \t]+to(?=[ \t]+(?P<verb>[a-z]+)\b(?:[ \t]+(?P<next>[A-Za-z]+))?)")
HELP_IS_NOUN = {'the', 'a', 'an', 'of', 'with', 'for', 'some', 'any', 'no', 'more', 'extra', 'great', 'little', 'much', 'real',
                'your', 'my', 'his', 'her', 'their', 'our', 'its', 'this', 'that', 'without', 'professional', 'expert',
                'give', 'gives', 'gave', 'given', 'giving', 'offer', 'offers', 'offered', 'offering', 'provide', 'provides',
                'provided', 'providing', 'get', 'gets', 'got', 'getting', 'need', 'needs', 'needed', 'seek', 'seeks',
                'sought', 'receive', 'receives', 'received', 'bring', 'brings', 'brought', 'lend', 'lends', 'lent', 'find',
                'finds', 'found', 'want', 'wants', 'wanted', 'ask', 'asks', 'asked', 'send', 'sends', 'sent'}

HELP_VERB_BEFORE = {'to', 'can', 'could', 'will', 'would', 'may', 'might', 'must', 'should', 'shall', 'do', 'does', 'did',
                    'not', 'also', 'i', 'you', 'we', 'they', 'cannot', 'often', 'usually', 'always', 'still', 'all', 'both',
                    'and', 'or', 'who', 'that', 'which', 'really'}
# "It helps to automate your savings" says that automating helps: the "it" stands for what follows "to", which needs it
HELP_DUMMY_IT = re.compile(r"\bit(?:[ \t]+(?:can|could|may|might|will|would|should|does|did|also|often|really|usually|always|"
                           r"sometimes|certainly|definitely|generally))*[ \t]+\Z", re.I)

def help_to_plain(m, tone):
    verb, after = m.group('verb'), PHRASE_FUNCTION_WORD.match((m.group('next') or '').lower())
    word = word_before_match(m).lower().replace('’', "'")
    if verb == 'be' or not base_verb(verb) or word in HELP_IS_NOUN or HELP_DUMMY_IT.search(text_before(m, 40)):
        return m.group(0)
    if m.group('help') == 'help':
        classes = word_classes(word)
        plural_subject = word.endswith('s') and not word.endswith('ss') and 'NOUN' in classes and 'ADJ' not in classes
        if not (word in HELP_VERB_BEFORE or word.endswith(("n't", "'ll", 'ly')) or plural_subject):
            return m.group(0)
    if 'NOUN' in word_classes(verb) and not (after and after.group('determiner')):
        return m.group(0)
    return m.group('help')

# "Wages, however, increased" -> "Wages, though, increased": "however" between commas inside a sentence (or ending it)
# is the contrast "though" makes; "however much it costs" has no comma and isn't one. Not before another "though" in
# the sentence ("The results, however, though promising, ...").
HOWEVER_INSIDE = re.compile(r"(?<=,)(?P<space>[ \t]+)however(?=,[ \t]+\w|\.)")
THOUGH_FOLLOWS = re.compile(r"[^.!?\n]*?\b(?:though|although)\b", re.I)

def however_to_though(m, tone):
    return m.group(0) if THOUGH_FOLLOWS.match(m.string, m.end()) else m.group('space') + 'though'

# "walked at least 8,000 steps a day" -> "walked 8,000 steps or more a day": a count of things "or more" says the same.
# Only where the count's noun ends the phrase: before "of" "or more" would split it ("at least 150 minutes of
# exercise"), and before an adjective or a verb it would land in the wrong place ("at least 18 years old", "at least
# 20 people died"); a range ("at least three to six months") has no single count, and a hedge after it would repeat
# the "more" ("at least 20 volunteers, ideally more", "at least 3 attempts, if not 4").
AT_LEAST = re.compile(r"\b[Aa]t[ \t]+least[ \t]+(?P<count>(?:\d[\d,.]*|two|three|four|five|six|seven|eight|nine|ten|eleven|"
                      r"twelve|fifteen|twenty|thirty|forty|fifty|sixty|a[ \t]+hundred|hundreds|a[ \t]+thousand)"
                      r"[ \t]+(?P<noun>[a-z]+))(?=[ \t]*[,.;:!?)]|[ \t]*$|[ \t]+(?:a|an|per|each|every|in|on|at|for|with|"
                      r"before|after|from|during|to)\b)", re.M)

AT_LEAST_HEDGE = re.compile(r"[ \t]*,[ \t]*(?:ideally|maybe|perhaps|possibly|probably|if[ \t]+not|or|likely|even|"
                            r"preferably)\b", re.I)

def at_least_or_more(m, tone):
    noun = m.group('noun')
    if not noun.endswith('s') or 'NOUN' not in word_classes(noun) or AT_LEAST_HEDGE.match(m.string, m.end()):
        return m.group(0)
    return match_case(m.group(0), m.group('count') + ' or more')

# "Questions about the policy should be directed to ..." -> "... should go to ...": for questions, requests and
# complaints in the subject only ("Attention should be directed to the risks" and "After complaints, the floodlights
# should be directed to the pitch" stay). A line, a number or a desk is called or visited, not sent to: "should go to".
DIRECTED_TO = re.compile(r"\bshould[ \t]+be[ \t]+directed[ \t]+to\b")
DIRECTED_MESSAGES = re.compile(r"\b(?:questions|queries|enquiries|inquiries|complaints|requests|comments|feedback|concerns|"
                      r"applications)\b", re.I)
DIRECTED_OPTIONS = {'*': ['should go to', 'should be sent to'], 'professional': ['should be sent to'],
                    'academic': ['should be sent to']}

DIRECTED_TO_CALL = re.compile(r"[ \t]+(?:(?:the|our|your|their|a|an)[ \t]+)?(?:[\w-]+[ \t]+){0,2}(?:hotline|helpline|"
                              r"phone[ \t]+line|line|number|desk|counter|switchboard|reception)\b", re.I)

def directed_to_sent(m, tone):
    subject_start = max(m.string.rfind(c, 0, m.start()) for c in '.!?\n,') + 1
    if not DIRECTED_MESSAGES.search(m.string, subject_start, m.start()):
        return m.group(0)
    return 'should go to' if DIRECTED_TO_CALL.match(m.string, m.end()) else pick(DIRECTED_OPTIONS, tone)

# "was most likely caused by" -> "was probably caused by", "will most likely rise" -> "will probably rise": after a
# helping verb. "the most likely cause" and "is most likely to win" compare and stay, and so does "most likely" as what
# something is, before the end of the clause or a preposition ("Which outcome is most likely?", "Side effects are most
# likely during the first week").
MOST_LIKELY = re.compile(r"\bmost[ \t]+likely\b(?![ \t]+(?:to|that)\b)")
LIKELY_AFTER = {'is', 'are', 'was', 'were', 'be', 'been', 'will', 'would', 'could', 'can', 'should', 'may', 'might', 'has',
                'have', 'had'}
LIKELY_AS_ADJECTIVE = re.compile(r"[ \t]*(?:[.?!,;:)]|$)|[ \t]+(?:in|on|at|during|among|when|if|after|before|for|with|"
                                 r"between|under|over|within|where|while|once|from|until|since|as|than)\b", re.M)

def most_likely_probably(m, tone):
    before = word_before_match(m).lower()
    if before not in LIKELY_AFTER:
        return m.group(0)
    if before in ('is', 'are', 'was', 'were', 'be', 'been') and LIKELY_AS_ADJECTIVE.match(m.string, m.end()):
        return m.group(0)
    return 'probably'

# "When it comes to choosing a laptop, ..." -> "When choosing a laptop, ...": before a verb in -ing the frame says no
# more than "when". Before a noun it stays ("When it comes to nutrition", "When it comes to building design"), and so
# it does before a participle that describes a noun ("When it comes to existing customers", "rising prices").
WHEN_IT_COMES = re.compile(r"\b(?P<when>[Ww]hen)[ \t]+it[ \t]+comes[ \t]+to(?=[ \t]+(?P<doing>[a-z]+ing)\b"
                           r"(?:[ \t]+(?P<next>[A-Za-z]+))?)")

# Participles that often describe a noun are verbs only before a determiner or a preposition ("When it comes to running
# a business", "When it comes to working from home")
WHEN_PARTICIPLE_ADJECTIVES = {'rising', 'falling', 'growing', 'leading', 'emerging', 'developing', 'changing', 'declining',
                         'increasing', 'decreasing', 'existing', 'remaining', 'upcoming', 'ongoing', 'working', 'ageing',
                         'aging', 'competing', 'evolving', 'struggling', 'failing', 'returning', 'paying', 'visiting',
                         'moving', 'operating', 'running', 'starting', 'booming', 'soaring', 'shrinking', 'missing',
                         'outstanding', 'prevailing', 'surrounding', 'underlying', 'corresponding', 'following', 'living',
                         'aspiring', 'exciting', 'amazing', 'interesting'}
WHEN_VERB_READING_NEXT = {'in', 'on', 'at', 'from', 'to', 'into', 'with', 'for', 'about', 'out', 'up', 'over', 'through', 'around',
                     'without', 'by', 'off', 'down', 'back', 'away'}

def when_it_comes_doing(m, tone):
    doing, following = m.group('doing'), (m.group('next') or '').lower()
    after = PHRASE_FUNCTION_WORD.match(following)
    determiner = bool(after and after.group('determiner'))
    classes = word_classes(doing)
    if 'VERB' not in classes or ('NOUN' in classes and not determiner):
        return m.group(0)
    if ('ADJ' in classes or doing in WHEN_PARTICIPLE_ADJECTIVES) and not (determiner or following in WHEN_VERB_READING_NEXT):
        return m.group(0)
    return m.group('when')

# "The island is home to rare birds" -> "The island has rare birds", "was once home to a Roman fort" -> "once had a Roman
# fort": what a place is home to, it has. Only before a quantity, a number, a determiner or a describing word ("is home to
# several museums", "an estimated 300 species"), never after a person ("She is home to rest"), and never before people
# or a name, who live or play there rather than belong to it ("is home to nine million people", "a family of four", "a
# community of artists", "is home to the Gunners").
HOME_TO = re.compile(r"\b(?P<verb>is|are|was|were)(?P<adverb>[ \t]+(?:also|now|still|once))?[ \t]+home[ \t]+to"
                     r"(?=[ \t]+(?P<next>[\w$£€]+))", re.I)
HOME_TO_HAS = {'is': 'has', 'are': 'have', 'was': 'had', 'were': 'had'}
HOME_TO_LEADS = {'a', 'an', 'the', 'some', 'several', 'many', 'more', 'over', 'around', 'about', 'nearly', 'almost',
                 'roughly', 'hundreds', 'thousands', 'millions', 'dozens', 'its', 'their', 'one', 'two', 'three', 'four',
                 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'most', 'countless', 'numerous', 'estimated', 'than', 'of',
                 'up', 'to', 'at', 'least'}
HOME_TO_PEOPLE = {'family', 'families', 'couple', 'household', 'households', 'population', 'populations', 'community',
                  'communities', 'tribe', 'tribes', 'colony', 'generation', 'generations', 'crowd', 'congregation'}
HOME_TO_WORDS = re.compile(r"[ \t]+([\w$£€,.'’-]+)")

def home_to_has(m, tone):
    word = m.group('next')
    classes = word_classes(word)
    if not (word[0].isdigit() or word[0] in '$£€' or word.lower() in HOME_TO_LEADS
            or ('ADJ' in classes and not classes & {'VERB', 'NOUN'})):
        return m.group(0)
    if re.search(r"\b(?:I|you|he|she|we|they)[ \t]+\Z", text_before(m, 8), re.I):
        return m.group(0)
    position = m.end()
    for _ in range(8):
        found = HOME_TO_WORDS.match(m.string, position)
        if not found:
            break
        position = found.end()
        word = found.group(1).rstrip(',.')
        low = word.lower()
        if word[:1].isupper() or low in HOME_TO_PEOPLE or (low.isalpha() and word_names_people(low)):
            return m.group(0)
        if found.group(1)[-1] in ',.' or (PHRASE_FUNCTION_WORD.match(low) and low not in HOME_TO_LEADS):
            break
    has = HOME_TO_HAS[m.group('verb').lower()]
    has = match_case(m.group('verb'), has)
    return (m.group('adverb').strip() + ' ' + has.lower() if m.group('adverb') else has)

# "your message regarding the refund" -> "your message about the refund", "our policy regarding refunds" -> "our policy
# on refunds" (Standard and Casual; Professional and Academic keep it, as they keep "regarding" for "with regard to").
# Only after a word for a message, a question, a rule or a view, or after "us", "you" and the like ("contact us regarding
# your order"): elsewhere "regarding" may be the verb ("She stood regarding him coldly", "They began regarding him as a
# rival"), and at the start of a sentence "About the schedule, ..." would read as an approximation. Nor before "as" a few
# words on ("the team regarding him as a threat").
REGARDING = re.compile(r"\b(?P<before>[a-z]+)[ \t]+regarding(?=[ \t]+[\w$£€])")
REGARDING_ON_NOUNS = {'policy', 'policies', 'guidance', 'guidelines', 'rule', 'rules', 'law', 'laws', 'legislation',
                      'regulations', 'advice', 'position', 'stance', 'view', 'views', 'decision', 'decisions', 'report',
                      'reports', 'research', 'debate', 'talks', 'agreement', 'recommendations', 'consultation', 'statement',
                      'statements', 'data', 'figures', 'statistics', 'study', 'studies', 'update', 'updates'}
REGARDING_ABOUT_WORDS = {'question', 'questions', 'query', 'queries', 'enquiry', 'enquiries', 'inquiry', 'inquiries',
                         'concern', 'concerns', 'complaint', 'complaints', 'information', 'details', 'message', 'messages',
                         'email', 'emails', 'letter', 'letters', 'call', 'calls', 'request', 'requests', 'feedback',
                         'comment', 'comments', 'news', 'discussion', 'discussions', 'conversation',
                         'conversations', 'doubts', 'worries', 'fears', 'confusion', 'uncertainty', 'dispute', 'disputes',
                         'anything', 'something', 'nothing', 'everything', 'us', 'you', 'me', 'them', 'him', 'her'}
REGARDING_AS_FOLLOWS = re.compile(r"(?:[ \t]+[\w'’-]+){1,4}[ \t]+as\b")

def regarding_about(m, tone):
    before = m.group('before')
    if before not in REGARDING_ON_NOUNS and before not in REGARDING_ABOUT_WORDS:
        return m.group(0)
    if REGARDING_AS_FOLLOWS.match(m.string, m.end()):
        return m.group(0)
    return before + m.string[m.end('before'):m.end() - len('regarding')] + ('on' if before in REGARDING_ON_NOUNS else 'about')

RATED_PHRASE_RULES = [
    (re.compile(r"\bahead[ \t]+of\b", re.I), ahead_of_event, PHRASE_RATE_ALWAYS),
    (TEND_TO, tend_to_habit, {'standard': (0.8, 1.0), 'casual': (1.0, 1.0), 'professional': (0.6, 1.0), 'academic': (0.5, 0.8)}),
    (ALLOW_TO, allow_to_let, {'standard': (0.7, 1.0), 'casual': (1.0, 1.0), 'professional': (0.6, 0.9)}),
    (EXPECTED_TO, expected_to_due, PHRASE_RATE_MOSTLY),
    (MARKED_MILESTONE, marked_milestone, PHRASE_RATE_ALWAYS),
    (LAID_FOUNDATION, laid_foundation, PHRASE_RATE_ALWAYS),
    (STUDY_WHETHER, study_whether, PHRASE_RATE_MOSTLY),
    (CAUTION_THAT, caution_that, PHRASE_RATE_ALWAYS),
    (FAR_BEYOND, far_beyond, PHRASE_RATE_ALWAYS),
    (PLACED_IN, placed_in, PHRASE_RATE_MOSTLY),
    (COMPARED_AMOUNT, more_than_to_over, {'standard': (0.6, 1.0), 'casual': (0.8, 1.0), 'professional': (0.4, 0.8)}),
    (SUCH_AS, such_as_to_like, {'standard': (0.5, 1.0), 'casual': (1.0, 1.0)}),
    (PER_TIME, per_time_unit, {'standard': (0.6, 1.0), 'casual': (1.0, 1.0)}),
    (JUST_A_COUNT, just_to_only, PHRASE_RATE_MOSTLY),
    (WITH_JUST, just_to_only, PHRASE_RATE_MOSTLY),
    (SAYING_THAT, drop_reported_that, {'standard': (0.6, 1.0), 'casual': (1.0, 1.0)}),
    (CLAIMING_THAT, drop_reported_that, {'standard': (0.4, 0.8), 'casual': (0.8, 1.0)}),
    (FEELING_THAT, drop_that_after_feeling, {'standard': (0.5, 1.0), 'casual': (1.0, 1.0)}),
    (SO_THAT, so_that_to_so, {'standard': (0.5, 1.0), 'casual': (1.0, 1.0)}),
    (AROUND_THE_WORLD, around_the_world, {tone: (0.7, 1.0) for tone in TONES}),
    (EMPTY_INTENSIFIER, drop_intensifier, {tone: (0.0, 1.0) for tone in TONES}),
    (RESULT_IN, result_in_to_lead, PHRASE_RATE_MOSTLY),
    (HELP_TO, help_to_plain, {'standard': (0.7, 1.0), 'casual': (1.0, 1.0), 'professional': (0.5, 0.9)}),
    (HOWEVER_INSIDE, however_to_though, {'standard': (0.5, 0.9), 'casual': (0.8, 1.0), 'professional': (0.3, 0.7)}),
    (AT_LEAST, at_least_or_more, {'standard': (0.5, 1.0), 'casual': (0.7, 1.0), 'professional': (0.4, 0.8), 'academic': (0.3, 0.6)}),
    (DIRECTED_TO, directed_to_sent, PHRASE_RATE_ALWAYS),
    (MOST_LIKELY, most_likely_probably, PHRASE_RATE_MOSTLY),
    (WHEN_IT_COMES, when_it_comes_doing, PHRASE_RATE_MOSTLY),
    (HOME_TO, home_to_has, PHRASE_RATE_MOSTLY),
    (REGARDING, regarding_about, {'standard': (0.8, 1.0), 'casual': (1.0, 1.0)}),
]

# "is not only fast but also cheap" -> "is fast and cheap". Only the additive form: "but also" (after "not
# only/just/merely/simply", the comma before it optional), or a bare "but" right after "not only/just X"
# with no comma. "He did not simply walk away, but stayed" is corrective, and a "but" after a comma may start
# a contrasting clause ("..., but the board approved it anyway"), so those keep their "not".
# Lowercase "not" only: "Not only does it ..." uses inverted word order, which the structure layer handles.
NOT_ONLY = re.compile(r"\bnot (?:(?:only|just|merely|simply)\s+(?P<a>[^.;:!?\"“”()]{1,80}?),?\s+but\s+also"
                      r"|(?:only|just)\s+(?P<b>[^,.;:!?\"“”()]{1,80}?)\s+but)\s+")
# "not just one error but three" upgrades a count, and "do not just copy it but ..." would leave an emphatic
# "do": both keep their wording
COUNT_WORD = re.compile(r"(?:\d|(?:one|two|three|four|five|six|seven|eight|nine|ten|once|twice|thrice|all|both|many|"
                        r"several|dozens|hundreds|thousands|millions|a\s+few|a\s+couple)\b)", re.I)
DO_SUPPORT = re.compile(r"\b(?:do|does|did)\s+\Z", re.I)

def not_only(m):
    first = m.group('a') or m.group('b')
    if COUNT_WORD.match(first) or COUNT_WORD.match(m.string, m.end()) or DO_SUPPORT.search(text_before(m, 10)):
        return m.group(0)
    return first + ' ' + random.choice(['and', 'and also']) + ' '

# "Overall, 85% of ..." means "in total", not "to sum up"
NUMBER_FOLLOWS = re.compile(r'[$€£]?\d')
SUMMING_UP = 1

def pick_fresh(options, tone, used, avoid=()):
    """Like pick(), but avoids repeating a replacement already used in this text ("Also, ... Also, ..."), and
    the ones in `avoid` while there are others."""
    choices = options.get(tone) or options['*']
    choices = [c for c in choices if c not in avoid] or choices
    fresh = [c for c in choices if c not in used] or choices
    choice = random.choice(fresh)
    used.add(choice)
    return choice

def sentence_around(text, start, end):
    """The rest of the sentence a match sits in, without the match itself."""
    left = max(text.rfind('.', 0, start), text.rfind('!', 0, start), text.rfind('?', 0, start), text.rfind('\n', 0, start)) + 1
    ends = [p for p in (text.find('.', end), text.find('!', end), text.find('?', end), text.find('\n', end)) if p != -1]
    return text[left:start] + ' ' + text[end:min(ends) if ends else len(text)]

def past_tense(text):
    """True if a simple-past or past-perfect verb is in the text ("the empire collapsed", "had lost")."""
    if NLP is None:
        return False
    with _NLP_LOCK:
        doc = NLP(text)
    return any(t.tag_ == 'VBD' or (t.tag_ == 'VBN' and any(c.lower_ == 'had' for c in t.children)) for t in doc)

def time_guarded(match, new):
    """Keep the original if the rewrite would say "today" in a sentence that already says it ("today today"),
    in front of an "of" phrase ("in today's world of streaming"), or about the past ("In the digital age,
    newspapers lost most of their readers")."""
    if TIME_WORDS.search(new):
        rest = sentence_around(match.string, match.start(), match.end())
        if (TIME_WORDS.search(rest) or re.match(r"\s+of\b", match.string[match.end():])
                or past_tense(rest)):
            return match.group(0)
    return new

def shouting(text):
    """True for text in capitals ("MOREOVER"), whose replacement is written in capitals too."""
    letters = [c for c in text if c.isalpha()]
    return len(letters) > 1 and all(c.isupper() for c in letters)

# An opener echoes a word used nearby when the two share a stem ("Let's be honest: ... Truth be told," never becomes
# "Honestly,")
OPENER_WORD_ENDINGS = r"(?:ly|s|es|y|ity|ness|ed|d|ing)?\b"

def opener_echoes_nearby(word, nearby):
    return bool(re.search(r'\b' + re.sub(r'(?:ly|s)\Z', '', word) + OPENER_WORD_ENDINGS, nearby))

def match_in_a_title(m):
    """True if a match with a capital inside a sentence is part of a title or a name ("the film Far Beyond the Stars",
    "a guide called Ahead of the Curve")."""
    return m.group(0)[:1].isupper() and not opens_sentence(m)

def apply_phrase_rules(text, tone, deep_mode=False):
    # A text without quotations or code is searched for them once: no rule writes a quotation mark or a backtick, so it
    # stays without them, and that search is the slowest step on a long text that many rules change. In a text with
    # them each rule checks its matches.
    guarded_here = guarded if quoted_ranges(text) else (lambda fn: fn)
    text = NOT_ONLY.sub(guarded_here(not_only), text)

    used = set()
    for index, (pattern, chance, options, condition, keys) in enumerate(TRANSITION_RULES):
        lowered = text.lower()
        if not any(key in lowered for key in keys):
            continue

        def transition(m, options=options, chance=chance, index=index, condition=condition):
            if random.random() >= opener_chance(chance, deep_mode) or (condition and not condition(m)):
                return m.group(0)
            opener = m.group(0)[len(m.group('pre')):]
            follows = WORD_FOLLOWS.match(m.string, m.end())     # a word after it on the same line
            if (index == SUMMING_UP and NUMBER_FOLLOWS.match(m.string, m.end())
                    and re.match(r'(?:overall|all\s+in\s+all)\b', opener, re.I)):
                if tone == 'academic':
                    return m.group(0)
                new = 'All told,'
            elif follows and deep_mode and index in DROPPABLE_TRANSITIONS and random.random() < DROP_CHANCE:
                return m.group('pre') + '\x00'      # left out; the next word is capitalised below
            else:
                # "Moreover, the menu also lists ..." isn't "Also, the menu also lists ..."; nor is an opener replaced
                # by itself or by words used nearby ("For example," -> "For instance,", but "All things considered, the
                # key is balance" never becomes "On balance, the key is balance")
                choices = options.get(tone) or options['*']
                own = re.sub(r'\s+', ' ', opener.rstrip(', \t')).lower()
                nearby = (m.string[max(0, m.start() - 200):m.start()] + ' ' + m.string[m.end():m.end() + 200]).lower()
                echo = tuple(ALSO_OPENERS if SAYS_ALSO.search(sentence_around(m.string, m.start(), m.end())) else ()) + tuple(
                    c for c in choices if c.rstrip(',').lower() == own or any(
                        opener_echoes_nearby(word, nearby) for word in re.findall(r'[a-z]{4,}', c.lower())
                        if word not in OPENER_LINK_WORDS))
                if follows and index in DROPPABLE_TRANSITIONS and all(c in echo or c in used for c in choices):
                    return m.group('pre') + '\x00'      # every fitting opener is used: left out rather than repeated
                if all(c in echo or (index >= FIRST_PLAIN_OPENER and c in used) for c in choices):
                    return m.group(0)
                new = pick_fresh(options, tone, used, echo)
            return m.group('pre') + (new.upper() if shouting(opener) else new) + (' ' if follows else '')
        text = pattern.sub(guarded_here(transition), text)
    text = re.sub('\x00(\\W*)(\\w+)', lambda m: m.group(1) + sentence_capital(m.string, m.start(2), m.group(2)),
                  text).replace('\x00', '')

    signpost = FILLER_SIGNPOSTS.get(tone)

    def drop_filler(m):
        if not signpost:
            return m.group('pre') + sentence_capital(m.string, m.start('first'), m.group('first'))
        return m.group('pre') + (signpost.upper() if shouting(m.group(0)) else signpost) + m.group('first')
    text = FILLER_AT_START.sub(guarded_here(drop_filler), text)
    text = FILLER_INSIDE.sub(guarded_here(
        lambda m: m.group(0) if FILLER_DOUBTED.search(m.string, max(0, m.start() - 40), m.start()) else ''), text)

    today = {'*': ['today', 'these days'], 'casual': ['these days', 'nowadays'], 'professional': ['today'], 'academic': ['today']}
    text = TODAY_FILLER.sub(guarded_here(lambda m: time_guarded(m, match_case(m.group(0), pick(today, tone)))), text)
    for pattern, options, keys, other_sense_before in PHRASE_RULES:
        lowered = text.lower()
        if any(key in lowered for key in keys):
            text = pattern.sub(guarded_here(lambda m: time_guarded(m, rewrite_phrase(m, options, tone, other_sense_before))),
                               text)
    for pattern, rewrite in CONTEXT_RULES:
        text = pattern.sub(guarded_here(lambda m, rewrite=rewrite: m.group(0) if match_in_a_title(m) else rewrite(m, tone)), text)
    for pattern, rewrite, rates in RATED_PHRASE_RULES:
        def rated_rewrite(m, rewrite=rewrite, rates=rates):
            new = m.group(0) if match_in_a_title(m) else rewrite(m, tone)
            return new if new == m.group(0) or random.random() < phrase_rate(rates, tone, deep_mode) else m.group(0)
        text = pattern.sub(guarded_here(rated_rewrite), text)
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

# People contract in everyday writing; AI text rarely does. Standard mixes both, and in Chaos Mode leans towards them.
CONTRACT_CHANCE = {'casual': 1.0, 'standard': 0.5}
CHAOS_CONTRACT_CHANCE = {'casual': 1.0, 'standard': 0.8}

# After a preposition the pronoun belongs to the phrase before it, not to "is": "every second of it is
# memorable" can't become "of it's memorable". After "how" and a word the predicate has moved in front of the
# pronoun, so nothing follows the verb for it to lean on: "how likely you are to repay" can't become "how likely
# you're to repay".
PREPOSITION_BEFORE = re.compile(r"\b(?:of|about|for|with|in|on|to|from|by|at|without|than|like|into|over|under|"
                                r"through|after|before|behind|beyond|within|despite|around|toward|towards)\s+$"
                                r"|\bhow\s+[a-z]+\s+$", re.I)
# After a gerund or an infinitive, "it" or "that" is its object and the verb belongs to the whole phrase: "descaling it
# is quick" can't become "descaling it's quick", nor "To fix it will take a week" "To fix it'll take ...". Where the
# verb could take a clause instead ("Knowing it is wrong, he went ahead"), the writer's words stay too.
VERB_BEFORE_OBJECT = re.compile(r"\b(?:(?P<doing>[a-z]+ing)|to\s+(?P<do>[a-z]+))\s+\Z", re.I)

def pronoun_is_object(match):
    """True if the "it" or "that" a contraction would start with follows a gerund or "to" and a verb."""
    if match.group(0).split()[0].lower() not in ('it', 'that'):
        return False
    verb = VERB_BEFORE_OBJECT.search(match.string, max(0, match.start() - 40), match.start())
    if not verb:
        return False
    if verb.group('do'):
        return base_verb(verb.group('do').lower())
    classes = word_classes(verb.group('doing').lower())
    return 'VERB' in classes or not classes

# "it will not" is "it won't" or stays: never "it'll not"
WILL_NOT_FOLLOWS = re.compile(r"\s+not\b", re.I)

# Verse keeps its syllables: a contraction would drop one from the metre ("I know now it is mostly still," never becomes
# "it's mostly still,"). A paragraph is verse when it has three or more short lines, none of them a list item or a
# heading, at least half of them running on without a full stop, and a line after the first carries on in lower case.
PARAGRAPH_SPAN = re.compile(r"\S(?:[^\n]|\n(?![ \t]*(?:\n|\Z)))*")
VERSE_LINE_RUNS_ON = re.compile(r"(?:[,;]|[A-Za-z])[\"”’]?\Z")
LIST_OR_HEADING_LINE = re.compile(r"(?:[-*#>•]|\d+[.)])\s")

def verse_ranges(text):
    """Where the text is verse, as (start, end) pairs."""
    ranges = []
    for m in PARAGRAPH_SPAN.finditer(text):
        lines = [line.strip() for line in m.group(0).split('\n') if line.strip()]
        if len(lines) < 3 or any(len(line.split()) > 10 or LIST_OR_HEADING_LINE.match(line) for line in lines):
            continue
        running = sum(1 for line in lines if VERSE_LINE_RUNS_ON.search(line))
        if 2 * running >= len(lines) and any(line[:1].islower() for line in lines[1:]):
            ranges.append(m.span())
    return ranges

# In a question the subject comes after the helping verb: the formal form of "isn't it?" and "why don't we"
# is "is it not?" and "why do we not", never "is not it?"
INVERTED = re.compile(r"\b(?P<aux>do|does|did|is|are|was|were|ca|could|wo|would|should|has|have|had|must)"
                      r"n['’]t\s+(?P<subject>I|you|we|they|he|she|it|there)\b", re.I)
FULL_AUX = {'ca': 'can', 'wo': 'will'}
# With any other subject the question keeps its contraction ("Isn't that the point?", "Why didn't the council
# reply?"): "Is not that the point?" isn't English, and where "not" would go after the subject needs the parse.
# What may come before such a helping verb in its sentence or clause: a conjunction and a question word, and at the
# start of a line the marks of a list item, quote or heading ("- Isn't that ...?", "> Didn't the report ...?"),
# or of bold type ("**Isn't that ...?**", "**Q:** Isn't that ...?").
QUESTION_LEAD = re.compile(r"(?:(?:^|\n)[ \t]*(?:(?:[-*+•▪◦>#]+|\d+[.)]|[A-Za-z][.)])[ \t]+)*|[.!?][\"”’)\]*_]*\s+|"
                           r"[\"“‘(]|[,;:][*_]*\s+)[*_]*(?:(?:and|but|so|or|yet)\s+)?"
                           r"(?:(?P<wh>why|how|what|where|when|who|which)\s+)?\Z", re.I)
# A subject right after the helping verb: the question is inverted ("Isn't that ...?", "Don't people ...?")
QUESTION_SUBJECT = re.compile(r"\s+(?:(?-i:[A-Z])|(?:the|this|that|these|those|a|an|my|your|his|her|its|our|their|any|some|"
                              r"every|each|all|both|no|one|enough|many|much|most|such|people|anyone|anybody|anything|"
                              r"everyone|everybody|everything|someone|somebody|something|nobody|nothing)\b)", re.I)
QUESTION_NEXT_WORDS = re.compile(r"\s+([A-Za-z]+)(?:\s+([A-Za-z]+))?(?:\s+([A-Za-z]+))?")
# An order is a question only with a tag at its end ("Don't forget your ID, okay?")
QUESTION_TAG = re.compile(r",[ \t]*(?:okay|ok|right|alright|all right|yeah|yes|will you|won['’]t you|would you|could you|"
                          r"can you|eh)[ \t]*\?\Z", re.I)

def ends_in_question(text, end):
    """True if the sentence going on at `end` ends with a question mark."""
    stop = re.compile(r'[.!?\n]').search(text, end)
    return bool(stop) and stop.group(0) == '?'

def is_participle(word, tags):
    """True if `word` is a participle ('VBN', 'VBG') of a verb it may be the form of."""
    return any(word in getInflection(lemma, tag=tag) for lemma in getAllLemmas(word).get('VERB', ()) for tag in tags)

def uninverted_question(m, wh):
    """True if the helping verb matched by `m` has no subject after it, because the question word before it is
    the subject ("Who doesn't love a bargain?", "Who isn't a fan of ...?") or it gives an order ("Don't forget your
    ID, okay?"). The verb or adjective follows instead (after an adverb: "Who doesn't really ..."), and the full form
    reads ("Who does not love ...?", "Do not forget ..., okay?"). A question word is the subject only of a singular
    verb: "What don't police understand?" has its subject after the verb, and so does "What can't love conquer?",
    where a word that can only be a verb follows the noun, and "What can't hard work fix?", where a noun and a verb
    end the question. After "hasn't" the subject's own verb is a past participle ("What hasn't changed?", not "What
    hasn't time healed?"), after "isn't" an adjective or a participle ("Who isn't ready?")."""
    words = QUESTION_NEXT_WORDS.match(m.string, m.end())
    aux = m.group(0).lower()
    if (words and wh == 'who' and aux.startswith(('is', 'was'))
            and re.match(r'\s+an?\s', m.string[m.end():m.end() + 5], re.I)):
        return True                     # "Who isn't a fan of ...?": "who" is the subject, "a fan" what they are
    if not words or QUESTION_SUBJECT.match(m.string, m.end()) or 'getAllLemmas' not in globals():
        return False
    if not wh:
        stop = re.compile(r'[.!?\n]').search(m.string, m.end())
        return (aux.startswith('don') and bool(stop)
                and bool(QUESTION_TAG.search(m.string, max(m.end(), stop.start() - 16), stop.end())))
    if wh not in ('who', 'what', 'which') or aux.startswith(('don', 'are', 'were', 'have')):
        return False
    word, after, then = (w.lower() for w in words.groups(''))
    after_end = words.end(2)
    if after and 'ADV' in word_classes(word) and not word_classes(word) & {'VERB', 'AUX'}:
        word, after, after_end = after, then, words.end(3)            # "Who doesn't really care?"
    if aux.startswith(('has', 'had')):
        return is_participle(word, ('VBN',))
    if aux.startswith(('is', 'was')):
        return 'ADJ' in word_classes(word) or is_participle(word, ('VBN', 'VBG'))
    noun_then_verb = 'NOUN' in word_classes(word) and base_verb(after)
    if wh != 'who' and noun_then_verb and after_end > 0 and re.match(r'[ \t]*\?', m.string[after_end:after_end + 4]):
        return False                    # "What can't hard work fix?", "What doesn't time change?"
    # "What doesn't kill you", "Who doesn't want help?"; not "What can't love conquer?"
    return base_verb(word) and not (noun_then_verb and word_classes(after) & {'VERB', 'NOUN'} == {'VERB'})

def opens_question(m):
    """True if the helping verb matched by `m` opens a question ("Isn't that ...?", "Why didn't they ...?")."""
    lead = QUESTION_LEAD.search(m.string, max(0, m.start() - 30), m.start())
    return (bool(lead) and ends_in_question(m.string, m.end())
            and not uninverted_question(m, (lead.group('wh') or '').lower()))

def apply_tone_contractions(text, tone, deep_mode=False):
    curly = text.count('’') > text.count("'")       # write apostrophes the way the writer does
    verse = verse_ranges(text)

    def in_verse(match):
        return any(s <= match.start() < e for s, e in verse)

    def replace_from(table, chance=1.0, check_before=False):
        def replace(match):
            if in_quote(match) or random.random() >= chance or in_verse(match):
                return match.group(0)
            if check_before and (PREPOSITION_BEFORE.search(match.string[max(0, match.start() - 30):match.start()])
                                 or pronoun_is_object(match)
                                 or (match.group(0).lower().endswith('will') and WILL_NOT_FOLLOWS.match(match.string, match.end()))):
                return match.group(0)
            key = re.sub(r'\s+', ' ', match.group(0).lower()).replace('’', "'")
            if key.endswith("n't") and opens_question(match):
                return match.group(0)       # an expansion that would read "Is not that the point?"
            new = match_case(match.group(0), table[key])
            return new.replace("'", '’') if curly else new
        return replace

    def uninvert(m):
        if in_quote(m) or in_verse(m) or not ends_in_question(m.string, m.end()):
            return m.group(0)
        # "Who wasn't there?": "who" is the subject and "there" the place ("What isn't there to like?" is inverted)
        if (m.group('subject').lower() == 'there' and not re.match(r'\s+to\b', m.string[m.end():])
                and re.search(r'\b(?:who|what|which)\s+\Z', text_before(m, 12), re.I)):
            return m.group(0)
        aux = FULL_AUX.get(m.group('aux').lower(), m.group('aux').lower())
        if aux == 'are' and m.group('subject').lower() == 'i':
            aux = 'am'                      # "aren't I?" -> "am I not?"
        return match_case(m.group('aux'), aux) + ' ' + m.group('subject') + ' not'

    if tone in CONTRACT_CHANCE:
        chance = (CHAOS_CONTRACT_CHANCE if deep_mode else CONTRACT_CHANCE)[tone]
        text = CONTRACT_PATTERN.sub(replace_from(CONTRACTIONS, chance), text)
        return PRONOUN_PATTERN.sub(replace_from(PRONOUN_CONTRACTIONS, chance, check_before=True), text)
    return EXPAND_PATTERN.sub(replace_from(EXPANSIONS), INVERTED.sub(uninvert, text))

# ==========================================
#   STEP 3: WORD-LEVEL REWRITING
# ==========================================

UPOS_TO_WORDNET = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}
UNINFLECTED_TAGS = {'VB', 'VBP', 'NN', 'JJ', 'RB'}
SKIP_TAGS = {'JJR', 'JJS', 'RBR', 'RBS'}  # comparatives re-inflect badly ("beautifuler")

# --- NOUN SENSES (built offline with NLTK from WordNet by scripts/build_synonyms.py) ---
# kind -> the nouns whose main WordNet sense is of that kind: "person" (witness, host), "people" (audience,
# population), "group" (leadership, regiment), "clinician" (doctor, vet), "body" (chest, knee), "animal" (horse),
# "food" (cream, milk), "building" (factory, castle) and "stone" (diamond, crystal, prism). Some curated swaps only
# hold for one kind of noun
with gzip.open(os.path.join(DATA_DIR, 'noun_senses.json.gz'), 'rt', encoding='utf-8') as _f:
    NOUN_SENSES = {kind: frozenset(nouns) for kind, nouns in json.load(_f).items()}

def noun_sense_is(token, *kinds):
    """True if the token is a noun of one of the kinds, read from its lemma ("the hosts" -> person)."""
    lemma = token.lemma_.lower()
    return token.pos_ in ('NOUN', 'PROPN') and any(lemma in NOUN_SENSES[kind] for kind in kinds)

# The nouns spaCy's lemmatizer knows, from its index of WordNet's nouns (shipped with the spaCy model): a recipe's
# object that isn't one is an ingredient English dictionaries don't list ("Reduce the passata until thick")
LEMMATIZER_NOUNS = (frozenset(NLP.get_pipe('lemmatizer').lookups.get_table('lemma_index')['noun'])
                    if NLP is not None else frozenset())

# WordNet synonyms are guesses: about a quarter of them take another sense of the word ("sunny" -> "gay",
# "employs 4,500 people" -> "applies", "aged 18 to 24" -> "elderly"), and no word-pair statistic tells
# those apart. So a WordNet swap is only made when the Deep Rewrite model is loaded to check it in its
# sentence (see vet_swaps); without the model, only the hand-checked swaps below (AI_WORDS, PLAIN_WORDS)
# change words.
# How much rarer than the original a synonym may be (share of its corpus frequency); Chaos Mode reaches
# further, since every guess is checked.
SYNONYM_FLOOR = {'standard': 0.3, 'casual': 0.5, 'professional': 0.3, 'academic': 0.2}

# Share of eligible content words that get a (checked) WordNet synonym
WORD_RATE = {'standard': 0.5, 'casual': 0.55, 'professional': 0.35, 'academic': 0.35}

# Nouns carry most of a sentence's meaning and WordNet's noun synonyms often miss the context
# ("updated the details" is not "updated the points"), so only Chaos Mode swaps them, and only when the new
# word is attested next to a neighbour in real English. Chaos also tries more guesses, and rarer ones.
NOUN_WEIGHT_VETTED = {False: 0.0, True: 0.7}
CHAOS_WORD_RATE_VETTED, CHAOS_FLOOR_VETTED = 1.0, 0.05

# Curated adverb swaps that only read well in some positions ("manage time effectively" -> "well",
# but not "effectively reduces" -> "well reduces", nor "is effectively a ban" -> "is well a ban"; "truly matters"
# but not "truly unique" -> "really unique")
AFTER_VERB_ONLY = {'effectively|ADV', 'tirelessly|ADV'}
VERB_HEAD_ONLY = {'truly|ADV'}
# ...and verb swaps that only work with a direct object ("reduce costs" -> "cut costs", but
# "provide that ..." or "establish whether ..." can't take "give" or "build", and "the team was eliminated in
# the first round" wasn't "removed")
OBJECT_ONLY = {'reduce|VERB', 'pose|VERB', 'provide|VERB', 'organize|VERB', 'implement|VERB', 'establish|VERB',
               'maintain|VERB', 'identify|VERB', 'examine|VERB', 'analyze|VERB', 'analyse|VERB', 'investigate|VERB',
               'acquire|VERB', 'emphasize|VERB', 'highlight|VERB', 'mitigate|VERB', 'hoist|VERB', 'eliminate|VERB',
               'boast|VERB', 'exacerbate|VERB', 'alleviate|VERB', 'expedite|VERB', 'procure|VERB', 'delineate|VERB'}
# Adjectives swapped only in front of a noun ("a significant change" -> "a major change", but
# "statistically significant" and "is significant for" keep their technical sense)
BEFORE_NOUN_ONLY = {'significant|ADJ', 'substantial|ADJ', 'various|ADJ', 'fundamental|ADJ', 'appropriate|ADJ',
                    'consistent|ADJ', 'traditional|ADJ', 'salient|ADJ', 'subsequent|ADJ'}
# ...and adjectives swapped only after "is"/"are" ("exercise is essential for health" -> "necessary",
# but "essential oils" and "vital signs" are fixed terms)
PREDICATE_ONLY = {'essential|ADJ', 'vital|ADJ', 'imperative|ADJ'}
# ...and "readily", which means "easily" before an adjective ("readily available") but "willingly" before a
# verb ("she readily agreed")
BEFORE_ADJECTIVE_ONLY = {'readily|ADV'}
# ...and "ensure", whose replacement "make sure" takes a clause ("ensure (that) the door is locked"), never a
# noun object or a passive ("ensure comfort" can't become "make sure comfort", nor "is ensured" "is made sure")
CLAUSE_ONLY = {'ensure|VERB'}
# ...and "frequently", which before a describing word and its noun is part of a fixed term ("frequently
# accessed data"; "stores often accessed data" reads wrongly)
NOT_BEFORE_DESCRIBER = {'frequently|ADV'}
# ..."nearly", which after a negation or degree word is part of an idiom ("not nearly enough" means far from
# enough; "not almost enough" and "very almost" aren't English), also with a verb between them ("didn't get
# nearly enough sleep", "have not risen nearly as fast")
NOT_AFTER_NEGATION = {'nearly|ADV'}
NEARLY_NEGATIONS = {'not', "n't", 'n’t', 'never', 'no', 'nowhere', 'nobody', 'none', 'nothing', 'neither', 'nor',
                    'hardly', 'barely', 'scarcely', 'without'}
# ..."sufficient"/"adequate", since "enough" can't follow an article ("an adequate supply" -> "an enough
# supply") and "adequate" means good enough as well as much enough ("adequate housing" isn't "enough housing")
ENOUGH_KEYS = {'sufficient|ADJ', 'adequate|ADJ'}
# ...swaps that hold for one sense of the verb only, told apart by its object: "cultivate trust" -> "build
# trust", but "cultivate rice" is farming; "establish a company" -> "set up", but "established a link
# between sleep and memory" is proving one, and so is "establish the direction of the effect", "the extent of the
# damage" or "the safety of the vaccine"
OBJECT_SENSES = {'cultivate|VERB'}
EPISTEMIC_OBJECTS = {'link', 'connection', 'correlation', 'association', 'causality', 'causation', 'cause', 'fact',
                     'pattern', 'effect', 'identity', 'existence', 'validity', 'truth', 'guilt', 'innocence', 'whereabouts',
                     'direction', 'extent', 'nature', 'origin', 'source', 'date', 'age', 'scale', 'timing', 'sequence',
                     'mechanism', 'magnitude', 'likelihood', 'probability', 'reason', 'circumstance', 'detail',
                     'authenticity', 'ownership', 'liability', 'responsibility', 'paternity', 'provenance', 'size',
                     'location', 'position', 'impact', 'role', 'relevance', 'efficacy', 'effectiveness', 'safety',
                     'accuracy', 'reliability'}
# ...and what "establish" sets up or builds, per replacement (the object's lemma): a body or a service is set up
# ("establish a charity", "a new fund", "a helpline"), a standing or a tie is built up over time ("establish her
# career", "his reputation", "a strong online presence", and "trust" with no article: "a trust" is a fund). Anything
# else keeps the word: "set up her career" and "build a fund" read wrongly, and "set up rules" or "build boundaries"
# aren't said
ESTABLISHED_FITS = {
    'set up': {'company', 'business', 'firm', 'charity', 'fund', 'committee', 'panel', 'board', 'council', 'commission',
               'inquiry', 'taskforce', 'group', 'team', 'unit', 'office', 'branch', 'centre', 'center', 'clinic',
               'scheme', 'programme', 'program', 'service', 'system', 'account', 'website', 'shop', 'store',
               'restaurant', 'bakery', 'partnership', 'venture', 'startup', 'organisation', 'organization', 'agency',
               'union', 'club', 'society', 'association', 'network', 'camp', 'factory', 'workshop', 'studio', 'hub',
               'hotline', 'helpline', 'register', 'database', 'school', 'college', 'library', 'museum', 'subsidiary',
               'department'},
    'build': {'career', 'reputation', 'relationship', 'rapport', 'presence', 'following', 'brand', 'credibility', 'bond',
              'friendship', 'partnership', 'network', 'audience', 'business', 'company', 'empire', 'foundation', 'habit',
              'routine', 'community', 'alliance', 'clientele', 'readership', 'fanbase', 'trust'},
}
CULTIVATE_OBJECTS = {'relationship', 'culture', 'habit', 'skill', 'mindset', 'trust', 'sense', 'curiosity',
                     'environment', 'community', 'talent', 'network', 'loyalty', 'partnership', 'connection',
                     'creativity', 'leadership', 'atmosphere', 'attitude', 'discipline', 'interest', 'awareness',
                     'friendship', 'patience', 'resilience', 'reputation', 'image', 'following', 'audience',
                     'presence', 'spirit', 'love', 'empathy', 'confidence', 'tie', 'rapport', 'goodwill', 'support',
                     'practice', 'appreciation', 'understanding', 'respect', 'teamwork', 'collaboration', 'growth'}
# ...and "cornerstone", a figure only for an idea, a plan or a practice ("the cornerstone of any good relationship",
# "Trust remains a cornerstone in every partnership"). The stone itself belongs to a building, is laid, carved,
# cracked or opened, bears a date, and holds things hidden inside it ("the cornerstone of the old mill", "a box
# inside the cornerstone", "The cornerstone, laid in 1885, ..."). Verbs that also take the figure ("would damage the
# cornerstone of our justice system", "replace the cornerstone of the welfare state") only count for a cornerstone of
# nothing ("Workers removed the cornerstone")
LAID_STONE_VERBS = {'lay', 'unveil', 'dedicate', 'set', 'place', 'carve', 'engrave', 'inscribe', 'bless', 'hide', 'crack',
                    'bury', 'seal', 'insert', 'uncover', 'chisel'}
GENERAL_STONE_VERBS = {'remove', 'open', 'damage', 'replace', 'restore', 'read', 'bear'}
INSIDE_PREPOSITIONS = {'inside', 'behind', 'under', 'beneath', 'below', 'underneath', 'within', 'into'}
# (with the buildings whose main WordNet sense is another: a mansion is first a sign of the zodiac, a gymnasium a
# school and a clinic a medical team)
CORNERSTONE_BUILDINGS = {'church', 'cathedral', 'bank', 'library', 'building', 'house', 'temple', 'school', 'hall',
                         'museum', 'tower', 'bridge', 'monument', 'courthouse', 'mosque', 'synagogue', 'chapel', 'station',
                         'hospital', 'college', 'university', 'theatre', 'theater', 'office', 'block', 'wall', 'structure',
                         'site', 'wing', 'mansion', 'gym', 'gymnasium', 'clinic', 'pavilion', 'depot', 'campus', 'lodge',
                         'hangar', 'terminal', 'arena', 'barracks', 'manor', 'villa', 'shrine', 'headquarters', 'jail',
                         'gaol', 'firehouse', 'schoolhouse', 'farmhouse', 'parliament', 'capitol', 'market',
                         'marketplace', 'cinema', 'shop', 'store', 'mall', 'inn', 'tavern', 'saloon', 'hotel', 'motel',
                         'restaurant', 'cafe', 'garage', 'bakery', 'pharmacy', 'laboratory', 'observatory', 'arcade'}
# "promote" pushes an idea or a practice forward ("promote innovation", "promote recycling"), but a person who is
# promoted gets a better job: "encourage" becomes "promote" only before an abstract noun, never a person or a body
# of people ("the witness", "the audience", "the next generation"; the kinds come from WordNet). A word that names
# a body acting as one is people when the text points at a particular one ("the leadership", "its membership", "the
# regiment", "the opposition"), and an idea otherwise ("encourage teamwork and leadership")
PROMOTABLE_ENDINGS = ('tion', 'sion', 'ment', 'ity', 'ness', 'ance', 'ence', 'ing', 'ism', 'cy', 'ship', 'ure', 'th', 'ty')
DEFINITE_DETERMINERS = {'the', 'its', 'their', 'our', 'his', 'her', 'your', 'my', 'this', 'that', 'these', 'those'}
NOT_PROMOTABLE = {'youth', 'faculty', 'deputy', 'sibling', 'offspring', 'king', 'darling', 'party', 'being', 'creature',
                  'government', 'management', 'authority', 'majority', 'minority', 'celebrity', 'community', 'charity',
                  'society', 'agency', 'ministry', 'university', 'city', 'county', 'talent', 'congregation', 'delegation',
                  'nation'}
# "sharply" grades a change in amount ("sharply increased"), not "sharply transformed" or "sharply improved"
SHARP_CHANGE_VERBS = {'increase', 'decrease', 'rise', 'fall', 'drop', 'decline', 'grow', 'shrink', 'cut', 'reduce',
                      'raise', 'lower', 'climb', 'jump', 'slow', 'accelerate', 'expand', 'contract', 'plunge', 'soar',
                      'surge', 'tumble', 'widen', 'narrow', 'boost', 'curb', 'shorten', 'lengthen', 'diverge', 'differ',
                      'contrast', 'divide', 'split'}
# Verbs whose curated replacement has another sense, which the object gives away: "retain a lawyer" (hire), "retain
# the information" (remember), "the body retains water" (holds back), "acquire a rival firm" (buy), "acquire a
# language" (learn), "images were acquired" (recorded), "identify the attacker" (name), "examine the dog" (as a vet
# does), "highlight the row" (mark on screen), "enable notifications" (switch on), "reduce the sauce" (boil down),
# "foster two children" (look after), "eliminate the champions" (knock out), "organize a union" (unionise); "keep",
# "gain", "find", "study", "show", "allow", "cut", "support", "remove" and "arrange" miss each
OTHER_SENSE_OBJECTS = {
    'retain|VERB': {'lawyer', 'solicitor', 'barrister', 'counsel', 'attorney', 'accountant', 'consultant', 'adviser',
                    'advisor', 'agent', 'firm', 'architect', 'information', 'knowledge', 'detail', 'fact', 'material',
                    'lesson', 'vocabulary', 'word', 'name', 'memory', 'water', 'fluid', 'moisture', 'heat', 'salt',
                    'sodium', 'urine', 'fat'},
    'acquire|VERB': {'company', 'firm', 'business', 'rival', 'startup', 'stake', 'share', 'competitor', 'brand', 'chain',
                     'bank', 'property', 'land', 'house', 'building', 'right', 'club', 'team', 'subsidiary', 'maker',
                     'language', 'image', 'data', 'datum', 'signal', 'scan', 'sample', 'spectrum', 'recording',
                     'measurement', 'reading', 'grammar', 'vocabulary'},
    'organize|VERB': {'union', 'strike', 'protest', 'campaign', 'movement', 'party', 'boycott', 'rally', 'worker',
                      'labour', 'labor', 'resistance', 'opposition', 'march', 'walkout'},
    'identify|VERB': {'attacker', 'suspect', 'victim', 'body', 'killer', 'man', 'woman', 'person', 'driver', 'gunman',
                      'robber', 'thief', 'offender', 'culprit', 'remains', 'caller', 'boy', 'girl', 'child'},
    'examine|VERB': {'dog', 'cat', 'patient', 'animal', 'horse', 'child', 'baby', 'body', 'wound', 'injury', 'eye', 'ear',
                     'throat', 'skin', 'you', 'him', 'her', 'them', 'me', 'us'},
    'highlight|VERB': {'text', 'line', 'word', 'cell', 'row', 'column', 'selection', 'paragraph', 'item', 'field',
                       'passage', 'sentence', 'file', 'folder', 'link', 'code', 'range'},
    'enable|VERB': {'notification', 'setting', 'option', 'feature', 'cookie', 'javascript', 'authentication', 'mode',
                    'logging', 'permission', 'location', 'bluetooth', 'sync', 'backup', 'extension', 'plugin', 'macro',
                    'firewall', 'encryption', 'filter', 'alert', 'service', 'verification', 'update', 'tracking',
                    'syncing', 'sharing', 'autoplay', 'autosave', 'hotspot', 'roaming', 'wifi', 'vpn', 'lock', 'password',
                    'camera', 'microphone'},
    'reduce|VERB': {'sauce', 'liquid', 'stock', 'wine', 'syrup', 'mixture', 'juice', 'gravy', 'broth', 'vinegar',
                    'water'},
    'foster|VERB': {'child', 'kid', 'baby', 'dog', 'cat', 'puppy', 'kitten', 'animal', 'pet', 'teenager', 'girl', 'boy',
                    'youngster', 'sibling', 'refugee'},
    'eliminate|VERB': {'team', 'side', 'champion', 'rival', 'opponent', 'player', 'contestant', 'candidate', 'club',
                       'competitor', 'favourite', 'favorite', 'holder', 'seed', 'giant', 'act', 'contender',
                       'underdog', 'finalist', 'semifinalist'},
}
# ...and the same by the kind of object (NOUN_SENSES), in curated_fits: a doctor or a vet examines "her knee" or "the
# horse", and a contest eliminates "the hosts", "the finalists", "the 2014 runners-up" or "the world number one". A
# contest knocks a side out "in the second round", "in the final", "on penalties", "from the tournament", "in
# overtime" or "over two legs" (not "in the latest round of layoffs"), and "in the group stage", "in straight sets"
# or "in extra time" (not "at an early stage" or "in her spare time"). A leg is a contest's only when counted
# ("over two legs", "in the second leg", "the return leg"; not "the pain in his left leg"), and extra time only
# when something happens in it ("in/after/during extra time"; not "the need for extra time")
CONTEST_STAGES = {'round', 'final', 'semifinal', 'quarterfinal', 'playoff', 'tournament', 'championship', 'penalty',
                  'shootout', 'overtime', 'leg', 'tiebreak', 'tiebreaker', 'decider', 'qualifier'}
CONTEST_STAGE_MODIFIERS = {'stage': {'group', 'knockout', 'qualifying', 'final'}, 'set': {'straight'}, 'time': {'extra'}}
CONTEST_LEG_MODIFIERS = {'first', 'second', 'return', 'opening', 'home', 'away', 'final'}
# ...but a doctor who examines "the scan results" or "the records of 2,000 patients" reads them ("studied" fits), and
# a firm that eliminates "300 jobs in the first round" or "the middleman" cuts them out, not a rival
DOCUMENT_OBJECTS = {'record', 'result', 'scan', 'report', 'file', 'chart', 'note', 'image', 'test', 'finding',
                    'history', 'document', 'data', 'datum', 'evidence', 'sample', 'slide'}
JOB_NOUNS = {'job', 'post', 'position', 'role'}
INTERMEDIARY_NOUNS = {'middleman', 'intermediary'}
# A recipe boils a liquid down: "reduce" keeps that sense when the sentence has a cooking cue and its object is a
# food, or a word the dictionary doesn't know in a bare instruction with no diet in sight ("Reduce coconut milk by
# half", "Reduce the passata until thick"; not "We will reduce our ad spend by half" or "Reduce your carbs by half"),
# when a bare instruction reduces "the" food with no diet in sight ("Reduce the cream", not "Reduce the sugar in
# your tea"), and for "it" after a food in a cooking step ("Add the stock, then reduce it"). A diet is in sight when
# the sentence speaks of one, or when "you"/"your" is in reduce's own clause ("the sugar in your tea", "the salt
# you add"), not in a clause joined to it ("Reduce the cream, then pour it over your pasta")
COOKING_CUES = re.compile(r"\b(?:simmer\w*|boil\w*|by (?:half|a third|a quarter|two-thirds|three-quarters|about half)|"
                          r"until (?:it |they |the \w+ )?(?:is |are |has |have )?(?:thick\w*|syrupy|glossy|reduced|"
                          r"coats?|jammy|sticky)|over (?:a |the )?(?:low|medium|high|gentle|moderate|medium-high|"
                          r"medium-low) heat|in (?:a|the) (?:pan|saucepan|pot|skillet|wok)|to a (?:glaze|syrup)|"
                          r"for (?:\d+|a few|two|three|four|five|ten|fifteen|twenty) minutes)\b", re.I)
RECIPE_STEP_VERBS = {'add', 'pour', 'stir', 'simmer', 'boil'}
DIET_WORDS = {'diet', 'intake', 'eat', 'drink', 'consumption'}
DIET_PRONOUNS = {'you', 'your'}
# ...and "terminate", which only means "end" for an agreement or a process ("the employee was terminated" was
# dismissed)
CONTRACT_NOUNS = {'contract', 'agreement', 'employment', 'lease', 'tenancy', 'subscription', 'membership', 'service',
                  'licence', 'license', 'account', 'plan', 'policy', 'deal', 'relationship', 'call', 'session', 'process',
                  'program', 'programme', 'pregnancy', 'project', 'trial', 'scheme', 'partnership', 'arrangement'}
# "comprise" means "make up" before a share, also after "under", "over" or "about" ("women comprise nearly half of the
# workforce", "which comprised under 3% of all responses"), and "is comprised of" can't become "is included of"
COMPRISE_SHARES = {'half', 'third', 'quarter', 'majority', 'minority', 'proportion', 'share', 'percent', 'percentage',
                   'fraction', 'bulk', 'part', 'fifth', 'tenth', '%'}
# "immediately after a meal" can't become "at once after a meal" or "right away after a meal"; nor can "immediately
# below the knee" or "immediately obvious", and neither phrase stands before its verb ("He did not right away grasp
# ...", "The company at once recalled ...")
SEQUENCE_WORDS = {'after', 'before', 'following', 'upon', 'on', 'prior', 'afterwards', 'afterward'}
POSITION_DEPENDENT = (AFTER_VERB_ONLY | VERB_HEAD_ONLY | OBJECT_ONLY | BEFORE_NOUN_ONLY | PREDICATE_ONLY | CLAUSE_ONLY
                      | NOT_BEFORE_DESCRIBER | NOT_AFTER_NEGATION | ENOUGH_KEYS | OBJECT_SENSES | BEFORE_ADJECTIVE_ONLY
                      | set(OTHER_SENSE_OBJECTS)
                      | {'cornerstone|NOUN', 'dramatically|ADV', 'encourage|VERB', 'substantially|ADV', 'facet|NOUN',
                         'profound|ADJ', 'require|VERB', 'anticipate|VERB', 'comprise|VERB', 'cease|VERB', 'decisive|ADJ',
                         'paramount|ADJ', 'immediately|ADV', 'consequently|ADV', 'terminate|VERB', 'illustrate|VERB',
                         'demonstrate|VERB', 'valuable|ADJ', 'unprecedented|ADJ', 'comprehensive|ADJ', 'initial|ADJ',
                         'tapestry|NOUN', 'elevate|VERB', 'significantly|ADV', 'continuously|ADV', 'currently|ADV',
                         'component|NOUN', 'ameliorate|VERB', 'proficient|ADJ', 'adept|ADJ', 'disseminate|VERB',
                         'vibrant|ADJ', 'innovative|ADJ'})
# Comparisons need the original word: "the most significant issue" can't become "the most big issue"
DEGREE_WORDS = {'more', 'most', 'less', 'least', 'very', 'so', 'too', 'as'}
NEGATION_OR_DEGREE = DEGREE_WORDS | {'not', "n't", 'n’t', 'never', 'nowhere', 'pretty', 'quite', 'far'}
# "adequate" -> "enough" only for amounts ("adequate funding", "the budget is adequate")
AMOUNT_NOUNS = {'time', 'sleep', 'rest', 'funding', 'money', 'water', 'food', 'space', 'room', 'resource', 'staff',
                'staffing', 'notice', 'evidence', 'information', 'support', 'training', 'protection', 'ventilation',
                'lighting', 'nutrition', 'hydration', 'capacity', 'coverage', 'compensation', 'pay', 'data'}
# Replacements that only work in front of a noun ("a major step", but not "it is major for us to ...")
ATTRIBUTIVE_ONLY = {'major', 'big', 'core'}
# A replacement verb must not form a phrasal verb with the word after it ("providing up to five hours" can't
# become "giving up to five hours")
PARTICLE_CLASHES = {
    'give': {'up', 'in', 'out', 'off', 'over', 'away', 'back'}, 'get': {'up', 'over', 'out', 'off', 'by', 'on', 'back',
    'away', 'down', 'in'}, 'end': {'up'}, 'cut': {'up', 'out', 'off', 'back', 'down', 'in'}, 'keep': {'up', 'on', 'off',
    'out', 'back', 'down'}, 'change': {'over'}, 'show': {'up', 'off'}, 'use': {'up'}, 'stop': {'over', 'off', 'by'},
    'start': {'up', 'off', 'out', 'over'}, 'help': {'out'}, 'build': {'up'}, 'try': {'out', 'on'}, 'find': {'out'},
    'look': {'up', 'out', 'over', 'in', 'back', 'on'}, 'set': {'up', 'out', 'off', 'back', 'in', 'about'},
    'lower': {'down'}, 'raise': {'up'}, 'spot': {'on'}, 'plan': {'on', 'out'}, 'arrange': {'up'},
}
# Degree adverbs before a comparative ("significantly more") must be ones that grade a comparison
COMPARATIVE_DEGREE = {'considerably', 'substantially', 'much', 'far'}
# "significantly reduce" can become "greatly reduce", but "hugely" reads oddly before a verb
NOT_BEFORE_VERB = {'hugely'}
# "tap into" fits sources ("tap into their expertise"), not plans ("tap into new strategies")
TAP_OBJECTS = {'resource', 'expertise', 'knowledge', 'talent', 'skill', 'strength', 'experience', 'potential', 'power',
               'energy', 'creativity', 'market', 'audience', 'network', 'demand', 'data', 'datum', 'insight', 'community',
               'pool', 'reserve', 'source', 'fund', 'wisdom'}
# "substantial"/"significant" mean size before amounts and money ("a large sum", "a large safety net") but
# weight before changes and effects ("a major change"); "a major number of" and "large evidence" are wrong
SIZE_NOUNS = {'number', 'amount', 'sum', 'portion', 'proportion', 'share', 'quantity', 'body', 'income', 'salary',
              'savings', 'saving', 'fund', 'net', 'cushion', 'reserve', 'budget', 'loan', 'debt', 'payment', 'discount',
              'loss', 'profit', 'sample', 'population', 'majority', 'margin', 'investment', 'cost', 'sale', 'revenue',
              'audience', 'crowd', 'area', 'house', 'garden', 'meal', 'collection', 'volume', 'stock', 'supply'}
WEIGHT_NOUNS = {'change', 'shift', 'improvement', 'progress', 'challenge', 'implication', 'impact', 'effect', 'role',
                'risk', 'factor', 'problem', 'issue', 'concern', 'obstacle', 'barrier', 'contribution', 'step',
                'difference', 'advantage', 'benefit', 'threat', 'development', 'milestone', 'breakthrough',
                'achievement', 'influence', 'reason', 'cause', 'event', 'moment', 'decision', 'setback', 'gain',
                'upgrade', 'boost', 'reform', 'overhaul', 'turning', 'player', 'part', 'source', 'driver'}

# "a significant correlation" is a statistical claim, not a large one; so is "a significant difference
# between the two groups (p < 0.01)"
STATISTICS_NOUNS = {'correlation', 'association', 'relationship', 'interaction', 'predictor', 'covariate', 'variance'}
MEASURED_NOUNS = {'difference', 'effect', 'increase', 'decrease', 'reduction', 'improvement', 'change', 'gain', 'loss',
                  'decline', 'rise', 'drop', 'link', 'trend', 'result', 'finding', 'variation', 'deviation'}
STATISTICAL = re.compile(r"\b(?:p\s*[<=>≤]|statistical|confidence interval|CI\b|control group|compared (?:with|to)|"
                         r"participants?|respondents?|patients|subjects|sample|trial|regression|odds ratio|"
                         r"standard deviation|between the (?:two |three )?groups)", re.I)
# "significantly" in a report of evidence may be the statistical sense too: "Studies have shown that losing 5% of your
# body weight can lower blood sugar significantly", "Research shows that exercise significantly reduces the risk", "The
# results improved significantly", "fell significantly, by 8 mmHg". The evidence is named in the sentence or the one
# before it (the lemmas below), or the sentence measures something (a percentage, a quantity or a figure)
SIGNIFICANCE_EVIDENCE = {'study', 'research', 'researcher', 'trial', 'result', 'finding', 'data', 'datum', 'evidence',
                         'experiment', 'survey', 'analysis', 'measurement', 'test', 'scientist', 'review'}
# "raise" a question or doubt, but "pose a threat" isn't "raise a threat"
RAISED_OBJECTS = {'question', 'concern', 'issue', 'doubt', 'objection', 'point', 'possibility', 'query'}
# "profound" is "far-reaching" or "major" before an effect ("a profound impact") and "deep" before a feeling or a
# thought ("a profound sense of loss"); elsewhere it stays ("profound hearing loss" is a clinical grade). An effect
# on one person isn't far-reaching ("a profound impact on my life" -> "a major impact", "a profound change in how
# he spoke")
FAR_REACHING_HEADS = {'effect', 'impact', 'implication', 'consequence', 'influence', 'change', 'shift', 'transformation'}
PERSONAL_TARGETS = {'me', 'you', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their', 'i', 'he', 'she', 'we',
                    'they', 'myself', 'yourself', 'himself', 'herself', 'ourselves', 'themselves'}
DEEP_HEADS = {'sense', 'gratitude', 'respect', 'sadness', 'grief', 'love', 'understanding', 'knowledge', 'insight',
              'question', 'thought', 'silence', 'sleep', 'sorrow', 'regret', 'admiration', 'affection', 'belief', 'faith',
              'emotion', 'feeling', 'connection', 'appreciation', 'thinker', 'wisdom', 'concern'}
# "decisive" means resolute only for actions and leaders ("take decisive action" -> "firm action"); "a decisive
# goal" or "the decisive factor" settles the outcome, which "firm" and "strong" don't say
RESOLUTE_HEADS = {'action', 'leadership', 'leader', 'response', 'measure', 'step', 'approach', 'manner', 'stance',
                  'move', 'intervention', 'decision', 'style', 'tone', 'management', 'government'}
# "valuable" means useful for help and knowledge ("a valuable lesson"); "valuable paintings" are worth money and
# "valuable time" is precious
USEFUL_HEADS = {'insight', 'information', 'experience', 'lesson', 'feedback', 'advice', 'skill', 'tool', 'resource',
                'contribution', 'input', 'support', 'guidance', 'tip', 'opportunity', 'knowledge', 'perspective', 'data',
                'service', 'work', 'help', 'addition', 'member', 'partner', 'source', 'evidence', 'research', 'reference',
                'training', 'practice', 'idea', 'suggestion', 'feature', 'lead'}
# ...but said of a noun ("Your feedback is valuable", "Their advice proved valuable") only of advice and know-how:
# "Personal information is valuable" and "Our members are valuable to the club" are worth something, not "helpful"
PREDICATE_USEFUL_HEADS = {'insight', 'experience', 'lesson', 'feedback', 'advice', 'skill', 'input', 'guidance', 'tip',
                          'knowledge', 'perspective', 'contribution', 'help', 'training', 'practice', 'idea',
                          'suggestion', 'support', 'research', 'evidence', 'reference', 'tool', 'technique', 'method',
                          'feature'}
# "record" is an amount ("unprecedented profits" -> "record profits"), not "unprecedented times"
RECORD_HEADS = {'profit', 'sale', 'level', 'high', 'low', 'number', 'turnout', 'temperature', 'growth', 'revenue',
                'attendance', 'demand', 'price', 'rate', 'amount', 'loss', 'heat', 'score', 'investment'}
# Fixed terms: "a comprehensive school" is a kind of school
FIXED_TERM_NOUNS = {
    'comprehensive': {'school', 'exam', 'examination', 'insurance'},
}
# "traditional" is "conventional" only for a way of doing things that newer ones replace ("traditional methods",
# "traditional banks", "traditional cars"); a tradition's own things keep it ("traditional carols", "a traditional
# wooden boat", "traditional medicine", whose "conventional" opposite is the other medicine), and "standard" fits a
# method only ("the standard approach", never "standard roles")
CONVENTIONAL_NOUNS = {'method', 'approach', 'model', 'technique', 'procedure', 'process', 'format', 'design', 'tool',
                      'system', 'way', 'medium', 'media', 'bank', 'banking', 'lender', 'retailer', 'retail', 'store',
                      'classroom', 'teaching', 'school', 'education', 'learning', 'office', 'workplace', 'job', 'role',
                      'career', 'business', 'industry', 'sector', 'car', 'vehicle', 'engine', 'fuel', 'loan', 'mortgage',
                      'investment', 'publishing', 'publisher', 'advertising', 'marketing', 'broadcaster', 'television',
                      'newspaper', 'employment', 'structure', 'hierarchy'}
STANDARD_METHOD_NOUNS = {'method', 'approach', 'model', 'technique', 'procedure', 'process', 'format', 'design', 'tool',
                         'system'}
# "appropriate" is "fitting" for an honour or an ending ("a fitting tribute", "a fitting end"); "take fitting action"
# isn't said ("suitable" fits the rest)
FITTING_NOUNS = {'tribute', 'end', 'ending', 'conclusion', 'memorial', 'reward', 'finale', 'farewell', 'climax',
                 'honour', 'honor', 'monument', 'testament', 'reminder', 'send-off', 'way', 'name', 'title', 'symbol'}
# "initial" is "first" only for the start of a series or a process ("the initial stage", "my initial reaction", "the
# initial results"); an amount or a state at the start isn't the first of anything ("the initial investment", "their
# initial capacity", "initial conditions", "an initial public offering")
INITIAL_STAGE_NOUNS = {'stage', 'phase', 'step', 'round', 'wave', 'meeting', 'visit', 'consultation', 'contact',
                       'interview', 'impression', 'reaction', 'response', 'thought', 'version', 'draft', 'plan', 'idea',
                       'design', 'attempt', 'trial', 'test', 'result', 'finding', 'report', 'assessment', 'review',
                       'diagnosis', 'symptom', 'outbreak', 'shock', 'day', 'week', 'month', 'year', 'hour', 'minute',
                       'moment', 'period', 'release', 'batch', 'episode', 'season', 'chapter', 'session', 'lesson'}
# Terms a plain word would break: "continuously differentiable", "the first principal component"
CONTINUOUS_TERMS = {'differentiable', 'variable', 'distributed', 'compounded', 'valued', 'integrable'}
# "acquire" -> "gain" only for what is built up ("acquired new skills", "acquired a reputation"); records, land or a
# firm are got or bought ("Researchers acquired detailed records" can't be "gained detailed records")
GAINED_QUALITIES = {'skill', 'experience', 'knowledge', 'understanding', 'insight', 'expertise', 'reputation', 'status',
                    'popularity', 'momentum', 'confidence', 'access', 'habit', 'taste', 'fluency', 'competence', 'ability',
                    'qualification', 'immunity', 'resistance', 'meaning', 'importance', 'significance', 'influence',
                    'power', 'control', 'recognition', 'fame', 'following', 'trust', 'respect'}
# "ameliorate" -> "improve" only for a state that gets better ("ameliorate living conditions"); "ameliorate the side
# effects", "the pain" or "poverty" makes them less bad, which "improve the side effects" doesn't say
IMPROVABLE_STATES = {'condition', 'situation', 'circumstance', 'life', 'quality', 'health', 'wellbeing', 'standard',
                     'relation', 'relationship', 'outcome', 'access', 'environment', 'performance', 'design'}
# "reduce the heat" -> "lower the heat", never "cut the heat"
LOWER_ONLY_OBJECTS = {'heat', 'temperature', 'flame', 'gas', 'oven', 'power', 'speed', 'volume'}
# ...and the symptoms a treatment lowers and the loads a measure lowers ("can reduce inflammation" -> "can lower
# inflammation", "reduce emissions" -> "lower emissions", "reduce latency"); other objects of "reduce" are cut, not
# lowered ("cut the menu to ten dishes", "cut waiting lists")
LOWERED_SYMPTOMS = {'inflammation', 'swelling', 'anxiety', 'symptom', 'fever', 'burden', 'emission', 'latency', 'load',
                    'consumption', 'intake', 'usage', 'exposure', 'demand'}
# Police "look into" a death; they don't "study" or "explore" it
INCIDENT_OBJECTS = {'death', 'murder', 'crime', 'incident', 'case', 'attack', 'fire', 'complaint', 'allegation', 'theft',
                    'robbery', 'fraud', 'accident', 'disappearance', 'shooting', 'collision', 'breach', 'leak', 'killing'}
# "organize a meeting" -> "plan a meeting", but files and folders are arranged, not planned
PLANNED_EVENTS = {'event', 'meeting', 'trip', 'party', 'conference', 'festival', 'workshop', 'wedding', 'tour', 'visit',
                  'campaign', 'fundraiser', 'protest', 'competition', 'match', 'concert', 'exhibition', 'march', 'program',
                  'programme'}
# "elevate the injured leg" -> "raise", never "improve" or "boost"
LIMB_OBJECTS = {'leg', 'arm', 'foot', 'feet', 'head', 'ankle', 'knee', 'hand', 'limb', 'wrist', 'hip'}
# A separable phrasal verb ("speed up", "smooth out", "set up", "figure out", "bring in") puts a pronoun object before
# its particle: "expedite them" -> "speed them up", never "speed up them"
SEPARABLE_PARTICLES = {'up', 'out', 'off', 'down', 'over', 'back', 'away', 'in'}
OBJECT_PRONOUNS = {'it', 'them', 'him', 'her', 'me', 'us', 'you', 'this', 'that', 'these', 'those', 'itself',
                   'themselves', 'himself', 'herself', 'myself', 'ourselves', 'yourself', 'yourselves'}
# "tapestry" is a figure for a mix ("a rich cultural tapestry", "a tapestry of cultures") unless the text is about the
# cloth: "the tapestry" or "a fine tapestry" with nothing that makes it a figure, an old or woven one, one with a date
# ("a 16th-century tapestry", "a tapestry from 1520"), one that hangs or is restored, one with a scene on it, or one
# that shows a scene ("a medieval tapestry that shows a royal hunt"). A tapestry of many things is a mix whatever
# is done with it ("draws on a rich tapestry of traditions", "sells a tapestry of local crafts")
FIGURE_DESCRIBERS = {'rich', 'cultural', 'vibrant', 'diverse', 'complex', 'social', 'linguistic', 'ethnic',
                     'multicultural', 'human', 'varied', 'eclectic', 'dynamic', 'urban', 'global', 'intellectual',
                     'musical', 'culinary'}
WOVEN_DESCRIBERS = {'medieval', 'woven', 'silk', 'wool', 'woollen', 'woolen', 'flemish', 'embroidered', 'antique',
                    'tattered', 'faded', 'huge', 'large', 'giant', 'enormous', 'priceless', 'hanging'}
CLOTH_VERBS = {'hang', 'restore', 'embroider', 'sew', 'stitch', 'drape', 'commission', 'steal', 'clean', 'repair',
               'conserve', 'auction', 'unroll', 'frame', 'buy', 'sell', 'adorn', 'own', 'donate'}
SCENE_VERBS = {'show', 'depict', 'portray', 'illustrate', 'display', 'exhibit', 'weave'}
SURFACE_PREPOSITIONS = {'on', 'behind', 'under', 'beneath', 'above', 'below', 'across'}
# "a profound impact on one person" isn't far-reaching: "on me", "in his mood", "on the patient", "for Maria",
# "within her", "inside him", "to his voice"
TARGET_PREPOSITIONS = {'on', 'upon', 'in', 'for', 'within', 'inside', 'to'}
# "vibrant" is "bright" for a colour or a coloured thing ("vibrant colours", "vibrant blooms", never "lively fabrics"),
# and "lively" for the rest: a vibrant city, music scene or culture is lively, and a bright one is well lit or clever
VIBRANT_COLOUR_NOUNS = {'colour', 'color', 'hue', 'shade', 'tint', 'paint', 'palette', 'print', 'dye', 'pigment',
                        'flower', 'bloom', 'blossom', 'petal', 'plumage', 'feather', 'mural', 'fabric', 'textile', 'tile',
                        'red', 'blue', 'green', 'yellow', 'orange', 'pink', 'purple'}
# "innovative" is "creative" (or Casual's "fresh") only for an idea, a way of working or the people who have them ("an
# innovative approach", "innovative thinking"); a vaccine, a device or a product isn't creative, and a "fresh" one is
# newly made
INNOVATIVE_CREATIVE_HEADS = {'approach', 'idea', 'solution', 'design', 'strategy', 'thinking', 'thinker', 'campaign',
                             'way', 'method', 'team', 'artist', 'writer', 'teacher', 'teaching', 'project', 'response',
                             'concept', 'format', 'mind', 'leader', 'leadership', 'culture', 'programme', 'program',
                             'use', 'twist', 'take', 'angle', 'perspective', 'recipe', 'menu', 'workshop'}
# "consistent" is "steady" only for what goes on over time ("consistent effort", "consistent progress"); consistent
# formatting, style or results are the same everywhere, which "steady" doesn't say
STEADY_OVER_TIME_NOUNS = {'effort', 'progress', 'growth', 'income', 'pace', 'practice', 'improvement', 'supply', 'rise',
                          'decline', 'increase', 'decrease', 'flow', 'rhythm', 'routine', 'training', 'exercise',
                          'pressure', 'demand', 'stream', 'gain', 'climb', 'fall', 'sleep', 'rate', 'output', 'support',
                          'attendance', 'contact', 'saving', 'investment'}
# A treatment eases pain and the like ("While the drug reduces pain" -> "eases pain"); "cuts pain" and "lowers back
# pain" read wrongly
EASED_SYMPTOMS = {'pain', 'ache', 'headache', 'discomfort', 'soreness', 'stiffness', 'itching', 'itch', 'cramp',
                  'nausea', 'irritation', 'symptom', 'swelling', 'inflammation', 'tenderness', 'bloating'}
# Foods one slices: "cut the meat" or "cut the bread" reads as slicing it, where "cut the sugar" or "cut the alcohol"
# can only mean having less (and so does "cut red meat", with no article)
CUT_UP_FOODS = {'meat', 'bread', 'cheese', 'cake', 'fruit', 'vegetable', 'chicken', 'beef', 'pork', 'lamb', 'fish',
                'steak', 'ham', 'bacon', 'sausage', 'potato', 'onion', 'apple', 'pie', 'pizza', 'butter', 'chocolate',
                'biscuit', 'pastry', 'carrot', 'tomato', 'turkey', 'salmon', 'tuna', 'melon', 'banana', 'loaf',
                'sandwich', 'burger', 'egg'}

def separable_object(token):
    """A verb's pronoun object when it comes after the verb ("streamline it"), or None."""
    return next((c for c in token.children if c.dep_ == 'dobj' and c.lower_ in OBJECT_PRONOUNS and c.i > token.i), None)

def particle_fits(pronoun):
    """True if a particle can follow a pronoun object: one right after its verb and on its own ("streamline it
    for us" -> "speed it up for us", but not "streamline them all"), and not "that", which may open a clause the
    parser misread ("establish that the claims hold")."""
    doc = pronoun.doc
    return (pronoun.i == pronoun.head.i + 1 and not list(pronoun.children) and pronoun.lower_ != 'that'
            and not (pronoun.i + 1 < len(doc) and doc[pronoun.i + 1].lower_ in ('all', 'both', 'each')))

def particle_split(token, word):
    """A separable phrasal verb as two edits around the verb's pronoun object: "sped up" before "them" ->
    ("sped", them, "them up"); None for any other word."""
    verb, _, particle = word.rpartition(' ')
    obj = separable_object(token) if verb and particle in SEPARABLE_PARTICLES else None
    return (verb, obj, f'{obj.text} {particle}') if obj is not None and particle_fits(obj) else None

def woven_tapestry(token):
    """True if "tapestry" is the cloth itself, not a figure for a mix."""
    kids = list(token.children)
    of = next((c for c in kids if c.dep_ == 'prep' and c.lower_ == 'of'), None)
    whole = next((c for c in of.children if c.dep_ == 'pobj'), None) if of is not None else None
    if any(c.dep_ in ('amod', 'compound') and c.lemma_.lower() in WOVEN_DESCRIBERS for c in kids):
        return True     # "a medieval tapestry"
    figure_word = any(c.dep_ == 'amod' and c.lemma_.lower() in FIGURE_DESCRIBERS for c in kids)
    # what the tapestry does or has done to it: "hangs", "was stolen", "that shows a royal hunt" ("weaves a rich
    # tapestry of voices" and "shows the tapestry of local life" are figures)
    verbs = [c for c in kids if c.dep_ in ('relcl', 'acl')]
    if token.dep_ in ('nsubj', 'nsubjpass', 'dobj'):
        verbs.append(token.head)
    # a named or living subject ("a tapestry of saints", "of the Apostles", "of hunters and hounds") or a verb only
    # the cloth takes ("hangs", "was restored"; buy, sell, own and adorn also take the figure) keeps it the cloth
    literal = whole is not None and not figure_word and (
        whole.pos_ == 'PROPN' or noun_sense_is(whole, 'person', 'animal')
        or any(v.lemma_.lower() in CLOTH_VERBS - {'buy', 'sell', 'own', 'adorn'} for v in verbs))
    if whole is not None and not literal and (
            whole.tag_ in ('NNS', 'NNPS') or any(c.dep_ == 'conj' for c in whole.children)):
        return False    # "a tapestry of cultures", "of sounds, colours and flavours", "of joy and sorrow": a mix
    # a date before it or after "from"/"dating" ("a rich tapestry spanning 2,000 years" is still a figure)
    dating = [*token.doc[token.left_edge.i:token.i],
              *(t for c in kids if c.dep_ in ('prep', 'acl') and c.lower_ in ('from', 'dating') for t in c.subtree)]
    if any(t.like_num or t.ent_type_ in ('DATE', 'ORDINAL') for t in dating):
        return True     # "a 16th-century tapestry", "a tapestry from 1520"
    if of is None and not figure_word:
        return True     # "The tapestry was restored in 1990", "The museum owns a fine tapestry"
    # on its surface: "the figures on the tapestry", "spent years on the tapestry of the battle" (not "reflects on the
    # tapestry of human experience")
    scene = whole is not None and whole.tag_ in ('NN', 'NNP') and any(
        d.dep_ == 'det' and d.lower_ == 'the' for d in whole.children)
    if token.dep_ == 'pobj' and token.head.lower_ in SURFACE_PREPOSITIONS and (
            of is None or scene or token.head.head.pos_ in ('NOUN', 'PROPN')):
        return True
    return any(v.lemma_.lower() in CLOTH_VERBS or (v.lemma_.lower() in SCENE_VERBS and of is None) for v in verbs)

def on_one_person(noun):
    """True if an effect or a change is on or in one person: "a profound change in me", "in his mood", "on the
    patient", "on Maria". The parser may hang the phrase on the verb ("brought a profound change in my thinking")."""
    targets = [c for c in noun.children if c.dep_ == 'prep' and c.lower_ in TARGET_PREPOSITIONS]
    if noun.dep_ == 'dobj':
        targets += [c for c in noun.head.children if c.dep_ == 'prep' and c.lower_ in TARGET_PREPOSITIONS
                    and c.i == noun.right_edge.i + 1]
    return any(t.lower_ in PERSONAL_TARGETS or t.ent_type_ == 'PERSON'
               or (t.tag_ == 'NN' and noun_sense_is(t, 'person')) for target in targets for t in target.subtree)

def predicate_subject(token):
    """The noun a describing word is said of after its verb, or None: "Your feedback is valuable", "The impact of the
    reforms was profound", "Their advice proved valuable", "kept the data valuable", "made [the feedback valuable]"."""
    own = next((c for c in token.children if c.dep_ in ('nsubj', 'nsubjpass')), None)
    if own is not None or token.dep_ not in ('acomp', 'oprd'):
        return own
    kids = list(token.head.children)
    return (next((c for c in kids if c.dep_ == 'dobj'), None) if token.dep_ == 'oprd' else None) or next(
        (c for c in kids if c.dep_ in ('nsubj', 'nsubjpass')), None)

def context_choices(choices, token):
    """The curated replacements that read correctly where the token stands."""
    doc = token.doc
    nxt = doc[token.i + 1] if token.i + 1 < len(doc) else None
    lemma = token.lemma_.lower()
    described = token.head.lemma_.lower() if token.dep_ == 'amod' else ''      # the noun an adjective describes
    said_of = predicate_subject(token)     # ...or the one it is said of
    if lemma == 'significantly' and statistical_sense(token.sent):
        return []       # "significantly lower (p < 0.01)" claims statistical significance, not a large effect
    if lemma == 'decisive' and described not in RESOLUTE_HEADS:
        return []
    noun = token.head if token.dep_ == 'amod' else said_of
    kind = noun.lemma_.lower() if noun is not None else ''
    if lemma == 'vibrant':      # "a lively city", "bright colours"
        choices = [c for c in choices if (c == 'bright') == (kind in VIBRANT_COLOUR_NOUNS)]
    if lemma == 'innovative' and kind not in INNOVATIVE_CREATIVE_HEADS:
        choices = [c for c in choices if c not in ('creative', 'fresh')]    # "a new vaccine", "a creative approach"
    if lemma == 'consistent' and kind not in STEADY_OVER_TIME_NOUNS:
        return []       # "steady effort", but "consistent formatting" is the same everywhere
    if lemma == 'valuable' and described not in USEFUL_HEADS and (
            said_of is None or said_of.lemma_.lower() not in PREDICATE_USEFUL_HEADS):
        return []
    if lemma in FIXED_TERM_NOUNS and described in FIXED_TERM_NOUNS[lemma]:
        return []
    if lemma == 'traditional' and described not in CONVENTIONAL_NOUNS:
        return []
    if lemma == 'traditional' and described not in STANDARD_METHOD_NOUNS:
        choices = [c for c in choices if c != 'standard']
    if lemma == 'appropriate' and token.dep_ == 'amod' and any(
            c.dep_ in ('det', 'poss') and c.lower_ not in ('a', 'an') for c in token.head.children):
        return []       # "an appropriate response" -> "a suitable response", but "the appropriate dose" is the right one
    if lemma == 'appropriate' and described not in FITTING_NOUNS:
        choices = [c for c in choices if c != 'fitting']
    if lemma == 'initial' and described not in INITIAL_STAGE_NOUNS:
        return []
    if lemma in ('proficient', 'adept') and any(t.ent_type_ in ('LANGUAGE', 'NORP') or t.lemma_.lower() == 'language'
                                                for c in token.children if c.dep_ == 'prep' for t in c.subtree):
        return []       # "proficient in English" or "in a foreign language" is fluent in it, not "skilled in English"
    if (lemma == 'continuously' and nxt is not None and nxt.lower_ in CONTINUOUS_TERMS) or (
            lemma == 'component' and any(c.lower_ in ('principal', 'independent') for c in token.children)):
        return []       # "continuously differentiable", "the first principal component" (see CONTINUOUS_TERMS)
    if lemma == 'currently' and nxt is not None and nxt.tag_ == 'VBN' and nxt.dep_ == 'amod':
        return []       # "the currently accepted model" isn't "the now accepted model" (accepted only lately)
    if lemma == 'disseminate' and any(c.dep_ in ('prep', 'dative') and c.lower_ == 'to' for c in token.children):
        return []       # "spread best practice across schools", but data sent "to all schools" weren't "spread to" them
    if lemma == 'tapestry' and woven_tapestry(token):
        return []
    if lemma == 'profound':
        if kind in FAR_REACHING_HEADS:
            choices = [c for c in choices if c != 'deep' and not (c == 'far-reaching' and on_one_person(noun))]
        elif kind in DEEP_HEADS:
            choices = [c for c in choices if c == 'deep']
        else:
            return []
    if lemma == 'unprecedented' and described not in RECORD_HEADS:
        choices = [c for c in choices if c != 'record']
    if token.pos_ == 'VERB':
        objects = {c.lemma_.lower() for c in token.children if c.dep_ == 'dobj'}
        if lemma == 'reduce':
            choices = [c for c in choices if c in reduce_fits(token, objects)]
        elif lemma == 'establish':
            # "establish a charity" -> "set up", "establish her career" -> "build" (see ESTABLISHED_FITS); "trust" only
            # with no article ("build trust", but "a trust" is a fund)
            article = any(c.dep_ == 'dobj' and any(g.dep_ in ('det', 'nummod') for g in c.children)
                          for c in token.children)
            choices = [c for c in choices if objects & ESTABLISHED_FITS.get(c, set())
                       and not ('trust' in objects and article)]
        elif lemma == 'investigate' and objects & INCIDENT_OBJECTS:
            choices = [c for c in choices if c == 'look into']
        elif lemma == 'organize' and not objects & PLANNED_EVENTS:
            choices = [c for c in choices if c != 'plan']
        elif lemma == 'elevate' and objects & LIMB_OBJECTS:
            choices = [c for c in choices if c == 'raise']
        elif lemma == 'acquire' and not objects & GAINED_QUALITIES:
            choices = [c for c in choices if c != 'gain']
        elif lemma == 'highlight' and acting_body(swap_subject(token)):
            choices = []        # "The results highlight the need" -> "show", but teachers who highlight a value stress it
        elif lemma == 'ameliorate' and not objects & IMPROVABLE_STATES:
            choices = []        # "ameliorate conditions" -> "improve conditions", but improved side effects would be worse
        elif lemma == 'empower':
            # "enables women to lead", but "supports young carers" ("enables young carers" to do what?); the parser
            # may hang the "to" clause as adverbial ("empower staff to make decisions" can't become "support staff to
            # make decisions", which reads as the noun "support staff")
            to_clause = any(c.dep_ in ('xcomp', 'ccomp') or (c.dep_ == 'advcl' and c.i > token.i and any(
                g.dep_ == 'aux' and g.lower_ == 'to' for g in c.children)) for c in token.children)
            choices = [c for c in choices if c != ('support' if to_clause else 'enable')]
        # a separable phrasal verb splits around a pronoun object ("speed them up") where the particle fits after
        # it; "find it out" isn't idiomatic ("found him out" exposed him), so "determine it" or "figure it out" is used
        pronoun = separable_object(token)
        if pronoun is not None and not particle_fits(pronoun):
            choices = [c for c in choices if c.split()[-1] not in SEPARABLE_PARTICLES]
        elif pronoun is not None:
            choices = [c for c in choices if c != 'find out']
        # "resonated strongly" can't become "struck a chord strongly" or "clicked strongly", nor "deeply resonated"
        # "deeply clicked"; nor (an echo fallback) "is underpinned by" become "is underlain by"
        if any(c.dep_ == 'advmod' and (c.i > token.i or c.lower_.endswith('ly')) for c in token.children):
            choices = [c for c in choices if c not in ('strike a chord', 'click')]
        if token.tag_ == 'VBN':
            choices = [c for c in choices if c != 'underlie']
    if 'further' in choices and not (token.i == token.sent.start and nxt is not None and nxt.text == ','):
        # "Further, members can claim ...", not "can further claim" or "Further members can claim" (more members)
        choices = [c for c in choices if c != 'further']
    if token.pos_ == 'ADJ' and token.dep_ != 'amod':
        choices = [c for c in choices if c not in ATTRIBUTIVE_ONLY]
        # "Exercise is a must" and "Sleep is needed for recovery", but not "It's a must that you ..." or "It is
        # needed to ..."
        if any(c.dep_ in ('ccomp', 'xcomp') for c in token.children) or any(
                c.dep_ in ('nsubj', 'expl') and c.lower_ == 'it' for c in token.head.children):
            choices = [c for c in choices if c not in ('a must', 'needed')]
        # "It is crucial for schools to ..." can't become "It is central for schools to ..."; "central to X" can
        if not any(c.dep_ == 'prep' and c.lower_ == 'to' for c in token.children):
            choices = [c for c in choices if c != 'central']
    if token.pos_ == 'ADJ' and token.lemma_.lower() in ('substantial', 'significant') and token.dep_ == 'amod':
        head = token.head.lemma_.lower()
        if token.lemma_.lower() == 'significant' and (head in STATISTICS_NOUNS or (
                head in MEASURED_NOUNS and STATISTICAL.search(token.sent.text))):
            choices = []
        elif head in SIZE_NOUNS:
            choices = [c for c in choices if c == 'large']
        elif head in WEIGHT_NOUNS:
            choices = [c for c in choices if c == 'major']
        else:
            choices = []        # "substantial evidence" is neither "large" nor "major" evidence
    if 'raise' in choices and not any(c.dep_ == 'dobj' and c.lemma_.lower() in RAISED_OBJECTS for c in token.children):
        choices = [c for c in choices if c != 'raise' or token.lemma_.lower() != 'pose']
    if token.pos_ == 'VERB' and nxt is not None:
        choices = [c for c in choices if ' ' in c or nxt.lower_ not in PARTICLE_CLASHES.get(c, ())]
    if token.pos_ == 'ADV':
        comparative = token.head.tag_ in ('JJR', 'RBR') or (nxt is not None and (
            nxt.tag_ in ('JJR', 'RBR') or nxt.lower_ in ('more', 'less', 'fewer')))
        if comparative:
            choices = [c for c in choices if c in COMPARATIVE_DEGREE]
        elif token.head.pos_ == 'VERB' and token.i < token.head.i:
            choices = [c for c in choices if c not in NOT_BEFORE_VERB]
    if ({'tap', 'tap into'} & set(choices)) and not any(
            c.dep_ == 'dobj' and (c.lemma_.lower() in TAP_OBJECTS or c.lower_ in TAP_OBJECTS) for c in token.children):
        choices = [c for c in choices if c not in ('tap', 'tap into')]
    return choices

def curated_choices(options, tone, token):
    """An AI word's replacements in this tone that fit the token's place; when none of this tone's do, the
    other formal tones' ("improve ... and improve" -> "strengthen")."""
    choices = context_choices(options.get(tone) or options['*'], token)
    if choices:
        return choices
    wider = [c for t, opts in options.items() if t != 'casual' or tone == 'casual' for c in opts]
    return context_choices(list(dict.fromkeys(wider)), token)

# Plain-English swaps Academic doesn't make (its readers expect the technical word), and the research verbs
# Professional keeps ("This study looks into ..." is too chatty for a report), with "enormous", whose "huge" is too
# casual for it ("a profession that demands huge energy")
ACADEMIC_KEEPS = {'significant|ADJ', 'substantial|ADJ', 'fundamental|ADJ', 'effective|ADJ', 'traditional|ADJ',
                  'analyze|VERB', 'analyse|VERB', 'investigate|VERB', 'examine|VERB', 'identify|VERB'}
TONE_KEEPS = {'academic': ACADEMIC_KEEPS,
              'professional': frozenset({'analyze|VERB', 'analyse|VERB', 'examine|VERB', 'investigate|VERB',
                                         'enormous|ADJ'})}
# Verbs that state how strong the evidence for a claim is: their synonyms change it ("the findings indicate"
# is neither "show" nor "suggest"; "a widely cited review" isn't "widely mentioned"), so they are never swapped
CLAIM_VERBS = {'indicate', 'suggest', 'imply', 'cite'}

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
    'mitigate|VERB': {'*': ['reduce', 'ease'], 'academic': ['reduce', 'lessen']},
    'cornerstone|NOUN': {'*': ['foundation']},
    # The replacements avoid "key", "vital" and "essential", which are on the same lists of overused words
    'crucial|ADJ': {'standard': ['important', 'major'], 'casual': ['important', 'big'], 'professional': ['important', 'major'],
                    'academic': ['important', 'central']},
    'pivotal|ADJ': {'standard': ['central', 'major'], 'casual': ['big', 'major'], 'professional': ['central', 'major'],
                    'academic': ['central']},
    'paramount|ADJ': {'*': ['all-important', 'most important'], 'casual': ['huge', 'all-important']},
    # A necessity stays a necessity ("essential for most living organisms" isn't merely "important")
    'essential|ADJ': {'*': ['necessary'], 'casual': ['necessary', 'a must']},
    'vital|ADJ': {'*': ['necessary']},
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
    'ensure|VERB': {'*': ['make sure'], 'academic': ['ensure']},
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
    'pressing|ADJ': {'*': ['urgent']},          # ("serious" is a different point: grave, not urgent)
    'decisive|ADJ': {'*': ['firm', 'strong'], 'academic': ['firm', 'resolute']},
    'viable|ADJ': {'*': ['workable', 'practical'], 'academic': ['feasible', 'practical']},
    'valuable|ADJ': {'*': ['useful', 'helpful'], 'academic': ['useful', 'important']},
    'utmost|ADJ': {'*': ['greatest', 'highest']},
    'tirelessly|ADV': {'*': ['hard', 'nonstop'], 'professional': ['hard'], 'academic': ['persistently']},
    'effectively|ADV': {'*': ['well'], 'academic': ['effectively', 'well']},
    'truly|ADV': {'*': ['really'], 'professional': ['genuinely'], 'academic': ['genuinely']},
    'imperative|ADJ': {'*': ['necessary']},
    'facet|NOUN': {'*': ['side', 'part']},
    'profound|ADJ': {'*': ['far-reaching', 'major', 'deep']},
    'exacerbate|VERB': {'*': ['worsen'], 'academic': ['worsen', 'aggravate']},
    'alleviate|VERB': {'*': ['ease', 'relieve']},
    'prevalent|ADJ': {'*': ['common', 'widespread']},
}
# When every option of an AI word is already nearby ("We utilize cookies. You can use the settings page ..."),
# these are tried before the word is kept; every AI word with a single option in some tone has some ("These
# tools facilitate learning and help students ..." in Casual -> "support learning"). A need stays a need, not a
# rule ("Sleep is vital for recovery" -> "needed", never "required")
ECHO_FALLBACKS = {'utilize|VERB': ['employ', 'make use of'], 'utilise|VERB': ['employ', 'make use of'],
                  'leverage|VERB': ['draw on', 'make use of'], 'harness|VERB': ['draw on', 'make use of'],
                  'utilization|NOUN': ['usage'], 'utilisation|NOUN': ['usage'], 'delve|VERB': ['dig', 'look'],
                  'facilitate|VERB': ['support', 'enable'], 'bolster|VERB': ['strengthen', 'shore up'],
                  'endeavor|VERB': ['aim', 'seek'], 'endeavour|VERB': ['aim', 'seek'],
                  'enhance|VERB': ['boost', 'strengthen'], 'streamline|VERB': ['speed up', 'smooth out'],
                  'empower|VERB': ['help', 'support', 'enable'], 'spearhead|VERB': ['head', 'drive'],
                  'underpin|VERB': ['underlie'], 'resonate|VERB': ['strike a chord'],
                  'cultivate|VERB': ['nurture', 'build'], 'exacerbate|VERB': ['aggravate', 'intensify'],
                  'cornerstone|NOUN': ['basis', 'mainstay'], 'pivotal|ADJ': ['major', 'central'],
                  'essential|ADJ': ['needed'], 'vital|ADJ': ['needed'], 'imperative|ADJ': ['needed'],
                  'seamless|ADJ': ['effortless', 'easy'], 'groundbreaking|ADJ': ['original', 'pioneering'],
                  'invaluable|ADJ': ['very useful', 'extremely useful'], 'holistic|ADJ': ['integrated', 'complete'],
                  'transformative|ADJ': ['major', 'far-reaching'], 'seamlessly|ADV': ['easily', 'effortlessly'],
                  'meticulously|ADV': ['thoroughly', 'painstakingly'], 'additionally|ADV': ['further'],
                  'tirelessly|ADV': ['persistently', 'hard'], 'truly|ADV': ['really', 'genuinely'],
                  'effectively|ADV': ['successfully']}

# Formal words with plain everyday equivalents, the kind of edit a copy editor makes.
# Every swap here reads correctly in almost any sentence, so they can be applied freely.
PLAIN_WORDS = {
    'individual|NOUN': ['person'],
    'numerous|ADJ': ['many'],
    'approximately|ADV': ['about', 'around'],
    'sufficient|ADJ': ['enough'],
    'require|VERB': ['need'],
    'demonstrate|VERB': ['show'],
    'assist|VERB': ['help'],
    'obtain|VERB': ['get'],
    'purchase|VERB': ['buy'],
    'attempt|VERB': ['try'],
    'inquire|VERB': ['ask'],
    'possess|VERB': ['have'],
    'reside|VERB': ['live'],
    'modify|VERB': ['change'],
    'eliminate|VERB': ['remove'],          # "cut" means reduce: "the new law cut the tax" isn't "removed" it
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
    'comprehend|VERB': ['understand'],
    'expedite|VERB': ['speed up'],
    'procure|VERB': ['get', 'obtain'],
    'disseminate|VERB': ['spread'],         # not "share", which takes "with" ("share the findings to ...")
    'entail|VERB': ['involve'],
    'delineate|VERB': ['outline'],
    'enumerate|VERB': ['list'],
    'reiterate|VERB': ['repeat', 'restate'],
    'adhere|VERB': ['stick'],
    'ameliorate|VERB': ['improve'],
    'exemplify|VERB': ['show', 'illustrate'],
    'predominantly|ADV': ['mainly', 'mostly'],
    'readily|ADV': ['easily'],
    'swiftly|ADV': ['quickly'],
    'markedly|ADV': ['noticeably'],
    'substantially|ADV': ['greatly', 'considerably'],
    'enormous|ADJ': ['huge'],
    'akin|ADJ': ['similar'],
    'adept|ADJ': ['skilled'],
    'proficient|ADJ': ['skilled'],
    'arduous|ADJ': ['difficult', 'hard'],
    'detrimental|ADJ': ['harmful'],
    'advantageous|ADJ': ['helpful', 'useful'],
    'pertinent|ADJ': ['relevant'],
    'requisite|ADJ': ['required', 'necessary'],
    'ubiquitous|ADJ': ['widespread'],
    'salient|ADJ': ['main', 'key'],
    'subsequent|ADJ': ['later'],
    'dramatically|ADV': ['sharply'],
    'encourage|VERB': ['promote'],
    'assistance|NOUN': ['help'],
    'methodology|NOUN': ['method'],
    'objective|NOUN': ['goal', 'aim'],
    'component|NOUN': ['part'],
    'nevertheless|ADV': ['still'],
    'nonetheless|ADV': ['still'],
    'reduce|VERB': ['cut', 'lower', 'ease', 'cut down on'],    # (each for its own objects: see reduce_fits)
    'pose|VERB': ['present', 'raise'],
    'provide|VERB': ['give', 'offer'],
    'organize|VERB': ['arrange', 'plan'],
    'implement|VERB': ['apply', 'use'],
    'establish|VERB': ['build', 'set up'],
    'identify|VERB': ['spot', 'find'],
    # not "look at": an audit or an analysis that was "looked at" sounds casual and vaguer
    'examine|VERB': ['study'],
    'analyze|VERB': ['study', 'examine'],
    'analyse|VERB': ['study', 'examine'],
    'investigate|VERB': ['look into', 'study', 'explore'],
    'acquire|VERB': ['get', 'gain'],
    'emphasize|VERB': ['stress', 'highlight'],
    'highlight|VERB': ['show'],
    'enable|VERB': ['allow'],
    # "regular" for frequent/consistent means periodic, "working" for effective and "keep" for maintain
    # ("keep your health") read wrongly, and "big" is too casual for most nouns after "significant"
    'significant|ADJ': ['major', 'large'],
    'substantial|ADJ': ['large', 'major'],
    'various|ADJ': ['different', 'several'],
    'fundamental|ADJ': ['basic', 'core'],
    'appropriate|ADJ': ['suitable', 'fitting'],
    'consistent|ADJ': ['steady'],
    'traditional|ADJ': ['conventional', 'standard'],
    # (not frequent -> "common": "frequent travel" happens often, "common travel" is widespread)
    'immediately|ADV': ['right away', 'at once'],
    'primarily|ADV': ['mainly', 'mostly'],
    'particularly|ADV': ['especially'],
    'typically|ADV': ['usually'],
    'rapidly|ADV': ['quickly', 'fast'],
    'continuously|ADV': ['constantly'],
    'nearly|ADV': ['almost'],
    'hoist|VERB': ['raise'],
}
# Chance of making a plain-English swap; Academic keeps its formal vocabulary
PLAIN_CHANCE = {'casual': 1.0, 'standard': 0.85, 'professional': 0.5, 'academic': 0.0}
IRREGULAR_PLURALS = {'person': 'people'}
# ...except the plain words a paper uses too ("demonstrate" -> "show", "numerous" -> "many", "subsequently" ->
# "later"): Academic swaps these at ACADEMIC_PLAIN_CHANCE (in Chaos Mode wherever they fit), and its last-resort swap
# (force_word_edit) takes only these, never a casual word or a loose one ("obtain" stays, not "get"; "implement a
# scheme", "establish a panel" and "eliminated smallpox" keep the paper's word, and so does "initial", a term in
# "initial conditions", "the initial velocity" and "the initial investment")
ACADEMIC_PLAIN = {
    'numerous|ADJ': ['many'], 'demonstrate|VERB': ['show'], 'require|VERB': ['need'], 'assist|VERB': ['help'],
    'purchase|VERB': ['buy'], 'attempt|VERB': ['try'], 'modify|VERB': ['change'],
    'terminate|VERB': ['end'], 'anticipate|VERB': ['expect'], 'frequently|ADV': ['often'],
    'subsequently|ADV': ['later'], 'currently|ADV': ['now'], 'illustrate|VERB': ['show'], 'exhibit|VERB': ['show'],
    'necessitate|VERB': ['need'], 'retain|VERB': ['keep'], 'cease|VERB': ['stop'], 'comprehend|VERB': ['understand'],
    'disseminate|VERB': ['spread'], 'entail|VERB': ['involve'], 'delineate|VERB': ['outline'],
    'enumerate|VERB': ['list'], 'reiterate|VERB': ['repeat', 'restate'], 'ameliorate|VERB': ['improve'],
    'exemplify|VERB': ['illustrate'], 'predominantly|ADV': ['mainly', 'mostly'], 'readily|ADV': ['easily'],
    'swiftly|ADV': ['quickly'], 'substantially|ADV': ['considerably'], 'akin|ADJ': ['similar'],
    'adept|ADJ': ['skilled'], 'proficient|ADJ': ['skilled'], 'arduous|ADJ': ['difficult'],
    'detrimental|ADJ': ['harmful'], 'advantageous|ADJ': ['useful'], 'pertinent|ADJ': ['relevant'],
    'requisite|ADJ': ['required', 'necessary'], 'ubiquitous|ADJ': ['widespread'], 'salient|ADJ': ['main'],
    'subsequent|ADJ': ['later'], 'encourage|VERB': ['promote'], 'assistance|NOUN': ['help'], 'objective|NOUN': ['aim'],
    'component|NOUN': ['part'], 'pose|VERB': ['present', 'raise'],
    'emphasize|VERB': ['stress'], 'highlight|VERB': ['show'], 'enable|VERB': ['allow'],
    'appropriate|ADJ': ['suitable'], 'primarily|ADV': ['mainly'], 'particularly|ADV': ['especially'],
    'typically|ADV': ['usually'], 'rapidly|ADV': ['quickly'], 'continuously|ADV': ['constantly'],
    'nearly|ADV': ['almost'], 'reduce|VERB': ['lower'], 'dramatically|ADV': ['sharply'], 'organize|VERB': ['arrange'],
    'acquire|VERB': ['gain'], 'procure|VERB': ['obtain'], 'hoist|VERB': ['raise'], 'individual|NOUN': ['person'],
    'expedite|VERB': ['speed up'],
}
ACADEMIC_PLAIN_CHANCE = 0.35

# --- More curated swaps: AI-favourite words and everyday words in stock frames ---
# AI text leans on a few showy words ("stunning views", "remarkable growth", "renowned for its tea houses", "delivers
# exceptional sound") and on everyday words in stock frames ("offers benefits", "discover that", "remained stable",
# "occasionally"). Each has a plain equivalent in some of its senses only, so every entry below holds
# only where swap_choices finds that sense in the parse ("a stunning defeat", "exceptional circumstances", "offered
# her a job", "contained the blaze" keep their words). They are kept out of AI_WORDS and PLAIN_WORDS on purpose:
# the Deep Rewrite check counts those tables' replacements as keeping a describing word, and lets their adverbs be
# dropped, which swaps that hold in one sense only must not loosen.
# AI-favourite words, replaced wherever they fit, like AI_WORDS (an empty list keeps the word in that tone)
AI_FAVOURITES = {
    'stunning|ADJ': {'standard': ['beautiful', 'striking'], 'casual': ['gorgeous', 'beautiful'],
                     'professional': ['striking', 'beautiful'], 'academic': ['striking']},
    'breathtaking|ADJ': {'*': ['spectacular', 'magnificent']},
    'remarkable|ADJ': {'*': ['striking', 'impressive', 'extraordinary'], 'academic': ['striking', 'notable']},
    'remarkably|ADV': {'*': ['unusually', 'strikingly']},
    'renowned|ADJ': {'*': ['famous', 'well-known'], 'academic': ['well-known', 'noted']},
    'exceptional|ADJ': {'*': ['excellent', 'outstanding']},
    'impressive|ADJ': {'*': ['strong', 'solid'], 'academic': ['strong']},
    'notably|ADV': {'*': ['especially', 'particularly', 'noticeably'], 'academic': ['particularly', 'noticeably']},
    'ultimately|ADV': {'*': ['eventually', 'finally']},
    'nestle|VERB': {'*': ['tuck away', 'set']},
    'bustling|ADJ': {'*': ['busy', 'lively']},
    'unforgettable|ADJ': {'*': ['memorable']},
    'unparalleled|ADJ': {'*': ['unmatched']},
    'intuitive|ADJ': {'*': ['easy-to-use', 'simple']},
    'compelling|ADJ': {'*': ['convincing', 'strong', 'gripping']},
    'impactful|ADJ': {'*': ['effective', 'powerful']},
    'effortlessly|ADV': {'*': ['easily']},
    'undoubtedly|ADV': {'*': ['certainly', 'no doubt'], 'academic': ['certainly']},
    'profoundly|ADV': {'*': ['deeply']},
    'incredibly|ADV': {'*': ['extremely'], 'casual': ['really', 'extremely']},
    'vast|ADJ': {'*': ['huge', 'great'], 'academic': ['large', 'great']},
    'thrive|VERB': {'*': ['do well', 'flourish'], 'academic': ['flourish']},
    'strive|VERB': {'*': ['aim', 'try'], 'academic': ['aim', 'seek']},
    'impact|VERB': {'*': ['affect']},
    'unwavering|ADJ': {'*': ['steady', 'firm']},
    'esteemed|ADJ': {'*': ['respected']},
    'bespoke|ADJ': {'*': ['custom', 'tailor-made']},
    'unrivalled|ADJ': {'*': ['unmatched']},
    'unrivaled|ADJ': {'*': ['unmatched']},
    'picturesque|ADJ': {'*': ['pretty', 'scenic'], 'professional': ['scenic'], 'academic': ['scenic']},
    'serene|ADJ': {'*': ['peaceful', 'calm']},
    'tranquil|ADJ': {'*': ['peaceful', 'calm']},
    'iconic|ADJ': {'*': ['famous']},
    'captivating|ADJ': {'*': ['gripping']},
}
# Everyday and formal words with a plainer or fresher equivalent in some senses, swapped at EVERYDAY_CHANCE (in Chaos
# Mode wherever they fit). An empty list keeps the word in that tone ("remained stable" stays in a paper)
EVERYDAY_SWAPS = {
    'begin|VERB': {'*': ['start']},
    'deliver|VERB': {'*': ['provide', 'produce', 'give']},
    'discover|VERB': {'*': ['find', 'find out'], 'academic': ['find']},
    'offer|VERB': {'*': ['have', 'give']},
    'feature|VERB': {'*': ['have', 'include']},
    'contain|VERB': {'*': ['have', 'hold'], 'academic': []},
    'recommend|VERB': {'*': ['suggest', 'advise']},
    'caution|VERB': {'*': ['warn']},
    'benefit|VERB': {'*': ['gain', 'help']},
    'remain|VERB': {'*': ['stay'], 'academic': []},
    'locate|VERB': {'*': ['find']},
    'participate|VERB': {'*': ['take part']},
    'collaborate|VERB': {'*': ['work', 'work together']},
    'relocate|VERB': {'*': ['move']},
    'construct|VERB': {'*': ['build']},
    'manufacture|VERB': {'*': ['make']},
    'produce|VERB': {'*': ['make', 'give off']},
    'select|VERB': {'*': ['choose', 'pick'], 'professional': ['choose'], 'academic': ['choose']},
    'request|VERB': {'*': ['ask for', 'ask']},
    'notify|VERB': {'*': ['inform', 'tell'], 'professional': ['inform'], 'academic': ['inform']},
    'inform|VERB': {'*': ['tell'], 'academic': []},
    'submit|VERB': {'*': ['send']},
    'seek|VERB': {'*': ['try', 'aim', 'get'], 'professional': ['aim'], 'academic': ['aim']},
    'persist|VERB': {'*': ['last', 'continue']},
    'occur|VERB': {'*': ['happen'], 'academic': []},
    'depict|VERB': {'*': ['show']},
    'display|VERB': {'*': ['show']},
    'reveal|VERB': {'*': ['show']},
    'state|VERB': {'*': ['say'], 'academic': []},
    'note|VERB': {'*': ['point out'], 'academic': ['observe']},
    'alter|VERB': {'*': ['change']},
    'employ|VERB': {'*': ['use']},
    'incorporate|VERB': {'*': ['include']},
    'conclude|VERB': {'*': ['end']},
    'achieve|VERB': {'*': ['reach']},
    'resolve|VERB': {'*': ['fix', 'settle']},
    'address|VERB': {'*': ['tackle', 'deal with']},
    'minimize|VERB': {'*': ['cut', 'lower']},
    'minimise|VERB': {'*': ['cut', 'lower']},
    'generate|VERB': {'*': ['produce', 'bring in'], 'academic': ['produce']},
    'conduct|VERB': {'*': ['carry out', 'run']},
    'host|VERB': {'*': ['hold']},
    'attract|VERB': {'*': ['draw']},
    'accommodate|VERB': {'*': ['hold']},
    'expand|VERB': {'*': ['extend', 'grow']},
    'monitor|VERB': {'*': ['track']},
    'maintain|VERB': {'*': ['keep']},
    'receive|VERB': {'*': ['get'], 'professional': [], 'academic': []},
    'transform|VERB': {'*': ['turn', 'reshape']},
    'increase|VERB': {'*': ['raise', 'rise', 'grow', 'add to']},
    'decrease|VERB': {'*': ['lower', 'cut', 'fall', 'drop']},
    'explore|VERB': {'*': ['look at'], 'academic': ['examine', 'consider']},
    'complete|VERB': {'*': ['finish']},
    'repair|VERB': {'*': ['fix'], 'academic': []},
    'rise|VERB': {'*': ['climb', 'go up'], 'professional': ['climb'], 'academic': []},
    'fall|VERB': {'*': ['drop', 'go down'], 'professional': ['drop'], 'academic': []},
    'launch|VERB': {'*': ['start', 'open']},
    'demand|VERB': {'*': ['call for'], 'academic': []},
    'invariably|ADV': {'*': ['always']},
    'occasionally|ADV': {'standard': ['sometimes', 'now and then', 'from time to time'],
                         'casual': ['sometimes', 'now and then'],
                         'professional': ['sometimes', 'at times', 'from time to time'],
                         'academic': ['sometimes', 'at times']},
    'roughly|ADV': {'*': ['about', 'around']},
    'largely|ADV': {'*': ['mostly', 'mainly']},
    'solely|ADV': {'*': ['only']},
    'merely|ADV': {'*': ['only', 'just'], 'academic': ['only']},
    'commonly|ADV': {'*': ['often']},
    'promptly|ADV': {'*': ['quickly']},
    'initially|ADV': {'*': ['at first', 'first']},
    'highly|ADV': {'*': ['very']},
    'increasingly|ADV': {'*': ['more and more'], 'professional': [], 'academic': []},
    'considerable|ADJ': {'*': ['large', 'major']},
    'entire|ADJ': {'*': ['whole']},
    'extensive|ADJ': {'*': ['broad', 'large']},
    'rapid|ADJ': {'*': ['fast', 'quick']},
    'swift|ADJ': {'*': ['quick']},
    'primary|ADJ': {'*': ['main']},
    'principal|ADJ': {'*': ['main']},
    'superb|ADJ': {'*': ['excellent']},
    'outstanding|ADJ': {'*': ['excellent']},
    'spacious|ADJ': {'*': ['roomy', 'large'], 'academic': ['large']},
    'perfect|ADJ': {'*': ['ideal']},
    'unique|ADJ': {'*': ['distinctive', 'rare']},
    'beverage|NOUN': {'*': ['drink']},
    'attire|NOUN': {'*': ['clothing']},
    'outcome|NOUN': {'*': ['result']},
    'manner|NOUN': {'*': ['way']},
    'opportunity|NOUN': {'*': ['chance']},
    'aspect|NOUN': {'*': ['part']},     # (not "side": "every side of the course" reads as its edges)
}
# Chance of an everyday swap when the dice decide; Chaos Mode makes every one that fits, in every tone
EVERYDAY_CHANCE = {'standard': 0.5, 'casual': 0.6, 'professional': 0.4, 'academic': 0.3}

# The nouns that select a describing word's plain sense, per replacement. "stunning" is "beautiful" for a view, a
# beach or a building, but "a stunning defeat" is a surprising one (and "a stunning finish" to a race a dramatic one);
# "exceptional sound quality" is "excellent", but "exceptional circumstances" are unusual ones; "impressive results"
# are "strong", but "an impressive menu" isn't
SCENIC_NOUNS = {'view', 'vista', 'scenery', 'landscape', 'panorama', 'sunset', 'sunrise', 'coastline', 'coast', 'beach',
                'bay', 'lake', 'mountain', 'peak', 'valley', 'garden', 'park', 'architecture', 'building', 'cathedral',
                'church', 'palace', 'castle', 'temple', 'pavilion', 'mosque', 'bridge', 'skyline', 'interior',
                'photo', 'photograph', 'painting', 'mural', 'artwork', 'colour', 'color', 'scene', 'setting',
                'backdrop', 'location', 'island', 'waterfall', 'cliff', 'gorge', 'canyon', 'forest', 'countryside',
                'dress', 'gown', 'outfit', 'decor', 'facade', 'ceiling', 'mosaic', 'fresco', 'sculpture', 'statue',
                'home', 'house', 'villa', 'kitchen', 'bathroom', 'terrace', 'balcony', 'lagoon', 'river', 'fjord',
                'glacier', 'harbour', 'harbor', 'square', 'village', 'town', 'city', 'design', 'marble', 'stonework',
                'window', 'glass', 'staircase', 'courtyard', 'fountain', 'tower', 'dome', 'gate'}
# ...but a picture is "stunning" for what it shows as well as for its looks: "a stunning photo of the crash site" is a
# striking one, not a beautiful one
DEPICTION_NOUNS = {'photo', 'photograph', 'painting', 'mural', 'artwork'}
EXCELLENCE_NOUNS = {'quality', 'service', 'value', 'performance', 'experience', 'food', 'meal', 'dish', 'cuisine',
                    'cooking', 'taste', 'flavour', 'flavor', 'sound', 'audio', 'comfort', 'craftsmanship', 'workmanship',
                    'care', 'work', 'job', 'skill', 'talent', 'hospitality', 'teacher', 'student', 'athlete', 'player',
                    'leader', 'chef', 'cook', 'writer', 'scholar', 'employee', 'candidate', 'colleague', 'manager',
                    'team', 'result', 'grade', 'design', 'accuracy', 'clarity', 'reliability', 'durability',
                    'support', 'education', 'teaching', 'writing', 'acting', 'cast', 'menu', 'selection', 'example'}
STRONG_RESULT_NOUNS = {'result', 'performance', 'growth', 'record', 'number', 'figure', 'sale', 'return', 'gain',
                       'showing', 'start', 'debut', 'turnout', 'earnings', 'profit', 'revenue', 'score', 'season',
                       'quarter', 'finish', 'progress'}
# "remarkable": "a striking similarity", "impressive growth", "an extraordinary woman" (a remarkable spot, step or
# menu stays)
STRIKING_NOUNS = {'similarity', 'resemblance', 'contrast', 'difference', 'consistency', 'coincidence', 'example',
                  'feature', 'fact', 'sight', 'change', 'shift', 'transformation', 'increase', 'rise', 'decline', 'fall',
                  'drop', 'degree', 'extent', 'range', 'variety', 'diversity', 'pattern', 'parallel', 'detail'}
ACHIEVEMENT_NOUNS = {'growth', 'progress', 'achievement', 'success', 'feat', 'career', 'result', 'performance',
                     'recovery', 'turnaround', 'record', 'accomplishment', 'effort', 'skill', 'ability', 'talent',
                     'resilience', 'speed', 'accuracy', 'comeback', 'improvement', 'efficiency', 'precision',
                     'durability'}
EXTRAORDINARY_NOUNS = {'woman', 'man', 'person', 'people', 'individual', 'figure', 'life', 'story', 'journey', 'chapter', 'tale',
               'history', 'adventure', 'discovery', 'invention', 'career', 'event', 'moment', 'period', 'era', 'animal',
               'creature', 'voyage', 'expedition'}
# "vast": "the great majority", "a huge network"
VAST_SIZE_NOUNS = {'network', 'amount', 'number', 'collection', 'area', 'range', 'array', 'sum', 'quantity', 'empire',
                   'territory', 'expanse', 'desert', 'ocean', 'forest', 'plain', 'wealth', 'fortune', 'library',
                   'archive', 'database', 'scale', 'crowd', 'army', 'estate', 'region', 'continent', 'distance',
                   'reserve', 'supply', 'volume', 'complex', 'market', 'audience', 'selection', 'catalogue', 'catalog',
                   'sea', 'space', 'site', 'field', 'swathe', 'swath', 'stretch', 'tract', 'herd', 'flock'}
# "an intuitive interface" is easy to use ("the intuitive system" of psychology is the fast way of thinking)
INTERFACE_NOUNS = {'interface', 'design', 'app', 'application', 'control', 'layout', 'menu', 'navigation', 'dashboard',
                   'touchscreen', 'screen', 'software', 'tool', 'platform', 'program', 'website', 'site',
                   'setup', 'editor', 'keyboard', 'remote', 'button', 'workflow', 'display'}
ARGUMENT_NOUNS = {'argument', 'evidence', 'case', 'reason', 'data', 'datum', 'explanation', 'proof', 'rationale',
                  'justification', 'answer', 'finding'}
NARRATIVE_NOUNS = {'story', 'narrative', 'read', 'novel', 'film', 'movie', 'drama', 'book', 'documentary', 'thriller',
                   'plot', 'series', 'show', 'podcast', 'account', 'tale', 'memoir', 'biography', 'performance'}
# ("a broad range of skills", but "a large network of cycle paths": a network is only large)
BROAD_KNOWLEDGE_NOUNS = {'experience', 'knowledge', 'range', 'background', 'expertise', 'repertoire', 'portfolio',
               'understanding', 'skill', 'vocabulary'}
SIZED_COLLECTIONS = {'collection', 'selection', 'menu', 'list', 'catalogue', 'catalog', 'library', 'archive', 'network',
                     'database', 'garden', 'grounds'}
# "rapid growth" -> "fast growth", but "a rapid test" and "rapid transit" are terms, and "fast" doesn't go with a
# process that unfolds ("fast expansion", "fast development" and "fast urbanisation" aren't said; "a quick transition" is)
RAPID_CHANGE_NOUNS = {'growth', 'change', 'increase', 'rise', 'decline', 'fall', 'progress', 'spread', 'pace', 'shift',
               'improvement', 'recovery', 'decrease', 'drop', 'heartbeat', 'pulse', 'breathing', 'turnover', 'response'}
QUICK_CHANGE_NOUNS = {'change', 'shift', 'transition', 'rollout', 'recovery', 'improvement', 'decline', 'response', 'succession',
               'progress', 'rise'}
# "the primary reason" -> "the main reason", but a primary school, primary care, a primary source and a trial's
# primary outcome are terms
MAIN_PURPOSE_NOUNS = {'reason', 'goal', 'aim', 'objective', 'purpose', 'focus', 'concern', 'cause', 'factor', 'driver', 'role',
              'function', 'responsibility', 'task', 'challenge', 'issue', 'motivation', 'benefit', 'advantage', 'risk',
              'target', 'means', 'method', 'way', 'use', 'interest', 'emphasis', 'obstacle', 'barrier', 'difference',
              'problem', 'question', 'theme', 'argument', 'contribution', 'finding', 'priority'}
# "the perfect spot" -> "the ideal spot", but a perfect score, circle or tense is a perfect one (and "conditions" only
# in the plural: "in perfect condition" is undamaged, not suitable)
IDEAL_PLACE_NOUNS = {'place', 'spot', 'space', 'setting', 'location', 'venue', 'base', 'gift', 'choice', 'option', 'companion',
               'partner', 'solution', 'time', 'way', 'opportunity', 'destination', 'accompaniment', 'match', 'addition',
               'size', 'candidate', 'tool', 'introduction', 'escape', 'retreat', 'getaway', 'weekend', 'day',
               'temperature', 'condition', 'environment', 'home', 'stop', 'pairing', 'complement'}
# "unique": "a distinctive style", "a rare opportunity" (a unique identifier or code stays)
DISTINCTIVE_NOUNS = {'style', 'character', 'flavour', 'flavor', 'taste', 'charm', 'atmosphere', 'design', 'look', 'voice',
                     'sound', 'identity', 'personality', 'culture', 'architecture', 'feel', 'aroma', 'scent', 'shape',
                     'pattern', 'appearance', 'blend', 'mix', 'combination'}
RARE_CHANCE_NOUNS = {'opportunity', 'chance', 'glimpse'}
# "unwavering commitment" -> "firm commitment", "an unwavering gaze" -> "a steady gaze" ("firm dedication" and "steady
# loyalty" aren't said, and the staff's "firm support" reads as strict)
COMMITMENT_NOUNS = {'commitment', 'belief', 'resolve', 'stance', 'position', 'conviction', 'principle', 'opposition'}
STEADY_GAZE_NOUNS = {'gaze', 'hand', 'voice', 'focus', 'presence', 'support', 'commitment', 'pace'}
# "spacious" is "roomy" for an inside space or a thing that holds things ("a roomy boot", "roomy suites"), not for a
# balcony or a garden ("large"), nor for a room ("roomy rooms")
ROOMY_NOUNS = {'suite', 'cabin', 'boot', 'trunk', 'interior', 'bag', 'backpack', 'rucksack', 'suitcase', 'compartment',
               'pocket', 'wardrobe', 'cupboard', 'car', 'van', 'tent', 'seat', 'apartment', 'flat', 'house', 'home',
               'cottage', 'villa', 'lodge', 'caravan', 'motorhome', 'kitchen', 'hall', 'hallway', 'loft', 'attic', 'shed',
               'garage', 'office', 'studio', 'fridge', 'fit', 'jacket', 'coat'}
# "serene temples" -> "peaceful temples", "tranquil waters" -> "calm waters" ("serene confidence" stays)
QUIET_PLACE_NOUNS = SCENIC_NOUNS | {'temple', 'atmosphere', 'spot', 'place', 'retreat', 'setting', 'surroundings',
                                    'environment', 'neighbourhood', 'neighborhood', 'street', 'corner', 'area', 'pond',
                                    'cove', 'escape', 'oasis', 'haven', 'sanctuary', 'life', 'morning', 'evening'}
CALM_WATER_NOUNS = {'water', 'sea', 'lake', 'pond', 'river', 'bay', 'lagoon', 'morning', 'evening', 'atmosphere', 'mood',
              'expression', 'face', 'smile', 'voice'}
# "the iconic Eiffel Tower" -> "the famous Eiffel Tower" (an iconic sign, in the study of signs, stays)
FAMOUS_THING_NOUNS = {'landmark', 'brand', 'logo', 'dish', 'song', 'album', 'film', 'movie', 'photo', 'monument', 'figure',
                      'character', 'scene', 'moment', 'car', 'guitar', 'restaurant', 'hotel', 'product', 'shoe', 'bag',
                      'poster', 'store', 'shop', 'band', 'singer', 'actor', 'tune', 'line', 'quote', 'speech', 'game',
                      'team', 'club', 'stadium', 'route', 'trail', 'market', 'cafe', 'pub', 'bar', 'museum', 'gallery'}
# A describing word's replacements and the nouns each one fits (as swap_described_noun finds them)
DESCRIBED_FITS = {
    'stunning|ADJ': {'beautiful': SCENIC_NOUNS - DEPICTION_NOUNS, 'striking': SCENIC_NOUNS,
                     'gorgeous': SCENIC_NOUNS - DEPICTION_NOUNS},
    'breathtaking|ADJ': {'spectacular': SCENIC_NOUNS, 'magnificent': SCENIC_NOUNS},
    'remarkable|ADJ': {'striking': STRIKING_NOUNS, 'impressive': ACHIEVEMENT_NOUNS, 'extraordinary': EXTRAORDINARY_NOUNS,
                       'notable': STRIKING_NOUNS | ACHIEVEMENT_NOUNS},
    'exceptional|ADJ': {'excellent': EXCELLENCE_NOUNS, 'outstanding': EXCELLENCE_NOUNS},
    # "outstanding work", "three outstanding jobs" and "the outstanding results" may be still to be done or reported
    'outstanding|ADJ': {'excellent': EXCELLENCE_NOUNS - {'work', 'job', 'result'}},
    'impressive|ADJ': {'strong': STRONG_RESULT_NOUNS, 'solid': STRONG_RESULT_NOUNS},
    'vast|ADJ': {'great': {'majority'}, 'huge': VAST_SIZE_NOUNS, 'large': VAST_SIZE_NOUNS},
    'intuitive|ADJ': {'easy-to-use': INTERFACE_NOUNS, 'simple': INTERFACE_NOUNS},
    'compelling|ADJ': {'convincing': ARGUMENT_NOUNS, 'strong': ARGUMENT_NOUNS, 'gripping': NARRATIVE_NOUNS},
    'extensive|ADJ': {'broad': BROAD_KNOWLEDGE_NOUNS, 'large': SIZED_COLLECTIONS},
    'rapid|ADJ': {'fast': RAPID_CHANGE_NOUNS, 'quick': QUICK_CHANGE_NOUNS},
    'primary|ADJ': {'main': MAIN_PURPOSE_NOUNS},
    'principal|ADJ': {'main': MAIN_PURPOSE_NOUNS},
    'perfect|ADJ': {'ideal': IDEAL_PLACE_NOUNS},
    'unique|ADJ': {'distinctive': DISTINCTIVE_NOUNS, 'rare': RARE_CHANCE_NOUNS},
    'considerable|ADJ': {'large': SIZE_NOUNS - {'cost'}, 'major': WEIGHT_NOUNS},
    'unwavering|ADJ': {'steady': STEADY_GAZE_NOUNS, 'firm': COMMITMENT_NOUNS},
    'serene|ADJ': {'peaceful': QUIET_PLACE_NOUNS, 'calm': CALM_WATER_NOUNS},
    'tranquil|ADJ': {'peaceful': QUIET_PLACE_NOUNS, 'calm': CALM_WATER_NOUNS},
    'iconic|ADJ': {'famous': SCENIC_NOUNS | FAMOUS_THING_NOUNS},
    'captivating|ADJ': {'gripping': NARRATIVE_NOUNS},
}
# Replacements that read well after "more"/"most" ("the most striking chapter"); the others don't ("more strong")
MOST_FORM_DESCRIBERS = {'beautiful', 'striking', 'famous', 'memorable', 'extraordinary', 'spectacular', 'magnificent',
                'convincing', 'gripping', 'effective', 'powerful', 'distinctive', 'lively', 'gorgeous', 'notable'}

# The objects that select a verb's plain sense, per replacement (the object, or a passive's subject). A product that
# "offers 15 settings" has them, a tour that "offers a glimpse" gives one; "headphones deliver great sound" produce
# it, a teacher "delivers a lesson" gives it, but a courier delivers a parcel and a midwife a baby
OFFER_HAVE_NOUNS = {'view', 'wardrobe', 'parking', 'facility', 'amenity', 'room', 'suite', 'bedroom', 'bathroom',
                    'kitchen', 'garden', 'terrace', 'balcony', 'pool', 'gym', 'spa', 'restaurant', 'bar', 'cafe', 'shop',
                    'beach', 'trail', 'feature', 'function', 'option', 'setting', 'mode', 'menu', 'range', 'selection',
                    'choice', 'variety', 'mix', 'something', 'everything', 'plenty', 'port', 'slot', 'storage', 'space',
                    'seating', 'screen', 'display', 'camera', 'warranty', 'plan', 'tier', 'version', 'model', 'size',
                    'colour', 'color', 'flavour', 'flavor', 'dish', 'wine', 'seafood', 'course', 'class', 'program',
                    'programme', 'degree', 'module', 'benefit', 'advantage', 'perk', 'discount', 'deal', 'tour',
                    'activity', 'attraction', 'museum', 'gallery', 'market', 'landmark', 'temple', 'castle', 'church',
                    'entry', 'admission', 'breakfast', 'shuttle', 'service', 'delivery', 'shipping', 'wifi', 'lounge',
                    'playground', 'library', 'cinema', 'sauna', 'pitch', 'court', 'route', 'walk', 'hike'}
OFFER_GIVE_NOUNS = {'glimpse', 'introduction', 'overview', 'insight', 'chance', 'opportunity', 'look', 'sense', 'taste',
                    'break', 'reminder', 'lesson', 'example', 'answer', 'boost', 'edge', 'peace', 'comfort', 'relief',
                    'protection', 'hope', 'reassurance', 'perspective', 'snapshot', 'preview', 'advantage', 'guidance',
                    'advice', 'support', 'help', 'picture', 'clue', 'hint'}
DELIVER_PROVIDE_NOUNS = {'service', 'care', 'support', 'experience', 'value', 'training', 'education', 'protection',
                         'coverage', 'solution', 'benefit', 'information', 'help', 'advice', 'feedback', 'guidance',
                         'healthcare', 'treatment', 'programme', 'program', 'course', 'aid', 'relief', 'insight'}
# (only what is turned out, not a quality: headphones "deliver sound quality" and a router "fast speeds", but they
# don't produce them)
DELIVER_PRODUCE_NOUNS = {'result', 'outcome', 'sound', 'picture', 'image', 'power', 'output', 'return',
                         'growth', 'improvement', 'gain', 'saving', 'profit', 'revenue', 'torque',
                         'bass', 'reduction', 'increase'}
# (a performance is given by a performer or a team: "The orchestra delivered a stunning performance" -> "gave", but a
# chip that "delivers fast performance" keeps its word)
DELIVER_GIVE_NOUNS = {'speech', 'talk', 'lecture', 'presentation', 'address', 'sermon', 'keynote', 'eulogy', 'lesson',
                      'class', 'workshop', 'seminar', 'briefing', 'update', 'verdict', 'warning', 'message', 'statement',
                      'performance', 'reading', 'toast', 'ultimatum'}
ADDRESSED_PROBLEMS = {'issue', 'problem', 'concern', 'challenge', 'need', 'gap', 'shortage', 'question', 'crisis', 'risk',
                 'cause', 'threat', 'barrier', 'inequality', 'rent', 'price', 'cost', 'inflation', 'poverty',
                 'homelessness', 'unemployment', 'crime', 'pollution', 'emission', 'deficit', 'debt', 'backlog', 'delay',
                 'complaint', 'weakness', 'imbalance', 'bias', 'disparity', 'misconception', 'flaw', 'shortcoming',
                 'loneliness', 'congestion', 'obesity', 'waste', 'vulnerability', 'bottleneck'}
CONDUCTED_STUDIES = {'survey', 'study', 'research', 'interview', 'test', 'experiment', 'review', 'audit', 'analysis',
               'investigation', 'inspection', 'assessment', 'evaluation', 'trial', 'search', 'check', 'poll', 'census',
               'exercise', 'consultation', 'questionnaire', 'simulation', 'drill'}
RUN_STUDIES = {'survey', 'test', 'trial', 'study', 'experiment', 'poll', 'workshop', 'session', 'class', 'course',
             'check', 'search', 'simulation', 'drill', 'campaign', 'pilot'}
# (a presenter hosts a show or a quiz, and a home side hosts a match or a game: neither "holds" it)
HOSTED_EVENTS = {'event', 'festival', 'conference', 'meeting', 'workshop', 'competition', 'tournament',
                   'olympics', 'cup', 'championship', 'concert', 'exhibition', 'fair', 'market', 'party', 'dinner',
                   'reception', 'ceremony', 'summit', 'forum', 'talk', 'session', 'class', 'course', 'debate',
                   'parade', 'auction', 'screening', 'gala', 'contest', 'convention', 'expo'}
GENERATED_OUTPUT = {'electricity', 'power', 'energy', 'heat', 'emission', 'carbon', 'dioxide', 'waste', 'gas', 'noise',
                'light', 'output', 'signal', 'steam', 'pollution', 'smoke'}
EARNED_MONEY = {'revenue', 'income', 'profit', 'sale', 'cash', 'money', 'return', 'fund', 'funding', 'earnings', 'fee',
               'donation'}
EMPLOYED_METHODS = {'technique', 'method', 'approach', 'strategy', 'tool', 'model', 'design', 'system', 'tactic', 'measure',
                'metaphor', 'device', 'algorithm', 'procedure', 'process', 'framework', 'language', 'style', 'material',
                'test', 'scale', 'sample', 'analysis', 'software', 'instrument'}
ACHIEVED_TARGETS = {'goal', 'target', 'aim', 'objective', 'milestone', 'agreement', 'consensus', 'level', 'standard', 'score',
                'balance', 'peak', 'zero', 'compromise', 'deal', 'settlement', 'threshold', 'total', 'figure', 'record',
                'audience', 'quota', 'limit', 'height'}
RESOLVED_FAULTS = {'issue', 'problem', 'bug', 'error', 'fault', 'glitch', 'defect', 'leak', 'crash'}
# (a detective who "resolved the case" solved it; a settled case was ended by an agreement)
SETTLED_DISPUTES = {'dispute', 'conflict', 'disagreement', 'argument', 'claim', 'matter', 'quarrel', 'row', 'strike'}
MINIMIZED_COSTS = {'cost', 'spending', 'waste', 'expense', 'delay', 'downtime', 'time', 'loss', 'error',
              'paperwork', 'travel', 'use', 'usage', 'consumption', 'footprint'}
MINIMIZED_RISKS = {'risk', 'chance', 'impact', 'effect', 'damage', 'exposure', 'harm', 'likelihood', 'stress', 'pain',
              'discomfort', 'noise', 'friction', 'injury'}
EXTENDED_REACH = {'network', 'line', 'zone', 'area', 'service', 'route', 'programme', 'program', 'scheme', 'coverage',
                'hour', 'range', 'reach', 'boundary', 'limit', 'border', 'lane', 'track', 'rail', 'system',
                'grid', 'footprint', 'presence'}
GROWN_BUSINESSES = {'business', 'company', 'team', 'operation', 'workforce', 'audience', 'base', 'portfolio', 'firm',
                  'brand', 'membership', 'following', 'customer', 'revenue', 'economy', 'empire', 'readership'}
# (not "keep contact with" or "keep quality", which want "keep in contact" and "keep up quality")
MAINTAINED_STATES = {'weight', 'balance', 'distance', 'control', 'order', 'pace', 'temperature', 'focus', 'posture',
              'standard', 'level', 'routine', 'silence', 'composure', 'schedule', 'speed',
              'momentum', 'record', 'diary', 'log', 'eye', 'calm', 'hydration', 'habit'}
SUBMITTED_PAPERS = {'application', 'form', 'request', 'report', 'claim', 'proposal', 'paper', 'essay', 'assignment',
               'document', 'entry', 'bid', 'complaint', 'appeal', 'payment', 'invoice', 'manuscript', 'abstract',
               'coursework', 'homework', 'cv', 'resume', 'letter', 'statement', 'return', 'ticket', 'feedback',
               'expense', 'timesheet', 'portfolio', 'thesis', 'dissertation', 'article', 'design', 'plan', 'question'}
# (shelter is taken, not got: "Seek shelter" stays)
SOUGHT_HELP = {'help', 'advice', 'treatment', 'care', 'attention', 'support', 'guidance', 'assistance',
              'permission', 'input', 'feedback', 'counselling', 'counseling', 'therapy', 'information'}
CONSTRUCTED_THINGS = {'building', 'bridge', 'road', 'house', 'home', 'wall', 'tower', 'dam', 'railway', 'line', 'tunnel',
               'school', 'hospital', 'church', 'castle', 'fort', 'harbour', 'harbor', 'port', 'canal', 'stadium',
               'museum', 'library', 'factory', 'plant', 'model', 'shelter', 'cathedral', 'temple', 'palace', 'station',
               'airport', 'office', 'hall', 'pipeline', 'fence', 'barn', 'mill', 'lighthouse', 'pier', 'track',
               'highway', 'motorway', 'runway', 'apartment', 'flat', 'block', 'prototype', 'replica', 'nest', 'boat',
               'ship', 'monument', 'memorial', 'extension', 'facility', 'centre', 'center', 'complex'}
# (not what is grown, pumped or written: "Brazil produces coffee", "the wells produce oil", "produced a paper")
MADE_GOODS = {'wine', 'beer', 'cheese', 'car', 'vehicle', 'chip', 'phone', 'device',
               'goods', 'product', 'steel', 'furniture', 'clothes', 'clothing', 'cloth', 'toy', 'pottery',
               'glass', 'bread', 'chocolate', 'whisky', 'whiskey', 'cider', 'jam', 'soap', 'candle', 'shoe', 'bike',
               'bicycle', 'engine', 'part', 'component', 'battery', 'charger', 'case', 'tile', 'brick', 'textile',
               'fabric', 'jewellery', 'jewelry', 'watch', 'instrument', 'tool', 'machine', 'computer', 'laptop',
               'cake', 'pastry', 'sauce', 'spirit', 'perfume', 'cosmetic', 'medicine', 'drug', 'vaccine', 'fertilizer'}
FABRICATED_NOUNS = {'consent', 'evidence', 'crisis', 'outrage', 'story', 'excuse', 'controversy', 'scandal', 'drama',
                    'quote', 'alibi', 'charge', 'grievance', 'demand'}
# What a cleared-up or ended event looks like ("The talks concluded on Friday", "concluded the meeting"; a deal or a
# contract that is concluded is agreed, and "concluded that ..." draws a conclusion)
CONCLUDED_EVENTS = {'meeting', 'session', 'visit', 'tour', 'speech', 'talk', 'presentation', 'ceremony', 'event', 'series',
                 'season', 'programme', 'program', 'conference', 'hearing', 'show', 'concert', 'interview', 'discussion',
                 'debate', 'evening', 'day', 'festival'}
# "remain" -> "stay" for a state something keeps ("costs remained stable", "the doors remain open"), not for one that
# is merely found ("the cause remains unknown", "it remains unclear whether ...")
STAY_STATES = {'open', 'closed', 'stable', 'calm', 'quiet', 'active', 'healthy', 'fit', 'safe', 'warm', 'cool', 'dry',
               'fresh', 'loyal', 'focused', 'positive', 'strong', 'high', 'low', 'unchanged', 'steady', 'flat', 'constant',
               'competitive', 'relevant', 'silent', 'busy', 'awake', 'connected', 'alert', 'motivated',
               'popular', 'intact', 'cheap', 'affordable', 'firm', 'neutral', 'close', 'level', 'small', 'independent',
               'hydrated', 'consistent', 'flexible', 'humble', 'grounded', 'optimistic', 'faithful', 'committed',
               'engaged', 'available', 'single', 'sober', 'cautious', 'vigilant', 'upbeat',
               'operational', 'online', 'free', 'empty', 'full', 'clean', 'tidy', 'sharp', 'productive', 'comfortable'}
# What "contain" holds back rather than holds ("Firefighters contained the blaze", "could barely contain her
# excitement", "contain costs")
CONTAINED_THREATS = {'fire', 'blaze', 'flame', 'spread', 'outbreak', 'epidemic', 'pandemic', 'virus', 'disease',
                     'infection', 'damage', 'cost', 'inflation', 'anger', 'rage', 'fury', 'laughter', 'excitement',
                     'emotion', 'tear', 'grief', 'joy', 'enthusiasm', 'threat', 'situation', 'crisis', 'protest', 'riot',
                     'unrest', 'violence', 'crowd', 'leak', 'spill', 'flood', 'problem', 'risk', 'insurgency',
                     'rebellion', 'panic', 'impatience', 'frustration', 'disappointment', 'smile', 'glee', 'surprise',
                     'contagion', 'fallout', 'expansion', 'influence', 'growth'}
CONTAINER_NOUNS = {'jar', 'tank', 'box', 'bottle', 'bag', 'folder', 'drawer', 'cupboard', 'cabinet', 'bin', 'container',
                   'crate', 'case', 'can', 'cup', 'glass', 'bowl', 'pot', 'pan', 'tin', 'chest', 'pouch', 'envelope',
                   'packet', 'package', 'barrel', 'vial', 'tube', 'carton', 'basket', 'suitcase', 'backpack', 'wallet',
                   'purse', 'reservoir', 'vessel', 'cartridge', 'jug', 'flask', 'kettle', 'tray', 'shelf', 'safe',
                   'locker', 'trunk', 'bucket', 'sack'}
# "discover" -> "find" with a thing as object only for someone visiting or using something ("Visitors can discover
# 1,600 temples", "you will discover a quiet courtyard"); "Fleming discovered penicillin" was the first to find it
VISITOR_NOUNS = {'visitor', 'traveller', 'traveler', 'guest', 'tourist', 'reader', 'user', 'customer', 'shopper',
                 'diner', 'walker', 'hiker', 'fan', 'player', 'viewer', 'listener', 'learner', 'beginner', 'buyer'}
# A person or a body that "persists" keeps trying; an illness or a problem that persists lasts
PERSIST_ADVERBIALS = {'in', 'with', 'into'}
# "commonly known as" and its kin are fixed ("often known as" isn't said), and land that is "commonly owned" or "held"
# is owned jointly
NAMING_PARTICIPLES = {'known', 'called', 'referred', 'named', 'termed', 'accepted', 'held', 'believed', 'thought',
                      'assumed', 'regarded', 'described', 'owned', 'shared'}
# A number or amount "roughly" can come before ("roughly half", "roughly the same", "roughly as big as")
ROUGH_AMOUNT_LEADS = {'half', 'a', 'an', 'the', 'twice', 'double', 'triple', 'equal', 'one', 'once', 'as', 'every', 'per',
                  'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'twelve', 'twenty',
                  'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million',
                  'billion', 'dozen', 'quarter', 'third'}
# What "strikingly" goes before ("strikingly similar", "strikingly beautiful"; "remarkably short" and "remarkably well"
# are "unusually short" and "unusually well")
STRIKING_DESCRIBERS = {'similar', 'different', 'beautiful', 'handsome', 'modern', 'clear', 'consistent', 'original',
                       'simple', 'bold', 'vivid', 'bright', 'familiar', 'absent', 'alike', 'elegant', 'colourful',
                       'colorful', 'dramatic', 'uniform'}
# Verbs after which "ultimately" means "in the end" ("was ultimately abandoned", "ultimately lead to"); "the decision
# ultimately rests with the board" and "what ultimately determines the price" mean "fundamentally", which "eventually"
# isn't
ULTIMATE_OUTCOME_VERBS = {'lead', 'result', 'cause', 'become', 'end', 'prove', 'decide', 'win', 'lose', 'fail', 'succeed', 'cost',
                 'pay', 'save', 'help', 'make', 'turn', 'reach', 'improve', 'reduce', 'increase', 'produce', 'create',
                 'replace', 'abandon', 'collapse', 'benefit', 'transform', 'force',
                 'settle', 'agree', 'choose', 'leave', 'return', 'emerge', 'contribute', 'strengthen'}
# ...and verbs that keep the "fundamentally" sense in any tense: "is ultimately owned by a holding company", "is
# ultimately derived from Latin", "the decision ultimately rested with the board"; so do causes named with "by" ("most
# cancers are ultimately caused by damage to DNA") and anything traced back "from" a source
ROOTED_STATES = {'own', 'derive', 'base', 'link', 'fund', 'control', 'root', 'trace', 'stem', 'connect', 'relate', 'tie',
                 'attribute', 'govern', 'rest', 'lie', 'depend', 'owe', 'originate', 'descend', 'belong'}
ROOT_CAUSE_VERBS = {'cause', 'determine', 'drive', 'shape', 'explain', 'influence', 'dictate'}
# Describing words that leave "outcome" a plain result ("better long-term outcomes")
PLAIN_OUTCOME_MODIFIERS = {'good', 'bad', 'better', 'best', 'poor', 'worse', 'worst', 'positive', 'negative',
                           'long-term', 'short-term', 'term', 'long', 'short', 'final', 'likely', 'possible', 'desired',
                           'mixed', 'different', 'similar', 'overall', 'eventual', 'ultimate', 'actual', 'expected',
                           'successful', 'happy', 'fair', 'uncertain'}
# Who recommends a thing as advice rather than as a formal proposal ("Doctors recommend 150 minutes of exercise"; a
# board that recommends a dividend proposes one)
ADVISING_NOUNS = {'doctor', 'expert', 'guideline', 'dietitian', 'dietician', 'nutritionist', 'researcher', 'scientist',
                  'physician', 'dentist', 'specialist', 'adviser', 'advisor', 'study', 'guide', 'we', 'i', 'you',
                  'nurse', 'pharmacist', 'coach', 'trainer', 'teacher', 'author', 'chef', 'reviewer', 'critic',
                  'official', 'agency', 'organization', 'organisation', 'charity', 'manufacturer', 'site', 'app'}
# What "raise" and "lower" take for "increase" and "decrease" ("increase the risk" -> "raise the risk")
RAISED_LEVELS = {'price', 'rate', 'tax', 'wage', 'salary', 'pay', 'fee', 'cost', 'risk', 'level', 'awareness', 'standard',
               'productivity', 'output', 'limit', 'threshold', 'temperature', 'heat', 'volume', 'pressure', 'chance',
               'likelihood', 'odds', 'number', 'share', 'bar', 'cap', 'ceiling', 'minimum', 'age', 'rent', 'tariff',
               'dividend', 'profile', 'morale', 'quality', 'speed', 'cholesterol', 'premium', 'charge', 'interest',
               'expectation', 'concentration', 'dose', 'amount', 'budget', 'target', 'requirement', 'barrier'}
# ...which, with the risks and the symptoms a treatment lowers, are what "reduce" may become "lower" for
LOWERED_THINGS = RAISED_LEVELS | MINIMIZED_RISKS | LOWERED_SYMPTOMS
# A burden that something "increases" is added to ("which may increase stress" -> "add to stress")
STRAIN_NOUNS = {'stress', 'anxiety', 'tension', 'frustration', 'confusion', 'uncertainty', 'pressure', 'strain',
                'burden', 'workload', 'congestion', 'loneliness', 'fatigue', 'worry', 'noise', 'clutter', 'debt'}
# "produce no exhaust fumes" -> "give off no exhaust fumes" (not gas, which a gas field or a country produces)
GIVEN_OFF_NOUNS = {'fume', 'emission', 'smoke', 'heat', 'light', 'odour', 'odor', 'smell', 'radiation', 'steam',
                   'vapour', 'vapor', 'scent', 'glow', 'spark'}
# "complete the course" -> "finish the course" (a deal or a set that is completed is closed or made whole, and a form
# or a survey that is completed is filled in)
FINISHED_TASKS = {'task', 'course', 'training', 'degree', 'project', 'work', 'assignment',
              'program', 'programme', 'module', 'test', 'exam', 'review', 'homework', 'essay',
              'puzzle', 'race', 'marathon', 'lap', 'journey', 'renovation', 'installation', 'upgrade', 'setup', 'step',
              'level', 'game', 'internship', 'apprenticeship', 'placement', 'shift', 'tutorial', 'lesson',
              'chapter', 'draft', 'report', 'thesis', 'dissertation', 'interview', 'treatment'}
# "This essay explores the role of ..." -> "looks at the role of" (visitors who explore the old town walk around it)
EXPLORED_TOPICS = {'role', 'impact', 'effect', 'relationship', 'link', 'connection', 'question', 'idea', 'theme', 'issue',
               'history', 'use', 'option', 'possibility', 'potential', 'factor', 'cause', 'benefit', 'challenge', 'risk',
               'evidence', 'topic', 'concept', 'implication', 'influence', 'experience', 'perspective', 'approach',
               'strategy', 'trend', 'pattern', 'difference', 'association', 'origin', 'meaning', 'nature', 'extent',
               'consequence', 'dynamic', 'tension', 'debate', 'argument', 'value', 'limit', 'limitation'}
# "launch the server" -> "start the server" for a user starting a program (a launched product, campaign or rocket
# stays, and so does an app or a game a firm launches: it releases it)
LAUNCHED_PROGRAMS = {'server', 'app', 'application', 'program', 'browser', 'terminal', 'game', 'script', 'process',
                 'instance', 'container', 'editor', 'tool', 'wizard', 'console', 'shell', 'session', 'window', 'file',
                 'installer', 'emulator', 'simulator', 'debugger', 'notebook', 'service'}
# What rises and falls in amount ("Prices rose sharply" -> "climbed"; the sun rises, and a team falls behind)
AMOUNT_NOUNS_MOVING = {'price', 'revenue', 'rate', 'cost', 'sale', 'profit', 'share', 'number', 'temperature', 'level',
                       'wage', 'rent', 'inflation', 'unemployment', 'demand', 'value', 'stock', 'figure', 'income',
                       'spending', 'fee', 'tax', 'salary', 'pay', 'earnings', 'output', 'production', 'turnover', 'debt',
                       'deficit', 'attendance', 'enrolment', 'enrollment', 'membership', 'count', 'total', 'index',
                       'score', 'bill', 'premium', 'tariff', 'volume', 'emission', 'yield', 'return', 'margin', 'growth',
                       'population', 'traffic', 'usage', 'use', 'case', 'death', 'infection', 'admission', 'subscriber'}
# Ties that are mended rather than fixed ("repair trust", "repair the relationship")
MENDED_TIES = {'relationship', 'trust', 'reputation', 'image', 'friendship', 'marriage', 'tie', 'bond', 'rift', 'heart',
               'confidence', 'faith', 'link', 'alliance'}
# (with the lemmas the tagger gives the plural-only words: "the clothes" -> clothe, "jeans" -> jean)
CLOTHING_NOUNS = {'dress', 'suit', 'trousers', 'jacket', 'garment', 'clothes', 'skirt', 'shirt', 'gown', 'hem', 'sleeve',
                  'coat', 'jeans', 'pants', 'uniform', 'costume', 'outfit', 'clothe', 'trouser', 'jean', 'pant'}
# Pronouns for people: who offers or contains something in their other sense ("We offered her a job", "she could
# barely contain her excitement")
PRONOUNS_FOR_PEOPLE = {'i', 'we', 'you', 'he', 'she', 'they', 'who', 'whom', 'me', 'us', 'him', 'her', 'them', 'someone',
                   'somebody', 'everyone', 'everybody', 'anyone', 'anybody', 'nobody', 'one'}
# A text that says "he" or "she" is about people, so a name in it with no entity type may be one (see unnamed_person)
THIRD_PERSON_WORDS = {'he', 'she', 'him', 'her', 'his', 'hers', 'himself', 'herself'}
# A body that offers a plan, a deal or a choice puts it forward ("The government offered a new plan" didn't have one
# already), while a place or a product has what it offers ("The phone offers a choice of three colours")
PROPOSAL_NOUNS = {'plan', 'deal', 'option', 'choice'}
# A railway or a power line is built; a line in geometry is drawn ("construct a line perpendicular to AB")
BUILT_LINE_KINDS = {'railway', 'rail', 'tram', 'metro', 'tube', 'subway', 'underground', 'train', 'bus', 'power',
                    'pipe', 'telephone', 'phone', 'transmission'}
# "monitor" -> "track" only for a measure followed over time ("track your intake", "track inflation"): nurses who
# monitor patients and guards who monitor a car park watch them, and tracking them would follow where they go
TRACKED_MEASURES = {'progress', 'intake', 'spending', 'expense', 'budget', 'weight', 'symptom', 'level', 'pressure',
                    'temperature', 'rate', 'price', 'inflation', 'performance', 'sale', 'usage', 'use', 'consumption',
                    'trend', 'change', 'growth', 'number', 'count', 'sleep', 'step', 'habit', 'mood', 'glucose', 'sugar',
                    'pulse', 'calorie', 'activity', 'traffic', 'output', 'emission', 'cost', 'score', 'result', 'safety',
                    'spread', 'development', 'fitness', 'health', 'time', 'data', 'datum', 'metric', 'figure', 'demand'}
# "reveal that" -> "show that" when evidence is the subject ("The survey revealed that ..."); a person, a firm or a
# government that reveals something discloses it ("The minister revealed that she would step down")
REVEALING_SOURCES = {'study', 'survey', 'scan', 'analysis', 'data', 'datum', 'test', 'report', 'research', 'investigation',
                     'inquiry', 'audit', 'review', 'poll', 'census', 'experiment', 'trial', 'result', 'finding', 'record',
                     'document', 'file', 'footage', 'image', 'photo', 'photograph', 'search', 'examination', 'inspection',
                     'autopsy', 'excavation', 'evidence', 'figure', 'statistic', 'map', 'model', 'simulation',
                     'measurement', 'sample', 'screening', 'assessment', 'evaluation', 'observation', 'interview',
                     'questionnaire', 'letter', 'diary', 'email', 'log', 'recording', 'video', 'check'}
# "impact" -> "affect" for an effect ("will impact first-time buyers"); an asteroid that impacted the Earth hit it, and
# impacted wisdom teeth are stuck
COLLIDING_BODIES = {'asteroid', 'meteor', 'meteorite', 'comet', 'car', 'vehicle', 'truck', 'lorry', 'bus', 'train',
                    'plane', 'aircraft', 'jet', 'bullet', 'missile', 'rocket', 'projectile', 'shell', 'ball', 'object',
                    'debris', 'probe', 'lander', 'ship', 'boat', 'wave', 'tsunami', 'bomb', 'shrapnel', 'hammer'}
IMPACTED_TARGETS = {'tooth', 'teeth', 'molar', 'bowel', 'stool', 'colon', 'earwax', 'wax', 'ground', 'surface', 'earth',
                    'planet', 'moon', 'wall', 'barrier', 'target', 'pavement', 'windscreen', 'windshield'}
# "occur" -> "happen" for events; a letter, a name or a species that occurs somewhere is found there ("The letter E
# occurs more often than any other letter", "The species occurs throughout Europe")
FOUND_THING_NOUNS = {'letter', 'word', 'name', 'surname', 'term', 'phrase', 'symbol', 'character', 'sequence', 'pattern',
                     'value', 'gene', 'mutation', 'species', 'mineral', 'element', 'variant', 'spelling', 'vowel',
                     'consonant', 'motif', 'theme', 'plant', 'bird', 'tree', 'animal', 'fossil', 'metal', 'compound'}
# What "rises" or "falls" to must be a number or a level ("fell to 4 percent", "rose to a record high"; an estate
# "fell to her brother"), and a singular figure, bill, count or case may be a person, a law or a charge rather than
# an amount ("A tall figure rose from the armchair", "The bill fell at its second reading")
AMOUNT_LEVELS = {'level', 'low', 'high', 'record', 'minimum', 'maximum', 'zero', 'point', 'mark', 'percent', 'cent',
                 'average', 'peak', 'trough', 'bottom', 'floor', 'half', 'third', 'quarter'}
AMBIGUOUS_MOVERS = {'figure', 'bill', 'count', 'case'}
# Readings that rise but don't grow ("The temperature rose to 40 degrees", never "grew")
MEASURED_READINGS = {'temperature', 'speed', 'pressure', 'price', 'rate', 'level', 'volume', 'heat', 'dose',
                     'concentration', 'cholesterol', 'fee', 'tariff', 'premium', 'charge', 'rent'}
# "caution" -> "warn" for advice ("Experts caution that ..."); a referee who cautions a player books him, and the
# police who caution a suspect give a formal caution
CAUTIONING_OFFICIALS = {'referee', 'umpire', 'police', 'officer', 'constable', 'court', 'magistrate', 'judge',
                        'policeman', 'policewoman'}
# "locate" -> "find" for a search ("locate the nearest exit"); a firm that locates its headquarters in Austin or its
# factory near the port puts them there
SITED_FACILITIES = {'headquarters', 'factory', 'office', 'plant', 'store', 'branch', 'warehouse', 'base', 'hub',
                    'facility', 'centre', 'center', 'depot', 'operation', 'business', 'company', 'firm', 'campus', 'site',
                    'studio', 'school', 'hospital', 'clinic', 'mill', 'refinery'}
SITE_PLACES = {'city', 'town', 'area', 'region', 'country', 'district', 'centre', 'center', 'suburb', 'state', 'county',
               'village', 'park', 'zone', 'street', 'port', 'coast', 'capital', 'outskirts', 'estate'}
LOCATING_PREPOSITIONS = {'in', 'at', 'near', 'to', 'on', 'beside', 'outside', 'inside'}
# "incorporate" -> "include" for a part taken in ("The design incorporates feedback"); a business or a town that is
# incorporated becomes a company or a municipality ("She incorporated her bakery last year")
INCORPORATED_BODIES = {'company', 'business', 'firm', 'bakery', 'shop', 'charity', 'partnership', 'practice', 'venture',
                       'subsidiary', 'entity', 'startup', 'enterprise', 'organisation', 'organization', 'club', 'society',
                       'association', 'trust', 'restaurant', 'cafe', 'store', 'town', 'city', 'village'}
# "transform" -> "reshape" for a change of form ("transformed the way we shop"); data that is transformed has a function
# applied to it ("We transformed the variables using a log function")
TRANSFORMED_DATA = {'data', 'datum', 'variable', 'score', 'value', 'signal', 'image', 'feature', 'matrix', 'vector',
                    'function', 'equation', 'coordinate', 'input', 'output', 'text', 'file', 'column', 'series',
                    'distribution', 'measure', 'sample', 'variance', 'response'}
# "Discover our new range" invites a look, not a search ("Find our range" would be looking for it)
SHOWCASED_RANGES = {'range', 'collection', 'selection', 'world', 'magic', 'beauty', 'charm', 'story', 'secret',
                    'flavour', 'flavor', 'taste', 'delight', 'wonder', 'line', 'menu'}
# "attract" -> "draw" for interest ("The festival attracts thousands"); a magnet or a charge attracts in the physical
# sense ("Opposite charges attract each other")
PHYSICAL_ATTRACTORS = {'magnet', 'charge', 'mass', 'particle', 'planet', 'pole', 'ion', 'electron', 'proton',
                       'molecule', 'atom', 'nucleus', 'opposite', 'gravity'}
# A question or an issue that "remains open" is unresolved, which "stays open" doesn't say
OPEN_QUESTION_NOUNS = {'question', 'issue', 'matter', 'debate', 'case', 'possibility', 'problem', 'mystery'}
# "demand" -> "call for" for a public demand ("Residents have demanded stronger protection"); a landlord who demands
# payment claims money owed, and customers who demand quality expect it (lemmas: "arrears" -> arrear)
DEMANDED_PAYMENTS = {'payment', 'repayment', 'money', 'compensation', 'refund', 'arrear', 'rent', 'damage', 'fee',
                     'ransom', 'debt', 'sum'}
EXPECTED_QUALITIES = {'quality', 'value', 'excellence', 'perfection', 'convenience', 'reliability', 'speed', 'result',
                      'service', 'choice', 'flexibility', 'respect', 'loyalty', 'attention', 'consistency', 'obedience'}
# A verb's replacements and the objects each one fits (as swap_object finds them)
OBJECT_FITS = {
    'offer|VERB': {'have': OFFER_HAVE_NOUNS, 'give': OFFER_GIVE_NOUNS},
    'deliver|VERB': {'provide': DELIVER_PROVIDE_NOUNS, 'produce': DELIVER_PRODUCE_NOUNS, 'give': DELIVER_GIVE_NOUNS},
    'address|VERB': {'tackle': ADDRESSED_PROBLEMS - {'need'}, 'deal with': ADDRESSED_PROBLEMS},   # needs aren't tackled
    'conduct|VERB': {'carry out': CONDUCTED_STUDIES, 'run': RUN_STUDIES},
    'host|VERB': {'hold': HOSTED_EVENTS},
    'generate|VERB': {'produce': GENERATED_OUTPUT, 'bring in': EARNED_MONEY},
    'employ|VERB': {'use': EMPLOYED_METHODS},
    'achieve|VERB': {'reach': ACHIEVED_TARGETS},
    'resolve|VERB': {'fix': RESOLVED_FAULTS, 'settle': SETTLED_DISPUTES},
    'minimize|VERB': {'cut': MINIMIZED_COSTS, 'lower': MINIMIZED_RISKS},
    'minimise|VERB': {'cut': MINIMIZED_COSTS, 'lower': MINIMIZED_RISKS},
    'expand|VERB': {'extend': EXTENDED_REACH, 'grow': GROWN_BUSINESSES},
    'maintain|VERB': {'keep': MAINTAINED_STATES},
    'submit|VERB': {'send': SUBMITTED_PAPERS},
    'construct|VERB': {'build': CONSTRUCTED_THINGS},
    'produce|VERB': {'make': MADE_GOODS, 'give off': GIVEN_OFF_NOUNS},
    'complete|VERB': {'finish': FINISHED_TASKS},
    'explore|VERB': {'look at': EXPLORED_TOPICS, 'examine': EXPLORED_TOPICS, 'consider': EXPLORED_TOPICS},
    'launch|VERB': {'start': LAUNCHED_PROGRAMS - {'file', 'window', 'browser', 'terminal', 'editor'},
                    'open': LAUNCHED_PROGRAMS - {'server', 'process', 'instance', 'container', 'script', 'session'}},
}

def relative_antecedent(word):
    """The noun a relative pronoun stands for ("views that are stunning" -> views); any other word as it is."""
    if word is not None and word.lower_ in ('that', 'which', 'who', 'whom') and word.head.dep_ == 'relcl':
        return word.head.head
    return word

def swap_described_noun(token):
    """The noun a describing word is about: the one it stands before ("stunning views"), also as the second of two
    ("bright and spacious rooms"), or the one it is said of ("The views are stunning", "views that are stunning")."""
    if token.dep_ == 'amod' and token.head.pos_ in ('NOUN', 'PROPN'):
        return token.head
    if token.dep_ == 'conj' and token.head.dep_ == 'amod' and token.head.head.pos_ in ('NOUN', 'PROPN'):
        return token.head.head
    if token.dep_ == 'conj' and token.head.dep_ in ('acomp', 'oprd'):
        token = token.head
    return relative_antecedent(predicate_subject(token))

def swap_subject(token):
    """A verb's subject: through a relative pronoun to its noun, and the first verb's when two share it ("The jar holds
    two liters and contains no plastic"); None for an instruction or a verb after "to"."""
    subjects = [c for c in token.children if c.dep_ in ('nsubj', 'csubj')]
    if not subjects and token.dep_ == 'conj':
        subjects = [c for c in token.head.children if c.dep_ in ('nsubj', 'csubj')]
    # the one nearest the verb: the parser may take an opening phrase's noun for a second subject ("All things
    # considered, prices rose")
    return relative_antecedent(max(subjects, key=lambda c: c.i)) if subjects else None

def swap_object(token):
    """A verb's object, or the subject of its passive ("The survey was conducted ..."), through a relative pronoun to
    its noun ("the courses that the college offers" -> courses)."""
    return relative_antecedent(next((c for c in token.children if c.dep_ in ('dobj', 'nsubjpass')), None))

def swap_person_word(word):
    """True if a word stands for a person or people: a pronoun, a name or a noun such as "teacher" or "residents"
    (not a firm, a place or a thing)."""
    return word is not None and (word.lower_ in PRONOUNS_FOR_PEOPLE or word.ent_type_ == 'PERSON'
                                 or noun_sense_is(word, 'person', 'people')
                                 or (word.pos_ == 'NOUN' and (word.lemma_.lower() in PERSON_WORDS
                                                              or word.lemma_.lower() in PEOPLE_NOUNS)))

def clause_asks(clause):
    """True if a clause is an indirect question: "which hours work best", "whether it helps", "how it works"."""
    first = clause.doc[clause.left_edge.i]
    return first.tag_ in ('WDT', 'WP', 'WP$', 'WRB') or first.lower_ in ('whether', 'if')

def to_clause_verb(clause):
    """True if a clause is a "to" verb ("strive to improve")."""
    return clause.pos_ in ('VERB', 'AUX') and any(c.dep_ == 'aux' and c.lower_ == 'to' for c in clause.children)

def acting_body(word):
    """True if a word names someone who acts and decides: a person, a body of people, an organisation or a country
    ("the minister", "the council", "Apple", "Brazil"), not a place or a thing ("the hotel", "the phone")."""
    return word is not None and (swap_person_word(word) or word.ent_type_ in ('ORG', 'GPE', 'NORP')
                                 or noun_sense_is(word, 'group') or word.lemma_.lower() in GROUP_NOUNS)

def program_user(token, subject):
    """True if "launch" is a user starting a program: an instruction ("Launch the app", "Run the setup script to launch
    the server") or a user doing it ("Once you launch the editor", "Players can launch the game from the menu"). A firm
    that launches an app or a game releases it ("Nintendo launched the game", "plans to launch the app in Europe")."""
    verb, doer = token, subject
    while doer is None and verb.dep_ in ('xcomp', 'advcl', 'conj') and verb.head.pos_ in ('VERB', 'AUX'):
        verb = verb.head
        doer = swap_subject(verb)
    if doer is None:    # no one does it: an instruction ("To get started, launch the app")
        return token.tag_ == 'VB' and not any(c.dep_ in ('aux', 'auxpass') and c.lower_ != 'to' for c in token.children)
    return doer.lower_ == 'you' or doer.lemma_.lower() in VISITOR_NOUNS

def gives_amount(prep):
    """True if a preposition's object is a number or an amount ("to 4 percent", "to $5", "by 40 degrees", "to a record
    high", "to its lowest level")."""
    target = next((c for c in prep.children if c.dep_ == 'pobj'), None)
    return target is not None and (target.like_num or target.ent_type_ in ('PERCENT', 'MONEY', 'CARDINAL', 'QUANTITY')
                                   or target.lemma_.lower() in AMOUNT_LEVELS
                                   or any(c.dep_ in ('nummod', 'quantmod') and c.like_num for c in target.children))

def amount_moves(mover, kids):
    """True if what "rises" or "falls" moves as an amount (see AMOUNT_LEVELS): a "to" phrase gives a number or a level
    ("fell to 4 percent", not "fell to her brother"), a singular figure, bill, count or case is one of amounts ("Sales
    figures rose", "The figure rose to 12 percent", not "A tall figure rose from the armchair"), and "a number of" is
    "several", not a count ("A number of trees fell in the storm")."""
    if any(c.dep_ == 'prep' and c.lower_ == 'to' and not gives_amount(c) for c in kids):
        return False
    lemma = mover.lemma_.lower()
    if lemma == 'number' and any(c.dep_ == 'det' and c.lower_ == 'a' for c in mover.children):
        return False
    if lemma in AMBIGUOUS_MOVERS and mover.tag_ != 'NNS' and not any(
            c.dep_ in ('compound', 'poss') for c in mover.children):
        return lemma == 'figure' and any(
            c.dep_ == 'prep' and c.lower_ in ('to', 'by') and gives_amount(c) for c in kids)
    return True

def locate_sites(obj, kids):
    """True if "locate" puts something somewhere rather than finds it: a place after in/at/near/to on the verb or on its
    object ("locate its new headquarters in Austin", "locate the plant on the coast"), or a facility put anywhere
    ("locate the new factory near the port"); "Engineers located the fault in the cable" found it."""
    for prep in [*kids, *(obj.children if obj is not None else [])]:
        if prep.dep_ != 'prep' or prep.lower_ not in LOCATING_PREPOSITIONS:
            continue
        target = next((c for c in prep.children if c.dep_ == 'pobj'), None)
        if (obj is not None and SITED_FACILITIES & {obj.lemma_.lower(), obj.lower_}) or (target is not None and (
                target.ent_type_ in ('GPE', 'LOC', 'FAC') or target.lemma_.lower() in SITE_PLACES)):
            return True
    return False

def unnamed_person(word):
    """True if a subject is likely a person the entity tagger missed: a name with no type that marks a place, a body or
    a thing, no digit and no number or named word joined to it, in a text that speaks of "he" or "she" ("Seacole
    travelled to London and offered her services as a nurse"; "The Lumen T3 offers seven heating modes" and "Lisbon
    offers a wide choice of restaurants" are things and places)."""
    if word is None or word.tag_ != 'NNP' or (word.ent_type_ and word.ent_type_ != 'PERSON') or any(
            ch.isdigit() for ch in word.text) or any(c.dep_ == 'compound' and (c.like_num or c.ent_type_)
                                                     for c in word.children):
        return False
    doc = word.doc
    if 'third_person' not in doc.user_data:
        doc.user_data['third_person'] = any(t.lower_ in THIRD_PERSON_WORDS for t in doc)
    return doc.user_data['third_person']

def offered_own(obj):
    """True if what is offered is someone's own: "offered her services as a nurse", "offered his help", "Maria's
    flat". It is put at another's disposal, which "had her services" or "gave his help" doesn't say."""
    return obj is not None and any(c.dep_ == 'poss' and (c.lower_ in PERSONAL_TARGETS or c.ent_type_ == 'PERSON')
                                   for c in obj.children)

def swap_verb_choices(token, key, choices):
    """swap_choices for a verb: the replacements whose sense the verb's object, subject or complement selects."""
    kids = list(token.children)
    deps = {c.dep_ for c in kids}
    passive = bool(deps & {'auxpass', 'nsubjpass'})
    obj = swap_object(token)
    thing = obj.lemma_.lower() if obj is not None and obj.pos_ == 'NOUN' and obj.ent_type_ not in NAME_ENTITIES else ''
    preps = {c.lower_ for c in kids if c.dep_ == 'prep'}
    subject = swap_subject(token)
    clause = next((c for c in kids if c.dep_ == 'ccomp'), None)
    if any(c.dep_ == 'dative' and c.lower_ == 'to' for c in kids):
        preps.add('to')     # "offers support to families": a "to" phrase, not "offered her a job"
        deps = {c.dep_ for c in kids if not (c.dep_ == 'dative' and c.lower_ == 'to')}
    if key == 'expand|VERB' and preps & {'to', 'into', 'across'}:
        choices = [c for c in choices if c != 'grow']     # "expand its network to Asia" -> "extend", not "grow"
    if key in OBJECT_FITS:
        if 'dative' in deps or (key == 'offer|VERB' and (swap_person_word(subject) or unnamed_person(subject)
                                                         or offered_own(obj))):
            return []       # "offered her a job", "We offered to help" (offering may be turned down; having can't)
        if key == 'explore|VERB' and clause is not None and clause_asks(clause):
            return choices  # "This essay explores how cities grew" -> "looks at how"
        if key == 'launch|VERB' and (passive or not program_user(token, subject) or (obj is not None and any(
                c.lower_ in ('new', 'first', 'latest', 'its', 'their') for c in obj.children))):
            return []       # "The bank launched a new app", "The app was launched in 2021": released, not started
        if key == 'host|VERB' and swap_person_word(subject):
            return []       # "She hosts a popular cooking show" presents it
        if key == 'employ|VERB' and obj is not None and (
                any(c.dep_ == 'nummod' for c in obj.children) or any(swap_person_word(c) for c in obj.conjuncts)
                or (thing == 'model' and obj.tag_ == 'NNS')):
            return []       # "The agency employs twelve models and four photographers" hires them
        if key == 'construct|VERB' and thing == 'line' and not any(
                c.dep_ == 'compound' and c.lemma_.lower() in BUILT_LINE_KINDS for c in obj.children):
            return []       # "construct a line perpendicular to AB" draws it ("a new railway line" is built)
        choices = [c for c in choices if thing in OBJECT_FITS[key].get(c, ())]
        # every part of a joined object must take the new verb: the list layer may put the second next to it
        # ("deliver rich sound and impressive battery life" can't become "produce impressive battery life ...")
        for part in obj.conjuncts if obj is not None else ():
            part_thing = part.lemma_.lower() if part.pos_ == 'NOUN' and part.ent_type_ not in NAME_ENTITIES else ''
            choices = [c for c in choices if part_thing in OBJECT_FITS[key].get(c, ())]
        if key == 'offer|VERB' and ('to' in preps or passive
                                    or (token.tag_ == 'VBG' and any(c.dep_ == 'aux' and c.lemma_ == 'be' for c in kids))
                                    or (thing in PROPOSAL_NOUNS and acting_body(subject))):
            # "offers support to families" -> "gives support to"; "Guided tours are offered daily" and "The museum is
            # offering free entry" can't become "are had" or "is having", and "The government offered a new plan" put
            # one forward
            choices = [c for c in choices if c != 'have']
        if key == 'deliver|VERB' and thing == 'performance' and not (
                acting_body(subject) and any(c.dep_ in ('det', 'poss') for c in obj.children)):
            choices = [c for c in choices if c != 'give']   # "a moving performance" is given; a chip's performance isn't
        return choices
    if key == 'begin|VERB':
        return choices      # "Construction will begin in March"
    if key == 'caution|VERB':
        # "The authors caution that ...", "cautioned investors against ..." -> "warn"; a referee who cautions a player
        # for dissent books him, and the police who caution someone give a formal caution ("He was cautioned and
        # released")
        doers = [subject, *(g for c in kids if c.dep_ == 'agent' for g in c.children if g.dep_ == 'pobj')]
        if 'for' in preps or any(d is not None and d.lemma_.lower() in CAUTIONING_OFFICIALS for d in doers) or (
                passive and clause is None and not preps & {'against', 'about'} and 'xcomp' not in deps):
            return []
        return choices
    if key == 'discover|VERB':
        # "discover that they spend ..." -> "find that ...", "discover which hours work best" -> "find out which ..."
        if clause is not None:
            return [c for c in choices if c == ('find out' if clause_asks(clause) else 'find')]
        visitor = subject is not None and (subject.lower_ == 'you' or subject.lemma_.lower() in VISITOR_NOUNS)
        instruction = subject is None and token.tag_ == 'VB' and token.dep_ in ('ROOT', 'conj') and not (
            deps & {'aux', 'auxpass'})
        # "Discover our new range" and "Visitors can discover our collection" invite a look, which "find" doesn't say
        showcase = obj is not None and (thing in SHOWCASED_RANGES or any(
            c.dep_ == 'poss' and c.lower_ == 'our' for c in obj.children))
        if thing and not passive and (visitor or instruction) and not showcase:
            return [c for c in choices if c == 'find']
        return []
    if key == 'feature|VERB':
        # "The menu features seasonal dishes" -> "has"/"includes"; "The film features Tom Hanks" stars him, and one
        # thing is had, not included ("features a sleek aluminium frame" -> "has a sleek ...", never "includes")
        if passive or not thing or token.tag_ in ('VBG', 'VBN') or swap_person_word(subject):
            return []
        if obj.tag_ != 'NNS' and not any(c.dep_ in ('conj', 'nummod') for c in obj.children):
            choices = [c for c in choices if c != 'include']
        return choices
    if key == 'contain|VERB':
        # "Each tablet contains 500 mg" -> "has", "The jar contains two liters" -> "holds"; a person or a body that
        # contains something holds it back ("contained the blaze", "contain costs"), and "a can of cola" that contains
        # nine teaspoons of sugar is a serving, not a vessel that holds them
        if (passive or not thing or thing in CONTAINED_THREATS or subject is None or swap_person_word(subject)
                or subject.ent_type_ in ('ORG', 'NORP', 'GPE') or noun_sense_is(subject, 'group')
                or subject.lemma_.lower() in GROUP_NOUNS):
            return []
        container = subject.lemma_.lower() in CONTAINER_NOUNS and not any(
            c.dep_ == 'prep' and c.lower_ == 'of' for c in subject.children)
        return [c for c in choices if c == 'have' or (c == 'hold' and container)]
    if key == 'recommend|VERB':
        # "Experts recommend saving ..." -> "suggest saving"/"advise saving", "recommend that you rest" -> "suggest
        # that"; a recommendation of a thing to someone is an endorsement ("I would highly recommend this blender to
        # anyone")
        if passive or 'dative' in deps or any(c.dep_ == 'advmod' for c in kids):
            return []
        if any(c.tag_ == 'VBG' and c.dep_ in ('xcomp', 'dobj', 'ccomp') for c in kids) or (
                clause is not None and clause.tag_ != 'VBG'):
            return choices
        if (thing and obj.dep_ == 'dobj' and 'to' not in preps and subject is not None
                and subject.lemma_.lower() in ADVISING_NOUNS):
            return [c for c in choices if c == 'suggest']
        return []
    if key == 'benefit|VERB':
        # "benefited the most from walking" -> "gained the most from", "may not benefit younger students" -> "help"
        if passive:
            return []
        sources = [token, *(c for c in kids if c.lower_ in ('most', 'more', 'least', 'lot', 'much', 'greatly'))]
        if any(g.dep_ == 'prep' and g.lower_ == 'from' for s in sources for g in s.children):
            return [c for c in choices if c == 'gain']
        if obj is not None and obj.dep_ == 'dobj' and obj.lower_ not in ('most', 'more', 'least', 'lot', 'much'):
            return [c for c in choices if c == 'help']
        return []
    if key == 'remain|VERB':
        # "Prices remained stable" -> "stayed stable", "The shop will remain open" -> "will stay open"; not a state that
        # is merely found ("remains unknown"), a question left unresolved ("The question of funding remains open"),
        # or a state now ("the reserves remain healthy" says they still are; "stay healthy" says they will keep being)
        state = next((c for c in kids if c.dep_ == 'acomp'), None)
        if (state is None or state.lemma_.lower() not in STAY_STATES or state.tag_ in ('JJR', 'JJS')
                or deps & {'ccomp', 'xcomp', 'expl'} or token.tag_ in ('VBZ', 'VBP')
                or (subject is not None and (subject.lower_ == 'it' or subject.lemma_.lower() in OPEN_QUESTION_NOUNS))):
            return []
        return choices
    if key == 'locate|VERB':
        # "locate the file" -> "find the file"; a house that "is located" somewhere stands there, and a firm that
        # "locates its headquarters in Austin" puts them there
        return choices if thing and not passive and token.tag_ != 'VBN' and not locate_sites(obj, kids) else []
    if key in ('participate|VERB', 'relocate|VERB', 'depict|VERB', 'select|VERB', 'thrive|VERB'):
        if key == 'thrive|VERB' and 'on' in preps:
            return []       # "thrive on pressure" enjoys it
        return choices
    if key == 'collaborate|VERB':
        # "collaborate with partners" -> "work with", "teams that collaborate" -> "work together"
        together = 'with' in preps or any(c.lower_ == 'together' for c in kids)
        return [c for c in choices if c == ('work' if together else 'work together')]
    if key == 'manufacture|VERB':
        return choices if thing and thing not in FABRICATED_NOUNS else []
    if key == 'request|VERB':
        # "request a refund" -> "ask for a refund"; "You are requested to arrive early", "requested that the
        # meeting be moved" -> "asked"
        if passive:
            return [c for c in choices if c == 'ask'] if any(
                c.dep_ in ('xcomp', 'advcl') and to_clause_verb(c) for c in kids) else []
        if clause is not None or any(c.dep_ == 'xcomp' for c in kids):
            return [c for c in choices if c == 'ask']
        return [c for c in choices if c == 'ask for'] if obj is not None and obj.dep_ == 'dobj' else []
    if key == 'notify|VERB':
        if obj is None:
            return []
        if passive or (preps | {c.lower_ for c in obj.children if c.dep_ == 'prep'}) & {'of', 'about'}:
            return [c for c in choices if c != 'tell']     # "notify you of changes" -> "inform you of changes"
        return choices
    if key == 'inform|VERB':
        # "inform the manager" -> "tell the manager"; research that "informs policy" shapes it
        if obj is None or (preps | {c.lower_ for c in obj.children if c.dep_ == 'prep'}) & {'of', 'about'} or not (
                swap_person_word(obj) or noun_sense_is(obj, 'group') or obj.lemma_.lower() in GROUP_NOUNS
                or obj.ent_type_ == 'ORG'):
            return []
        return choices
    if key == 'seek|VERB':
        # "seek to improve" -> "try to improve" or "aim to improve" (a paper or a report "aims to"); "seek medical
        # help" -> "get medical help" in advice only: an instruction or "should"/"must" ("someone who sought help" or
        # "has the right to seek advice" may not get it)
        if any(c.dep_ == 'xcomp' and to_clause_verb(c) for c in kids):
            return [c for c in choices if c in ('try', 'aim')]
        advice = token.tag_ == 'VB' and (
            any(c.dep_ == 'aux' and c.lower_ in ('should', 'must') for c in kids)
            or (subject is None and token.dep_ in ('ROOT', 'conj') and not deps & {'aux', 'auxpass'}
                and (token.dep_ == 'ROOT' or token.head.tag_ == 'VB')))
        if thing in SOUGHT_HELP and not passive and advice:
            return [c for c in choices if c == 'get']
        return []
    if key == 'persist|VERB':
        # "If symptoms persist" -> "last"/"continue"; a person who persists keeps trying
        later = {c.lower_ for c in kids if c.dep_ == 'prep' and c.i > token.i}    # "In rare cases, the pain may persist"
        if (obj is not None or later & PERSIST_ADVERBIALS or 'xcomp' in deps or subject is None
                or swap_person_word(subject) or subject.ent_type_ in ('ORG', 'NORP', 'GPE')):
            return []
        return choices
    if key == 'occur|VERB':
        # "Accidents occur at home" -> "happen"; "occurs naturally in fish" is found there, "it occurred to me" came
        # to mind, and a letter, a name or a species that occurs somewhere is found there ("The letter E occurs more
        # often than any other letter", "Iron occurs as an oxide")
        # (the parser may hang the verb on the noun as a relative clause: "The letter [E occurs] ...")
        found = [w for w in (subject, token.head if token.dep_ == 'relcl' else None) if w is not None]
        if (obj is not None or preps & {'in', 'to', 'within', 'among', 'as', 'throughout'}
                or any(c.lower_ == 'naturally' for c in kids)
                or any(w.lemma_.lower() in FOUND_THING_NOUNS or w.lower_ in FOUND_THING_NOUNS or any(
                    c.dep_ == 'compound' and c.lemma_.lower() in FOUND_THING_NOUNS for c in w.children) for w in found)):
            return []
        return choices
    if key in ('display|VERB', 'alter|VERB', 'monitor|VERB', 'attract|VERB', 'impact|VERB'):
        if obj is None or (key == 'alter|VERB' and thing in CLOTHING_NOUNS) or (
                key in ('monitor|VERB', 'attract|VERB') and passive):
            return []
        if key == 'monitor|VERB' and thing not in TRACKED_MEASURES:
            return []       # "track your intake", but nurses monitor patients (see TRACKED_MEASURES)
        if key == 'attract|VERB' and ((subject is not None and subject.lemma_.lower() in PHYSICAL_ATTRACTORS)
                                      or obj.lower_ in ('other', 'another')):
            return []       # "Opposite charges attract each other"
        if key == 'impact|VERB' and ((subject is not None and subject.lemma_.lower() in COLLIDING_BODIES)
                                     or obj.lemma_.lower() in IMPACTED_TARGETS or noun_sense_is(obj, 'body')):
            return []       # "The asteroid impacted the Earth", "Her wisdom teeth were impacted"
        return choices
    if key in ('reveal|VERB', 'state|VERB', 'note|VERB'):
        # "The survey revealed that ..." -> "showed that", "The contract states that ..." -> "says that", "Analysts noted
        # that ..." -> "pointed out that"; not an instruction ("Note that the key is secret") or "as noted above". A
        # person or a firm that reveals something discloses it ("The minister revealed that she would step down"), and
        # one person who "noted that the window had been forced" saw it (Academic's "observe" fits both)
        if clause is None or passive or token.tag_ == 'VBN' or subject is None:
            return []
        if key == 'reveal|VERB' and subject.lemma_.lower() not in REVEALING_SOURCES:
            return []
        if key == 'note|VERB' and (subject.lower_ in PRONOUNS_FOR_PEOPLE or subject.ent_type_ == 'PERSON' or (
                subject.tag_ in ('NN', 'NNP') and swap_person_word(subject))):
            choices = [c for c in choices if c != 'point out']
        return choices
    if key == 'incorporate|VERB':
        # "The new design incorporates feedback" -> "includes"; "Incorporate the butter into the flour" mixes it in, and
        # a business that is incorporated becomes a company ("She incorporated her bakery last year")
        return choices if (obj is not None and not passive and not preps & {'into', 'in'} and obj.ent_type_ != 'ORG'
                           and thing not in INCORPORATED_BODIES) else []
    if key == 'conclude|VERB':
        # "The talks concluded on Friday" -> "ended", "concluded the meeting"; a concluded deal was agreed, and
        # "concluded that ..." drew a conclusion
        if passive or clause is not None or preps & {'from', 'that'}:
            return []
        doer = thing if obj is not None else (subject.lemma_.lower() if subject is not None else '')
        return choices if doer in CONCLUDED_EVENTS else []
    if key == 'accommodate|VERB':
        # "The hall accommodates 500 guests" -> "holds"; accommodating a request meets it, and a council or a charity
        # that "can accommodate 30 families" houses them
        return choices if (obj is not None and any(c.dep_ == 'nummod' for c in obj.children) and subject is not None
                           and not acting_body(subject)) else []
    if key == 'receive|VERB':
        # "will receive a certificate" -> "get"; "have received" is ambiguous with "have got" (own), and receiving
        # guests is welcoming them
        if passive or obj is None or swap_person_word(obj) or any(c.dep_ == 'aux' and c.lemma_ == 'have' for c in kids):
            return []
        return choices
    if key == 'transform|VERB':
        # "transform carbon dioxide into glucose" -> "turn ... into", "transformed the way we work" -> "reshaped"; data
        # that is transformed has a function applied to it ("transformed the variables using a log function")
        if any(c.dep_ == 'prep' and c.lower_ == 'into' for c in [*kids, *(obj.children if obj is not None else [])]):
            return [c for c in choices if c == 'turn']
        return [c for c in choices if c == 'reshape'] if (obj is not None and not passive
                                                          and thing not in TRANSFORMED_DATA) else []
    if key in ('increase|VERB', 'decrease|VERB'):
        # "increase the risk" -> "raise the risk", "Wages increased by 0.4 percent" -> "rose"/"grew" ("raise stress",
        # "raise engagement" and "cut the heat" aren't said; a temperature or a price rises but doesn't grow, and
        # tickets that "increased in price" cost more, which "grew in price" doesn't say)
        if obj is not None:
            return [c for c in choices if (c in ('raise', 'lower') and thing in RAISED_LEVELS)
                    or (c == 'cut' and thing in MINIMIZED_COSTS) or (c == 'add to' and thing in STRAIN_NOUNS and not passive)]
        if 'in' in preps:
            measure = next((g for c in kids if c.dep_ == 'prep' and c.lower_ == 'in' for g in c.children
                            if g.dep_ == 'pobj'), None)
            if measure is not None and measure.lemma_.lower() in ('price', 'value', 'cost'):
                return []
            return [c for c in choices if c == 'grow']    # "increased in size" -> "grew in size"
        if subject is not None and subject.lemma_.lower() in MEASURED_READINGS:
            choices = [c for c in choices if c != 'grow']
        return [c for c in choices if c in ('rise', 'grow', 'fall', 'drop')]
    if key == 'repair|VERB':
        # "the town repaired its walls" -> "fixed its walls"; "repair trust" mends it
        return choices if obj is not None and thing not in MENDED_TIES else []
    if key in ('rise|VERB', 'fall|VERB'):
        # "Prices rose sharply" -> "climbed sharply"/"went up sharply", also "in the spring", "on Monday" or "to 4
        # percent"; "The sun rose", "rose to power", "fell in love", "rose in value", "fell behind", "fell short", "The
        # rent falls due", "The cost falls on the tenant", "Its fees fall within the range" and "Most scores fall
        # between 580 and 740" stay
        mover = subject if subject is not None else (token.head if token.dep_ == 'relcl' else None)
        if (mover is None or mover.lemma_.lower() not in AMOUNT_NOUNS_MOVING or obj is not None or 'prt' in deps
                or preps & {'into', 'under', 'out', 'off', 'through', 'behind', 'apart', 'between', 'within', 'outside',
                            'among', 'upon'}
                or any(c.dep_ == 'prep' and c.lower_ in ('in', 'on') and not heads_time_phrase(c) for c in kids)
                or any(c.dep_ in ('advmod', 'acomp', 'oprd') and c.lower_ in (
                    'short', 'behind', 'apart', 'back', 'through', 'flat', 'silent', 'ill', 'asleep', 'due', 'open',
                    'vacant') for c in kids)
                or not amount_moves(mover, kids)):
            return []
        if preps & {'below', 'above'}:
            choices = [c for c in choices if c not in ('go up', 'go down')]    # "dropped below", not "went down below"
        return choices
    if key == 'demand|VERB':
        # "Residents have demanded stronger protection" -> "called for", "The job demands patience" -> "calls for"; a
        # landlord who "may demand payment" claims money owed, and customers who "demand quality" expect it
        if (obj is None or passive or clause is not None or swap_person_word(obj) or thing in DEMANDED_PAYMENTS
                or (thing in EXPECTED_QUALITIES and acting_body(subject))):
            return []
        return choices
    if key == 'nestle|VERB':
        # "Nestled among wooded hills, ..." -> "Tucked away among ..."; "she nestled her head on his shoulder" stays
        return choices if token.tag_ == 'VBN' and 'dobj' not in deps and preps else []
    if key == 'strive|VERB':
        if any(c.dep_ == 'xcomp' and to_clause_verb(c) for c in kids):
            return choices
        return [c for c in choices if c == 'aim'] if 'for' in preps else []
    return []

def swap_describer_choices(token, key, choices):
    """swap_choices for a describing word: the replacements the noun it describes selects ("stunning views" ->
    "beautiful views", but "a stunning defeat" stays)."""
    doc = token.doc
    before = doc[token.i - 1].lower_ if token.i > 0 else ''
    noun = swap_described_noun(token)
    lemma = noun.lemma_.lower() if noun is not None else ''
    attributive = token.dep_ == 'amod' or (token.dep_ == 'conj' and token.head.dep_ == 'amod')
    if before in ('more', 'most', 'less', 'least'):
        choices = [c for c in choices if c in MOST_FORM_DESCRIBERS]     # "the most striking chapter", not "the most strong"
    if any(c.dep_ == 'prep' and c.lower_ == 'to' for c in token.children):
        return []           # "unique to this region" (a "to" phrase belongs to the original word)
    if key in DESCRIBED_FITS:
        fits = DESCRIBED_FITS[key]
        food = noun is not None and noun_sense_is(noun, 'food')
        choices = [c for c in choices if lemma in fits.get(c, ())
                   or (food and key in ('exceptional|ADJ', 'outstanding|ADJ'))
                   # "a remarkable scientist" -> "an extraordinary scientist"
                   or (c == 'extraordinary' and noun is not None and noun.pos_ == 'NOUN' and swap_person_word(noun))
                   # "It is remarkable that so few people noticed" -> "It is striking that ..."
                   or (key == 'remarkable|ADJ' and c in ('striking', 'notable') and lemma == 'it'
                       and any(k.dep_ == 'ccomp' for k in [*token.children, *token.head.children]))
                   # "perfect for families" -> "ideal for families"
                   or (key == 'perfect|ADJ' and any(p.dep_ == 'prep' and p.lower_ == 'for' for p in [
                       *token.children, *(token.head.children if token.dep_ == 'acomp' else [])]))]
    if key == 'remarkable|ADJ' and lemma == 'difference' and noun.head.lemma_.lower() == 'make':
        return []           # "can make a remarkable difference" is a big one, not a striking one
    if key in ('entire|ADJ', 'extensive|ADJ', 'primary|ADJ', 'principal|ADJ', 'considerable|ADJ') and not attributive:
        return []
    if key == 'impressive|ADJ' and lemma in ('figure', 'number') and (noun.tag_ != 'NNS' or any(
            c.dep_ == 'prep' and c.lower_ == 'of' for c in noun.children)):
        return []           # "He cut an impressive figure", "an impressive number of people" ("strong sales figures")
    if key in ('exceptional|ADJ', 'outstanding|ADJ') and noun is not None and noun.dep_ == 'pobj' and (
            noun.head.lower_ == 'with'):
        return []           # "Handle the samples with exceptional care" is a manner, not a service
    if key == 'perfect|ADJ' and lemma == 'condition' and noun.tag_ != 'NNS':
        return []           # "in perfect condition" is undamaged ("ideal conditions for sailing" are suitable)
    if key == 'unique|ADJ' and any(t.dep_ == 'det' and t.lower_ in ('each', 'every') for t in token.sent):
        return []           # "Each snowflake has a unique shape": no two alike, which "distinctive" doesn't say
    if key == 'entire|ADJ' and lemma in ('agreement', 'contract', 'understanding'):
        return []           # "the entire agreement between the parties" names a clause of a contract
    if key == 'spacious|ADJ' and lemma not in ROOMY_NOUNS:
        choices = [c for c in choices if c != 'roomy']      # "large rooms", "a large balcony" (see ROOMY_NOUNS)
    if key == 'bespoke|ADJ' and not attributive:
        choices = [c for c in choices if c != 'custom']     # "The furniture is bespoke" -> "is tailor-made"
    if key == 'renowned|ADJ' and (any(c.dep_ == 'prep' and c.lower_ == 'as' for c in token.children) or (
            attributive and not (noun is not None and swap_person_word(noun))) or (
            not attributive and not any(c.dep_ == 'prep' and c.lower_ == 'for' for c in token.children))):
        # "famous as the birthplace", "a well-known bakery", and "which is renowned" alone, where "noted" reads unfinished
        choices = [c for c in choices if c != 'noted']
    if key in ('renowned|ADJ', 'intuitive|ADJ') and not attributive:
        # "is renowned for" -> "is well known for", "is intuitive" -> "is easy to use"
        choices = [{'well-known': 'well known', 'easy-to-use': 'easy to use'}.get(c, c) for c in choices]
    return choices

def swap_adverb_choices(token, key, choices):
    """swap_choices for an adverb: the replacements that fit where it stands ("occasionally crashes" -> "sometimes
    crashes", "crashes occasionally" -> "now and then"; "roughly half" -> "about half", but "handled roughly" stays)."""
    doc = token.doc
    head = token.head
    nxt = doc[token.i + 1] if token.i + 1 < len(doc) else None
    before = doc[token.i - 1] if token.i > 0 else None
    opening = token.i == token.sent.start
    right_before_head = head.i == token.i + 1
    if key == 'occasionally|ADV':
        # "occasionally crashes" -> "sometimes crashes", "are occasionally late" -> "are sometimes late"; a phrase
        # ("now and then", "from time to time") only opens the sentence or ends a clause ("crashes now and then,"),
        # or follows a verb ("We meet now and then for lunch", not "We meet at times for lunch"); after a verb's
        # object "at times" doesn't read as how often ("clean them now and then", never "clean them at times")
        ends_clause = nxt is None or nxt.is_punct or nxt.pos_ in ('CCONJ', 'SCONJ') or nxt.tag_ in ('WDT', 'WRB')
        if opening:
            return choices
        if token.i > head.i and ends_clause:
            return [c for c in choices if c != 'at times']
        if token.i > head.i and head.pos_ == 'VERB' and nxt.pos_ in ('ADP', 'DET', 'NOUN', 'PRON'):
            return [c for c in choices if c in ('now and then', 'from time to time')]
        return [c for c in choices if c == 'sometimes']
    if key == 'largely|ADV' and not (nxt is not None and (nxt.pos_ in ('ADP', 'SCONJ') or nxt.lower_ in ('due', 'thanks'))):
        # "largely due to demand" -> "mainly"/"mostly", but "have largely welcomed", "largely untouched" and "largely
        # contained" -> "mostly" ("mainly contained" reads as the main way it was contained)
        return [c for c in choices if c == 'mostly']
    if key == 'roughly|ADV':
        # "roughly half" -> "about half", "roughly $3 million"; "handled roughly" is rough handling, and so is "grabbed
        # his arm roughly as he fell", where "as" opens a clause, not a comparison
        if nxt is None or not (nxt.like_num or nxt.is_currency or nxt.lower_ in ROUGH_AMOUNT_LEADS):
            return []
        compared = nxt.i + 1 < len(doc) and doc[nxt.i + 1].pos_ in ('ADJ', 'ADV', 'DET', 'NUM')    # "roughly as big as"
        if not (nxt.like_num or nxt.is_currency) and (
                (head.pos_ in ('VERB', 'AUX') and head.i < token.i) or nxt.dep_ == 'mark' or nxt.pos_ == 'SCONJ'
                or (nxt.lower_ == 'as' and not compared)):
            return []
        return choices
    if key == 'solely|ADV':
        # "based solely on" -> "based only on", "rely solely on"; "solely responsible" isn't "only responsible"
        return choices if nxt is not None and nxt.pos_ == 'ADP' else []
    if key == 'commonly|ADV':
        # "is commonly used" -> "is often used"; "commonly known as" is fixed (see NAMING_PARTICIPLES), and "the most
        # commonly used language" can't be "the most often used"
        if (nxt is not None and nxt.lower_ in NAMING_PARTICIPLES) or (
                before is not None and before.lower_ in ('most', 'more', 'less', 'least')):
            return []
        return choices
    if key == 'promptly|ADV':
        # "respond promptly" -> "quickly"; "promptly at 9" is on time
        return [] if nxt is not None and (nxt.lower_ in ('at', 'on', 'after', 'before', 'by') or nxt.like_num) else choices
    if key == 'initially|ADV':
        # "Initially, sales were slow" -> "At first, ..."; "was initially developed" -> "was first developed" ("It
        # initially faced opposition" can't be "It first faced", which reads as "for the first time")
        if opening and nxt is not None and nxt.text == ',':
            return [c for c in choices if c == 'at first']
        # (not with a doer after "by", which the structure layer may make the subject: "was initially funded by a
        # donor" would end as "A donor first funded the scheme")
        if not opening and right_before_head and head.tag_ in ('VBN', 'VBD') and any(
                c.dep_ in ('aux', 'auxpass') for c in head.children) and not any(c.dep_ == 'agent' for c in head.children):
            return [c for c in choices if c == 'first']
        return []
    if key == 'highly|ADV':
        # "highly effective" -> "very effective"; a participle keeps it ("highly regarded", "highly qualified", "highly
        # paid", "highly sought after"), and so do the statistical "highly significant" and "highly available", a term
        # for systems that keep running
        return choices if (nxt is not None and nxt.tag_ == 'JJ' and nxt.lemma_.lower() != 'significant'
                           and not nxt.lower_.endswith('ed') and nxt.lower_ not in ('available', 'paid', 'sought')) else []
    if key in ('increasingly|ADV', 'remarkably|ADV', 'incredibly|ADV'):
        if nxt is None or nxt.pos_ not in ('ADJ', 'ADV') or opening:
            return []
        if key == 'increasingly|ADV' and head.dep_ == 'amod':
            return []       # "has become more and more popular", but "an increasingly popular choice" stays
        return [c for c in choices if c != 'strikingly' or nxt.lemma_.lower() in STRIKING_DESCRIBERS]
    if key == 'notably|ADV':
        # "several cities, notably Paris and Lyon" -> "especially Paris and Lyon"; "notably higher" -> "noticeably
        # higher"; "most notably" and a sentence's opening "Notably," stay
        if opening or (before is not None and before.lower_ in ('most', 'more', 'less', 'least')):
            return []
        if nxt is not None and nxt.tag_ in ('JJR', 'RBR'):
            return [c for c in choices if c == 'noticeably']
        if before is not None and before.text in (',', '(', ';') and head.pos_ in ('NOUN', 'PROPN', 'NUM', 'ADP', 'PRON'):
            return [c for c in choices if c != 'noticeably']
        return []
    if key == 'ultimately|ADV':
        # "was ultimately abandoned" -> "eventually"; "the decision ultimately rests with the board", "is ultimately owned
        # by" and "ultimately derived from Latin" say what is so at root, not when (see ROOTED_STATES; "Ultimately,"
        # opening a sentence is the phrase layer's)
        if opening or head.pos_ != 'VERB' or token.i > head.i:
            return []
        verb = head.lemma_.lower()
        if verb in ROOTED_STATES or any(c.dep_ == 'prep' and c.lower_ == 'from' for c in head.children) or (
                verb in ROOT_CAUSE_VERBS and any(c.dep_ == 'agent' for c in head.children)):
            return []
        if head.tag_ in ('VBD', 'VBN'):
            return choices
        # "habits ultimately lead to results" -> "eventually lead to" ("finally lead to" is a past event's)
        return [c for c in choices if c != 'finally'] if verb in ULTIMATE_OUTCOME_VERBS else []
    if key == 'profoundly|ADV':
        return choices if head.pos_ == 'VERB' and token.i < head.i else []
    if key == 'merely|ADV' and (opening or head.dep_ in ('csubj', 'nsubj', 'nsubjpass')):
        # "Merely writing down three tasks can help" -> "Just writing down ..."; "Only writing down ..." would say that
        # nothing else helps
        return [c for c in choices if c != 'only']
    if key == 'effortlessly|ADV' and (head.pos_ == 'ADJ' or (nxt is not None and nxt.pos_ == 'ADJ')):
        return []           # "effortlessly syncs" -> "easily syncs", but "effortlessly elegant" isn't "easily elegant"
    if key == 'invariably|ADV' and opening:
        return []           # "Invariably, someone forgets" can't open with "Always,"
    return choices      # "largely due to", "merely", "undoubtedly", "effortlessly syncs", "invariably reduces"

def swap_noun_choices(token, key, choices):
    """swap_choices for a noun: "in a similar manner" -> "way", "the opportunity to discuss" -> "chance", "every aspect
    of life" -> "part"."""
    kids = list(token.children)
    doc = token.doc
    if key == 'manner|NOUN':
        # "her manner" is how she behaves, and "manners" are etiquette
        in_phrase = (token.dep_ == 'pobj' and token.head.lower_ == 'in' and any(c.dep_ == 'det' for c in kids)
                     and any(c.dep_ == 'amod' for c in kids))
        in_which = token.i + 2 < len(doc) and doc[token.i + 1].lower_ == 'in' and doc[token.i + 2].lower_ == 'which'
        return choices if token.tag_ == 'NN' and not any(c.dep_ == 'poss' for c in kids) and (in_phrase or in_which) else []
    if key == 'opportunity|NOUN':
        # "the opportunity to discuss" -> "the chance to discuss"; "equal opportunities" is a term, and
        # "opportunities for learning" reads oddly as "chances for learning"
        return choices if any(c.dep_ in ('acl', 'relcl') and to_clause_verb(c) for c in kids) and not any(
            c.lower_ == 'equal' for c in kids) else []
    if key == 'outcome|NOUN':
        # "the outcome of the election" -> "the result", "better long-term outcomes" -> "results"; "patient outcomes",
        # "clinical outcomes" and "learning outcomes" are terms of their fields
        modifiers = [c for c in kids if c.dep_ in ('amod', 'compound', 'nmod') and c.i < token.i]
        if any(c.lemma_.lower() not in PLAIN_OUTCOME_MODIFIERS for c in modifiers) or (
                token.i > 0 and doc[token.i - 1].pos_ in ('NOUN', 'PROPN')):
            return []
        return choices
    if key == 'aspect|NOUN':
        return choices if any(c.dep_ == 'prep' and c.lower_ == 'of' for c in kids) else []
    return choices      # "beverage", "attire"

def swap_choices(token, key, tone):
    """The replacements from AI_FAVOURITES or EVERYDAY_SWAPS that read correctly where the token stands, in this tone:
    [] when it is in a sense, a frame or a fixed term its plain words don't share."""
    table = AI_FAVOURITES if key in AI_FAVOURITES else EVERYDAY_SWAPS
    choices = list(table[key][tone] if tone in table[key] else table[key]['*'])
    pos = key.split('|')[1]
    if not choices:
        return []
    if pos == 'VERB':
        if token.dep_ in ('amod', 'acl', 'compound'):
            return []       # "the recommended dose", "courses offered by top universities", "selected works"
        choices = swap_verb_choices(token, key, choices)
        return context_choices(choices, token) if choices else []
    if pos == 'ADJ':
        return swap_describer_choices(token, key, choices)
    if pos == 'ADV':
        return swap_adverb_choices(token, key, choices)
    return swap_noun_choices(token, key, choices)

def swap_edits(token, key, replacement):
    """Token index -> new text for a chosen replacement: the word re-inflected ("offers" -> "has", "conducted" ->
    "carried out"), split around a pronoun object ("conducted it" -> "carried it out"); None if it can't be
    inflected. A describing word or an adverb keeps its form whatever its tag ("breathtaking" -> "spectacular")."""
    word = inflect(replacement, {'ADJ': 'JJ', 'ADV': 'RB'}.get(key.split('|')[1], swap_verb_tag(token)))
    if not word:
        return None
    split = particle_split(token, word)
    return {token.i: split[0], split[1].i: split[2]} if split else {token.i: word}

def swap_verb_tag(token):
    """A verb's tag, corrected where the tagger mixes up the past tense and the participle, which an irregular
    replacement tells apart: "More than 50 schools participated" (tagged as a participle, with a subject and no helping
    verb) is "took part", never "taken part"; "has completed" is a participle whatever its tag."""
    helpers = {c.lemma_.lower() for c in token.children if c.dep_ in ('aux', 'auxpass')}
    if token.tag_ == 'VBN' and not helpers and any(c.dep_ == 'nsubj' for c in token.children):
        return 'VBD'
    if token.tag_ == 'VBD' and helpers & {'have', 'be'}:
        return 'VBN'
    return token.tag_

def swap_key(token):
    """The token's AI_FAVOURITES or EVERYDAY_SWAPS key, or None (also when the word may not be swapped at all). A
    describing word the tagger took for a verb or a noun is read as what it is ("enjoy breathtaking views", "The food
    was superb", "is renowned for")."""
    key = f'{token.lemma_.lower()}|{token.pos_}'
    if key in AI_FAVOURITES or key in EVERYDAY_SWAPS:
        return key if can_rewrite(token) else None
    key = f'{token.lower_}|ADJ'
    if ((key in AI_FAVOURITES or key in EVERYDAY_SWAPS) and token.tag_ in ('VBG', 'VBN', 'NN')
            and token.dep_ in ('amod', 'acomp', 'conj') and can_rewrite(token, describer=True)):
        return key
    return None

def neighbours(token):
    """The lowercase words either side of a token (None next to punctuation or at the edge)."""
    doc = token.doc
    before = doc[token.i - 1].lower_ if token.i > 0 and doc[token.i - 1].is_alpha else None
    after = doc[token.i + 1].lower_ if token.i + 1 < len(doc) and doc[token.i + 1].is_alpha else None
    return before, after

def context_score(before, word, after):
    words = word.lower().split()
    return pair_count(before, words[0]) + pair_count(words[-1], after)

def choose_synonym(token, entry, tone, floor, avoid, need_evidence, vetted=False):
    """(lemma, inflected word) for a WordNet synonym in the tone's style that fits the token's neighbours,
    or (None, None). If the original is an established pairing in the Brown corpus ("rely on", seen
    at least twice), the synonym must be attested there too; nouns always need that evidence. When the
    swap will be `vetted` by the model, the model judges the fit instead (Brown is small)."""
    lemma_frequency, candidates = entry
    # Never a word the curated tables replace ("use" -> "utilize", "start" -> "commence", "buy" -> "purchase",
    # "beautiful" -> "stunning")
    pool = [word for word, frequency in candidates if frequency >= floor * lemma_frequency and word not in avoid
            and f'{word}|{token.pos_}' not in AI_WORDS and f'{word}|{token.pos_}' not in PLAIN_WORDS
            and f'{word}|{token.pos_}' not in AI_FAVOURITES]
    if tone == 'casual':
        order = sorted(pool[:4], key=len) + pool[4:]
    elif tone in ('academic', 'professional'):
        order = pool            # the closest synonyms first (longest-first picked "peculiar" over "special")
    else:
        order = random.sample(pool[:3], len(pool[:3])) + pool[3:]
    before, after = neighbours(token)
    attested = context_score(before, token.lower_, after) >= 2
    # A noun right after its modifier ("remote sensing") must keep a known pairing with it
    modified = need_evidence and token.i > 0 and token.doc[token.i - 1].head == token and \
        token.doc[token.i - 1].dep_ in ('amod', 'compound', 'nmod')
    # An "-ly" adverb can't become a word that is also an adjective ("adapt quickly" -> "adapt quick")
    ly_adverb = token.pos_ == 'ADV' and token.lower_.endswith('ly')
    for lemma in order:
        if ly_adverb and not lemma.endswith('ly') and 'ADJ' in getAllLemmas(lemma):
            continue
        word = inflect(lemma, token.tag_)
        if not word or word.lower() == token.lower_:
            continue
        if modified and not pair_count(before, word.lower().split()[0]):
            continue
        if not vetted and (attested or need_evidence) and not context_score(before, word, after):
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

# "go away", "put away", "melted away": spaCy often tags the particle an adverb, and its WordNet synonyms
# ("off", "forth") make another phrasal verb ("go off", "put off")
DIRECTIONAL_PARTICLES = {'away', 'off', 'out', 'up', 'down', 'back', 'forth', 'along', 'around', 'over', 'apart',
                         'aside', 'ahead', 'forward', 'through', 'about', 'together', 'behind'}

def can_rewrite(token, describer=False):
    """True if a word may be swapped at all; `describer` for a describing word the tagger took for a verb ("a
    breathtaking view"), which the verb_as_noun check below would turn away."""
    doc = token.doc
    in_hyphenated_word = ((token.i > 0 and doc[token.i - 1].text == '-' and not doc[token.i - 1].whitespace_)
                          or (token.i + 1 < len(doc) and doc[token.i + 1].text == '-' and not token.whitespace_))
    in_compound_term = token.dep_ == 'compound' or any(c.dep_ == 'compound' for c in token.children)
    particle = token.lower_ in DIRECTIONAL_PARTICLES and token.head.pos_ == 'VERB' and 0 < token.i - token.head.i <= 2
    # a noun the tagger took for a verb ("a longer stay"): its verb synonyms don't fit ("a longer remain")
    verb_as_noun = token.pos_ == 'VERB' and (any(c.dep_ in ('det', 'amod') for c in token.children)
                                             or (token.i > 0 and doc[token.i - 1].pos_ == 'DET'))
    return (token.pos_ in UPOS_TO_WORDNET and token.is_alpha and len(token.text) >= 4
            and not token.is_stop and token.ent_type_ not in NAME_ENTITIES
            and not (token.is_upper and len(token.text) > 1)          # acronyms
            and not in_compound_term                                  # terms like "data pipeline"
            and token.tag_ not in SKIP_TAGS and not in_hyphenated_word
            and token.lemma_.lower() not in CLAIM_VERBS
            and not particle and not (verb_as_noun and not describer)
            and not (token.lower_ in PHRASE_LAYER_OPENERS and linking_adverb(token))     # an opener it kept
            and token.i not in quoted_tokens(doc))                    # someone's exact words

# The sentence openers the phrase layer rewrites ("Initially," -> "At first,"; see TRANSITIONS). Where it keeps one, its
# options were used nearby, so the word layer keeps it too ("At first, ... At first, ..." would echo)
PHRASE_LAYER_OPENERS = frozenset(w for words, _, _ in TRANSITIONS for w in words if ' ' not in w)

def takes_preposition(token):
    """Verbs and adjectives bound to a preposition ("rely on", "aware of", "carry out"). Their WordNet
    synonyms rarely take the same one ("trust on"), so only the curated swaps may change them."""
    return token.pos_ in ('VERB', 'ADJ') and any(
        child.dep_ in ('prep', 'prt') and 0 < child.i - token.i <= 2 for child in token.children)

def enough_fits(token, key):
    """"sufficient"/"adequate" -> "enough" only where "enough" can stand: not after an article or a degree word,
    and for "adequate" only about an amount ("adequate funding", "the budget is adequate")."""
    doc = token.doc
    if token.i > 0 and (doc[token.i - 1].pos_ == 'DET' or doc[token.i - 1].lower_ in DEGREE_WORDS):
        return False
    if token.dep_ == 'amod':
        if any(c.dep_ in ('det', 'poss', 'predet') and c.i < token.i for c in token.head.children):
            return False
        noun = token.head
    elif token.dep_ in ('acomp', 'attr'):
        noun = next((c for c in token.head.children if c.dep_ in ('nsubj', 'nsubjpass')), None)
    else:
        return False
    return key == 'sufficient|ADJ' or (noun is not None and noun.lemma_.lower() in AMOUNT_NOUNS | SIZE_NOUNS)

def negated_before(token):
    """A negation a few words before the token in its sentence, or on a word it hangs from or that word's helping
    verb: "We do not have nearly enough nurses", "I didn't get nearly enough sleep", "The hospital hasn't, in recent
    years, had nearly enough beds" (the "n't" hangs from "has")."""
    doc = token.doc
    if any(t.lower_ in NEARLY_NEGATIONS or t.dep_ == 'neg' for t in doc[max(token.sent.start, token.i - 6):token.i]):
        return True
    kids = [g for word in [token.head, *list(token.head.ancestors)[:2]] for c in word.children
            for g in [c, *(c.children if c.dep_ in ('aux', 'auxpass') else [])]]
    return any((g.dep_ == 'neg' or g.lower_ in NEARLY_NEGATIONS) and g.i < token.i for g in kids)

def laid_cornerstone(token):
    """True if "cornerstone" is the stone itself: a building's ("the cornerstone of the old mill"), one with
    something inside or behind it, or one that is laid, carved, cracked or bears a date ("The cornerstone, laid in
    1885, still bears the name of the mayor", "carved on the cornerstone"). A verb that also takes the figure
    ("damage", "replace") only counts when the stone is the cornerstone of nothing ("Workers removed the cornerstone",
    but not "would damage the cornerstone of our justice system")."""
    of = next((c for c in token.children if c.dep_ == 'prep' and c.lower_ == 'of'), None)
    whole = next((c for c in of.children if c.dep_ == 'pobj'), None) if of is not None else None
    if whole is not None and (whole.lemma_.lower() in CORNERSTONE_BUILDINGS or noun_sense_is(whole, 'building')):
        return True
    if token.dep_ == 'pobj' and token.head.lower_ in INSIDE_PREPOSITIONS:
        return True
    verbs = [c for c in token.children if c.dep_ in ('acl', 'relcl')]
    if token.dep_ in ('nsubj', 'nsubjpass', 'dobj'):
        verbs.append(token.head)
    elif token.dep_ == 'pobj':
        verbs.append(token.head.head)       # "carved on the cornerstone"
    stone_verbs = LAID_STONE_VERBS if whole is not None else LAID_STONE_VERBS | GENERAL_STONE_VERBS
    return any(v.pos_ == 'VERB' and v.lemma_.lower() in stone_verbs for v in verbs)

def contest_stage(token):
    """True if a verb places what it does in a contest: "in the second round", "in the final", "on penalties",
    "from the tournament", "in the group stage", "in straight sets" (but not "in the latest round of layoffs"). The
    parser may hang the phrase on the object ("eliminated [last year's runners-up on penalties]"). Also true for an
    object that only a contest has: "the 2014 runners-up" (the parser's object is "up", with "runners-" before it, unlike
    "customer complaints"), a body of people named by its role ("the host nation", "the host country") and "the
    world number one". Jobs are never in a contest ("eliminated 300 jobs in the first round")."""
    doc = token.doc
    objects = [c for c in token.children if c.dep_ == 'dobj']
    if any(c.lemma_.lower() in JOB_NOUNS for c in objects):
        return False

    def stage(prep, g):
        lemma = g.lemma_.lower()
        if lemma == 'leg':
            return any(x.like_num or x.lower_ in CONTEST_LEG_MODIFIERS for x in g.children
                       if x.dep_ in ('nummod', 'amod', 'compound'))
        if lemma == 'time' and prep.lower_ not in ('in', 'after', 'during'):
            return False
        return lemma in CONTEST_STAGES or any(
            x.dep_ in ('amod', 'compound') and x.lower_ in CONTEST_STAGE_MODIFIERS.get(lemma, ()) for x in g.children)
    if any(g.dep_ == 'pobj' and not any(x.dep_ == 'prep' and x.lower_ == 'of' for x in g.children) and stage(c, g)
           for head in [token, *objects] for c in head.children if c.dep_ == 'prep' for g in c.children):
        return True
    return any(any(x.dep_ == 'compound' and ((x.i < c.i and doc[x.i + 1].text == '-') or noun_sense_is(c, 'group', 'people'))
                   and noun_sense_is(x, 'person') for x in c.children)
               or (c.lemma_.lower() == 'number' and any(x.dep_ == 'nummod' for x in c.children)) for c in objects)

def studied_group(noun):
    """True if a plural is a group under study rather than one being examined: no "the" or possessive ("examined
    500 students", "teenagers who use social media", "200 mice")."""
    return noun.tag_ == 'NNS' and not any(c.dep_ in ('det', 'poss') for c in noun.children)

def denotes_people(noun):
    """True if a noun names a person or a body of people ("the witness", "the audience", "the next generation of
    engineers", "the population of the town", "the leadership of the club", "its membership"), not the act or the
    quality the same word names ("the generation of renewable energy", "the gathering of data", "leadership")."""
    if noun_sense_is(noun, 'person'):
        return True
    people = noun_sense_is(noun, 'people')
    if not people and not (noun_sense_is(noun, 'group') and any(
            (c.dep_ == 'det' and c.lower_ in DEFINITE_DETERMINERS) or c.dep_ == 'poss' for c in noun.children)):
        return False
    if not people and any(c.dep_ == 'prep' and c.lower_ == 'between' for c in noun.children):
        return False    # a tie between two sides: "the partnership between schools and businesses"
    of = next((c for c in noun.children if c.dep_ == 'prep' and c.lower_ == 'of'), None)
    what = next((c for c in of.children if c.dep_ == 'pobj'), None) if of is not None else None
    return (what is None or (people and any(c.dep_ == 'amod' for c in noun.children)) or bool(what.ent_type_)
            or noun_sense_is(what, 'person', 'people', 'animal')
            or any(c.dep_ in ('det', 'poss') for c in what.children))

def cooked_food(noun):
    """True if a noun is a food or a liquid a recipe boils down ("the cream", "the stock")."""
    return noun_sense_is(noun, 'food') or noun.lemma_.lower() in OTHER_SENSE_OBJECTS['reduce|VERB']

def recipe_context(sent):
    """What boiled_down and about_diet read from a sentence, worked out once per sentence of a parsed text (a long text
    can say "reduce" hundreds of times): whether it has a cooking cue, whether it is about a diet, whether it or the
    sentence before is a cooking step, and the first food noun from the sentence before on (what "it" can be)."""
    doc = sent.doc
    known = doc.user_data.setdefault('recipe', {})
    if sent.start not in known:
        first = doc[sent.start - 1].sent.start if sent.start > 0 else sent.start
        window = doc[first:sent.end]
        known[sent.start] = (
            bool(COOKING_CUES.search(sent.text)),
            any(t.lower_ in DIET_WORDS for t in sent),
            bool(COOKING_CUES.search(window.text)) or any(t.lemma_.lower() in RECIPE_STEP_VERBS for t in window),
            next((t.i for t in window if t.pos_ == 'NOUN' and cooked_food(t)), None))
    return known[sent.start]

def boiled_down(token):
    """True if "reduce" boils a liquid down in a recipe: "Reduce coconut milk by half", "Reduce the passata until
    thick", "Reduce the cream", "Add the stock, then reduce it" (a diet that "reduces sugar" or "the sugar in your
    tea" cuts it)."""
    cue, _, cooking_step, first_food = recipe_context(token.sent)
    objects = [c for c in token.children if c.dep_ == 'dobj']
    instruction = (token.tag_ == 'VB' and not any(c.dep_ in ('nsubj', 'aux') for c in token.children)
                   and (token.dep_ == 'ROOT' or (token.dep_ == 'conj' and token.head.dep_ == 'ROOT')))
    diet = about_diet(token)
    if cue and any(cooked_food(o) or (instruction and not diet and o.pos_ == 'NOUN'
                                      and o.lemma_.lower() not in LEMMATIZER_NOUNS) for o in objects):
        return True
    if instruction and not diet and any(
            cooked_food(o) and any(d.dep_ == 'det' and d.lower_ == 'the' for d in o.children) for o in objects):
        return True
    # "it" after a food in a cooking step
    return (cooking_step and any(o.lower_ in ('it', 'them') for o in objects)
            and first_food is not None and first_food < token.i)

def about_diet(token):
    """True if "reduce" speaks of what someone eats or drinks: the sentence names a diet ("in your diet", "eat more
    beans"), or "you"/"your" is in reduce's own clause ("the sugar in your tea", "the salt you add"), not in a clause
    joined to it ("Reduce the cream, then pour it over your pasta")."""
    return recipe_context(token.sent)[1] or any(
        t.lower_ in DIET_PRONOUNS for c in token.children
        if c.dep_ not in ('conj', 'advcl', 'cc', 'parataxis', 'dep') for t in c.subtree)

def reduce_fits(token, objects):
    """The replacements "reduce" takes with its objects (lemmas): "ease" for pain and the like ("While the drug eases
    pain"; it doesn't cut or lower it), "cut down on" for the food one slices, in a diet ("Reduce the meat in your
    diet" -> "Cut down on the meat"; "cut the meat" would slice it, and with no diet in sight it keeps its verb),
    "lower" for the heat, a level, a risk or a symptom, and "cut" for the rest ("cut the menu", "cut the sugar in your
    tea", "cut red meat", never "lowered the menu")."""
    if objects & EASED_SYMPTOMS:
        return {'ease'}
    if any(c.dep_ == 'dobj' and c.lemma_.lower() in CUT_UP_FOODS and any(g.dep_ in ('det', 'poss') for g in c.children)
           for c in token.children):
        return {'cut down on'} if about_diet(token) else set()
    if objects & LOWER_ONLY_OBJECTS:
        return {'lower'}
    return {'cut', 'lower'} if objects & LOWERED_THINGS else {'cut'}

def statistical_sense(sent):
    """True if "significantly" may claim statistical significance in a sentence: it reads like a statistical report
    (STATISTICAL), it or the sentence before names a study, a trial or results (SIGNIFICANCE_EVIDENCE), or it measures
    something ("by 8 mmHg", "losing just 5% of your body weight"). Worked out once per sentence of a parsed text."""
    doc = sent.doc
    known = doc.user_data.setdefault('statistical', {})
    if sent.start not in known:
        named = [sent, doc[sent.start - 1].sent] if sent.start > 0 else [sent]
        known[sent.start] = bool(
            STATISTICAL.search(sent.text) or any(t.lemma_.lower() in SIGNIFICANCE_EVIDENCE for s in named for t in s)
            or any(t.ent_type_ in ('PERCENT', 'QUANTITY') or (t.like_num and t.ent_type_ in ('', 'CARDINAL') and any(
                ch.isdigit() for ch in t.text)) for t in sent))
    return known[sent.start]

def enable_allows(token):
    """True if "enable" makes something possible, which "allow" says too: someone or something enables a "to" verb ("The
    grant will enable the school to hire two teachers") or a thing enables a thing ("Better data enables faster
    decisions"). In an instruction, with a person doing it or in the passive it switches a setting on ("Enable two-step
    verification", "To enable automatic updates, open the settings menu", "You can enable dark mode", "Location must be
    enabled"), which "allow" doesn't say; an instruction's "to" verb may be its purpose ("Enable cookies in your
    browser to continue")."""
    subject = swap_subject(token)
    if subject is None:
        return False
    return not swap_person_word(subject) or any(
        c.dep_ in ('xcomp', 'ccomp') and c.i > token.i and to_clause_verb(c) for c in token.children)

def curated_fits(token, key):
    doc = token.doc
    before = doc[token.i - 1].lower_ if token.i > 0 else ''
    nxt = doc[token.i + 1] if token.i + 1 < len(doc) else None
    after = nxt.lower_ if nxt is not None else ''
    deps = {c.dep_ for c in token.children}
    if key.endswith('|VERB') and token.dep_ == 'amod' and token.tag_ in ('VBN', 'VBG'):
        return False    # a participle used as a describing word has its own sense: "attempted murder" isn't "tried"
    if {w for c in token.children if c.dep_ in ('dobj', 'nsubjpass')
            for w in (c.lemma_.lower(), c.lower_)} & OTHER_SENSE_OBJECTS.get(key, set()):
        return False
    if key == 'reduce|VERB' and boiled_down(token):
        return False    # "cut the cream" or "lower the cream" would be using less of it
    if key == 'examine|VERB':
        # a doctor or a vet examines a patient, a rash, her knee or the horse; "studied" reads as research, which
        # "examined 500 students" is, and as reading, which "The doctor examined the scan results" is
        doers = [d for c in token.children if c.dep_ in ('nsubj', 'agent') for d in ([c] if c.dep_ == 'nsubj' else c.children)]
        # every object a document, its conjuncts too ("the scan and the patient" is still a patient)
        objects = [c for c in token.children if c.dep_ in ('dobj', 'nsubjpass')]
        document = bool(objects) and all(o.lemma_.lower() in DOCUMENT_OBJECTS for c in objects for o in (c, *c.conjuncts))
        if (not document and any(noun_sense_is(d, 'clinician') for d in doers)) or any(
                c.dep_ in ('dobj', 'nsubjpass') and noun_sense_is(c, 'body', 'animal', 'person') and not studied_group(c)
                for c in token.children):
            return False
    if key == 'require|VERB':
        # an obligation: "are required to give notice", "requires tenants to give notice" and "required by law"
        # aren't "needed" ("the skills required to excel" are). After a contraction the parser may hang a passive's
        # "to" clause as an adverbial one ("You'll be required to show photo ID")
        to_clause = 'xcomp' in deps or ('auxpass' in deps and any(
            c.dep_ == 'advcl' and c.i > token.i and any(g.dep_ == 'aux' and g.lower_ == 'to' for g in c.children)
            for c in token.children))
        if (to_clause and deps & {'auxpass', 'dobj'}) or {'auxpass', 'agent'} <= deps:
            return False
        # "need" takes no "that" clause: "The law requires that landlords give notice" can't become "needs that ..."
        if any(c.dep_ == 'ccomp' and any(g.dep_ == 'mark' and g.lower_ == 'that' for g in c.children)
               for c in token.children):
            return False
    if key == 'anticipate|VERB' and ((nxt is not None and nxt.tag_ == 'VBG') or any(
            c.dep_ in ('xcomp', 'ccomp', 'dobj') and c.tag_ == 'VBG' for c in token.children)):
        return False    # "anticipate hiring" can't become "expect hiring"
    if key == 'comprise|VERB' and ('auxpass' in deps or after == 'of' or any(
            c.dep_ == 'dobj' and (c.lemma_.lower() in COMPRISE_SHARES or c.like_num) for c in token.children) or any(
            c.dep_ == 'prep' and c.i > token.i and any(t.dep_ == 'pobj' and (
                t.lemma_.lower() in COMPRISE_SHARES or t.ent_type_ == 'PERCENT') for t in c.children)
            for c in token.children)):
        return False    # (see COMPRISE_SHARES: "which comprised under 3% of all responses")
    if key == 'cease|VERB' and ('xcomp' in deps or after == 'to'):
        return False    # "ceased to exist": "stopped to exist" would be stopping in order to exist
    if key == 'paramount|ADJ' and (after in ('importance', 'significance') or before == 'of'):
        return False    # "of paramount importance" can't become "of all-important importance"
    if key == 'immediately|ADV' and (after in SEQUENCE_WORDS or token.head.pos_ not in ('VERB', 'AUX')
                                     or token.i < token.head.i or (nxt is not None and nxt.pos_ in ('ADP', 'ADJ', 'ADV'))):
        return False    # (see SEQUENCE_WORDS) "contact your doctor immediately" -> "right away"
    if key == 'enable|VERB' and not enable_allows(token):
        return False
    if key == 'consequently|ADV' and token.i != token.sent.start and before not in ('and', 'but', 'or', 'so', ',', ';'):
        return False    # inside a clause "so" reads as a degree word ("the flight was so delayed by three hours")
    if key == 'terminate|VERB' and any(c.dep_ in ('dobj', 'nsubjpass') and c.lemma_.lower() not in CONTRACT_NOUNS
                                       for c in token.children):
        return False
    if key == 'illustrate|VERB' and 'agent' in deps:
        return False    # "illustrated by Quentin Blake" was drawn, not "shown"
    if key == 'demonstrate|VERB' and not deps & {'dobj', 'ccomp', 'xcomp', 'auxpass', 'nsubjpass', 'agent'}:
        return False    # "students demonstrated outside parliament" protested
    if key == 'eliminate|VERB' and (contest_stage(token) or any(
            c.dep_ == 'dobj' and ((noun_sense_is(c, 'person') and c.lemma_.lower() not in INTERMEDIARY_NOUNS)
                                  or c.ent_type_ in ('PERSON', 'ORG', 'GPE', 'NORP', 'LOC')
                                  or (c.pos_ == 'PROPN' and not c.ent_type_)) for c in token.children)):
        return False    # "Spain eliminated Italy", "the hosts", "the Gunners", "the top seed in the second round":
                        # knocked out of a contest
    if key in NOT_AFTER_NEGATION:
        return before not in NEGATION_OR_DEGREE and not negated_before(token)
    if key in ENOUGH_KEYS:
        return enough_fits(token, key)
    if key == 'cornerstone|NOUN':
        # a figure: "the cornerstone of any good relationship", "Trust remains a cornerstone in every partnership",
        # "serves as a cornerstone for"
        figure = (any(c.dep_ == 'prep' and c.lower_ == 'of' for c in token.children) or token.dep_ == 'attr'
                  or (token.dep_ == 'pobj' and token.head.lower_ == 'as'))
        return figure and not laid_cornerstone(token)
    if key == 'facet|NOUN':
        # "every facet of modern life" -> "side"/"part", but a cut stone's facets ("57 facets", "each facet of the
        # diamond", "of the prism") are neither. After a plural the parser may take the verb for a noun with the
        # stone in front, leaving the sentence without a verb ("The facets of the sapphire sparkle under the lamp"),
        # unlike "every facet of rock music". "Rock" with no article and not plural is the music ("every facet of
        # rock and roll"); any other stone is the stone ("each facet of polished quartz", "of the diamond")
        of = next((c for c in token.children if c.dep_ == 'prep' and c.lower_ == 'of'), None)
        wholes = [c for c in of.children if c.dep_ == 'pobj'] if of is not None else []
        if token.sent.root.pos_ not in ('VERB', 'AUX'):
            wholes += [g for c in wholes for g in c.children if g.dep_ == 'compound']
        return 'nummod' not in deps and of is not None and not any(
            noun_sense_is(w, 'stone') and (w.lemma_.lower() != 'rock' or w.tag_ in ('NNS', 'NNPS') or any(
                c.dep_ in ('det', 'poss') for c in (w.head if w.dep_ == 'compound' else w).children)) for w in wholes)
    if key in OBJECT_SENSES:
        objects = [c for c in token.children if c.dep_ in ('dobj', 'nsubjpass')]
        return bool(objects) and objects[0].lemma_.lower() in CULTIVATE_OBJECTS and not takes_preposition(token)
    if key in BEFORE_ADJECTIVE_ONLY:
        # "readily available" -> "easily available", but "readily apparent" isn't "easily apparent"
        return token.head.pos_ == 'ADJ' and token.i < token.head.i and token.head.lower_.endswith(('able', 'ible'))
    if key == 'substantially|ADV':
        # "substantially reduced" -> "greatly reduced", but "substantially similar" isn't "greatly similar"
        return token.head.pos_ == 'VERB' or (nxt is not None and (nxt.tag_ in ('JJR', 'RBR') or nxt.lower_ in ('more', 'less', 'fewer')))
    if key == 'dramatically|ADV':
        # "dramatically increased" -> "sharply", but not "paused dramatically" or "dramatically changed"
        return token.i < token.head.i and token.head.lemma_.lower() in SHARP_CHANGE_VERBS
    if key == 'encourage|VERB':
        # "encourage innovation" -> "promote", but a person who is promoted gets a better job, and "encourage
        # them to read" needs the "to"
        objects = [c for c in token.children if c.dep_ == 'dobj']
        thing = objects[0].lemma_.lower() if objects else ''
        return (len(objects) == 1 and not deps & {'xcomp', 'ccomp'} and objects[0].pos_ == 'NOUN'
                and (thing.endswith(PROMOTABLE_ENDINGS) or thing in CULTIVATE_OBJECTS)
                and thing not in GROUP_NOUNS | AGENT_NOUNS | NOT_PROMOTABLE and not denotes_people(objects[0])
                and objects[0].lower_ not in PERSONAL_OBJECTS and objects[0].ent_type_ not in ('PERSON', 'ORG', 'NORP'))
    if key == 'establish|VERB' and any(
            c.dep_ == 'dobj' and (c.lemma_.lower() in EPISTEMIC_OBJECTS or any(g.lower_ == 'between' for g in c.children))
            for c in token.children):
        return False
    if key in AFTER_VERB_ONLY:
        return token.head.pos_ == 'VERB' and token.i > token.head.i and before not in DEGREE_WORDS
    if key in VERB_HEAD_ONLY:
        return token.head.pos_ in ('VERB', 'AUX')
    if key in OBJECT_ONLY:
        # "provides the body with energy" can't become "gives the body with energy"
        return ('dobj' in deps and not takes_preposition(token)
                and not any(c.dep_ == 'prep' and c.lower_ == 'with' and c.i > token.i for c in token.children))
    if key in BEFORE_NOUN_ONLY:
        return (token.dep_ == 'amod' and token.head.i > token.i and not before.endswith('ly')
                and before not in DEGREE_WORDS)
    if key in PREDICATE_ONLY:
        # "vital to good health" would need "necessary for", so a "to" phrase keeps the word
        return (token.dep_ == 'acomp' and token.head.lemma_ == 'be' and before not in DEGREE_WORDS
                and not any(c.dep_ == 'prep' and c.lower_ == 'to' for c in token.children))
    if key in CLAUSE_ONLY:
        return bool(deps & {'ccomp', 'xcomp'}) and not deps & {'dobj', 'dative', 'nsubjpass', 'auxpass'}
    if key in NOT_BEFORE_DESCRIBER:
        # read from the tags, not the parse: spaCy takes "stores frequently accessed data" as subject + verb
        return not (token.i + 2 < len(doc) and doc[token.i + 1].tag_ in ('VBN', 'VBD', 'JJ')
                    and doc[token.i + 2].pos_ in ('NOUN', 'PROPN'))
    return True

def linking_adverb(token):
    """A sentence-opening adverb before a comma ("Finally, ...") links the sentence to the one before; its
    WordNet synonyms come from its other senses ("finally" -> "eventually" makes it a claim about time)."""
    doc = token.doc
    return (token.pos_ == 'ADV' and token.i == token.sent.start and token.i + 1 < len(doc)
            and doc[token.i + 1].text == ',')

# The phrase layer's own wording ("contributes a lot to", "goes a long way toward", "On top of that,") is chosen
# as a whole, so WordNet doesn't swap words inside it ("lends a lot to")
CURATED_PHRASE = re.compile(r"\b(?:" + "|".join(sorted(
    {re.escape(o.rstrip(',:').lower()).replace(r'\ ', r'\s+')
     for options in [entry[1] for entry in PHRASES] + [entry[2] for entry in TRANSITIONS] + CONTEXT_OPTIONS
     for choices in options.values() for o in choices if ' ' in o.strip()}, key=len, reverse=True)) + r")\b", re.I)

def phrase_tokens(doc):
    """Indices of the tokens inside the phrase layer's wording, worked out once per parsed text in one pass: the
    tokens and the matches both run in text order, so each token only looks at the first match not yet behind it."""
    if 'curated' not in doc.user_data:
        ranges = [(m.start(), m.end()) for m in CURATED_PHRASE.finditer(doc.text)]
        inside, k = set(), 0
        for t in doc:
            while k < len(ranges) and ranges[k][1] <= t.idx:
                k += 1
            if k < len(ranges) and ranges[k][0] <= t.idx:
                inside.add(t.i)
        doc.user_data['curated'] = frozenset(inside)
    return doc.user_data['curated']

def remember(avoid, replacement):
    """Note a replacement (and its first word) so it isn't used again nearby ("look into ... looked at")."""
    if replacement:
        avoid.add(replacement)
        avoid.add(replacement.split()[0])

def sentence_words(sent):
    return {t.lower_ for t in sent} | {t.lemma_.lower() for t in sent}

def word_forms(text):
    """The words of a piece of text in lower case with their lemmas ("looked at" -> looked, look, at)."""
    forms = set()
    for word in text.lower().split():
        forms.add(word)
        for lemmas in getAllLemmas(word).values():
            forms.update(lemmas)
    return forms

LINKING_WORDS = {'a', 'an', 'the', 'of', 'on', 'to', 'up', 'out', 'into', 'in', 'at', 'for', 'with', 'off'}

def fresh_choice(choices, avoid):
    """A replacement none of whose words is already used nearby ("This study studies", "lowers ... and
    lowers", "make use of ... use"), or None."""
    fresh = [c for c in choices if not (set(c.split()) - LINKING_WORDS) & avoid]
    return random.choice(fresh) if fresh else None

def echo_fallback(key, token, avoid):
    """Another way to say an AI word when all its usual replacements are already nearby, or None. A fallback's
    lemma counts as an echo too ("You need a visa ..." can't go on "... is needed")."""
    fresh = [c for c in context_choices(ECHO_FALLBACKS.get(key, []), token)
             if not (word_forms(c) - LINKING_WORDS) & avoid]
    return random.choice(fresh) if fresh else None

def adjectival_participle(token):
    """"fixed expenses", "children aged 5 to 12": a participle used as a describing word has its own sense,
    which the verb's synonyms miss ("repaired expenses", "elderly 5 to 12")."""
    return token.pos_ == 'VERB' and token.tag_ == 'VBN' and token.dep_ in ('amod', 'acl')

SUMMARY_DETERMINERS = {'the', 'this', 'that', 'these', 'those'}

def summary_noun(token):
    """"The shift cut their costs": a subject opening the sentence with only "the"/"this" before it sums up the
    sentence before, and takes its sense from there. The model checks a swap in this sentence alone, so it
    can't tell "The transformation cut their costs" is wrong; such nouns are left alone. Their place is enough
    when spaCy misreads the verb after them ("The effort paid off": "effort" as the root, "paid" hanging from it)."""
    if token.pos_ != 'NOUN':
        return False
    doc, start, children = token.doc, token.sent.start, list(token.children)
    return ((token.dep_ in ('nsubj', 'nsubjpass') and token.i - start <= 2 and len(children) == 1
             and children[0].dep_ == 'det' and children[0].lower_ in SUMMARY_DETERMINERS)
            or (token.i == start + 1 and doc[start].lower_ in SUMMARY_DETERMINERS
                and token.i + 1 < token.sent.end and doc[token.i + 1].pos_ in ('VERB', 'AUX')))

def fixed_frame(token):
    """Words whose synonyms rarely fit the same construction: "rely on"-style pairs and verbs
    with a to-infinitive ("struggle to manage" is not "fight to manage")."""
    return takes_preposition(token) or (token.pos_ == 'VERB' and any(c.dep_ == 'xcomp' for c in token.children))

def curated_key(token):
    """The AI_WORDS / PLAIN_WORDS key for a token, reading a verb the tagger took for a noun as the verb
    ("... and facilitate digestion")."""
    lemma = token.lemma_.lower()
    key = f'{lemma}|{token.pos_}'
    if (token.pos_ == 'NOUN' and key not in AI_WORDS and f'{lemma}|VERB' in AI_WORDS
            and f'{lemma}|n' not in SYNONYMS and token.tag_ == 'NN'):
        return f'{lemma}|VERB'
    return key

def plan_word_edits(doc, tone, deep_mode, from_model=frozenset(), unlock_model=False, vet=None):
    """Layer 3: token index -> replacement text, for curated and WordNet swaps. Sentences Deep Rewrite
    wrote (their start indices in `from_model`) only get the swaps for overused AI words, because the
    model has already chosen their wording; with `unlock_model` (Chaos Mode) they also get the plain-
    English and everyday swaps. WordNet swaps are only made when `vet` can check them with the model;
    `vet(sent, edits)` returns the edits to keep (all of a block's guesses are checked in one batch, through
    vet_many). Chaos Mode makes every everyday swap that fits, so a sentence may get a second or third change."""
    vetted = vet is not None
    rate = CHAOS_WORD_RATE_VETTED if deep_mode else WORD_RATE[tone]
    floor = CHAOS_FLOOR_VETTED if deep_mode else SYNONYM_FLOOR[tone]
    plain_words = ACADEMIC_PLAIN if tone == 'academic' else PLAIN_WORDS
    plain_chance = 1.0 if deep_mode else ACADEMIC_PLAIN_CHANCE if tone == 'academic' else PLAIN_CHANCE[tone]
    everyday_chance = 1.0 if deep_mode else EVERYDAY_CHANCE[tone]
    noun_weight = NOUN_WEIGHT_VETTED[deep_mode] if vetted else 0.0
    keeps = TONE_KEEPS.get(tone, ())
    curated_wording = phrase_tokens(doc)
    edits, previous, to_vet = {}, set(), []
    sents = list(doc.sents)
    for k, sent in enumerate(sents):
        here = sentence_words(sent)
        following = sentence_words(sents[k + 1]) if k + 1 < len(sents) else set()
        # No echo of a neighbouring sentence ("threaten ... threaten", "looks into ... looked at"): the last
        # one as it now reads and the next one; an echo inside the sentence itself is never allowed
        avoid = here | previous | following
        used_here = set(here)
        model_wrote = sent.start in from_model
        guessed = {}                    # WordNet swaps in this sentence, vetted before they are kept
        for token in sent:
            extra = swap_key(token)
            if not (extra or can_rewrite(token)) or token.i in curated_wording or token.i in edits:
                continue        # (a token already rewritten as part of another swap: "carried it out")
            lemma = token.lemma_.lower()
            key = curated_key(token)
            word = None
            if key in AI_WORDS and curated_fits(token, key):
                choices = curated_choices(AI_WORDS[key], tone, token)
                # (a describing word never falls back on echoing a neighbouring sentence: "a vibrant music scene. Its
                # lively bars ..." keeps "vibrant" rather than say "lively" twice)
                replacement = (fresh_choice(choices, avoid) or echo_fallback(key, token, avoid)
                               or (None if token.pos_ == 'ADJ' else fresh_choice(choices, used_here)
                                   or echo_fallback(key, token, used_here)))
                if replacement is None:
                    continue    # every option is already in the sentence ("improve mood ... and improve")
            elif extra in AI_FAVOURITES or (extra in EVERYDAY_SWAPS and (unlock_model or not model_wrote)
                                            and random.random() < everyday_chance):
                replacement = fresh_choice(swap_choices(token, extra, tone), avoid)
                new = swap_edits(token, extra, replacement) if replacement is not None else None
                if new and new[token.i].lower() != token.lower_:
                    edits.update((i, match_case(doc[i].text, text)) for i, text in new.items())
                    remember(avoid, replacement)
                    remember(used_here, replacement)
                continue
            elif model_wrote and not unlock_model:
                continue
            elif (key in plain_words and curated_fits(token, key) and random.random() < plain_chance
                  and key not in keeps):
                replacement = fresh_choice(context_choices(plain_words[key], token), avoid)
                if replacement is None:
                    continue
            elif (not vetted or token.ent_type_ or fixed_frame(token) or key in AI_WORDS or key in PLAIN_WORDS
                  or extra or not can_rewrite(token) or model_wrote or linking_adverb(token) or summary_noun(token)
                  or adjectival_participle(token)):
                continue  # number phrases, fixed constructions, curated words out of place and model wording stay
            elif random.random() < rate * (noun_weight if token.pos_ == 'NOUN' else 1.0):
                entry = SYNONYMS.get(f'{lemma}|{UPOS_TO_WORDNET[token.pos_]}')
                if not entry:
                    continue
                replacement, word = choose_synonym(token, entry, tone, floor, avoid, token.pos_ == 'NOUN',
                                                   vetted=vetted and deep_mode)
                if word and word.lower() != token.lower_:
                    guessed[token.i] = match_case(token.text, word)
                    remember(avoid, replacement)
                    remember(used_here, replacement)
                continue
            else:
                continue
            word = word or (inflect(replacement, token.tag_) if replacement else None)
            if word and word.lower() != token.lower_:
                split = particle_split(token, word)
                if split:       # "speed them up"
                    word, pronoun, with_particle = split
                    edits[pronoun.i] = with_particle
                edits[token.i] = match_case(token.text, word)
                remember(avoid, replacement)
                remember(used_here, replacement)
        if guessed and vet is not None:
            to_vet.append((sent, guessed))
        else:
            edits.update(guessed)
        # The sentence as it now reads: the words it keeps and the replacements (guesses included, in case)
        previous = ({w for t in sent if t.i not in edits and t.i not in guessed for w in (t.lower_, t.lemma_.lower())}
                    | (used_here - here))
    if to_vet:
        edits.update(vet.many(to_vet) if hasattr(vet, 'many') else vet_many(to_vet) if vet is vet_swaps else
                     {i: w for sent, guessed in to_vet for i, w in vet(sent, guessed).items()})

    for i in list(edits):
        # "a" / "an" must agree with a swapped word that follows it
        if i > 0 and edits[i] and (i - 1) not in edits and doc[i - 1].lower_ in ('a', 'an'):
            wanted = 'an' if starts_with_vowel_sound(edits[i]) else 'a'
            if doc[i - 1].lower_ != wanted:
                edits[i - 1] = match_case(doc[i - 1].text, wanted)
        # A plural's apostrophe needs an "s" when the new word doesn't end in one: "individuals' risk" -> "people's risk"
        if (i + 1 < len(doc) and doc[i + 1].tag_ == 'POS' and doc[i + 1].text in ("'", "’") and (i + 1) not in edits
                and edits[i] and not edits[i].lower().endswith('s')):
            edits[i + 1] = doc[i + 1].text + 's'
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
    standard English spacing. A full stop that a move put after a word already ending in one is
    written once ("..., according to Acme Inc." never ends "Inc.."); the writer's own text stays."""
    out, previous = [], None
    for index, text in pieces:
        if not text:
            continue
        if previous is not None:
            prev_index, prev_text = previous
            neighbours = prev_index is not None and index is not None and index == prev_index + 1
            if text == '.' and prev_text.endswith('.') and not neighbours:
                continue
            if neighbours:
                out.append(doc[prev_index].whitespace_)
            elif not (text in NO_SPACE_BEFORE or apostrophe_attached(index, text, doc) or prev_text in NO_SPACE_AFTER):
                out.append(' ')
        out.append(text)
        previous = (index, text)
    return ''.join(out)

def keeps_capital(token):
    if token.dep_ == 'det' and token.lower_ in ('the', 'a', 'an'):
        return False    # "The" opening a name ("The Industrial Revolution") is lowercase mid-sentence
    return (token.pos_ == 'PROPN' or token.ent_type_ in NAME_ENTITIES or token.text == 'I'
            or (token.is_upper and len(token.text) > 1) or any(c.isupper() for c in token.text[1:]))

def first_piece(pieces):
    return next((p for p in pieces if p[1]), None)

def first_word_piece(pieces):
    """The first piece with a word in it: past an elision mark, which spaCy may split off its word ("'" + "Til")."""
    live = [p for p in pieces if p[1]]
    if len(live) > 1 and live[0][1] in ("'", '‘', '’') and re.fullmatch(ELISIONS, live[1][1]):
        return live[1]
    return live[0] if live else None

def capitalize_first(pieces):
    piece = first_word_piece(pieces)
    if piece:
        first, space, rest = piece[1].partition(' ')
        mark = first[:1] if first[:1] in ("'", '‘', '’') else ''     # an elision kept on its word: "'til" -> "'Til"
        piece[1] = mark + capitalized(first[len(mark):]) + space + rest     # "eBay", "iPhone" keep their spelling

def lowercase_first(pieces, doc):
    """Lower-cases the first word of pieces moved inside a sentence, unless it keeps its capital anywhere (a name,
    "I", an acronym). A word written in capitals stays so ("IN 1902, THE CORNERSTONE ...", not "tHE"), and a word
    the writer wrote in lower case goes back to it after a template capitalised it ("Mass spectrometry analysed the
    samples" -> "In 2020, mass spectrometry analysed the samples")."""
    piece = first_word_piece(pieces)
    if piece is None:
        return
    word = piece[1].split(' ')[0]
    if len(word) > 1 and word.isupper():
        return
    if piece is not first_piece(pieces) or re.fullmatch("['‘’]" + ELISIONS, piece[1]):
        # An elided word ("'Til midnight", "'Cause") is lower case inside a sentence, whatever its tag: spaCy often
        # takes "Til" or "Nuff" for a name
        piece[1] = re.sub('[A-Z]', lambda m: m.group(0).lower(), piece[1], count=1)
    elif piece[0] is None or not keeps_capital(doc[piece[0]]) or doc[piece[0]].text[:1].islower():
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

def cut_off_opening_mark(token):
    """True for an opening mark that spaCy made a sentence of its own, cut off the word it belongs to ("We stay open
    late. 'Til midnight ..." -> "'" and "Til midnight ...")."""
    doc = token.doc
    return (token.text in ("'", '‘', '"', '“') and not token.whitespace_ and token.is_sent_start
            and token.i + 1 < len(doc) and doc[token.i + 1].is_sent_start)

def restructurable(sent):
    """Plain statements only: ends in a full stop, no quotes, brackets, colons, dashes or links, and not
    inside a quotation, nor opened by a mark cut off it (which has to stay in front of its word)."""
    texts = {t.text for t in sent}
    return (sent[-1].text == '.' and 6 <= len(sent) <= 45 and sent.root.pos_ in ('VERB', 'AUX')
            and not texts & {'"', '“', '”', '‘', '(', ')', '[', ']', ':', '—', '–'}
            and not any(t.like_url or t.like_email for t in sent)
            and not has_quoted(sent.doc, range(sent.start, sent.end))
            and not (sent.start > 0 and cut_off_opening_mark(sent.doc[sent.start - 1])))

# --- Moving adverbial phrases and clauses between the start and the end of a sentence ---

TIME_NOUNS = {'year', 'month', 'week', 'day', 'decade', 'century', 'time', 'period', 'past', 'future', 'era',
              'morning', 'evening', 'night', 'quarter', 'season', 'hour', 'minute', 'moment', 'age', 'today', 'recent'}
ALWAYS_MOVABLE_PREPS = {'during', 'after', 'before', 'since', 'throughout', 'despite', 'following', 'amid', 'until'}
SETTING_PREPS = {'in', 'on', 'at', 'over', 'within', 'across', 'for'}     # movable when they give a time or place
UNMOVABLE_PREPS = {'of', 'as', 'like', 'than', 'to'}
MOVABLE_MARKS = {'because', 'since', 'although', 'though', 'when', 'whenever', 'while', 'after', 'before', 'if',
                 'unless', 'once', 'whereas', 'until', 'whether'}
# Keep a comma when moved to the end ("..., whether it's a walk or a workout")
CONCESSIVE = {'although', 'though', 'whereas', 'while', 'despite', 'even', 'whether'}
DISCOURSE_PHRASES = {
    'in addition', 'in short', 'in conclusion', 'in sum', 'in particular', 'in fact', 'in general', 'in contrast',
    'in the end', 'in other words', 'on top of that', 'beyond that', 'beyond this', 'for example', 'for instance',
    'of course', 'by contrast', 'as a result', 'because of this', 'because of that', 'on the other hand',
    'at the same time', 'all in all', 'after all', 'above all', 'at the end of the day', 'in turn', 'to date', 'so far',
    'in practice', 'in theory', 'in reality', 'in total', 'in brief', 'in summary', 'at first', 'at last', 'at times',
    'by default', 'on average', 'on balance', 'for now', 'for this reason', 'to this end', 'in this case',
    'in that case', 'in this way', 'with this in mind', 'as such', 'by comparison', 'in comparison',
}
# ...and every opener the phrase layer itself puts in ("That said,", "All told,"), which must stay in front
DISCOURSE_PHRASES |= {o.rstrip(',').lower() for _, _, options in TRANSITIONS for choices in options.values() for o in choices}
PRONOUN_SUBJECTS = {'it', 'they', 'he', 'she', 'this', 'these', 'those', 'that', 'them', 'its', 'their'}
# Verbs whose "to ..." is part of their meaning, not a purpose ("ask someone to help", "want to go")
TO_COMPLEMENT_VERBS = {
    'ask', 'tell', 'allow', 'enable', 'encourage', 'help', 'want', 'expect', 'require', 'force', 'persuade', 'invite',
    'urge', 'remind', 'teach', 'cause', 'get', 'lead', 'permit', 'advise', 'instruct', 'order', 'warn', 'need',
    'try', 'plan', 'hope', 'decide', 'seem', 'tend', 'begin', 'start', 'continue', 'fail', 'manage', 'struggle',
    'attempt', 'aim', 'intend', 'agree', 'refuse', 'promise', 'prefer', 'like', 'love', 'hate', 'choose', 'learn',
    'forget', 'remember', 'offer', 'wish', 'appear', 'happen', 'have', 'use', 'go', 'used', 'train', 'prepare',
}
# ...and verbs whose object is the one who does the "to ..." ("empower employees to reach their potential")
OBJECT_CONTROL = {'empower', 'inspire', 'motivate', 'equip', 'push', 'drive', 'prompt', 'compel', 'oblige', 'convince',
                  'challenge', 'pressure', 'trust', 'pay', 'hire', 'assign', 'select', 'appoint', 'mobilize', 'mobilise'}
PERSONAL_OBJECTS = {'me', 'you', 'him', 'her', 'us', 'them', 'everyone', 'everybody', 'someone', 'somebody', 'people'}
# ...and verbs whose passive takes a "to ..." of obligation or request ("be required/asked/expected to ..."), which
# spaCy may label advcl ("Further research is needed to determine ..." is a purpose, so "need" isn't one)
PASSIVE_COMPLEMENT_VERBS = {'require', 'ask', 'expect', 'force', 'oblige', 'compel', 'tell', 'order', 'instruct', 'allow',
                            'permit', 'encourage', 'advise', 'urge', 'invite', 'remind', 'warn', 'persuade', 'request',
                            'suppose'}
# ...and nouns that take a "to ..." of their own, which a clause after the verb's object may belong to ("unveiled a
# plan on Wednesday to build 5,000 homes": the plan is to build them, not the unveiling)
INFINITIVE_NOUNS = {'plan', 'scheme', 'proposal', 'programme', 'program', 'campaign', 'initiative', 'bid', 'effort',
                    'attempt', 'drive', 'push', 'project', 'strategy', 'policy', 'measure', 'bill', 'law', 'rule', 'deal',
                    'agreement', 'contract', 'decision', 'motion', 'vote', 'call', 'appeal', 'request', 'offer', 'promise',
                    'pledge', 'commitment', 'target', 'goal', 'mission', 'mandate', 'fund', 'grant', 'budget', 'money',
                    'package', 'move', 'step', 'order', 'permission', 'right', 'licence', 'license', 'chance',
                    'opportunity', 'deadline', 'way', 'tool', 'app', 'platform', 'system', 'service', 'power', 'reason'}

def adjunct_span(child):
    span = child.doc[child.left_edge.i: child.right_edge.i + 1]
    if len(span) > 1 and span[-1].text == ',':
        span = span[:-1]    # spaCy sometimes hangs the clause's closing comma on the clause itself
    # "Whether it's X, Y, or Z" lists its options, so it may run longer than other clauses
    longest = 20 if span[0].lower_ == 'whether' else 14
    if span.text.lower() in DISCOURSE_PHRASES or not 2 <= len(span) <= longest:
        return None
    first = span[0]
    if child.dep_ == 'advcl' and first.dep_ in ('mark', 'advmod') and first.head == child and first.lower_ in MOVABLE_MARKS:
        # "because"/"although" are tagged as markers, "when"/"while" as adverbs of the clause's verb
        return span
    if child.dep_ in ('advcl', 'xcomp'):
        # A purpose clause: "... every year to mark the end of British rule". spaCy often labels it
        # xcomp; it is only a purpose clause if the verb doesn't take "to ..." itself ("want to go",
        # "asked the team to review") and already has its object, which isn't a person or group
        # doing the action ("empower their employees to reach ...")
        purpose = (first.lower_ == 'to' and first.dep_ == 'aux' and first.head == child and child.tag_ == 'VB'
                   and 'VERB' in getAllLemmas(child.lower_))     # "contribute substantially to healthcare" isn't one
        # "The dish takes forty minutes to prepare", "The update cost a fortune to install": the "to ..." completes
        # "take"/"cost" ("To prepare, the dish takes forty minutes" says something else), and a clause with its
        # own finite verb ("to prepare and covered every branch") is no single purpose
        if purpose and (child.head.lemma_.lower() in ('take', 'cost')
                        or any(t.dep_ == 'conj' and t.tag_ in ('VBD', 'VBZ', 'VBP') for t in span)):
            purpose = False
        # "You'll be required to show photo ID": never "To show photo ID, you'll be required"
        if (purpose and child.dep_ == 'advcl' and child.head.lemma_.lower() in PASSIVE_COMPLEMENT_VERBS
                and any(c.dep_ == 'auxpass' for c in child.head.children)):
            purpose = False
        # "She was the only member of the team to finish the race": after "be" and a noun, the "to ..." says which
        # one ("the first to arrive", "the best way to learn"); "To finish the race, she was the only member of the
        # team" makes it a purpose
        if purpose and child.head.lemma_ == 'be' and any(c.dep_ == 'attr' for c in child.head.children):
            purpose = False
        if purpose and child.dep_ == 'advcl' and any(c.dep_ in ('dobj', 'nsubjpass') and c.lemma_.lower() in INFINITIVE_NOUNS
                                                      for c in child.head.children):
            purpose = False
        if purpose and child.dep_ == 'xcomp':
            verb = child.head
            objects = [c for c in verb.children if c.dep_ == 'dobj' and c.i < child.i]
            purpose = (verb.lemma_.lower() not in TO_COMPLEMENT_VERBS | OBJECT_CONTROL and objects
                       and not any(o.lemma_.lower() in GROUP_NOUNS or o.lower_ in PERSONAL_OBJECTS
                                   or o.ent_type_ in ('PERSON', 'ORG', 'NORP') for o in objects))
        return span if purpose else None
    return span if child.dep_ == 'prep' else None

BACK_POINTERS = {'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'he', 'him', 'his', 'himself',
                 'she', 'her', 'hers', 'herself'}

def points_back(span):
    """True if a phrase refers back to something earlier in its sentence ("although it also ...", "during that
    time"): moved to the front, it would come before what it refers to."""
    for t in span:
        w = t.lower_
        if w in BACK_POINTERS or w == 'also':
            return True
        if t.tag_ == 'DT' and w in ('this', 'that', 'these', 'those') and (t.dep_ != 'det' or w in ('that', 'those')):
            return True
        if w in ('too', 'either') and t.dep_ == 'advmod' and t.head.pos_ in ('VERB', 'AUX'):
            return True
    return 'as well' in span.text.lower()

def mid_sentence_words(doc):
    """The words written somewhere other than at the start of a sentence, worked out once per parsed text."""
    if 'mid_words' not in doc.user_data:
        doc.user_data['mid_words'] = frozenset(t.text for t in doc if not t.is_sent_start)
    return doc.user_data['mid_words']

# Common words that are also the names of places, people or firms ("Reading", "Bath", "Will", "Rose", "Jordan",
# "Shell"): at the start of a sentence, the capital may be the name's, whatever the tagger says
OPENING_NAME_HOMOGRAPHS = {'reading', 'bath', 'nice', 'mobile', 'buffalo', 'hope', 'will', 'bill', 'mark', 'rose', 'may',
                           'june', 'april', 'august', 'grace', 'faith', 'joy', 'jack', 'dawn', 'summer', 'turkey', 'china',
                           'chad', 'jordan', 'victoria', 'florence', 'phoenix', 'sandy', 'rich', 'frank', 'pat', 'sue',
                           'ray', 'guy', 'gene', 'amber', 'ruby', 'holly', 'ivy', 'iris', 'lily', 'daisy', 'robin',
                           'drew', 'wade', 'chase', 'cliff', 'dean', 'don', 'herb', 'max', 'miles', 'nick', 'pierce',
                           'reed', 'rod', 'sky', 'storm', 'wells', 'banks', 'march', 'polish', 'apple', 'amazon', 'shell',
                           'orange', 'next', 'boots', 'virgin', 'sage', 'dove', 'ford', 'cork', 'derby', 'deal', 'rugby',
                           'sale', 'hull', 'gap', 'target', 'subway', 'oracle', 'corona', 'lincoln', 'jersey', 'wren',
                           'hunter', 'baker', 'mason', 'carter', 'cooper', 'fisher', 'porter', 'tanner', 'parker'}

def uncertain_capital(token):
    """A capitalised first word that spaCy takes for a name may only be capitalised because it starts the
    sentence ("Insulin is produced by ...", "Oxygen", "Green algae"). Such a word is uncertain when it is also
    an ordinary English word or isn't a recognised name, unless it is capitalised inside a sentence too or
    starts a longer name ("Herman Melville"). So is a common word that is also a name (OPENING_NAME_HOMOGRAPHS),
    whatever spaCy takes it for ("Reading raised its parking fees in 2020": the town). A transform that would move it
    inside the sentence is skipped: it would either keep a wrong capital or lowercase a real name."""
    doc = token.doc
    if (token.i == token.sent.start and token.text[:1].isupper() and token.text[1:].islower()
            and token.lower_ in OPENING_NAME_HOMOGRAPHS and token.text not in mid_sentence_words(doc)):
        return True
    if token.i != token.sent.start or not token.text[:1].isupper() or not keeps_capital(token):
        return False
    if (token.is_upper and len(token.text) > 1) or any(c.isupper() for c in token.text[1:]):
        return False        # an acronym, or "iPhone"
    nxt = doc[token.i + 1] if token.i + 1 < len(doc) else None
    if nxt is not None and nxt.pos_ == 'PROPN' and nxt.text[:1].isupper():
        return False
    if token.text in mid_sentence_words(doc):
        return False
    return bool(getAllLemmas(token.lower_)) or token.ent_type_ not in NAME_ENTITIES

def opens_with_phrase(sent):
    """True if the sentence already starts with a phrase or clause and a comma ("In 2020, ...", "These days, ..."),
    or with a connective and a comma ("Plus, ...", "And, ...", "That said, ...", which spaCy reads as a clause)."""
    root, doc = sent.root, sent.doc
    comma = next((t.i for t in doc[sent.start: min(sent.end, sent.start + 7)] if t.text == ','), sent.start)
    lead = doc[sent.start:comma]
    if len(lead) and (' '.join(t.lower_ for t in lead) in DISCOURSE_PHRASES or (
            len(lead) <= 3 and all(t.pos_ in ('CCONJ', 'ADV', 'SCONJ', 'INTJ', 'PART') for t in lead))):
        return True
    if len(lead) and comma < root.i:
        # Words before an early comma that don't hold the main verb's subject open the sentence too ("Truth be
        # told,", "Weather permitting,"); so do words the parser took a subject from when the main verb has
        # another one after the comma ("All things considered, prices rose ...")
        def has_subject(span):
            return any(t.dep_ in ('nsubj', 'nsubjpass', 'expl', 'csubj') and t.head.i == root.i for t in span)
        if not has_subject(lead) or has_subject(doc[comma + 1: root.i]):
            return True
    opener = next((c for c in root.lefts if c.left_edge.i == sent.start), None)
    if opener is None or opener.dep_ not in ('prep', 'advcl', 'npadvmod', 'advmod'):
        return False
    end = opener.right_edge.i
    if doc[end].text == ',':
        return True
    return end + 1 < sent.end and doc[end + 1].text == ','

# A sentence opening with one of these stays opened by it when a phrase is moved to the front ("So in 2021,
# the company moved ..."; "In 2021, so the company moved ..." isn't English)
LEADING_CONNECTIVES = {'then', 'so', 'and', 'but', 'yet', 'or', 'also', 'still', 'now', 'thus', 'hence', 'plus'}
# ...of which "also" and "still" take a comma before a phrase moved in after them ("Also in 2021, ..." would mean
# "in 2021 too"), and these adverbs before a purpose clause ("Then to fetch the data, use ..." reads like a list of
# steps; "Then, to fetch the data, use ..." doesn't)
ADVERB_CONNECTIVES = {'then', 'also', 'still', 'now', 'thus', 'hence'}

UNTIL_ADVERBS = {'recently', 'now', 'then', 'today', 'tonight', 'tomorrow', 'yesterday', 'lately', 'very', 'quite',
                 'fairly', 'only', 'just', 'relatively', 'comparatively'}

def misparsed_root(sent):
    """True if the parse of a sentence can't be right, so none of its phrases is moved: its main verb is a bare "-ing"
    form ("However, constantly putting other people's needs before your own can lead to stress", where spaCy made
    "putting" the main verb and "before your own can lead to stress" a clause of it), a participle with neither a
    helping verb nor a subject, or a bare verb with a subject ("The sauce thickens as it cools, so add a splash of
    water before serving", read as "the sauce ... add")."""
    root = sent.root
    if any(c.dep_ in ('aux', 'auxpass') for c in root.children):
        return False
    subject = next((c for c in root.children if c.dep_ in ('nsubj', 'nsubjpass', 'expl')), None)
    return (root.tag_ == 'VBG' or (root.tag_ == 'VBN' and subject is None)
            or (root.tag_ == 'VB' and subject is not None and subject.lower_ != 'you'))

# Words that qualify the time or the place right after them ("shortly after midnight", "only on Sundays", "well before
# the deadline", "even after the war"): in front, the phrase would leave them behind ("After midnight, the plane landed
# shortly"). "late" and "early" do so before one day or part of one ("late on Sunday evening": "On Sunday evening, the
# team located the hikers late"), but not before a season or days in general ("We stay open late in summer", "We
# work late on Fridays"), see PHRASE_LEAD_TIMES
PHRASE_LEAD_ADVERBS = {'shortly', 'soon', 'long', 'well', 'right', 'straight', 'immediately', 'directly', 'midway',
                       'halfway', 'exactly', 'precisely', 'approximately', 'roughly', 'nearly', 'almost', 'sometime',
                       'around', 'about', 'only', 'just', 'even', 'mainly', 'mostly', 'largely', 'partly', 'primarily',
                       'chiefly', 'especially', 'particularly', 'exclusively', 'solely', 'merely', 'simply', 'also', 'not'}
# ...of which these may come along with the phrase when the parser hangs them on its preposition ("Shortly after
# midnight, the plane landed", "Long before the war, ...")
PHRASE_LEAD_MOVABLE = {'shortly', 'soon', 'long', 'just', 'right', 'immediately', 'straight', 'directly', 'well'}
PHRASE_LEAD_TIMES = {'late', 'early', 'later', 'earlier'}
SEASON_WORDS = {'spring', 'summer', 'autumn', 'fall', 'winter'}

def qualifies_phrase(word, child):
    """True if `word`, right before the phrase headed by `child`, qualifies that phrase (see PHRASE_LEAD_ADVERBS)."""
    if word.lower_ in PHRASE_LEAD_ADVERBS:
        return True
    if word.lower_ not in PHRASE_LEAD_TIMES:
        return False
    obj = next((c for c in child.children if c.dep_ == 'pobj'), None) if child.dep_ == 'prep' else None
    return obj is None or not (obj.tag_ in ('NNS', 'NNPS') or obj.lemma_.lower() in SEASON_WORDS)

# Prepositions that count from a time, after which a length of time says how long before or after it ("hours before
# the final", "seven days or more before your next payment date")
TIME_MEASURE_PREPS = {'before', 'after', 'since', 'until', 'till', 'following', 'ahead', 'prior', 'into'}
# Verbs of lasting, whose "until ..." is all they say ("The consultation will run until the end of September": in
# front, "Until the end of September, a public consultation on the proposal will run" ends on a bare verb)
LASTING_VERBS = {'run', 'continue', 'last', 'stay', 'remain', 'operate', 'extend', 'stretch', 'persist', 'wait', 'open'}
# Words that set two clauses against each other, which a fronted ", while ..." needs to read as a contrast ("Some
# workers flourish in this setting, while others miss the office"); without them it compares two facts ("The octopus
# was tender, while the lamb melted in the mouth"), and in front it would read as a concession or a time
WHILE_COMPARISON_WORDS = {'not', "n't", 'n’t', 'no', 'never', 'others', 'other', 'some', 'most', 'few', 'only', 'still',
                          'less', 'fewer', 'more', 'rarely', 'little', 'nothing', 'none'}
# What one finds on arriving or waking ("came home to find the door open", "woke up to discover ..."): what happened
# next, not a purpose ("To find the door open, he came home"). "go", "get" and the like arrive only with "home",
# "back" or the like ("got home to discover ..."; "went to the library to find a book" is a purpose)
ARRIVING_VERBS = {'wake', 'awake', 'awaken', 'come', 'return', 'arrive', 'emerge', 'reach'}
HOMECOMING_VERBS = {'get', 'go', 'walk', 'turn', 'step', 'head'}
HOMECOMING_WORDS = {'home', 'back', 'round', 'around', 'outside', 'downstairs', 'upstairs', 'in', 'out'}
FINDING_INFINITIVES = {'find', 'discover', 'realize', 'realise', 'notice', 'learn'}

# Words that start a comparison a "than" after them completes ("better in low light than last year's model")
COMPARING_WORDS = {'more', 'less', 'fewer', 'rather', 'other', 'else', 'different', 'differently'}

def carries_clause(span):
    """True if a phrase holds a clause with a subject of its own ("after 23 minutes when Summerville curled a shot into
    the far corner", "in 1998 when its founder retired"): moved, it reads as a fragment and the time loses its anchor.
    A relative clause whose subject is its "that"/"which"/"who" doesn't count ("after the war that devastated
    Europe")."""
    return any(t.i != span.root.i and t.pos_ in ('VERB', 'AUX') and t.dep_ in ('relcl', 'advcl', 'acl', 'ccomp')
               and any(c.dep_ in ('nsubj', 'nsubjpass', 'expl') and c.tag_ not in ('WDT', 'WP') for c in t.children)
               for t in span)

def outcome_infinitive(child, root):
    """True if a trailing "to ..." says what happened next rather than what for: after arriving or waking ("came home
    from work to find the door wide open", "woke up to rain"), after "only" ("only to discover ..."), "grew up to be",
    or "to find" with what was found ("arrived to find the building already ablaze")."""
    doc, main, lemma = child.doc, root.lemma_.lower(), child.lemma_.lower()
    arriving = main in ARRIVING_VERBS or (main in HOMECOMING_VERBS and any(
        c.lower_ in HOMECOMING_WORDS and c.dep_ in ('advmod', 'prt', 'npadvmod') for c in root.children))
    if child.dep_ == 'prep':
        return child.lower_ == 'to' and main in ('wake', 'awake', 'awaken', 'return', 'come')
    if child.dep_ not in ('advcl', 'xcomp') or not any(c.lower_ == 'to' for c in child.children):
        return False
    return (lemma in ('rain', 'snow') or (lemma == 'be' and main in ('grow', 'live'))
            or any(t.lower_ == 'only' for t in doc[max(child.sent.start, child.left_edge.i - 1): child.i])
            or (arriving and lemma in FINDING_INFINITIVES)
            or (main in ('wake', 'awake', 'awaken') and lemma in ('see', 'hear'))
            or (lemma == 'find' and any(c.dep_ in ('oprd', 'ccomp', 'xcomp', 'acomp') for c in child.children)))

def phrase_belongs_elsewhere(sent, child, span):
    """True if the phrase or clause ending a sentence (`span`, headed by `child`) belongs to a word or a clause before
    it, or says what it says only where it is, so that in front it would mean something else:
    - a word qualifies it ("late on Sunday evening", "shortly after midnight", "only on Sundays"), or a length of time
      counts up to it ("hours before the final", "seven days or more before your next payment date"; but "simmer for
      twenty minutes until the rice is tender" says how long the verb goes on); inside the phrase, only a length of
      time without a bound ("At least seven days before your next payment date, requests should be submitted" ends
      on a bare verb) or "shortly"/"long"/"just" may come along ("rose over 4% after the announcement");
    - it ends a list of the same kind of phrase ("at baseline, at eight weeks and at a three-month follow-up") or
      follows "and"/"or";
    - it limits a superlative ("its highest level since 2019") or completes a comparison ("better in low light than
      last year's model");
    - it says how long something takes ("complete the degree in three years": in front, "In three years," is a time
      to come);
    - it carries a clause of its own ("after 23 minutes when Summerville curled a shot into the far corner");
    - it ends a range ("from 8am until 8pm"), or is the only thing a verb of lasting says ("will run until the end of
      September"), or holds a second time ("until ten during the summer holidays");
    - it compares, after a comma, with ", while ..." (see WHILE_COMPARISON_WORDS);
    - it says what happened next (see outcome_infinitive)."""
    doc, root = sent.doc, sent.root
    first = span[0]
    before = doc[span.start - 1] if span.start > sent.start else None
    earlier = doc[sent.start: span.start]
    if before is not None and (qualifies_phrase(before, child) or before.lower_ in ('and', 'or', 'nor', 'but') or (
            first.lower_ in TIME_MEASURE_PREPS and ((before.lemma_.lower() in RELATIVE_TIME_UNITS and not (
                before.dep_ == 'pobj' and before.head.lower_ in ('for', 'over', 'within', 'during'))) or (
                before.lower_ in ('more', 'less', 'fewer', 'longer', 'earlier', 'later')
                and span.start - 2 >= sent.start and doc[span.start - 2].lower_ == 'or')))):
        return True
    if child.dep_ == 'prep' and child.left_edge.i < child.i:
        lead = doc[child.left_edge.i: child.i]
        measured = (any(t.lemma_.lower() in RELATIVE_TIME_UNITS for t in lead)
                    and not any(t.lower_ in ('least', 'most', 'up') for t in lead)
                    and all(t.dep_ in ('det', 'nummod', 'amod', 'quantmod', 'npadvmod', 'advmod') for t in lead))
        if not (measured or (len(lead) == 1 and lead[0].lower_ in PHRASE_LEAD_MOVABLE)):
            return True
    if (before is not None and before.text == ',' and child.dep_ == 'prep'
            and any(t.dep_ == 'prep' and t.lower_ == child.lower_ for t in earlier)):
        return True
    if first.lower_ == 'since' and any(t.tag_ in ('JJS', 'RBS') or t.ent_type_ == 'ORDINAL'
                                       or t.lower_ in ('first', 'last', 'only', 'record', 'ever') for t in earlier):
        return True
    words = {t.lower_ for t in span}
    if (('than' in words and any(t.tag_ in ('JJR', 'RBR') or t.lower_ in COMPARING_WORDS for t in earlier))
            or ('as' in words and any(t.lower_ in ('as', 'same', 'such') or t.tag_ in ('JJR', 'RBR') for t in earlier))):
        return True     # (a comparison of its own stays: "after more than a decade")
    counted = any(t.lemma_.lower() in RELATIVE_TIME_UNITS and any(
        c.dep_ == 'nummod' or c.lower_ in ('a', 'an', 'few', 'several', 'couple') for c in t.lefts) for t in span)
    if (first.lower_ == 'in' and child.dep_ == 'prep' and counted
            and not any(t.lower_ in BARE_TIME_DETERMINERS | {'the', 'first'} or t.dep_ == 'poss' for t in span)):
        return True
    if child.dep_ == 'prep' and carries_clause(span):
        return True
    if first.lower_ in ('until', 'till', 'to', 'through', 'thru') and any(t.lower_ == 'from' for t in earlier):
        return True
    if first.lower_ in ('until', 'till', 'through', 'throughout'):
        if root.lemma_.lower() in LASTING_VERBS and all(c.dep_ in ('prep', 'npadvmod', 'advmod', 'prt', 'punct')
                                                         for c in root.rights):
            return True
        if any(t.dep_ == 'prep' and t.lower_ in ('during', 'throughout', 'in', 'on', 'at', 'over', 'for')
               for t in span[1:]):
            return True
    if (child.dep_ == 'advcl' and first.lower_ in ('while', 'whilst') and before is not None and before.text == ','
            and any(c.dep_ in ('nsubj', 'nsubjpass', 'expl') for c in child.children)
            and not any(t.lower_ in WHILE_COMPARISON_WORDS for t in sent)):
        return True
    return outcome_infinitive(child, root)

def front_trailing_adjunct(sent, pieces):
    """"Many studies lack data despite recent advances." -> "Despite recent advances, many studies lack data.\""""
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    child = next((c for c in reversed(list(root.rights)) if not c.is_punct), None)
    if child is None or child.right_edge.i != stop - 1:
        return None
    if any(c.dep_ in ('ccomp', 'parataxis', 'conj') for c in root.lefts) or any(
            c.dep_ == 'cc' and c.i < child.i and (c.lower_ == 'so' or doc[c.i - 1].text == ',') for c in root.rights):
        return None     # "We'll be open late, so grab them while stocks last", "The rooms fill up fast, so book a
                        # table before the weekend": the phrase belongs to the second clause, and in front of the
                        # first it would say when that one happens
    if any(c.dep_ == 'conj' and c.i < child.i for c in root.rights):
        return None     # "Reduce the heat and simmer, until the rice is tender": the phrase may belong to the last verb
    span = adjunct_span(child)
    if span is None or points_back(span) or uncertain_capital(sent[0]) or span[0].i in phrase_tokens(doc):
        return None     # (the phrase layer's own wording stays whole: "To healthcare, AI can contribute ...")
    if misparsed_root(sent) or phrase_belongs_elsewhere(sent, child, span):
        return None
    if [t.lower_ for t in span] == ['at', 'a', 'time'] or any(
            t.lower_ == 'time' and t.i < span.end - 1 and any(
                c.lower_ in ('first', 'last', 'only', 'second', 'third') or c.ent_type_ == 'ORDINAL' for c in t.lefts)
            for t in span):
        return None     # "for 20 minutes at a time" (each time); "landed on the moon for the first time in 1969" (the
                        # first time ever): in front, "At a time, ..." and "For the first time in 1969, ..." mean else
    if doc[span.start - 1].dep_ == 'prep' and not any(c.dep_ in ('pobj', 'pcomp') for c in doc[span.start - 1].children):
        return None     # "open until late on Fridays": the move would strand "until"
    dated = dating_phrase(child)
    if dated is not None and (time_stays_put(sent, dated) or (
            not doc[dated.start - 1].is_punct and dated.start - 1 != root.i and time_could_attach(doc[dated.start - 1], root))):
        return None     # a time that belongs where it is, the same as for time_to_front: "She wrote about the riots in
                        # 1968", "Rovers travel to Leeds next weekend, while City host Bristol on Tuesday"
    if (dated is None and child.dep_ == 'prep' and child.lower_ in SETTING_PREPS and not doc[span.start - 1].is_punct
            and inside_noun_phrase(doc[span.start - 1], root)):
        return None     # ...and a place or a time after a noun inside a phrase: "sheds light on the lives of women in
                        # wartime Britain" (the women's)
    first = sent[0]
    lead = []
    if first.pos_ in ('CCONJ', 'ADV', 'SCONJ') and not (first.i + 1 < sent.end and doc[first.i + 1].text == ','):
        if first.lower_ not in LEADING_CONNECTIVES or first.head != root:
            return None     # "Recently the team ...", "Although ...": nothing goes in front of these
        lead = take(pieces, first.i, first.i)
    # A place or an event makes "in"/"at"/"across" a setting, but "for" only a time: "laid the groundwork for the
    # Reformation" says what the groundwork was for
    settings = ('DATE', 'TIME') if child.lower_ == 'for' else ('DATE', 'TIME', 'GPE', 'LOC', 'EVENT')
    if child.dep_ == 'prep' and child.lower_ not in ALWAYS_MOVABLE_PREPS and not (
            child.lower_ in SETTING_PREPS
            and any(t.ent_type_ in settings or t.lemma_.lower() in TIME_NOUNS for t in span)):
        return None
    if span[0].lower_ in ('until', 'till'):
        # An elliptical "until soft" or "until melted" stays with its verb ("Until soft, heat the oil and fry the
        # onion"), but "until recently", "until midnight" or "until the rice is tender" is a time like any other
        after = doc[span.start + 1: span.end]
        if child.dep_ == 'advcl':
            complete = any(c.dep_ in ('nsubj', 'nsubjpass', 'expl') for c in child.children)
        else:
            complete = all(t.lower_ in UNTIL_ADVERBS for t in after) or any(
                t.pos_ in ('AUX', 'NOUN', 'PROPN', 'NUM', 'PRON') for t in after)
        if not complete:
            return None
    if span.start - sent.start < 4 or (span.start <= root.i + 2 and takes_preposition(root)
                                       and child.lower_ not in ALWAYS_MOVABLE_PREPS):
        return None     # "rely on X" keeps its "on"; "during" never belongs to the verb
    if opens_with_phrase(sent):
        return None     # two fronted phrases in a row read badly ("To stay ahead, in today's market, ...")
    moved = take(pieces, span.start, span.end - 1)
    main = take(pieces, sent.start + len(lead), span.start - 1)
    while main and main[-1][1] in (',', ''):
        main.pop()      # "X, although Y." -> "Although Y, X."
    if lead:
        lowercase_first(moved, doc)
        if first.lower_ in ('also', 'still') or (first.lower_ in ADVERB_CONNECTIVES and span[0].lower_ == 'to'):
            lead.append([None, ','])
    else:
        capitalize_first(moved)
    lowercase_first(main, doc)
    return lead + moved + [[None, ',']] + main + take(pieces, stop, stop)

STANCE_ADJECTIVES = {'fair', 'honest', 'clear', 'sure', 'frank', 'blunt', 'brief', 'candid', 'precise', 'exact',
                     'truthful', 'certain'}

def back_fronted_adjunct(sent, pieces):
    """"When a user asks, the system checks the cache." -> "The system checks the cache when a user asks.\""""
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    child = next((c for c in root.lefts if c.left_edge.i == sent.start), None)
    if child is None or (child.dep_ == 'prep' and child.lower_ in UNMOVABLE_PREPS):
        return None
    span = adjunct_span(child)
    if span is None or span.end >= stop or doc[span.end].text != ',' or stop - (span.end + 1) < 4:
        return None
    if span[0].i in phrase_tokens(doc):
        return None     # an opener the phrase layer chose stays in front
    if child.lemma_ == 'be' and any(c.dep_ == 'acomp' and c.lower_ in STANCE_ADJECTIVES for c in child.children):
        return None     # "To be fair, X" comments on X: "X to be fair" reads as a purpose
    if child.dep_ == 'advcl' and span[0].lower_ in ('while', 'whilst') and (
            child.lemma_ in ('be', 'have') or any(t.dep_ == 'neg' or t.lower_ in ('no', 'little', 'few') for t in span)):
        return None     # "While there is no cure, there are ways to manage it" concedes; at the end, "..., while there
                        # is no cure" reads as an afterthought or a time ("While you wait, ..." still moves)
    if misparsed_root(sent) or any(c.dep_ == 'conj' and (c.pos_ == 'AUX' or any(
            g.dep_ in ('nsubj', 'nsubjpass', 'aux', 'auxpass') for g in c.children)) for c in root.children):
        return None     # "Before the 1450s, books were copied by hand and were extremely expensive to produce": at the
                        # end, the phrase would date only the second statement (verbs that share "were" still move)
    if child.dep_ == 'prep' and len(span) < 3:
        return None     # short signposts ("In practice,", "At first,") read oddly at the end
    if any(t.tag_ == 'DT' and t.lower_ in ('this', 'that', 'these', 'those') for t in span):
        return None     # "On this day," links back to the sentence before; at the end it reads as an afterthought
    if child.dep_ == 'prep' and child.lower_ not in ALWAYS_MOVABLE_PREPS | {'by', 'through', 'without'} and not (
            child.lower_ in SETTING_PREPS and any(t.ent_type_ in ('DATE', 'TIME', 'GPE', 'LOC', 'EVENT')
                                                  or t.lemma_.lower() in TIME_NOUNS for t in span)):
        return None     # "From healthcare to finance," at the end would attach to the last noun
    subject = next((c for c in root.children if c.dep_ in ('nsubj', 'nsubjpass', 'expl', 'csubj')), None)
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

# Times of the day and of the year, which spaCy may leave untagged ("by the autumn", "at dawn", "in the summer")
DEADLINE_TIMES = {'morning', 'afternoon', 'evening', 'night', 'midday', 'noon', 'midnight', 'dawn', 'daybreak', 'dusk',
                  'nightfall', 'sunrise', 'sunset', 'sundown', 'weekend', 'weekday', 'spring', 'summer', 'autumn',
                  'fall', 'winter', 'midsummer', 'midwinter', 'semester', 'holiday', 'vacation'}
# ...and what else a time is named after ("lunchtime", "closing time", "half term")
TIME_NAME_PARTS = DEADLINE_TIMES | {'lunch', 'dinner', 'supper', 'tea', 'breakfast', 'meal', 'bed', 'closing',
                                    'opening', 'half', 'play', 'show', 'harvest', 'home', 'break'}
# A part of a time: "in the middle of the night", "in the space of a week" (but "in the middle of the road" is a
# place). The parts in TIME_ONLY_PARTS are times whatever they are of: "on the eve of the election", "at the turn of
# the century", "in the course of the trial"
TIME_PART_NOUNS = {'middle', 'start', 'end', 'beginning', 'space', 'depth', 'depths', 'half', 'part', 'rest'}
TIME_ONLY_PARTS = {'eve', 'course', 'turn', 'dead', 'crack', 'heat', 'height', 'close'}

def time_of_day_or_year(noun):
    """True for "the autumn", "dawn", "the weekend": not for "the fall of Rome" or "a fall in prices", events."""
    lemma = noun.lemma_.lower()
    return lemma in DEADLINE_TIMES and not (lemma == 'fall' and any(c.dep_ == 'prep' for c in noun.children))

# Events, whose start, middle or end is a time ("at the end of the war"), unlike places ("the end of the corridor")
EVENT_NOUNS = {'war', 'battle', 'siege', 'game', 'match', 'season', 'election', 'campaign', 'trial', 'meeting', 'conference',
               'festival', 'tournament', 'concert', 'show', 'race', 'storm', 'crisis', 'pandemic', 'lockdown', 'strike',
               'revolution', 'occupation', 'reign', 'session', 'shift', 'trip', 'journey', 'voyage', 'performance',
               'ceremony', 'project', 'semester', 'term'}

def heads_time_phrase(prep):
    """True if the object of a preposition is a time ("in 1925", "during the night") or a part of one ("in the
    middle of the night"): not "in the history of the 1920s" or "in age-related diseases", which name a field."""
    obj = next((c for c in prep.children if c.dep_ == 'pobj'), None)
    if obj is None:
        return False
    lemma = obj.lemma_.lower()
    if obj.ent_type_ in ('DATE', 'TIME') or lemma in TIME_NOUNS or time_of_day_or_year(obj):
        return True
    of = next((c for c in obj.children if c.dep_ == 'prep' and c.lower_ == 'of'), None)
    event = next((c for c in of.children if c.dep_ == 'pobj'), None) if of is not None else None
    return of is not None and (lemma in TIME_ONLY_PARTS or (lemma in TIME_PART_NOUNS and (heads_time_phrase(of) or (
        event is not None and (event.lemma_.lower() in EVENT_NOUNS or event.ent_type_ == 'EVENT')))))

def names_deadline(noun):
    """True if the noun after a passive's "by" is a time, so "by" gives a deadline: a time does nothing itself ("by
    the autumn", "by lunchtime", "by closing time", "by half term", "by that time"). "A fall", "the fall of Rome",
    "mountain springs", "screen time" and "downtime" can be doers."""
    lemma = noun.lemma_.lower()
    if (noun.tag_ not in ('NN', 'NNS') or (noun.tag_ == 'NNS' and lemma not in ('holiday', 'vacation', 'break'))
            or any(c.dep_ == 'det' and c.lower_ in ('a', 'an') for c in noun.children)):
        return False
    if lemma in ('time', 'term', 'break', 'recess', 'interval'):
        return any((c.dep_ == 'det' and lemma == 'time') or (c.dep_ in ('compound', 'amod') and c.lower_ in TIME_NAME_PARTS | {'coffee', 'school'})
                   for c in noun.children)
    return time_of_day_or_year(noun) or (lemma.endswith('time') and lemma[:-4] in TIME_NAME_PARTS)

# "signed by hand", "sent by email", "deleted by mistake", "collected by questionnaire", and any bare "-ing" ("executed
# by hanging"): a bare noun after "by" is the means or manner, not the doer ("Hand signed the documents"); causes
# ("destroyed by fire") are left to convert
MEANS_NOUNS = {'hand', 'email', 'e-mail', 'mail', 'post', 'phone', 'telephone', 'fax', 'text', 'message', 'courier',
               'card', 'cash', 'check', 'cheque', 'credit', 'air', 'sea', 'land', 'road', 'rail', 'train', 'bus', 'car',
               'taxi', 'boat', 'ship', 'plane', 'bike', 'bicycle', 'foot', 'truck', 'van', 'accident', 'chance',
               'mistake', 'default', 'design', 'force', 'law', 'nature', 'name', 'heart', 'birth', 'request',
               'appointment', 'invitation', 'machine', 'computer', 'radio', 'satellite', 'video', 'wire', 'letter',
               'luck', 'proxy', 'consensus', 'vote', 'ballot', 'referendum', 'decree', 'contract', 'subscription',
               'weight', 'volume', 'telegram', 'airmail', 'tradition', 'definition', 'necessity', 'choice', 'habit',
               'questionnaire', 'interview', 'telegraph', 'surgery', 'injection', 'irrigation', 'auction', 'lottery',
               'election', 'spectrometry', 'microscopy', 'survey', 'observation', 'experiment', 'simulation',
               'algorithm', 'drone', 'radar', 'microscope', 'telescope', 'scan', 'ultrasound', 'lot', 'draw', 'raffle',
               'rota', 'rotation', 'agreement', 'negotiation', 'arbitration', 'consent', 'prescription', 'transfusion',
               'tender'}
# "raised by 10 percent", "approved by a vote of 6 to 3", "passed by a narrow majority": an amount or a margin after
# "by" isn't a doer either
AMOUNT_DOERS = {'percent', 'percentage', 'point', 'factor', 'half', 'third', 'quarter', 'margin', 'vote', 'degree',
                'amount', 'fraction', 'total', 'unit', 'cent', 'dollar', 'pound', 'euro', 'yen', 'rupee', 'inch', 'foot',
                'metre', 'meter', 'mile', 'kilometre', 'kilometer', 'minute', 'hour', 'day', 'week', 'month', 'year',
                'majority', 'minority'}
# ...nor a deadline: "finished by the end of the week", "submitted by the deadline", and dates ("by Friday")
DEADLINE_NOUNS = {'end', 'deadline', 'date', 'start', 'beginning', 'close', 'middle', 'noon', 'midnight', 'dawn',
                  'dusk', 'nightfall', 'sunset', 'sunrise', 'weekend'}
TIME_PREPOSITIONS = {'in', 'on', 'during', 'since', 'at', 'throughout'}

# Words that single out or measure the doer ("is caused mainly by", "was mainly caused by", "was criticised even by its
# supporters", "was written entirely by volunteers"): in the active they would land on the verb or after the object
# ("X causes climate change mainly", "X mainly causes it"), which says something else
BY_PHRASE_FOCUS = {'mainly', 'mostly', 'largely', 'partly', 'partially', 'primarily', 'chiefly', 'principally',
                   'predominantly', 'entirely', 'solely', 'wholly', 'exclusively', 'purely', 'only', 'just', 'even',
                   'especially', 'particularly', 'overwhelmingly', 'heavily', 'disproportionately', 'merely'}

def possessive_taken_over(subject, doer, doer_span):
    """True if the passive's subject holds a possessive ("its", "their", "his", "her") that the doer, put in front of it
    as the new subject, could be taken to own: "Its success was driven by a highly organized army" -> "A highly
    organized army drove its success" (the army's?), "His design was praised by the judge" -> "The judge praised his
    design". A doer that can't own it leaves it clear ("The floods destroyed their homes", "The judges praised his
    design")."""
    owners = {t.lower_ for t in subject.subtree if t.tag_ == 'PRP$'} - {t.lower_ for t in doer_span}
    if not owners:
        return False
    plural = (doer.tag_ in ('NNS', 'NNPS') or doer.lower_ in PLURAL_PRONOUNS
              or any(c.dep_ == 'conj' for c in doer.children))
    person = (names_people(doer) or denotes_people(doer) or doer.ent_type_ == 'PERSON'
              or doer.lower_ in ('he', 'him', 'she', 'her', 'they', 'them'))
    return (('its' in owners and not plural) or ('their' in owners and plural and person)
            or (bool(owners & {'his', 'her'}) and not plural and person))

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
    # A "by" after another phrase may belong to a noun in it ("is achieved through the absorption of light by
    # chlorophyll": the plants do the achieving), so only adverbs may stand between the verb and "by"; nor
    # may a second "by" follow ("was caused by the purchase of land by investors" stays as written). A plain date
    # may stand between, as it has no noun the "by" could belong to: "was laid in 1902 by the mayor" -> "The mayor
    # laid the cornerstone in 1902"; and a second "by" may give a deadline: "was repaired by the council by the
    # autumn of 1946" -> "The council repaired the damage by the autumn of 1946"
    date = next((c for c in root.children if root.i < c.i < agent.i and c.dep_ == 'prep'), None)
    if (date is not None and (dating_phrase(date) is None or date.right_edge.i + 1 != agent.i
                              or any(t.pos_ in ('NOUN', 'PROPN') and t.ent_type_ not in ('DATE', 'TIME')
                                     and t.lemma_.lower() not in TIME_NOUNS | DEADLINE_TIMES | TIME_MOVE_DAY_WORDS
                                     for t in dating_phrase(date)))):
        date = None
    if any(t.dep_ not in ('advmod', 'prt') for t in doc[root.i + 1: agent.i]
           if date is None or not date.i <= t.i <= date.right_edge.i):
        return None
    if date is not None and doer.pos_ == 'NOUN' and doer.tag_ == 'NN' and not any(
            c.dep_ in ('det', 'poss', 'nummod', 'predet') for c in doer.children):
        return None     # "was born in 1902 by caesarean section", "collected in 2019 by questionnaire": a means
    if agent.right_edge.i + 1 < sent.end and doc[agent.right_edge.i + 1].lower_ == 'by':
        deadline = doc[agent.right_edge.i + 1]
        until = next((c for c in deadline.children if c.dep_ == 'pobj'), None)
        if (until is None or deadline.head.i != root.i or deadline.right_edge.i != stop - 1
                or not (until.ent_type_ in ('DATE', 'TIME') or until.lemma_.lower() in DEADLINE_NOUNS
                        or names_deadline(until))):
            return None
    if uncertain_capital(doc[subject.left_edge.i]) or any(
            t.lower_ == 'et' and t.i + 1 < sent.end and doc[t.i + 1].lower_ in ('al', 'al.') for t in doc[agent.i: sent.end]):
        return None     # (a citation stays as written: "The study was led by Garcia et al.")
    # "The bridge was designed by Eiffel and completed in 1889": the second verb shares the passive "was" and
    # has no doer of its own, so it can't follow an active "Eiffel designed the bridge"
    if any(c.dep_ == 'conj' and c.tag_ in ('VBN', 'VBD') and not any(
            g.dep_ in ('nsubj', 'nsubjpass', 'aux', 'auxpass') for g in c.children) for c in kids):
        return None
    # A time phrase the parser hung on the doer goes to the end: "by a worker every five minutes", "by a group
    # of local artists in 1925". The time word must head the phrase: "by experts in age-related diseases" and
    # "by specialists in time management" name the doers' field
    doer_end = agent.right_edge.i
    trailing = next((c for c in doer.rights if c.dep_ in ('appos', 'npadvmod') and is_time_phrase(c)), None)
    if trailing is not None:
        doer_end = trailing.left_edge.i - 1
    late = next((t for t in doc[doer.i + 1: agent.right_edge.i + 1] if t.dep_ == 'prep' and t.lower_ in TIME_PREPOSITIONS | {'within', 'over'}
                 and t.right_edge.i == agent.right_edge.i and heads_time_phrase(t)), None)
    if late is not None:
        doer_end = min(doer_end, late.i - 1)
    doer_span = doc[agent.i + 1: doer_end + 1]
    if any(t.dep_ == 'prep' and t.head.i == doer.i and t.lower_ in ('in', 'at', 'near', 'inside', 'outside')
           and any(c.dep_ == 'pobj' and c.ent_type_ in ('GPE', 'LOC', 'FAC') for c in t.children) for t in doer_span):
        return None     # "signed by the two presidents in Geneva": the place may be where it was signed, not theirs
    bare = not any(c.dep_ in ('det', 'poss', 'nummod', 'predet') for c in doer.children)
    # A means, an amount or a deadline: "Hand signed the documents", "10 percent raised the price", "The end of the
    # week finished the report", "The autumn adopted the rules", "Lunchtime cleaned the rooms"
    if ((doer.tag_ == 'NN' and bare and (doer.lemma_.lower() in MEANS_NOUNS or doer.lower_.endswith('ing')))
            or doer.like_num or doer.ent_type_ in ('PERCENT', 'MONEY', 'QUANTITY', 'CARDINAL', 'DATE', 'TIME')
            or doer.lemma_.lower() in AMOUNT_DOERS or doer.lemma_.lower() in DEADLINE_NOUNS or names_deadline(doer)
            or (doer.lemma_.lower() == 'show' and any(t.lower_ == 'hands' for t in doer.subtree))):
        return None     # (and "by a show of hands", a vote)
    if any(t.text in (',', ';') or t.dep_ == 'relcl' for t in doer_span):
        return None     # "by the board, which met on Monday" would lose its closing comma at the front
    if any(t.lower_ in BY_PHRASE_FOCUS for t in doc[auxpass[0].i + 1: agent.i + 1]) or any(
            c.dep_ == 'advmod' and c.lower_ in BY_PHRASE_FOCUS for c in agent.children):
        return None     # "is caused mainly by", "was mainly caused by", "was criticised even by its supporters"
    if any(t.lower_ in ('as', 'than') for t in doc[doer_end + 1: stop]) and (
            doc[agent.right_edge.i + 1].lower_ in ('as', 'than') or any(
                t.lower_ in ('same', 'such', 'other', 'different', 'similar') or t.tag_ in ('JJR', 'RBR')
                for t in doer_span)):
        return None     # "is caused by the same virus as chickenpox": the comparison would stay after the object ("The
                        # same virus causes it as chickenpox")
    if possessive_taken_over(subject, doer, doer_span):
        return None
    if any(c.dep_ == 'conj' for c in doer.children) and any(t.lower_ in SUBJECT_FORMS for t in doer_span):
        return None     # "by me and two other members" can't become "Me and two other members rejected ..."
    if auxpass[0].lower_ in ('was', 'were'):
        tag = 'VBD'
    else:
        plural = (doer.tag_ in ('NNS', 'NNPS') or doer.lower_ in PLURAL_PRONOUNS
                  or any(c.dep_ == 'conj' for c in doer.children))
        tag = 'VBP' if plural else 'VBZ'
    forms = getInflection(root.lemma_, tag=tag)
    if not forms:
        return None
    # A regular past is spelt like the participle the writer used: "was cancelled by" -> "cancelled", not the first
    # form lemminflect gives ("canceled")
    verb = next((f for f in forms if f.lower() == root.lower_), forms[0])
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
    return head + doers + adverbs + [[None, verb]] + receivers + rest + take(pieces, stop, stop)

# --- "Task managers allow you to plan your day." -> "With task managers, you can plan your day." ---

ENABLING_VERBS = {'allow', 'enable', 'let'}
# Places and venues, which permit things the way people and bodies do ("The museum allows visitors to take photos" isn't
# "With the museum, visitors can take photos")
PERMITTING_PLACE_NOUNS = {'museum', 'gallery', 'park', 'library', 'hotel', 'hostel', 'restaurant', 'cafe', 'bar', 'pub',
                          'airline', 'airport', 'gym', 'festival', 'venue', 'stadium', 'arena', 'zoo', 'theatre',
                          'theater', 'cinema', 'campsite', 'resort', 'beach', 'store', 'shop', 'supermarket', 'market',
                          'hospital', 'clinic', 'church', 'prison', 'office', 'building', 'site', 'campus', 'university',
                          'college', 'embassy', 'landlord', 'employer', 'host', 'organiser', 'organizer'}

def repeated_infinitive_marks(action, stop):
    """The "to" of every verb listed after an infinitive (`action`) before token `stop`, which goes when the first
    "to" goes: "to spend time outdoors and to meet people", "to plan sprints, to track bugs and to ship faster", "to
    pause the internet and to schedule bedtimes" (the parser may not join those verbs, or may take the verb for a
    noun, but it tags such a "to" as a verb's). A "to" before a noun stays ("to send money to friends and to
    family")."""
    doc = action.doc
    marks = {c.i - 1 for c in action.conjuncts if c.i > action.i and doc[c.i - 1].lower_ == 'to'}
    return marks | {t.i for t in doc[action.i + 1: stop] if t.lower_ == 'to' and doc[t.i - 1].lower_ in (',', 'and', 'or')
                    and (t.tag_ == 'TO' or (t.i + 1 < stop and doc[t.i + 1].tag_ == 'VB'))}

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
    if (subject.lower_ in PRONOUN_SUBJECTS or subject.left_edge.i != sent.start or subject.pos_ == 'PRON'
            or uncertain_capital(sent[0])):
        return None
    if root.lower_ == 'let' and (subject.tag_ in ('NNS', 'NNPS') or any(c.dep_ == 'conj' for c in subject.children)):
        return None     # "Online forms let customers renew ...": a plural "let" is present or past alike, whatever its
                        # tag, so neither "can" nor "could" is sure to keep the tense
    if ((names_people(subject) and not any(c.dep_ == 'compound' for c in subject.children))
            or subject.lemma_.lower() in PERMITTING_PLACE_NOUNS or subject.ent_type_ in ('GPE', 'FAC', 'LOC')):
        return None     # a person, a body or a place gives permission: "The court allowed the appeal to proceed", "My
                        # parents allowed me to stay out" aren't "With the court, ...", "With my parents, ..." (but
                        # "task managers" are tools)
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
    # "to spend time outdoors and to meet people" -> "can spend time outdoors and meet people"
    again = repeated_infinitive_marks(action, stop)
    return ([[None, 'With']] + means + [[None, ',']] + actor + tense + adverb_pieces
            + [p for p in take(pieces, action.i, stop - 1) if p[0] not in again] + take(pieces, stop, stop))

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
        if of.right_edge.i + 1 < len(doc) and doc[of.right_edge.i + 1].lower_ == 'by':
            continue    # "the absorption of light by chlorophyll" names its doer: "absorbing light by chlorophyll" loses it
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

# --- "has transformed the way students learn" -> "has transformed how students learn" ---
# Only "the way" followed by a clause with its own subject ("the way (that/in which) we work"), as an object,
# subject or complement, or after "about"/"of"/"to"/"at": "on the way" and "by the way" mean something else.

WAY_PREPOSITIONS = {'about', 'of', 'to', 'at'}
# "The way" as a route or meaning "like" stays: "find the way we came", "blocked the way we take to school",
# "closed the way the trucks use", "learn the way their parents do", "drives the way his father drove", "showed
# us the way the miners walked", "shortened the way we drive to work". These verbs only say how when they have an
# object of their own or are passive ("the way we do business", "the way people go about paying their bills", "the
# way things are done"), and "used to" does when it is a habit ("the way we used to work"), not a way used to get
# somewhere ("the way the farmers used to reach the market")
WAY_ROUTE_VERBS = {'come', 'go', 'take', 'do', 'use', 'follow'}
WAY_ARRIVAL_VERBS = {'reach', 'get', 'access', 'enter', 'cross'}
WAY_PATH_HEADS = {'block', 'find', 'lose', 'light', 'clear', 'bar', 'retrace', 'pave', 'cover', 'obstruct'}
WAY_MOTION_VERBS = {'walk', 'drive', 'travel', 'run', 'ride', 'fly', 'head', 'return', 'enter', 'leave', 'follow',
                    'cross', 'climb', 'sail', 'hike', 'cycle', 'commute', 'wander', 'march', 'move'}
WAY_SHOWING_HEADS = {'show', 'know', 'lead', 'point', 'tell', 'ask', 'mark', 'map'}
# ...or with a destination ("to work", "home"): then it is the route
WAY_DESTINATIONS = {'to', 'into', 'from', 'up', 'down', 'across', 'through', 'home', 'here', 'there'}
# spaCy often reads the object of "has changed the way people communicate" as an adverb (npadvmod); after a verb of
# changing that has no other object, it is the object
WAY_CHANGE_VERBS = {'change', 'transform', 'revolutionize', 'revolutionise', 'reshape', 'redefine', 'alter', 'reinvent',
                    'shape', 'influence', 'affect', 'improve', 'disrupt', 'rethink', 'reimagine'}

# Verbs that take "the way ..." as their object, which "how ..." can replace ("love the way she explains things",
# "explains the way the brain stores memories", "miss the way we used to talk"). After any other verb it may say "as"
# or "like" ("Prices shifted the way the model predicted", "The project turned out the way we hoped"), and the parser
# reads it as an object all the same
WAY_OBJECT_VERBS = {'love', 'like', 'hate', 'dislike', 'enjoy', 'admire', 'appreciate', 'miss', 'remember', 'recall',
                    'notice', 'see', 'watch', 'observe', 'understand', 'know', 'learn', 'teach', 'explain', 'describe',
                    'show', 'reveal', 'study', 'examine', 'analyse', 'analyze', 'explore', 'investigate', 'research',
                    'consider', 'question', 'challenge', 'criticise', 'criticize', 'determine', 'define', 'control',
                    'govern', 'regulate', 'reflect', 'mirror', 'capture', 'document', 'highlight', 'track', 'measure',
                    'monitor', 'assess', 'evaluate', 'review', 'discuss', 'illustrate', 'demonstrate', 'compare',
                    'model', 'simulate', 'map', 'trace', 'redesign', 'rework', 'modernise', 'modernize', 'streamline',
                    'simplify', 'standardise', 'standardize', 'optimise', 'optimize', 'adjust', 'adapt', 'modify',
                    'refine', 'overhaul', 'upgrade', 'imagine', 'picture', 'envision', 'resent', 'celebrate', 'praise',
                    'copy', 'imitate', 'mimic', 'replicate', 'dictate'}
# "the way X predicted/wanted/hoped (it would)" after "changed"/"improved" with no object says "as X predicted"
WAY_EXPECTING_VERBS = {'predict', 'expect', 'hope', 'want', 'wish', 'plan', 'intend', 'say', 'promise', 'fear', 'imagine',
                       'forecast', 'anticipate', 'suggest', 'describe', 'advise', 'ask', 'need', 'mean', 'envisage',
                       'envision', 'foresee', 'dream', 'require', 'recommend', 'tell', 'claim', 'warn', 'demand',
                       'propose', 'picture', 'project'}

def the_way_to_how(sent, pieces):
    doc = sent.doc
    current = {p[0]: p[1] for p in pieces if p[0] is not None}
    for token in sent:
        if token.lower_ != 'way' or token.tag_ != 'NN' or token.i < 1 or token.i + 2 >= sent.end:
            continue
        det = doc[token.i - 1]
        if det.lower_ != 'the' or det.head != token or current.get(det.i) != det.text or current.get(token.i) != token.text:
            continue
        if not (token.dep_ in ('nsubj', 'attr')
                or (token.dep_ == 'dobj' and token.head.lemma_.lower() in WAY_CHANGE_VERBS | WAY_OBJECT_VERBS)
                or (token.dep_ == 'pobj' and token.head.lower_ in WAY_PREPOSITIONS)
                or (token.dep_ == 'npadvmod' and token.head.lemma_.lower() in WAY_CHANGE_VERBS
                    and not any(c.dep_ == 'dobj' for c in token.head.children))):
            continue
        clause = next((c for c in token.children if c.dep_ == 'relcl'), None)
        if clause is None or any(c.dep_ not in ('det', 'relcl') for c in token.children):
            continue
        verb, head = clause.lemma_.lower(), token.head.lemma_.lower()
        if token.dep_ == 'npadvmod' and ({verb} | set(getAllLemmas(clause.lower_).get('VERB', ()))) & WAY_EXPECTING_VERBS and not any(
                c.dep_ in ('dobj', 'xcomp', 'ccomp') and c.tag_ != 'MD' for c in clause.children):
            continue    # "Sales improved the way the analysts predicted": as they predicted, not how they predict
        route = verb in WAY_ROUTE_VERBS and (not any(c.dep_ in ('dobj', 'nsubjpass', 'auxpass') or c.lower_ == 'about'
                                                     for c in clause.children)
                                             or (head not in WAY_CHANGE_VERBS and any(
                                                 c.dep_ in ('prep', 'advmod') and c.lower_ in WAY_DESTINATIONS
                                                 for c in clause.children)))
        purpose = next((c for c in clause.children if c.dep_ == 'xcomp'), None)
        if route and verb == 'use' and clause.tag_ == 'VBD' and purpose is not None:
            goal = purpose.lemma_.lower()
            route = goal in WAY_ARRIVAL_VERBS or head not in WAY_CHANGE_VERBS and (
                     (goal in WAY_ROUTE_VERBS and not any(c.dep_ in ('dobj', 'nsubjpass', 'auxpass', 'prt')
                                                             or c.lower_ == 'about' for c in purpose.children))
                     or (goal in WAY_MOTION_VERBS and any(c.dep_ in ('prep', 'advmod') and c.lower_ in WAY_DESTINATIONS
                                                          for c in purpose.children)))
        motion = verb in WAY_MOTION_VERBS and (head in WAY_SHOWING_HEADS or any(
            c.dep_ in ('prep', 'advmod') and c.lower_ in WAY_DESTINATIONS for c in clause.children))
        if route or motion or verb == head or (token.dep_ == 'dobj' and head in WAY_PATH_HEADS):
            continue
        start, skip = token.i + 1, set()
        if doc[start].lower_ == 'that':
            skip, start = {start}, start + 1
        elif doc[start].lower_ == 'in' and doc[start + 1].lower_ == 'which':
            skip, start = {start, start + 1}, start + 2
        subject = next((c for c in clause.children if c.dep_ in ('nsubj', 'nsubjpass')), None)
        if subject is None or subject.left_edge.i != start or has_quoted(doc, range(det.i, clause.i + 1)):
            continue
        if doc[start].lower_ in ('many', 'much', 'few', 'little'):
            continue    # "how many students access education" would ask how many do
        new = [list(p) for p in pieces]
        for piece in new:
            if piece[0] == det.i or piece[0] in skip:
                piece[1] = ''
            elif piece[0] == token.i:
                piece[1] = 'How' if det.i == sent.start else 'how'
        return new
    return None

# --- "Photosynthesis is the process by which plants make food." -> "Photosynthesis is how plants make food." ---

PROCESS_NOUNS = {'process', 'method', 'means', 'mechanism'}
# The endings of nouns that name a process ("Voting", "Taxation", "Diffusion", "Osmosis", "Government"), which "is
# how" describes; a thing used as the means reads oddly with it ("The gun was how he killed her")
PROCESS_ENDINGS = ('ing', 'tion', 'sion', 'sis', 'ment')

def process_to_how(sent, pieces):
    """"X is the process by which S" -> "X is how S" (also "the method/means/mechanism by/through which"). Only "the"
    and the bare noun as what "is" says the subject is, with the clause running to the end and its own subject right
    after "which"; only one process (not "These are the means by which we live": "These are how") named by its verb
    or a noun of a process (see PROCESS_ENDINGS). The dropped words' slots stay in `pieces`."""
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    if root.lemma_ != 'be' or any(c.dep_ in ('neg', 'aux') for c in root.children):
        return None
    noun = next((c for c in root.children if c.dep_ == 'attr' and c.i > root.i), None)
    if (noun is None or (noun.lemma_.lower() not in PROCESS_NOUNS and noun.lower_ not in PROCESS_NOUNS)
            or noun.i != root.i + 2 or noun.i + 4 > stop):
        return None
    topic = next((c for c in root.children if c.dep_ in ('nsubj', 'csubj') and c.i < root.i), None)
    if (topic is None or root.lower_ in ('are', 'were', "'re", '’re') or topic.tag_ in ('NNS', 'NNPS')
            or (noun.tag_ == 'NNS' and noun.lower_ != 'means')
            or not (topic.tag_ == 'VBG' or topic.dep_ == 'csubj'
                    or (topic.pos_ in ('NOUN', 'PROPN') and topic.lower_.endswith(PROCESS_ENDINGS)))):
        return None
    det, prep, which = doc[noun.i - 1], doc[noun.i + 1], doc[noun.i + 2]
    clause = next((c for c in noun.children if c.dep_ == 'relcl'), None)
    if (det.lower_ != 'the' or det.head.i != noun.i or prep.lower_ not in ('by', 'through') or which.lower_ != 'which'
            or clause is None or clause.left_edge.i != prep.i or clause.right_edge.i != stop - 1
            or any(c.dep_ not in ('det', 'relcl') for c in noun.children)):
        return None
    subject = next((c for c in clause.children if c.dep_ in ('nsubj', 'nsubjpass')), None)
    texts = {p[0]: p[1] for p in pieces if p[0] is not None}
    if (subject is None or not subject.left_edge.i <= which.i + 1 <= subject.i or has_quoted(doc, range(det.i, which.i + 1))
            or any(texts.get(i) != doc[i].text for i in range(det.i, which.i + 1))):
        return None
    new = [list(p) for p in pieces]
    for piece in new:
        if piece[0] in (det.i, prep.i, which.i):
            piece[1] = ''
        elif piece[0] == noun.i:
            piece[1] = 'how'
    return new

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
    if doc[relative.i - 1].is_punct:
        return None     # ", who help out" adds a comment; without "there are" it would read "volunteers, help out"
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

# Adjectives that also take "to" + a noun ("key to progress", "vital to growth"); "wise", "easy" or "hard" don't
TO_NOUN_ADJECTIVES = {'important', 'essential', 'crucial', 'necessary', 'vital', 'helpful', 'useful', 'common',
                      'critical', 'beneficial', 'better', 'good', 'key'}
GERUND_NOUN_READING = re.compile(r"\s*(?:[.,;:!?)]|$)|\s+(?:of|in|across|between)\b")

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
    if (adjective.lower_ in TO_NOUN_ADJECTIVES and 'NOUN' in getAllLemmas(action.lower_)
            and (GERUND_NOUN_READING.match(doc.text, action.idx + len(action.text))
                 or any(c.dep_ == 'conj' and c.pos_ in ('NOUN', 'PROPN') for c in action.children))):
        return None     # "It is key to progress." says what progress needs: it isn't "Progressing is key."
    if any(c.dep_ == 'ccomp' or (c.dep_ == 'xcomp' and any(g.dep_ == 'mark' or g.tag_ in ('WDT', 'WP', 'WP$', 'WRB')
                                                          for g in c.subtree)) for c in action.children):
        return None     # "It is easy to forget that most of these tools are new": the gerund would carry the whole
                        # clause before a short "is easy" ("Forgetting that most of these tools are new is easy")
    prefix = leading_signpost(sent, it.i)
    if prefix is None:
        return None
    subject = gerund_phrase(pieces, action, stop)
    if subject is None:
        return None
    verb = 'was' if root.lower_ == 'was' else 'is'
    head = take(pieces, prefix.start, prefix.end - 1) if len(prefix) else []
    if head:
        lowercase_first(subject, doc)
    else:
        capitalize_first(subject)
    return (head + subject + [[None, verb]] + take(pieces, adjective.left_edge.i, adjective.i)
            + take(pieces, stop, stop))

# --- "This highlights the importance of sleep." -> "This shows why sleep matters." ---

SHOW_VERBS = {'highlight', 'underscore', 'emphasize', 'emphasise', 'stress', 'demonstrate', 'show', 'illustrate',
              'reinforce', 'reveal', 'underline'}

def importance_of(sent, pieces):
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    if root.lemma_.lower() not in SHOW_VERBS or root.tag_ not in ('VBZ', 'VBP', 'VBD'):
        return None
    obj = next((c for c in root.children if c.dep_ == 'dobj'), None)
    if obj is None or obj.lemma_.lower() not in ('importance', 'significance', 'value') or obj.left_edge.i != root.i + 1:
        return None
    det = doc[obj.i - 1]
    of = doc[obj.i + 1] if obj.i + 1 < stop else None
    if det.lower_ != 'the' or det.head != obj or of is None or of.lower_ != 'of' or of.head != obj:
        return None
    thing = next((c for c in of.children if c.dep_ == 'pobj'), None)
    if thing is None or obj.right_edge.i != stop - 1 or any(c.dep_ not in ('det', 'prep') for c in obj.children):
        return None     # "the great importance of", "the importance of X for Y"
    if any(c.dep_ == 'prep' and c.i != of.i for c in obj.children) or any(
            t.i > thing.i and t.dep_ not in ('conj', 'cc', 'punct', 'det', 'amod', 'compound', 'nummod', 'poss', 'case')
            for t in thing.subtree):
        return None     # "the importance of hedgerows as corridors for wildlife", "... of trees in cooling city streets"
    forms = getInflection('show', tag=root.tag_)
    if not forms or any(c.dep_ in ('aux', 'neg') for c in root.children):
        return None
    plural = thing.tag_ in ('NNS', 'NNPS') or any(c.dep_ == 'conj' for c in thing.children)
    matter = 'mattered' if root.tag_ == 'VBD' else ('matter' if plural else 'matters')
    return (take(pieces, sent.start, root.i - 1) + [[None, forms[0]], [None, 'why']]
            + take(pieces, of.i + 1, obj.right_edge.i) + [[None, matter]] + take(pieces, stop, stop))

# --- "It is crucial for businesses to adapt." -> "Businesses need to adapt." ---

NEED_ADJECTIVES = {'essential', 'crucial', 'vital', 'important', 'necessary', 'imperative', 'critical'}
NP_PARTS = {'DET', 'ADJ', 'NOUN', 'PROPN', 'PRON', 'NUM', 'CCONJ'}

def essential_for_to(sent, pieces):
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    if root.lemma_ != 'be' or root.lower_ not in ('is', 'was', "'s", '’s') or any(c.dep_ == 'neg' for c in root.children):
        return None
    it = next((c for c in root.children if c.dep_ in ('nsubj', 'expl') and c.lower_ == 'it'), None)
    adjective = next((c for c in root.children if c.dep_ == 'acomp' and c.lower_ in NEED_ADJECTIVES), None)
    if it is None or adjective is None or it.i + 1 != root.i or adjective.i != root.i + 1:
        return None
    start = adjective.i + 1
    if start >= stop or doc[start].lower_ != 'for':
        return None
    to = next((j for j in range(start + 2, min(start + 8, stop - 1)) if doc[j].lower_ == 'to' and doc[j + 1].tag_ == 'VB'), None)
    if to is None:
        return None
    phrase = doc[start + 1: to]
    if not all(t.pos_ in NP_PARTS for t in phrase) or not any(t.pos_ in ('NOUN', 'PROPN', 'PRON') for t in phrase):
        return None
    prefix = leading_signpost(sent, it.i)
    if prefix is None:
        return None
    head = [t for t in phrase if t.pos_ in ('NOUN', 'PROPN', 'PRON')][-1]
    plural = (head.tag_ in ('NNS', 'NNPS') or head.lower_ in PLURAL_PRONOUNS or any(t.lower_ == 'and' for t in phrase))
    verb = 'needed' if root.lower_ == 'was' else ('need' if plural or head.lower_ == 'you' else 'needs')
    subject = take(pieces, start + 1, to - 1)
    if len(subject) == 1 and subject[0][1].lower() in SUBJECT_FORMS:
        subject[0][1] = SUBJECT_FORMS[subject[0][1].lower()]
    lead = take(pieces, prefix.start, prefix.end - 1) if len(prefix) else []
    if lead:
        lowercase_first(subject, doc)
    else:
        capitalize_first(subject)
    return lead + subject + [[None, verb]] + take(pieces, to, stop - 1) + take(pieces, stop, stop)

# --- "Caching is a technique that stores data." -> "Caching stores data." ---
# Defining something by a generic category ("a technique that", "a tool that") is a stock frame of AI text; what
# it does says the same. Only bare categories: "a powerful tool that" keeps its describing word, and its frame.

CATEGORY_NOUNS = {'technique', 'method', 'approach', 'strategy', 'tool', 'process', 'practice', 'mechanism', 'procedure',
                  'way', 'term', 'concept', 'tactic', 'destination', 'platform', 'solution', 'resource', 'service',
                  'skill', 'habit', 'framework', 'feature', 'program', 'programme', 'initiative', 'system', 'product',
                  'device', 'app', 'application', 'technology'}
# A word's frame only goes with "refers to": "Gaslighting is a term that describes manipulation" isn't
# "Gaslighting describes manipulation" (it is the manipulation)
WORD_NOUNS = {'term', 'concept', 'word', 'phrase', 'name'}
# Verbs that often end a clause, with no object ("is cheap and lasts", "saves time and works")
CLAUSE_END_VERBS = {'work', 'last', 'help', 'matter', 'pay', 'sell', 'run', 'cost', 'count', 'grow', 'fit', 'save'}

def agree(text, plural):
    """A present-tense verb (the first word of `text`) made to agree with a plural or a singular subject, or
    None if it can't be inflected. A contracted "'s" is the "is" of "a tool that's easy to use"; other
    contractions aren't guessed (lemminflect would turn "'s" into "am")."""
    head, space, rest = text.partition(' ')
    low = head.lower().replace('’', "'")
    if low == "'s":
        low = 'is'
    pairs = {'is': ('are', 'is'), 'are': ('are', 'is'), 'was': ('were', 'was'), 'were': ('were', 'was'),
             'has': ('have', 'has'), 'have': ('have', 'has'), 'does': ('do', 'does'), 'do': ('do', 'does')}
    if low in pairs:
        new = pairs[low][0 if plural else 1]
    elif low.startswith("'"):
        return None
    else:
        lemmas = getAllLemmas(low).get('VERB')
        forms = getInflection(lemmas[0], tag='VBP' if plural else 'VBZ') if lemmas else None
        if not forms:
            return None
        new = forms[0]
    return match_case(head, new) + space + rest

# A place's or an activity's category goes only where the subject already says what it is: a named place ("Lisbon is
# a city that rewards curious travelers" -> "Lisbon rewards curious travelers") or an activity named by its "-ing"
# ("Hiking is an activity that builds strength" -> "Hiking builds strength")
PLACE_CATEGORY_NOUNS = {'city', 'town', 'village', 'country', 'nation', 'state', 'region', 'island', 'place', 'spot'}
ACTIVITY_CATEGORY_NOUNS = {'activity', 'hobby', 'pastime', 'sport', 'exercise', 'workout', 'discipline'}

def category_fits(subject, noun):
    """True if "<subject> is a <noun> that ..." may lose its category noun."""
    lemma = noun.lemma_.lower()
    if lemma in CATEGORY_NOUNS:
        return True
    if lemma in PLACE_CATEGORY_NOUNS:
        return subject.pos_ == 'PROPN' and subject.ent_type_ in ('GPE', 'LOC')
    return lemma in ACTIVITY_CATEGORY_NOUNS and (subject.tag_ == 'VBG' or (subject.lower_.endswith('ing') and subject.tag_ == 'NN'))

def category_to_direct(sent, pieces):
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    if root.lemma_ != 'be' or any(c.dep_ in ('neg', 'aux') for c in root.children):
        return None
    subject = next((c for c in root.children if c.dep_ in ('nsubj', 'csubj') and c.i < root.i), None)
    noun = next((c for c in root.children if c.dep_ == 'attr' and c.i > root.i), None)
    if subject is None or noun is None or not category_fits(subject, noun):
        return None
    return category_from_parse(sent, pieces, subject, noun) or category_misread(sent, pieces, subject, noun)

def misread_first_verb(relative, first):
    """True for the misreading of "a technique that stores frequently accessed data": the verb after "that" taken for
    a noun ("stores") and the describing participle after it for the clause's verb ("accessed"), which then has two
    direct objects, "that" and "data" - a sign no correct reading gives."""
    verb = relative.head
    lemmas = getAllLemmas(first.lower_).get('VERB')
    forms = getInflection(lemmas[0], tag='VBZ') if lemmas else None
    return (relative.dep_ == 'dobj' and verb.dep_ == 'relcl' and first.dep_ == 'nsubj' and first.head.i == verb.i
            and first.tag_ in ('NNS', 'NN') and bool(forms) and forms[0] == first.lower_ and verb.tag_ in ('VBD', 'VBN')
            and sum(1 for c in verb.children if c.dep_ == 'dobj') >= 2
            and all(t.dep_ == 'advmod' and t.head.i == verb.i for t in relative.doc[first.i + 1: verb.i]))

def category_misread(sent, pieces, subject, noun):
    """The frame when spaCy misread the clause after "that" as in "a technique that stores frequently accessed data"
    (see misread_first_verb): "stores" is its verb, made to agree with the real subject if it must. Only a clause
    with no other verb of its own and no second relative ("... and that doctors trust")."""
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    relative = doc[noun.i + 1]
    bare = ((noun.i == root.i + 2 and doc[root.i + 1].lower_ in ('a', 'an')) or (noun.tag_ == 'NNS' and noun.i == root.i + 1))
    if (not bare or relative.lower_ not in ('that', 'which') or subject.left_edge.i != sent.start
            or subject.right_edge.i + 1 != root.i or not 3 <= stop - relative.i <= 20 or noun.lemma_.lower() in WORD_NOUNS):
        return None
    clause = doc[relative.i + 1: stop]
    texts = {p[0]: p[1] for p in pieces if p[0] is not None}
    if (not misread_first_verb(relative, clause[0]) or has_quoted(doc, range(relative.i, stop))
            or any(t.text in MOVE_BREAK_MARKS or texts.get(t.i) != t.text or t.lower_ in ('that', 'which', 'who')
                   or (t.i > clause.start and (t.tag_ in ('VBZ', 'VBP') or t.dep_ == 'conj')) for t in clause)):
        return None
    new = take(pieces, sent.start, subject.right_edge.i) + take(pieces, relative.i + 1, stop - 1) + take(pieces, stop, stop)
    if (root.tag_ == 'VBP' or root.lower_ in ('are', 'were')) != (noun.tag_ == 'NNS'):
        verb = agree(clause[0].text, root.tag_ == 'VBP' or root.lower_ in ('are', 'were'))
        if verb is None:
            return None
        for piece in new:
            if piece[0] == clause.start:
                piece[1] = verb
    return new

def category_from_parse(sent, pieces, subject, noun):
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    det = doc[noun.i - 1]
    clause = next((c for c in noun.children if c.dep_ == 'relcl'), None)
    bare = (det.dep_ == 'det' and det.lower_ in ('a', 'an') and det.i == root.i + 1) or (noun.tag_ == 'NNS' and noun.i == root.i + 1)
    if (not bare or clause is None or any(c.dep_ not in ('det', 'relcl') for c in noun.children)
            or clause.right_edge.i != stop - 1):
        return None     # "a technique that ..." or "tools that ...", with nothing else describing it
    relative = doc[noun.i + 1]
    if relative.lower_ not in ('that', 'which') or relative.dep_ != 'nsubj' or relative.head != clause:
        return None
    after = doc[relative.i + 1]
    if after.i != clause.i and not (after.head.i == clause.i and after.dep_ in ('advmod', 'aux', 'auxpass', 'neg')):
        return None     # "a city that tourists adore": with a noun after "that", "that" is the object
    if any(c.dep_ == 'dobj' and c.tag_.startswith('VB') for c in clause.children):
        return None     # "a sport that coaches love": a verb read as the object is a noun ("coaches") read as the verb
    if subject.left_edge.i != sent.start or subject.right_edge.i + 1 != root.i:
        return None
    if clause.tag_ not in ('VBZ', 'VBP', 'VBD') and not any(c.dep_ == 'aux' and c.tag_ == 'MD' for c in clause.children):
        return None     # "a technique that" needs a finite verb ("stores", "can block")
    if noun.lemma_.lower() in WORD_NOUNS and clause.lemma_.lower() != 'refer':
        return None
    new = take(pieces, sent.start, subject.right_edge.i) + take(pieces, relative.i + 1, stop - 1) + take(pieces, stop, stop)
    # A relative pronoun repeated after "and" goes with the first one ("a tool that protects people and that saves
    # lives" -> "... protect people and save lives"). After a comma ", and that saves lives" may be "and this saves
    # lives", so that "that" stays, with its verb as written
    repeated = set()
    for c in clause.conjuncts:
        if any(g.lower_ in ('that', 'which') and g.dep_ not in ('nsubj', 'nsubjpass') and g.i - 1 >= sent.start
               and doc[g.i - 1].lower_ in ('and', 'or', 'but') for g in c.children):
            return None     # "a tool that protects people and that doctors trust": "... and that doctors trust" dangles
        again = next((g for g in c.children if g.dep_ in ('nsubj', 'nsubjpass') and g.lower_ in ('that', 'which')), None)
        if (again is not None and again.i - 2 >= sent.start and doc[again.i - 1].lower_ in ('and', 'or', 'but')
                and not (again.lower_ == 'that' and doc[again.i - 2].text == ',')):
            repeated.add(again.i)
    for piece in new:
        if piece[0] in repeated:
            piece[1] = ''
    # The verb agreed with the category ("a tool that protects"); now it must agree with the subject, whose
    # number the "is"/"are" shows ("Vaccines are a tool that protects ..." -> "Vaccines protect ..."), and so must
    # every verb that shares the clause's subject ("... and saves lives" -> "... and save lives")
    plural = root.tag_ == 'VBP' or root.lower_ in ('are', 'were')
    if plural != (noun.tag_ == 'NNS'):
        verbs, k = [clause], 0
        while k < len(verbs):
            verbs += [c for c in verbs[k].children if c.dep_ == 'conj' and c.pos_ in ('VERB', 'AUX')
                      and not any(g.dep_ in ('nsubj', 'nsubjpass', 'expl') and g.i not in repeated for g in c.children)]
            k += 1
        agreed = set()
        for verb in verbs:
            agreed.add(verb.i)
            finite = verb if verb.tag_ in ('VBZ', 'VBP', 'VBD') else next(
                (c for c in verb.children if c.dep_ in ('aux', 'auxpass') and c.i < verb.i), None)
            if finite is None or not (finite.tag_ in ('VBZ', 'VBP') or finite.lower_ in ('was', 'were')):
                continue    # a past tense, a modal, or a verb that shares the first one's "can" or "has"
            agreed.add(finite.i)
            piece = next((p for p in new if p[0] == finite.i), None)
            text = agree(piece[1], plural) if piece else None
            if text is None:
                return None
            piece[1] = text
        # A verb the parser took for a noun ("gives adults a second chance and costs very little", "lends books and
        # hosts events") can't be made to agree, so the sentence keeps its frame. A word that ends the clause is a
        # plain object ("tracks sleep and steps"), unless it is a verb usually used without one ("and works")
        if any(doc[t.i - 1].lower_ in ('and', 'or', 'but') and t.i not in agreed and t.lower_.endswith('s')
               and t.tag_ in ('VBZ', 'NNS') and 'VERB' in getAllLemmas(t.lower_)
               and (t.tag_ == 'VBZ' or (t.i < clause.right_edge.i and doc[t.i + 1].pos_ not in ('PUNCT', 'CCONJ'))
                    or (t.i == clause.right_edge.i and (t.lemma_.lower() in CLAUSE_END_VERBS
                                                        or 'NOUN' not in getAllLemmas(t.lower_))))
               for t in doc[clause.i + 1: clause.right_edge.i + 1]):
            return None
    return new

# --- "A common approach is to set a limit." -> "Setting a limit is a common approach." ---

GOAL_NOUNS = {'approach', 'way', 'goal', 'aim', 'idea', 'plan', 'key', 'solution', 'strategy', 'option', 'step', 'trick',
              'answer', 'task', 'job', 'priority', 'challenge', 'purpose', 'objective', 'mission', 'secret', 'method',
              'technique', 'tactic'}
GERUND_MAX_TOKENS = 22

def gerund_phrase(pieces, action, stop):
    """The pieces of "to set a budget and (to) review it ..." as "setting a budget and reviewing it ...": every
    coordinated verb becomes a gerund, or None if one can't ("... and review it" left bare is a broken subject). The
    verbs of a longer list hang on each other ("to reduce traffic, cut air pollution and raise money": "raise" on
    "cut"), so the whole chain is taken."""
    doc = action.doc
    verbs = [action] + sorted((c for c in action.conjuncts if c.pos_ in ('VERB', 'AUX')), key=lambda t: t.i)
    heads = {v.i for v in verbs}
    if any(t.dep_ == 'conj' and t.pos_ not in ('VERB', 'AUX') and doc[t.i - 1].lower_ in ('and', 'or')
           and (t.head.i in heads or ((getAllLemmas(t.lower_).get('VERB') or ('',))[0] == t.lower_
                                     and not any(c.dep_ in ('det', 'poss', 'amod', 'compound') for c in t.children)
                                     and t.i + 1 < len(doc) and doc[t.i + 1].pos_ == 'ADP'))
           for t in action.subtree):
        return None     # "to invest in training and plan for the future": "plan" is a verb spaCy took for a noun
    if any(v.tag_ not in ('VB', 'VBP') for v in verbs[1:]):
        return None     # "to prepare and covered every branch": a verb of another clause
    if any(g.dep_ in ('nsubj', 'nsubjpass') for v in verbs for g in v.children):
        return None     # a coordinated clause with a subject of its own
    gerunds = {}
    for verb in verbs:
        current = next((p[1] for p in pieces if p[0] == verb.i), verb.text)
        gerund = inflect(current.lower(), 'VBG')
        if not gerund:
            return None
        gerunds[verb.i] = gerund
    drop = {doc[v.i - 1].i for v in verbs[1:] if doc[v.i - 1].lower_ == 'to'}    # "to set X and to clear Y"
    return [[i, gerunds.get(i, text)] for i, text in take(pieces, action.i, stop - 1) if i not in drop]

def goal_to_gerund(sent, pieces):
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    if root.lemma_ != 'be' or root.lower_ not in ('is', 'was') or any(c.dep_ in ('neg', 'aux') for c in root.children):
        return None
    subject = next((c for c in root.children if c.dep_ == 'nsubj' and c.i < root.i), None)
    action = next((c for c in root.children if c.dep_ == 'xcomp' and c.i > root.i), None)
    if (subject is None or action is None or subject.lemma_.lower() not in GOAL_NOUNS
            or subject.left_edge.i != sent.start or subject.right_edge.i + 1 != root.i or subject.tag_ != 'NN'):
        return None
    to = doc[action.i - 1]
    if (to.lower_ != 'to' or to.i != root.i + 1 or action.tag_ != 'VB' or action.right_edge.i != stop - 1
            or stop - action.i > GERUND_MAX_TOKENS):
        return None
    for t in doc[action.i: stop]:
        if t.pos_ in ('NOUN', 'PROPN'):
            break       # a pronoun after this may stand for it ("set a value ... and invalidate it")
        if t.lower_ in BACK_POINTERS or (t.lower_ in ('this', 'that', 'these', 'those') and t.dep_ != 'det'):
            return None     # "Practicing it ... is the best way to learn a language": "it" before what it stands for
    phrase = gerund_phrase(pieces, action, stop)
    if phrase is None:
        return None
    capitalize_first(phrase)
    rest = take(pieces, subject.left_edge.i, subject.right_edge.i)
    lowercase_first(rest, doc)
    return phrase + [[root.i, root.lower_]] + rest + take(pieces, stop, stop)

# --- "Not only do these tools save time, but they also cut costs." -> "These tools save time, and they also cut costs." ---

INVERTING_AUX = {'do', 'does', 'did', 'is', 'are', 'was', 'were', 'can', 'could', 'will', 'would', 'has', 'have', 'had',
                 'should', 'may', 'might', 'must'}

def not_only_inverted(sent, pieces):
    doc, stop = sent.doc, sent[-1].i
    if len(sent) < 8 or sent[0].lower_ != 'not' or sent[1].lower_ != 'only' or sent[2].lower_ not in INVERTING_AUX:
        return None
    aux = sent[2]
    comma = next((t.i for t in sent if t.text == ',' and t.i + 1 < stop and doc[t.i + 1].lower_ == 'but'), None)
    if comma is None or sum(1 for t in sent if t.lower_ == 'but') != 1:
        return None
    subject = next((t for t in doc[aux.i + 1: comma] if t.dep_ in ('nsubj', 'nsubjpass', 'expl')), None)
    if subject is None or subject.left_edge.i != aux.i + 1 or subject.right_edge.i >= comma - 1:
        return None
    after_subject = subject.right_edge.i + 1
    first = take(pieces, aux.i + 1, subject.right_edge.i)
    if len(first) == 1 and first[0][1].lower() in SUBJECT_FORMS:
        first[0][1] = SUBJECT_FORMS[first[0][1].lower()]
    capitalize_first(first)
    if aux.lower_ in ('do', 'does', 'did'):
        verb = doc[after_subject]
        if verb.tag_ != 'VB':
            return None
        current = next((p[1] for p in pieces if p[0] == verb.i), verb.text)
        forms = getInflection(current.split()[0].lower(), tag={'do': 'VBP', 'does': 'VBZ', 'did': 'VBD'}[aux.lower_])
        if not forms:
            return None
        first += [[None, forms[0] + current[len(current.split()[0]):]]] + take(pieces, verb.i + 1, comma - 1)
    else:
        first += [[aux.i, aux.text.lower()]] + take(pieces, after_subject, comma - 1)
    second = take(pieces, comma + 2, stop - 1)
    lowercase_first(second, doc)
    return first + [[None, ','], [None, 'and']] + second + take(pieces, stop, stop)

# --- "unity, peace, and progress" -> "peace, unity, and progress" ---
# Plain nouns or adjectives only: names, numbers and verbs (steps in a procedure) keep their order.

LIST_RATE = {'standard': 0.5, 'casual': 0.6, 'professional': 0.35, 'academic': 0.25}
# A list whose order is stated ("algebra, geometry, and calculus, in that order"; "cost, speed, and quality, in
# that order of importance"; "from most to least important") or ranked ("Rank the options by price, quality, and
# speed"; "Our priorities are ...", but not "high-ranking officials") keeps it
ORDERED = re.compile(r"\b(?:in\s+(?:that|this|the\s+same|the\s+following|descending|ascending|reverse|chronological|"
                     r"alphabetical|numerical)\s+(?:order|sequence)|in\s+order\s+of|respectively|in\s+sequence|"
                     r"in\s+turn|in\s+succession|one\s+after\s+(?:the\s+)?other|"
                     r"from\s+(?:the\s+)?(?:most|least|highest|lowest|best|worst|first|last|top|largest|smallest|"
                     r"biggest|oldest|newest|earliest|latest)\s+to\s+(?:the\s+)?\w+|"
                     r"(?<!-)rank(?:s|ed|ing)?|prioriti[sz](?:e|es|ed|ing)|priorit(?:y|ies))\b", re.I)
# ...and so does a list of the keys something is sorted by ("Sort the tasks by urgency, cost, and effort": urgency
# comes first)
SORTING_VERBS = {'sort', 'order', 'rank', 'arrange', 'list', 'group', 'organise', 'organize', 'classify',
                 'categorise', 'categorize', 'index', 'sequence', 'prioritise', 'prioritize'}

# Scales and series, whose order is part of what a list of their words says ("small, medium and large", "beginner,
# intermediate and advanced", "bronze, silver and gold", "mild, moderate or severe", "breakfast, lunch and dinner",
# "past, present and future"): two items from one of them keep the list's order, and so does a day, a month or an
# ordinal ("first, second and third")
LIST_SERIES_GROUPS = (
    {'tiny', 'small', 'medium', 'large', 'big', 'huge', 'giant', 'jumbo'},
    {'beginner', 'novice', 'intermediate', 'advanced', 'expert', 'basic', 'standard', 'premium', 'deluxe', 'pro'},
    {'bronze', 'silver', 'gold', 'platinum', 'diamond'},
    {'mild', 'moderate', 'severe', 'extreme', 'critical', 'low', 'medium', 'high'},
    {'slow', 'medium', 'fast'},
    {'short', 'medium', 'long'},
    {'light', 'medium', 'heavy'},
    {'budget', 'cheap', 'mid-range', 'expensive', 'luxury'},
    {'poor', 'good', 'better', 'best', 'excellent'},
    {'hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly', 'annual'},
    {'breakfast', 'brunch', 'lunch', 'tea', 'dinner', 'supper', 'starter', 'main', 'dessert'},
    {'infant', 'baby', 'toddler', 'child', 'teenager', 'teen', 'adult', 'birth', 'childhood', 'adolescence',
     'adulthood', 'death', 'young', 'old', 'elderly'},
    {'past', 'present', 'future', 'beginning', 'start', 'middle', 'end', 'finish'},
    {'north', 'south', 'east', 'west', 'northern', 'southern', 'eastern', 'western'},
    {'primary', 'secondary', 'tertiary', 'elementary', 'undergraduate', 'postgraduate', 'graduate', 'bachelor',
     'master', 'doctoral', 'freshman', 'sophomore', 'junior', 'senior'},
    {'local', 'regional', 'national', 'international', 'global', 'federal', 'state'},
    {'single', 'double', 'triple', 'minimum', 'average', 'maximum'},
    {'spring', 'summer', 'autumn', 'fall', 'winter', 'morning', 'afternoon', 'evening', 'night'},
)

def series_list(conjuncts):
    """True if the items of a list (their head words) form a scale or a series (see LIST_SERIES_GROUPS)."""
    words = [{c.lower_, c.lemma_.lower()} for c in conjuncts]
    if any(w & (TIME_MOVE_DAY_WORDS | TIME_MOVE_MONTHS) for w in words) or any(
            c.ent_type_ == 'ORDINAL' for c in conjuncts):
        return True
    return any(sum(1 for w in words if w & group) >= 2 for group in LIST_SERIES_GROUPS)

def list_items(token):
    """The items of a list headed by `token` as (start, end) token ranges, and whether the last item must stay
    last because a phrase follows it ("ecosystems, economies, and communities around the world"); or None."""
    doc = token.doc
    conjuncts = [token] + sorted(token.conjuncts, key=lambda t: t.i)
    if token.dep_ == 'conj' or len(conjuncts) < 3 or any(c.i < token.i for c in token.conjuncts):
        return None
    if token.pos_ not in ('NOUN', 'ADJ') or any(c.pos_ != token.pos_ for c in conjuncts):
        return None
    if ORDERED.search(token.sent.text) or (token.pos_ == 'NOUN' and token.dep_ in ('compound', 'nmod')):
        return None     # (nouns before a head they share, which may belong to the last alone: "by date, level and
                        # service name" isn't "by service, level and date name")
    if series_list(conjuncts):
        return None     # a scale or a series: "small, medium and large", "bronze, silver or gold"
    if token.dep_ == 'pobj' and token.head.lower_ == 'by':
        # "by" hangs on the verb ("Sort the tasks by ...") or on its object ("sort your emails by ..."). The people
        # who did it are no keys ("The festival was organised by artists, musicians, and volunteers"), but a person
        # can be one ("The files are organised by client, project, and month")
        verb = token.head.head if token.head.head.pos_ == 'VERB' else token.head.head.head
        if verb.lemma_.lower() in SORTING_VERBS and not (token.head.dep_ == 'agent'
                                                          and all(names_people(c) for c in conjuncts)):
            return None
    # "the distance, speed, and duration of every run": a "the"/"all"/"their" before the first item belongs to the
    # whole list, so it can't move with that item ("speed, the distance, and duration")
    determined = [any(t.dep_ in ('det', 'poss', 'predet', 'nummod') for t in c.lefts) for c in conjuncts]
    if any(determined) and not all(determined):
        return None
    items, fixed_last = [], False
    for n, c in enumerate(conjuncts):
        # Numbers or names; but a common noun opening the sentence may only look like a name to the tagger
        # ("Wildfires, floods, and droughts ...")
        if c.like_num or (c.ent_type_ and not (c.pos_ == 'NOUN' and c.i == c.sent.start
                                               and c.ent_type_ in ('PERSON', 'ORG', 'PRODUCT', 'WORK_OF_ART'))):
            return None
        if any(t.dep_ not in ('conj', 'cc', 'punct') for t in c.rights):
            if n < len(conjuncts) - 1:
                return None     # an item with a trailing modifier ("tools for X") inside the list
            fixed_last = True   # the phrase after the last item may belong to it alone, so it stays in place
        start = min([c.i] + [t.left_edge.i for t in c.lefts if t.dep_ not in ('cc', 'punct')])
        if c.i - start > 3:
            return None
        if any(t.dep_ == 'compound' and t.tag_ in ('NN', 'NNS') and 'NOUN' not in getAllLemmas(t.lower_) for t in c.lefts):
            return None     # a verb read as a noun: "set reminders, create lists, and share notes"
        items.append((start, c.i))
    for (_, end), (start, _) in zip(items, items[1:]):
        if any(t.lower_ not in (',', 'and', 'or') for t in doc[end + 1: start]):
            return None
    if sum(1 for (_, end), (start, _) in zip(items, items[1:]) for t in doc[end + 1: start] if t.lower_ in ('and', 'or')) > 1:
        return None     # "producer and songwriter and singer": a chain, not a list
    if items[0][0] > 0 and doc[items[0][0] - 1].lower_ in ('a', 'an'):
        return None     # a shared "a"/"an" would have to agree with whichever item comes first
    return items, fixed_last

# Things one can do, listed after "can"/"may", are options whose order is free ("Visitors can explore historic
# castles, relax in hot springs, and enjoy world-class cuisine"). Steps aren't ("You can preheat the oven, mix the
# batter and bake it"): a list with a word of sequence, a word standing for an earlier item, or a verb of a
# procedure keeps its order
VERB_LIST_MODALS = {'can', 'could', 'may', 'might'}
VERB_LIST_ORDER_WORDS = {'then', 'first', 'next', 'finally', 'after', 'afterwards', 'before', 'until', 'once', 'again',
                         'later', 'last', 'lastly', 'second', 'third', 'eventually', 'meanwhile', 'subsequently',
                         'when'}
VERB_LIST_PROCEDURE_VERBS = {'preheat', 'heat', 'mix', 'stir', 'add', 'pour', 'bake', 'cook', 'boil', 'simmer', 'chop',
                             'slice', 'wash', 'rinse', 'dry', 'drain', 'season', 'serve', 'install', 'download', 'open',
                             'click', 'select', 'enter', 'save', 'restart', 'press', 'insert', 'remove', 'attach',
                             'connect', 'plug', 'unplug', 'close', 'log', 'sign', 'submit', 'upload', 'copy', 'paste',
                             'run', 'type', 'fill', 'register', 'apply', 'pay', 'book', 'check', 'confirm', 'verify',
                             'wait', 'repeat', 'return', 'start', 'finish', 'complete', 'begin', 'end', 'leave',
                             'arrive', 'board', 'land', 'collect',
                             # getting something, which comes before using it ("buy a pass, ride the cable car, and
                             # admire the glacier"; "earn coins, unlock new characters, ...")
                             'buy', 'rent', 'hire', 'purchase', 'borrow', 'get', 'obtain', 'order', 'pick', 'grab',
                             'fetch', 'reserve', 'pack', 'earn', 'win', 'gain', 'receive', 'choose'}
# ...nor does one that goes somewhere first, where the other things are done ("visit the farm, feed the animals and pick
# strawberries"), or moves along the legs of a trip ("climb the tower, admire the view, and descend by the stairs")
VERB_LIST_SETTING_VERBS = {'visit', 'go', 'come', 'tour', 'travel', 'head', 'walk', 'drive', 'fly', 'enter', 'stop',
                           'stay', 'join', 'reach', 'ride', 'row', 'paddle', 'cycle', 'sail', 'catch', 'climb', 'ascend',
                           'descend', 'cross', 'hike'}
VERB_LIST_BACK_REFERENCES = {'it', 'them', 'this', 'that', 'these', 'those', 'one', 'ones', 'its', 'there', 'they', 'their'}

def verb_list_items(token):
    """The items of a list of three or more things the subject can do, headed by `token` (the first verb, after its
    "can"/"may"), or of what something is shown or meant to do ("has been shown to improve mood, reduce stress, and
    sharpen focus"), as (start, end) token ranges, with whether the last must stay last because a phrase after it may
    belong to every item ("... and enjoy the food in the old town"); or None. Every item has its own object or
    complement: a bare verb shares the last one's ("read, write and speak French") or is a stage of a process
    ("launch, grow and scale quickly", "buy, hold and sell shares")."""
    doc = token.doc
    if token.pos_ != 'VERB' or token.tag_ != 'VB' or token.dep_ == 'conj':
        return None
    options = (any(c.dep_ == 'aux' and c.lower_ in VERB_LIST_MODALS for c in token.children)
               and any(c.dep_ == 'nsubj' and c.i < token.i for c in token.children))
    effects = (token.dep_ == 'xcomp' and token.i > 0 and token.doc[token.i - 1].lower_ == 'to'
               and token.head.pos_ in ('VERB', 'ADJ') and token.head.lemma_.lower() not in VERB_LIST_PROCEDURE_VERBS)
    if not (options or effects):
        return None     # (things that can be done, or effects: "shown to improve mood, reduce stress and ...")
    conjuncts = [token] + sorted(token.conjuncts, key=lambda t: t.i)
    if (len(conjuncts) < 3 or any(c.tag_ != 'VB' or c.lemma_.lower() in VERB_LIST_PROCEDURE_VERBS | VERB_LIST_SETTING_VERBS
                                  for c in conjuncts) or ORDERED.search(token.sent.text)):
        return None
    items, fixed_last = [], False
    for n, c in enumerate(conjuncts):
        if n and any(t.dep_ not in ('cc', 'punct') for t in c.lefts):
            return None
        own = [t for t in c.rights if t.dep_ not in ('cc', 'punct', 'conj')]
        end = max([c.i] + [t.right_edge.i for t in own])
        if end - c.i > 7 or any(t.text in MOVE_BREAK_MARKS or t.lower_ in VERB_LIST_ORDER_WORDS | VERB_LIST_BACK_REFERENCES
                                or t.dep_ in ('advcl', 'ccomp', 'xcomp', 'relcl', 'acl') for t in doc[c.i: end + 1]):
            return None
        if n == len(conjuncts) - 1 and any(t.dep_ in ('prep', 'advmod', 'npadvmod') for t in doc[c.i + 1: end + 1]):
            fixed_last = True   # (anywhere in it: "act out stories in the club" may say where all of them happen)
        items.append((c.i, end))
    if any(start == end for start, end in items):
        return None
    for (_, end), (start, _) in zip(items, items[1:]):
        if start <= end or any(t.lower_ not in (',', 'and', 'or') for t in doc[end + 1: start]):
            return None
    if sum(1 for (_, end), (start, _) in zip(items, items[1:]) for t in doc[end + 1: start] if t.lower_ in ('and', 'or')) != 1:
        return None
    return items, fixed_last

# Two items swap only when each has its own describing words ("exceptional sound quality and impressive battery life",
# "more frequent and more severe"): two bare words are often a fixed pair ("salt and pepper", "trial and error"), and
# a word of order or degree keeps its place ("primary and secondary schools", "short-term costs and long-term gains")
PAIR_ORDER_ADJECTIVES = {'first', 'second', 'third', 'last', 'final', 'initial', 'early', 'earlier', 'late', 'later',
                         'previous', 'next', 'former', 'latter', 'primary', 'secondary', 'tertiary', 'basic',
                         'advanced', 'preliminary', 'subsequent', 'prior', 'upcoming', 'past', 'present', 'future',
                         'old', 'new', 'short', 'long', 'small', 'large', 'big', 'low', 'high', 'top', 'bottom',
                         'front', 'back', 'upper', 'lower', 'inner', 'outer', 'black', 'white', 'rich', 'poor', 'young',
                         'elder', 'junior', 'senior', 'minor', 'major', 'local', 'national', 'global', 'domestic',
                         'international', 'public', 'private', 'opening', 'closing', 'introductory', 'concluding',
                         'starting', 'beginning', 'ending', 'middle', 'morning', 'afternoon', 'evening'}
PAIRED_DEGREE_WORDS = {'more', 'less', 'increasingly'}
# Two stages of a process come in their order ("careful diagnosis and timely treatment", "careful planning and
# flawless execution")
PAIR_STAGE_NOUNS = {'diagnosis', 'detection', 'screening', 'testing', 'test', 'treatment', 'therapy', 'recovery',
                    'rehabilitation', 'planning', 'plan', 'design', 'research', 'development', 'construction',
                    'execution', 'implementation', 'delivery', 'preparation', 'prevention', 'response', 'analysis',
                    'collection', 'review', 'approval', 'assessment', 'evaluation', 'training', 'question', 'answer',
                    'problem', 'solution', 'cause', 'effect', 'input', 'output', 'production', 'distribution',
                    'installation', 'maintenance', 'investigation', 'prosecution', 'arrest', 'trial', 'conviction',
                    'recruitment', 'selection', 'application', 'admission', 'enrolment', 'enrollment', 'graduation'}

# Describing words that can reach over "and" to the next noun ("new rules and safety checks"); comparatives and
# participles ("higher", "rising", "reduced") are caught by their tags
PAIR_SHARED_ADJECTIVES = {'new', 'extra', 'additional', 'high', 'low', 'more', 'fewer', 'less'}

def pair_items(token):
    """The two items of "A and B" headed by `token`, as ((start, end), (start, end)), if both have describing words of
    their own and can swap; else None. Not two stages of a process (PAIR_STAGE_NOUNS), nor after "either", "both" or
    "neither", which would move with the first item ("a free replacement or either a full refund")."""
    doc = token.doc
    if token.dep_ == 'conj' or token.pos_ not in ('NOUN', 'ADJ') or len(token.conjuncts) != 1:
        return None
    other = token.conjuncts[0]
    if other.i < token.i or other.head.i != token.i or other.pos_ != token.pos_ or ORDERED.search(token.sent.text):
        return None
    if ({token.lemma_.lower(), other.lemma_.lower()} <= PAIR_STAGE_NOUNS or series_list([token, other])
            or any(t.dep_ == 'preconj' or t.lower_ in ('either', 'both', 'neither', 'whether') for t in token.subtree)
            or (token.left_edge.i > 0 and doc[token.left_edge.i - 1].lower_ in ('either', 'both', 'neither', 'whether'))):
        return None
    if any(t.dep_ not in ('cc', 'conj') for t in token.rights) or any(True for _ in other.rights):
        return None     # "X and Y in the region": what follows may describe both
    items = []
    for c in (token, other):
        lefts = [t for t in c.lefts if t.dep_ not in ('cc', 'punct')]
        start = min([c.i] + [t.left_edge.i for t in lefts])
        words = doc[start: c.i + 1]
        if (c.i - start > 3 or c.like_num or c.ent_type_ or any(
                t.like_num or t.ent_type_ or t.tag_ in ('PRP', 'POS', 'CD') or t.lower_ in PAIR_ORDER_ADJECTIVES
                or t.lower_.endswith('-term') or t.text in MOVE_BREAK_MARKS
                or (t.dep_ == 'compound' and t.tag_ == 'NNS') for t in words)):
            return None     # (a plural "describing" noun is a verb misread: "tracks energy use and flags waste")
        if c.pos_ == 'ADJ':
            degree = [t.lower_ for t in lefts if t.dep_ == 'advmod']
            if degree != [words[0].lower_] or degree[0] not in PAIRED_DEGREE_WORDS or len(words) != 2:
                return None     # "more frequent and more severe" (not "safe and sound")
        elif not any(t.dep_ in ('amod', 'compound', 'nmod') for t in lefts):
            return None
        items.append((start, c.i))
    if doc[items[0][1] + 1: items[1][0]].text.lower() not in ('and', 'or'):
        return None
    if token.pos_ == 'ADJ' and doc[items[0][0]].lower_ != doc[items[1][0]].lower_:
        return None
    determiners = [[t.lower_ for t in doc[s: e + 1] if t.dep_ in ('det', 'poss', 'predet')] for s, e in items]
    if determiners[0] != determiners[1]:
        return None     # "a five-year warranty and free shipping": the "a" may be shared
    if any(t.dep_ == 'amod' for t in doc[items[1][0]: items[1][1] + 1]) and not any(
            t.dep_ == 'amod' for t in doc[items[0][0]: items[0][1] + 1]):
        return None     # "blood pressure and harmful cholesterol": put first, "harmful" could seem to describe both
    if not any(t.dep_ == 'amod' for t in doc[items[1][0]: items[1][1] + 1]) and any(
            t.dep_ == 'amod' and (t.tag_ in ('JJR', 'JJS', 'VBG', 'VBN') or t.lower_ in PAIR_SHARED_ADJECTIVES)
            for t in doc[items[0][0]: items[0][1]]):
        return None     # "higher wages and energy bills": "higher" may describe both, and put second it no longer can
    if items[0][0] > 0 and doc[items[0][0] - 1].lower_ in ('a', 'an', 'between'):
        return None
    return items, False

def reorder_list(sent, pieces):
    doc, root = sent.doc, sent.root
    if not any(c.dep_ in ('nsubj', 'nsubjpass', 'expl', 'csubj', 'csubjpass') for c in root.children) and any(
            c.dep_ in ('aux', 'auxpass') and c.lower_ != 'to' for c in root.children):
        return None     # a main verb with a helping verb but no subject: the parser took the subject for more items of
                        # a list ("If you prefer not to use vinegar, lemon juice or citric acid will work just as well")
    for token in sent:
        found = list_items(token) or verb_list_items(token) or pair_items(token)
        if not found:
            continue
        items, fixed_last = found
        k, m = random.sample(range(len(items) - 1 if fixed_last else len(items)), 2)
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
                piece = first_word_piece(group)
                opener = doc[sent.start]
                if (piece is not None and piece[0] == sent.start and opener.pos_ == 'NOUN' and opener.ent_type_
                        and opener.text[1:].islower()):
                    piece[1] = piece[1][:1].lower() + piece[1][1:]    # "Wildfires" read as a name, inside the list
            capitalize_first(moved_first)
        return new
    return None

# --- "People across the country celebrate ..." -> "Across the country, people celebrate ..." ---

SPREAD_PREPOSITIONS = {'across', 'around', 'throughout'}

def front_subject_place(sent, pieces):
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    subject = next((c for c in root.children if c.dep_ == 'nsubj' and c.i < root.i), None)
    if subject is None or subject.left_edge.i != sent.start or subject.pos_ == 'PRON' or uncertain_capital(sent[0]):
        return None
    place = next((c for c in subject.rights if c.dep_ == 'prep' and c.lower_ in SPREAD_PREPOSITIONS), None)
    if place is None or place.right_edge.i != subject.right_edge.i or subject.right_edge.i + 1 != root.i:
        return None
    moved = take(pieces, place.i, place.right_edge.i)
    main = take(pieces, sent.start, place.i - 1) + take(pieces, root.i, stop - 1)
    capitalize_first(moved)
    lowercase_first(main, doc)
    return moved + [[None, ',']] + main + take(pieces, stop, stop)

# --- Moving a date or a time between the start and the end ---
# "In 1903, Marie Curie shared the prize." <-> "Marie Curie shared the prize in 1903."; "The office will move next
# spring." -> "Next spring, the office will move." A time can change ends when nothing at its new place could take it
# as its own describer ("wrote about the riots in 1968": the riots' year, or the writing's?), it doesn't land inside
# another clause ("decided to leave the company in 2019"), and the sentence has no other time to collide with
# ("Today, ships pass through every year"). These moves work on the pieces as they now stand, so in Chaos they can
# follow a passive turned active ("The cornerstone was laid in 1902 by the mayor." -> "The mayor laid the cornerstone
# in 1902." -> "In 1902, the mayor laid the cornerstone.").

# Prepositions that date an action by the time after them ("in 1998", "on Tuesday", "at noon", "during the night")
DATING_PREPOSITIONS = {'in', 'on', 'at', 'during'}
# "two weeks ago", "eight years later", "a decade earlier"
RELATIVE_TIME_ADVERBS = {'ago', 'later', 'earlier', 'afterwards', 'afterward'}
RELATIVE_TIME_UNITS = {'second', 'minute', 'hour', 'day', 'night', 'week', 'month', 'year', 'decade', 'century', 'generation'}
# The words that may come before a time noun used on its own ("last year", "next spring", "every morning")
BARE_TIME_DETERMINERS = {'last', 'next', 'this', 'that', 'every', 'each', 'same', 'following', 'previous', 'coming',
                         'past', 'early', 'late'}
TIME_MOVE_DAY_WORDS = {'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'mondays',
                       'tuesdays', 'wednesdays', 'thursdays', 'fridays', 'saturdays', 'sundays', 'day', 'days', 'today',
                       'tomorrow', 'yesterday', 'tonight'}
TIME_MOVE_MONTHS = {'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                    'november', 'december', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct',
                    'nov', 'dec'}
# Words that limit or count a time, which a move would detach from what they qualify ("only on Sundays", "twice a
# week", "not in 2019"), or that say it again ("also", "too")
TIME_PHRASE_LIMITERS = {'only', 'just', 'even', 'not', 'never', 'twice', 'once', 'times', 'per', 'also', 'too', 'again',
                        'still', 'already', 'until', 'till', 'since', 'all', 'whole', 'entire'}
# Times said by an adverb: a second time in the sentence, which a moved phrase would collide with
SECOND_TIME_ADVERBS = {'now', 'then', 'today', 'tonight', 'tomorrow', 'yesterday', 'currently', 'recently', 'lately',
                       'later', 'soon', 'already', 'once', 'ago', 'earlier', 'previously', 'formerly', 'nowadays',
                       'meanwhile', 'eventually', 'initially', 'originally', 'subsequently', 'presently', 'immediately',
                       'afterwards', 'afterward', 'beforehand', 'shortly'}
# Verbs whose "in"/"on"/"at" + a time is their complement, not when the action happened: "The film is set in 1920",
# "The book focuses on the 1990s", "Christmas falls on a Monday", "The talks were scheduled at noon"
TIME_COMPLEMENT_VERBS = {'set', 'focus', 'concentrate', 'centre', 'center', 'dwell', 'look', 'specialize', 'specialise',
                         'date', 'fall', 'schedule', 'reschedule', 'plan', 'slate', 'fix', 'book', 'reserve',
                         'postpone', 'delay', 'adjourn', 'defer', 'advance', 'time', 'arrange', 'result', 'culminate',
                         'last', 'span', 'stretch', 'range', 'base', 'trace', 'root', 'compare', 'peak', 'bottom',
                         'place', 'rank', 'hold'}
# ...and verbs of advice or need, whose time belongs to what they ask for: "Experts recommend 150 minutes of exercise
# each week" isn't "Each week, experts recommend ..."
TIME_OWNING_ADVICE_VERBS = {'recommend', 'advise', 'suggest', 'prescribe', 'urge', 'require', 'propose', 'encourage',
                            'expect', 'aim', 'target', 'budget', 'allow', 'allot', 'ask', 'tell', 'demand', 'request'}
# A word that singles out or denies, whose reach a moved time would change: "We only meet on Fridays" isn't "On
# Fridays, we only meet", and "He didn't die in 1934" isn't "In 1934, he didn't die"
TIME_MOVE_FOCUS_WORDS = {'only', 'just', 'even', 'mainly', 'mostly', 'largely', 'primarily', 'chiefly', 'especially',
                         'particularly', 'exclusively', 'solely', 'merely', 'simply', 'also', 'too', 'either', 'partly'}
TIME_MOVE_NEGATIONS = {'not', "n't", 'n’t', 'never', 'no', 'nobody', 'nothing', 'none', 'neither', 'nor', 'nowhere',
                       'without', 'hardly', 'barely', 'scarcely', 'few', 'little', 'cannot'}
# Nouns a time can describe ("the riots in 1968", "his visit on Monday", "the economy in the 1990s", "prices in
# 2020"): a time next to one may be its own, so it isn't moved away from it or next to it
TIME_DESCRIBED_NOUNS = EVENT_NOUNS | {
    'visit', 'trip', 'tour', 'talk', 'riot', 'attack', 'raid', 'crash', 'collapse', 'launch', 'release', 'opening',
    'closure', 'storm', 'flood', 'fire', 'earthquake', 'outbreak', 'merger', 'deal', 'sale', 'vote', 'protest',
    'speech', 'interview', 'survey', 'study', 'report', 'review', 'test', 'exam', 'repair', 'change', 'move', 'return',
    'death', 'birth', 'wedding', 'funeral', 'party', 'summit', 'debate', 'hearing', 'invasion', 'uprising', 'boom',
    'recession', 'depression', 'slump', 'economy', 'market', 'history', 'situation', 'life', 'culture', 'society',
    'politics', 'music', 'art', 'fashion', 'price', 'rate', 'level', 'figure', 'record', 'result', 'growth', 'decline',
    'rise', 'fall', 'drop', 'increase', 'policy', 'law', 'rule', 'budget', 'plan', 'schedule', 'deadline',
    'appointment', 'booking', 'reservation', 'ticket', 'flight', 'train', 'show', 'class', 'lesson', 'course', 'event',
    'final', 'round', 'goal', 'shot', 'win', 'defeat', 'loss', 'victory', 'score', 'arrival', 'approval',
    'removal', 'renewal', 'proposal', 'festival', 'survival', 'revival', 'refusal', 'burial', 'departure', 'failure',
    'exposure', 'procedure', 'treatment', 'surgery', 'operation', 'crisis', 'conflict', 'dispute', 'strike', 'era',
    'period', 'phase', 'stage', 'year', 'month', 'week', 'day', 'decade', 'century', 'version', 'edition', 'issue',
    'model', 'generation', 'cycle', 'term', 'session', 'contract', 'agreement', 'treaty', 'ban', 'cut', 'hike',
    'announcement', 'settlement', 'payment', 'development', 'movement', 'investment', 'shipment', 'enrolment',
    'enrollment', 'gathering', 'briefing', 'screening', 'sitting', 'bombing', 'shooting', 'killing', 'landing',
    'closing', 'scoring', 'signing', 'ruling', 'rising', 'reading', 'showing', 'viewing', 'crossing', 'founding',
    'beginning', 'ending', 'spending', 'trading', 'fighting', 'voting', 'polling', 'testing', 'training', 'counting',
    'meeting', 'recording', 'filming', 'shopping', 'sailing', 'racing', 'trial', 'inspection', 'run', 'trade',
    'industry', 'sector', 'population', 'climate', 'weather', 'condition', 'demand', 'supply', 'output',
    'traffic', 'crime', 'unemployment', 'harvest', 'crop'}
# Labels of the clauses and loose phrases a time may belong to when it comes after them
TIME_ATTACHING_DEPS = {'ccomp', 'xcomp', 'advcl', 'relcl', 'acl', 'conj', 'pcomp', 'csubj', 'csubjpass', 'parataxis',
                       'dep', 'appos', 'list'}
MOVE_BREAK_MARKS = {',', ';', ':', '—', '–', '(', ')', '"', '“', '”', '‘', '’', "'"}
MOVE_NUMBER_ENTITIES = {'DATE', 'TIME', 'MONEY', 'PERCENT', 'QUANTITY', 'CARDINAL', 'ORDINAL'}

def names_day_or_hour(obj):
    """True for a day, a date with its day, a clock time, a part of the day, or the end or start of a period ("on
    Tuesday", "on 3 May", "at 6pm", "at noon", "at the weekend", "at the end of the month"): not a year, a decade or a
    century, which "on"/"at" take as a topic ("focuses on the 1990s", "looks at the 1920s")."""
    lemma = obj.lemma_.lower()
    if obj.ent_type_ == 'TIME' or lemma in TIME_MOVE_DAY_WORDS or obj.lower_ in TIME_MOVE_DAY_WORDS or lemma in DEADLINE_TIMES:
        return True
    if lemma in TIME_PART_NOUNS | TIME_ONLY_PARTS:
        return any(c.dep_ == 'prep' and c.lower_ == 'of' for c in obj.children)
    return lemma in TIME_MOVE_MONTHS and any(t.like_num and len(t.text) <= 2 for t in obj.subtree)

def dating_phrase(child):
    """The span of a phrase hanging on the main verb that says when its action happened, or None: "in 1998", "on
    Tuesday", "at the end of the month", "during the night", "over time", "last year", "next spring", "every
    morning", "today", "two weeks ago", "eight years later". Not how long ("in a decade", "in two days", "within 30
    days"), how often ("twice a day", "a day"), "in time" or "on time" (other senses), nor a time with a word that
    limits it ("only on Sundays")."""
    doc = child.doc
    span = doc[child.left_edge.i: child.right_edge.i + 1]
    if (len(span) > 8 or span.text.lower() in DISCOURSE_PHRASES
            or any(t.text in MOVE_BREAK_MARKS or t.lower_ in TIME_PHRASE_LIMITERS for t in span)):
        return None
    if child.dep_ == 'prep':
        obj = next((c for c in child.children if c.dep_ == 'pobj'), None)
        if obj is None or child.left_edge.i != child.i:
            return None
        if child.lower_ == 'over' and obj.lower_ == 'time' and len(span) == 2:
            return span     # "over time"
        if child.lower_ == 'from':
            # "from 1979 to 1990": two dates
            to = next((c for c in list(child.children) + list(obj.children) if c.dep_ == 'prep' and c.lower_ == 'to'), None)
            end = next((c for c in to.children if c.dep_ == 'pobj'), None) if to is not None else None
            return (span if end is not None and end.i == span.end - 1 and all(
                t.ent_type_ == 'DATE' for t in span if t.i not in (child.i, to.i)) else None)
        if child.lower_ not in DATING_PREPOSITIONS or obj.lemma_.lower() == 'time' or not heads_time_phrase(child):
            return None
        if any(t.lower_ in ('a', 'an') for t in span) or (
                child.lower_ == 'in' and any(c.dep_ == 'nummod' for c in obj.children)):
            return None     # "in a decade", "in two days": how long, not when
        if child.lower_ in ('on', 'at') and not names_day_or_hour(obj):
            return None     # "focuses on the 1990s"
        return span
    if child.dep_ == 'advmod' and child.lower_ in RELATIVE_TIME_ADVERBS:
        amount = next((c for c in child.children if c.dep_ == 'npadvmod'), None)
        if (amount is not None and amount.lemma_.lower() in RELATIVE_TIME_UNITS and amount.left_edge.i == span.start
                and all(t.dep_ in ('det', 'nummod', 'amod') for t in doc[span.start: amount.i])):
            return span     # "two weeks ago", "a year later"
        return None
    if child.dep_ == 'npadvmod' and child.pos_ in ('NOUN', 'PROPN') and child.lemma_.lower() not in ('time', 'moment'):
        if any(t.lower_ in ('a', 'an', 'per') for t in span):
            return None     # "a day": how often
        if all(t.ent_type_ in ('DATE', 'TIME') for t in span) or (
                child.lemma_.lower() in TIME_NOUNS | DEADLINE_TIMES | TIME_MOVE_DAY_WORDS
                and all(c.dep_ in ('det', 'amod') and c.lower_ in BARE_TIME_DETERMINERS for c in child.children)):
            return span
    return None

# "in rare cases", "in most cases": a hedge on the whole statement, which reads as well at either end
CASE_HEDGE_WORDS = {'rare', 'most', 'some', 'many', 'severe', 'extreme', 'exceptional', 'certain', 'serious'}

def case_hedge(child):
    """The span "in rare cases" (or "most", "some", ...) of a phrase hanging on the main verb, or None."""
    obj = next((c for c in child.children if c.dep_ == 'pobj'), None) if child.dep_ == 'prep' else None
    if (obj is None or child.lower_ != 'in' or obj.lower_ != 'cases' or child.right_edge.i != obj.i
            or obj.i != child.i + 2 or child.doc[child.i + 1].lower_ not in CASE_HEDGE_WORDS):
        return None
    return child.doc[child.i: obj.i + 1]

def framing_phrase(child):
    """A time phrase or a case hedge hanging on the main verb (see dating_phrase and case_hedge), or None."""
    return dating_phrase(child) or case_hedge(child)

# Words that count the subject ("most of Gen Z", "many families", "three students")
SUBJECT_QUANTIFIERS = {'most', 'many', 'some', 'few', 'several', 'all', 'each', 'every', 'no', 'none', 'half', 'majority',
                       'minority', 'both', 'any', 'more', 'fewer', 'less', 'everyone', 'everybody', 'nobody', 'someone',
                       'somebody', 'anyone', 'anybody'}

def recurring_over_quantity(span, root):
    """True for a recurring time ("every day", "each week") and a counted subject: moving one past the other changes
    which counts for which ("Most of Gen Z uses TikTok every day": the same majority daily; "Every day, most of Gen Z
    uses TikTok": a majority each day, maybe not the same people)."""
    if not any(t.lower_ in ('every', 'each') for t in span):
        return False
    subject = next((c for c in root.children if c.dep_ in ('nsubj', 'nsubjpass', 'expl', 'csubj')), None)
    return subject is not None and any(t.lower_ in SUBJECT_QUANTIFIERS or t.like_num for t in subject.subtree)

def limits_main_verb(root):
    """True if a negation or a word that singles something out hangs on the main verb ("did not die", "never
    visited", "only meet", "also hired"): a time moved past it would change what it denies or singles out ("He didn't
    die in 1934" isn't "In 1934, he didn't die"). On a noun it stays with the noun ("Even small savings add up over
    time", "No one won the award in 2019")."""
    return any(c.lower_ in TIME_MOVE_NEGATIONS | TIME_MOVE_FOCUS_WORDS for c in root.children)

def dated_noun(token):
    """True for a noun a time can describe: an event, a stretch of time or something that changes with time
    ("the riots", "the visit", "the scoring", "the economy", "prices", "the election"), or a named event or date."""
    lemma = token.lemma_.lower()
    return (lemma in TIME_DESCRIBED_NOUNS or token.lower_ in TIME_DESCRIBED_NOUNS or lemma in NOMINALIZATIONS
            or token.ent_type_ in ('EVENT', 'DATE', 'TIME') or lemma.endswith(('tion', 'sion')))

# Prepositions whose phrase on a noun names a place, a topic or a part, which a time after it may describe ("a book
# about Berlin in 1995": Berlin as it was then?; "the coach of the national team in 2019"); not "with", "to" or "for"
# ("shared the prize with Henri Becquerel in 1903")
NOUN_PLACE_PREPS = {'about', 'of', 'on', 'from', 'in', 'at', 'near', 'beside', 'by', 'around', 'inside', 'outside',
                    'under', 'over', 'across', 'within', 'behind', 'along', 'among', 'between', 'into', 'onto', 'off',
                    'above', 'below', 'beneath', 'past', 'through', 'throughout', 'beyond', 'opposite'}

def time_could_attach(token, root, nouns=True):
    """True if a time phrase next to `token` might belong to it or to a clause or a phrase on its way up to the main
    verb, rather than to the main verb: "decided to leave the company in 2019", "wrote about the riots in 1968",
    "collapsed after a run on its deposits in 2008", and a place or a topic on a noun (NOUN_PLACE_PREPS: "published a
    book about Berlin in 1995", "will ban smoking in parks in May"). With `nouns` only: a case hedge describes no
    noun."""
    steps = 0
    while token.i != root.i and steps < 40:
        if (token.dep_ in TIME_ATTACHING_DEPS or (nouns and token.pos_ in ('NOUN', 'PROPN') and dated_noun(token))
                or (nouns and token.dep_ == 'prep' and (token.lower_ in ALWAYS_MOVABLE_PREPS or (
                    token.lower_ in NOUN_PLACE_PREPS and token.head.pos_ in ('NOUN', 'PROPN', 'PRON'))))):
            return True
        token, steps = token.head, steps + 1
    return token.i != root.i

def inside_noun_phrase(token, root):
    """True if `token` sits inside a phrase on a noun or inside a clause on its way up to the main verb ("the lives of
    women", "a book about Berlin", "decided to leave the company"), so a phrase right after it may be its own. Unlike
    time_could_attach, the main verb's own object is no owner ("changed its refund policy in March 2022")."""
    steps = 0
    while token.i != root.i and steps < 40:
        if token.dep_ in TIME_ATTACHING_DEPS or (token.dep_ == 'prep' and token.lower_ in NOUN_PLACE_PREPS
                                                 and token.head.pos_ in ('NOUN', 'PROPN', 'PRON')):
            return True
        token, steps = token.head, steps + 1
    return token.i != root.i

# Subjects with a time of their own: a story's, a film's or a picture's time is when it is set ("The novel ends in the
# spring of 1919", "The map shows Europe in 1914", "The film follows a family in Berlin in 1945"); in front, the time
# would say when the book or the map did something
TIME_STORY_NOUNS = {'story', 'novel', 'novella', 'film', 'movie', 'book', 'play', 'series', 'episode', 'chapter', 'show',
                    'poem', 'song', 'narrative', 'plot', 'tale', 'saga', 'action', 'musical', 'opera', 'drama', 'comedy',
                    'memoir', 'documentary', 'scene', 'trilogy', 'sequel', 'script', 'photo', 'photograph', 'picture',
                    'map', 'painting', 'portrait', 'image', 'video', 'footage', 'diary', 'letter', 'mural', 'drawing',
                    'poster', 'sculpture', 'exhibition', 'exhibit', 'display', 'biography', 'chronicle', 'article',
                    'essay', 'report', 'chart', 'graph', 'album'}
# Verbs whose time is all they say ("The accident happened on Monday"), and the times "lives in the past" isn't
TIME_EVENT_VERBS = {'happen', 'occur'}
FIGURATIVE_TIMES = {'past', 'moment', 'present', 'future'}

def time_stays_put(sent, span):
    """True if a time after the main verb (`span`) must stay after it, whatever would move it: when the verb takes it
    as all it says ("The festival takes place in July", "It happened on Monday", "Payment is due at the end of the
    month"), when it's no time ("She still lives in the past"), when a story, a film or a picture has it as its own
    time (TIME_STORY_NOUNS), or when another time comes before it, which it may count from ("The ship, which left
    Southampton on April 10, sank four days later": in front, "Four days later" would count from the sentence before;
    "Rovers travel to Leeds next weekend, while City host Bristol on Tuesday"). A case hedge ("in rare cases") stays
    only for the verb's sake."""
    doc, root = sent.doc, sent.root
    lemma = root.lemma_.lower()
    if (lemma in TIME_EVENT_VERBS or (lemma == 'take' and any(c.dep_ == 'dobj' and c.lower_ == 'place' for c in root.children))
            or (lemma == 'be' and any(c.dep_ == 'acomp' and c.lower_ in ('due', 'overdue', 'payable') for c in root.children))):
        return True
    if case_hedge(span[0]) is not None:
        return False
    subject = next((c for c in root.children if c.dep_ in ('nsubj', 'nsubjpass', 'expl', 'csubj')), None)
    if ((lemma in ('live', 'dwell') and any(t.lemma_.lower() in FIGURATIVE_TIMES for t in span))
            or (subject is not None and subject.lemma_.lower() in TIME_STORY_NOUNS
                and not any(c.dep_ == 'auxpass' for c in root.children))):
        return True
    return any(t.ent_type_ in ('DATE', 'TIME') or t.lower_ in SECOND_TIME_ADVERBS | TIME_MOVE_DAY_WORDS
               or (t.pos_ == 'NOUN' and t.lemma_.lower() in TIME_NOUNS) for t in doc[sent.start: span.start])

def piece_run(pieces, doc, start, end):
    """The positions (first, last) in `pieces` of the words of tokens start..end (inclusive), if all of them are still
    there, side by side, with nothing else of the sentence's own between them; else None (a template run before may
    have moved or dropped them). Their own order inside may have changed ("the city's history")."""
    wanted = {t.i for t in doc[start:end + 1] if not t.is_space}
    found = [k for k, p in enumerate(pieces) if p[0] in wanted]
    if not found or len(found) != len(wanted) or {pieces[k][0] for k in found} != wanted:
        return None
    if any(p[0] is not None and p[0] not in wanted for p in pieces[found[0]: found[-1] + 1]):
        return None
    return found[0], found[-1]

def live_neighbour(pieces, position, step):
    """The nearest piece with words before (step -1) or after (step 1) a position, or None."""
    k = position + step
    while 0 <= k < len(pieces):
        if pieces[k][1]:
            return pieces[k]
        k += step
    return None

def time_to_end(sent, pieces):
    """"In 1903, Marie Curie shared the prize with Henri Becquerel." -> "Marie Curie shared the prize with Henri
    Becquerel in 1903."; "Last year, the company moved ..." -> "The company moved ... last year." The clause must have
    its subject first, no comma (the time would go with its last part: "..., the maker of Call of Duty in 2023"), no
    other time, no negation ("In 2019, nobody won" isn't "Nobody won in 2019") or focus word, and no second verb
    sharing the subject ("opened a store and hired staff in 2010" may date only the hiring). Its last word must hang
    on the main verb through objects and phrases only, none a noun the time could describe. A case hedge moves the
    same way ("In rare cases, the pain may persist for a month." -> "The pain may persist for a month in rare
    cases."), with a time in the clause allowed."""
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    child = next((c for c in root.lefts if c.left_edge.i == sent.start), None)
    span = framing_phrase(child) if child is not None else None
    if span is None or span.end + 4 > stop or doc[span.end].text != ',' or span[0].i in phrase_tokens(doc):
        return None
    hedge = case_hedge(child) is not None
    if recurring_over_quantity(span, root) or misparsed_root(sent):
        return None
    main = doc[span.end + 1: stop]
    subject = next((c for c in root.children if c.dep_ in ('nsubj', 'nsubjpass', 'expl', 'csubj')), None)
    if (subject is None or subject.left_edge.i != main.start or subject.like_num or subject.ent_type_ in MOVE_NUMBER_ENTITIES
            or any(c.dep_ in ('conj', 'cc', 'neg') for c in root.children) or limits_main_verb(root)
            or any(t.text in MOVE_BREAK_MARKS for t in main)
            or (not hedge and any(t.lower_ in SECOND_TIME_ADVERBS or t.ent_type_ in ('DATE', 'TIME')
                                  or t.lemma_.lower() in TIME_NOUNS for t in main))):
        return None
    if subject.pos_ == 'PRON' and any(t.pos_ == 'PROPN' and t.ent_type_ not in ('DATE', 'TIME') for t in span):
        return None     # "On Napoleon's birthday, he ...": "he" may stand for the name
    run = piece_run(pieces, doc, span.start, span.end - 1)
    if (run is None or live_neighbour(pieces, run[0], -1) is not None or run[1] + 1 >= len(pieces)
            or pieces[run[1] + 1][0] != span.end):
        return None     # the phrase no longer opens the sentence, with its comma after it
    rest = [list(p) for p in pieces[run[1] + 2:]]
    if not rest or rest[-1][0] != stop:
        return None
    last = live_neighbour(rest, len(rest) - 1, -1)
    if last is None or last[0] is None or time_could_attach(doc[last[0]], root, not hedge):
        return None
    moved = [list(p) for p in pieces[run[0]: run[1] + 1]]
    lowercase_first(moved, doc)
    if sent[0].text[:1].isupper():
        capitalize_first(rest)      # (a line the writer began in lower case stays so: "in 1903, marie curie ...")
    return rest[:-1] + moved + rest[-1:]

def time_to_front(sent, pieces):
    """"The London office will move to a larger building next spring." -> "Next spring, the London office will move
    to a larger building."; a time in the middle comes forward too ("She died in 1934 from an illness ..." -> "In
    1934, she died from an illness ..."). Only the main verb's one time, when the verb doesn't take it as its
    complement ("The meeting is on Monday", "The film is set in 1920"; see also time_stays_put), with no negation or
    focus word, and not from after a noun, a phrase on a noun or a clause it might belong to ("described the riots in
    1968", "wrote a book about Berlin in 1989", "decided to leave in 2019")."""
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    if (opens_with_phrase(sent) or uncertain_capital(sent[0]) or not sent[0].text[:1].isupper()
            or root.lemma_.lower() in TIME_COMPLEMENT_VERBS | TIME_OWNING_ADVICE_VERBS
            or any(c.dep_ in ('conj', 'cc', 'neg') for c in root.children)
            or (root.lemma_ == 'be' and not any(c.dep_ in ('acomp', 'attr', 'oprd') for c in root.children))
            or limits_main_verb(root)):
        return None
    times = [c for c in root.children if c.lower_ in SECOND_TIME_ADVERBS or framing_phrase(c) is not None]
    if len(times) != 1 or times[0].i < root.i or framing_phrase(times[0]) is None:
        return None
    span = framing_phrase(times[0])
    if span[0].i in phrase_tokens(doc) or recurring_over_quantity(span, root) or time_stays_put(sent, span):
        return None
    if misparsed_root(sent) or qualifies_phrase(doc[span.start - 1], times[0]) or carries_clause(span):
        return None     # "located the hikers late on Sunday evening", "in 1998 when its founder retired" (see
                        # phrase_belongs_elsewhere)
    if span.end < stop and doc[span.end].lower_ in TIME_MEASURE_PREPS and any(
            t.like_num or t.lower_ in ('a', 'an', 'few', 'several', 'couple') for t in span):
        return None     # "Sales doubled two weeks after the launch": the length counts from what follows ("Two weeks,
                        # sales doubled after the launch")
    if span[0].lower_ in ('every', 'each') and any(t.like_num or t.ent_type_ in MOVE_NUMBER_ENTITIES
                                                   for t in doc[root.i + 1: span.start]):
        return None     # "sends two reminders every day": a rate, which may belong to the amount
    first, lead = sent[0], 0
    if first.pos_ in ('CCONJ', 'ADV', 'SCONJ') and not (first.i + 1 < sent.end and doc[first.i + 1].text == ','):
        if first.lower_ not in LEADING_CONNECTIVES or first.head.i != root.i:
            return None     # "Recently the team ...": nothing goes in front of it
        lead = 1
    subject = next((c for c in root.children if c.dep_ in ('nsubj', 'nsubjpass', 'expl', 'csubj')), None)
    if (subject is None or subject.i > root.i or subject.left_edge.i != sent.start + lead or subject.like_num
            or subject.ent_type_ in MOVE_NUMBER_ENTITIES):
        return None
    run = piece_run(pieces, doc, span.start, span.end - 1)
    if run is None:
        return None
    before, after = live_neighbour(pieces, run[0], -1), live_neighbour(pieces, run[1], 1)
    if (before is None or before[0] is None or doc[before[0]].is_punct or after is None
            or (after[0] != stop and (after[0] is None or doc[after[0]].is_punct))):
        return None
    # What the time may belong to is read from the sentence as written: after a passive turned active ("The mayor
    # laid the cornerstone of the old library in 1902") it was the verb's ("was laid in 1902 by the mayor")
    if time_could_attach(doc[span.start - 1], root, case_hedge(times[0]) is None):
        return None
    if before[0] == root.i and after[0] == stop and root.i - sent.start > 5:
        return None     # "The new cycle lanes, which make riding safer, opened in March": a long subject and a bare verb
    live = [k for k, p in enumerate(pieces) if p[1]]
    if lead and pieces[live[0]][0] != first.i:
        return None
    head = [list(pieces[live[0]])] if lead else []
    moved = [list(p) for p in pieces[run[0]: run[1] + 1]]
    rest = [list(p) for k, p in enumerate(pieces) if not run[0] <= k <= run[1] and not (lead and k == live[0])]
    if lead:
        lowercase_first(moved, doc)
    else:
        capitalize_first(moved)
    lowercase_first(rest, doc)
    return head + moved + [[None, ',']] + rest

# --- "Critics argued that the plan does nothing." -> "The plan does nothing, critics argued." ---
# Opening with who said it is a stock frame of AI text ("Research suggests that ...", "Studies show that ..."); the
# claim can come first, with its source after it. The tense, the hedges and the source stay as written.

CLAIM_REPORTING_VERBS = {'argue', 'say', 'suggest', 'find', 'show', 'note', 'claim', 'report', 'believe', 'think',
                         'warn', 'caution', 'conclude', 'estimate', 'predict', 'explain', 'maintain', 'contend', 'add',
                         'observe', 'acknowledge', 'admit', 'agree', 'announce', 'confirm', 'reveal', 'stress',
                         'emphasize', 'emphasise', 'write', 'state', 'indicate', 'demonstrate', 'calculate',
                         'forecast', 'assert', 'allege', 'reckon', 'concede', 'fear', 'expect', 'feel'}
# The third-person pronouns, which in the claim would come before the source they may stand for ("Her team would win,
# she said")
CLAIM_SPEAKER_PRONOUNS = {'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they',
                          'them', 'their', 'theirs', 'themselves'}
CLAIM_GROUP_PRONOUNS = {'they', 'them', 'their', 'theirs', 'themselves'}
CLAIM_MAX_TOKENS = 24
# Words in the source or on its verb that deny, count down or discredit what it says ("No study has shown that X",
# "Few analysts expect that X", "Only a few scientists believe that X", "Critics falsely claimed that X", "Scientists
# once believed that X"): put first, X would read as a fact the source then takes back
CLAIM_DOUBT_WORDS = {'only', 'fewer', 'minority', 'noone', 'wrongly', 'falsely', 'mistakenly', 'incorrectly',
                     'erroneously', 'wrongfully', 'naively', 'once', 'formerly', 'previously', 'originally', 'initially',
                     'rarely', 'seldom', 'supposedly', 'allegedly'}
# Verbs that can report a demand as well as a statement: after them a bare verb is a demand ("The guidelines suggest
# that adults exercise daily", "The rules state that players wear helmets", "Advocates argue that employers pay more")
CLAIM_MANDATE_VERBS = {'suggest', 'stress', 'emphasize', 'emphasise', 'warn', 'caution', 'argue', 'agree', 'conclude',
                       'contend', 'maintain', 'say', 'state', 'write', 'add', 'note', 'explain'}

def claim_to_end(sent, pieces):
    """"Research suggests that ten minutes of practice may lower blood pressure." -> "Ten minutes of practice may
    lower blood pressure, research suggests." Only a plain source (no comma, at most six words, not "it"/"this")
    and a reporting verb without a modal or a negation ("might argue", "didn't say"), followed by nothing but its
    claim; nothing in the source or on its verb that denies or discredits the claim (CLAIM_DOUBT_WORDS); the claim a
    statement with its own subject that runs to the end, with no comma, no question, not a question of its own ("shows
    how much can change"), no second "that" ("said that X and that Y"), not a demand ("He suggested that she take a
    break", see CLAIM_MANDATE_VERBS) and no pronoun that may stand for the source."""
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    if root.lemma_.lower() not in CLAIM_REPORTING_VERBS or root.pos_ != 'VERB' or root.tag_ == 'VB':
        return None
    subject = next((c for c in root.children if c.dep_ == 'nsubj' and c.i < root.i), None)
    clause = next((c for c in root.children if c.dep_ == 'ccomp' and c.i > root.i), None)
    if (subject is None or clause is None or subject.left_edge.i != sent.start or uncertain_capital(sent[0])
            or sent.start in phrase_tokens(doc) or subject.right_edge.i - sent.start > 5
            or subject.lower_ in ('it', 'this', 'that', 'these', 'those', 'which', 'what', 'who', 'there', 'one')):
        return None
    if any(c.dep_ not in ('nsubj', 'aux', 'advmod', 'ccomp', 'punct') for c in root.children):
        return None     # "told reporters that", "said on Monday that", "didn't say"
    doubting = TIME_MOVE_NEGATIONS | CLAIM_DOUBT_WORDS
    if any(t.lower_ in doubting or t.dep_ == 'neg' for t in subject.subtree) or any(
            c.dep_ == 'advmod' and c.lower_ in doubting for c in root.children):
        return None     # "Nobody thinks the plan will work", "Not all scientists agree that ...", "wrongly believe"
    if any(not ((t.dep_ == 'aux' and t.lemma_ in ('have', 'be')) or (t.dep_ == 'advmod' and t.lower_ not in doubting))
           for t in doc[subject.right_edge.i + 1: root.i]):
        return None     # only "have"/"be" and adverbs between: "have long argued", not "might argue"
    start = root.i + 1
    if doc[start].lower_ == 'that':
        if not (doc[start].dep_ == 'mark' or (doc[start].dep_ == 'det' and doc[start].head.tag_ in ('NNS', 'NNPS'))):
            return None     # "shows that sleep matters" may be "that (kind of) sleep"
        start += 1
    if clause.left_edge.i not in (root.i + 1, start) or clause.right_edge.i != stop - 1 or not 3 <= stop - start <= CLAIM_MAX_TOKENS:
        return None
    body = doc[start:stop]
    claimant = next((c for c in clause.children if c.dep_ in ('nsubj', 'nsubjpass', 'expl')), None)
    if claimant is None or claimant.left_edge.i != start or claimant.lower_ in ('who', 'what', 'which'):
        return None
    if doc[start].tag_ in ('WDT', 'WP', 'WP$', 'WRB') or doc[start].lower_ in ('whether', 'if'):
        return None     # a question: "shows how much can change", "show which projects are on track"
    if any(t.lower_ in ('that', 'whether') and t.dep_ == 'mark' and (doc[t.i - 1].lower_ in ('and', 'or', 'but', ',')
                                                                    or (t.head.dep_ == 'conj' and t.head.head.i == clause.i))
           for t in body):
        return None     # "said that the road will reopen and that traffic will return to normal"
    auxiliaries = [c for c in clause.children if c.dep_ in ('aux', 'auxpass')]
    # A finite verb: "suggests", "had", "are", "may lower", "is needed", or a past tense the tagger took for a
    # participle ("noted that margins improved")
    finite = (clause.tag_ in ('VBZ', 'VBD') or clause.lower_ in ('are', "'re", '’re', 'am')
              or any(c.tag_ in ('MD', 'VBZ', 'VBP', 'VBD') for c in auxiliaries)
              or (clause.tag_ == 'VBN' and claimant.dep_ == 'nsubj' and not auxiliaries))
    # A bare present after a plural is a statement too ("Studies show that children learn best through play"), but
    # not after a verb that can report a demand ("suggested that she take a break", "The guidelines suggest that
    # adults exercise daily")
    if not finite and (root.lemma_.lower() in CLAIM_MANDATE_VERBS or not (
            clause.tag_ == 'VBP' and claimant.tag_ in ('NNS', 'NNPS', 'PRP') and claimant.lower_ not in ('he', 'she', 'it'))):
        return None
    if any(t.text in MOVE_BREAK_MARKS or t.text == '?' for t in body):
        return None
    plural = subject.tag_ in ('NNS', 'NNPS') or any(c.dep_ == 'conj' for c in subject.children)
    if subject.lower_ in ('he', 'she', 'they'):
        speaker = CLAIM_SPEAKER_PRONOUNS
    elif subject.pos_ == 'PRON':
        speaker = set()
    else:
        speaker = CLAIM_GROUP_PRONOUNS if plural else CLAIM_SPEAKER_PRONOUNS
    if any(t.lower_ in speaker for t in body):
        return None
    source, claim = piece_run(pieces, doc, sent.start, root.i), piece_run(pieces, doc, start, stop - 1)
    if source is None or claim is None or claim[0] != source[1] + 1 + (start - root.i - 1):
        return None
    said = [list(p) for p in pieces[claim[0]: claim[1] + 1]]
    who = [list(p) for p in pieces[source[0]: source[1] + 1]]
    if sent[0].text[:1].isupper():
        capitalize_first(said)
    lowercase_first(who, doc)
    return said + [[None, ',']] + who + [list(p) for p in pieces[claim[1] + 1:]]

# --- "According to the council, rents have doubled." <-> "Rents have doubled, according to the council." ---

def source_phrase(child):
    """The span "according to X" of a phrase hanging on the main verb (X up to eight words, no comma), or None."""
    if child is None or child.lower_ != 'according' or child.dep_ != 'prep' or child.left_edge.i != child.i:
        return None
    doc = child.doc
    to = doc[child.i + 1] if child.i + 1 < len(doc) else None
    if to is None or to.lower_ != 'to' or to.head.i != child.i or not any(c.dep_ == 'pobj' for c in to.children):
        return None
    span = doc[child.i: child.right_edge.i + 1]
    if len(span) > 10 or any(t.text in MOVE_BREAK_MARKS for t in span):
        return None
    return span

def plain_main_clause(sent, span_start, span_end, root):
    """True if the words of the sentence outside a phrase (span_start..span_end - 1) are one clause with its subject
    first and no comma, quote or colon, so the phrase can stand at either end of it."""
    doc, stop = sent.doc, sent[-1].i
    words = [t for t in doc[sent.start:stop] if not span_start <= t.i < span_end]
    subject = next((c for c in root.children if c.dep_ in ('nsubj', 'nsubjpass', 'expl', 'csubj') and c.i < root.i), None)
    return (subject is not None and len(words) >= 5 and not any(t.text in MOVE_BREAK_MARKS for t in words[1:-1])
            and words[0].i == subject.left_edge.i)

def source_to_end(sent, pieces):
    """"According to the chief technology officer, no customer data was lost." -> "No customer data was lost,
    according to the chief technology officer." Not when the clause has a third-person pronoun, which would come
    before the source it may stand for ("According to the firm, its profits rose")."""
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    child = next((c for c in root.lefts if c.left_edge.i == sent.start), None)
    span = source_phrase(child)
    if span is None or span.end >= stop or doc[span.end].text != ',' or span[0].i in phrase_tokens(doc):
        return None
    if not plain_main_clause(sent, span.start, span.end + 1, root) or any(
            t.lower_ in CLAIM_SPEAKER_PRONOUNS for t in doc[span.end + 1: stop]):
        return None
    run = piece_run(pieces, doc, span.start, span.end - 1)
    if (run is None or live_neighbour(pieces, run[0], -1) is not None or run[1] + 1 >= len(pieces)
            or pieces[run[1] + 1][0] != span.end):
        return None
    rest = [list(p) for p in pieces[run[1] + 2:]]
    if not rest or rest[-1][0] != stop:
        return None
    moved = [list(p) for p in pieces[run[0]: run[1] + 1]]
    lowercase_first(moved, doc)
    if sent[0].text[:1].isupper():
        capitalize_first(rest)
    return rest[:-1] + [[None, ',']] + moved + rest[-1:]

# What "according to" names when it gives a source: people and bodies (names_people), and what they publish or pass
# on ("according to the report", "according to government data", "according to legend"). With anything else it may
# mean "depending on" ("Fees differ, according to age"; "The houses will be built, according to the plans"), which
# can't open the sentence
SOURCE_NAMING_NOUNS = {'report', 'study', 'survey', 'data', 'datum', 'figure', 'estimate', 'statistic', 'poll', 'census',
                       'record', 'document', 'source', 'forecast', 'projection', 'analysis', 'research', 'evidence',
                       'review', 'account', 'article', 'paper', 'website', 'statement', 'filing', 'memo', 'letter',
                       'database', 'archive', 'calculation', 'count', 'tally', 'model', 'legend', 'tradition', 'myth',
                       'folklore', 'lore', 'rumour', 'rumor', 'scripture', 'bible', 'prophecy', 'press', 'media',
                       'newspaper', 'biography', 'diary', 'police', 'government', 'ministry', 'department', 'agency',
                       'company', 'firm', 'bank', 'court', 'council', 'committee', 'board', 'authority', 'spokesperson',
                       'spokesman', 'spokeswoman', 'official', 'expert', 'analyst', 'researcher', 'scientist', 'witness',
                       'insider', 'observer', 'economist', 'reporter', 'journalist', 'historian', 'critic', 'author',
                       'auditor', 'organiser', 'organizer', 'owner', 'operator', 'regulator', 'watchdog'}

def source_to_front(sent, pieces):
    """"The plan will add 1,500 homes by 2030, according to Mayor Ted Wheeler." -> "According to Mayor Ted Wheeler,
    the plan will add 1,500 homes by 2030." Only when it names a source (see SOURCE_NAMING_NOUNS), and not when the
    source has a pronoun, which would then come before what it stands for ("according to its director")."""
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    span = next((source_phrase(c) for c in root.rights if source_phrase(c) is not None), None)
    if (span is None or span.end != stop or doc[span.start - 1].text != ',' or span[0].i in phrase_tokens(doc)
            or opens_with_phrase(sent) or uncertain_capital(sent[0]) or not sent[0].text[:1].isupper()
            or any(t.lower_ in CLAIM_SPEAKER_PRONOUNS for t in span)):
        return None
    named = next((c for c in doc[span.start + 1].children if c.dep_ == 'pobj'), None)
    if named is None or not (names_people(named) or named.lemma_.lower() in SOURCE_NAMING_NOUNS
                             or named.ent_type_ in ('PERSON', 'ORG', 'NORP', 'GPE', 'WORK_OF_ART', 'LAW', 'FAC')):
        return None
    if not plain_main_clause(sent, span.start - 1, span.end, root):
        return None
    run = piece_run(pieces, doc, span.start, span.end - 1)
    if (run is None or run[0] < 1 or run[1] + 1 >= len(pieces) or pieces[run[0] - 1][0] != span.start - 1
            or pieces[run[1] + 1][0] != stop):
        return None
    moved = [list(p) for p in pieces[run[0]: run[1] + 1]]
    rest = [list(p) for p in pieces[:run[0] - 1]]
    capitalize_first(moved)
    lowercase_first(rest, doc)
    return moved + [[None, ',']] + rest + [list(p) for p in pieces[run[1] + 1:]]

# --- "Electric buses, which produce no fumes, now run on every route." -> "Electric buses produce no fumes and now
# run on every route." ---
# An aside set off by commas between a subject and its verb is a stock frame of AI text; its two statements about
# the subject can be joined with "and" ("The bridge, which was built in 1890, carries ..." -> "The bridge was built
# in 1890 and carries ..."), unless a present aside would come before a past verb ("The lanes, which make riding
# safer, opened in March" -> "make riding safer and opened in March"), which reads oddly, or the two statements
# contrast, where "and" would lose the "but": an aside of seeming or intent ("The plan, which seemed sensible, did not
# work"; "The hotel, which promised sea views, faced a car park") or a negation or a word of contrast on the main
# verb ("The drug, which was approved in 2010, is no longer sold")
ASIDE_EXPECTING_VERBS = {'seem', 'appear', 'look', 'sound', 'feel', 'promise', 'suppose', 'claim', 'expect', 'pledge',
                         'vow', 'hope', 'tout', 'want', 'wish'}
# ...and verbs of intent or labelling, with what was intended ("which was meant to last a century", "which was
# advertised as waterproof"; not "which was designed by Eiffel")
ASIDE_INTENT_VERBS = {'mean', 'intend', 'design', 'plan', 'aim', 'try', 'bill', 'market', 'advertise', 'sell', 'build'}
ASIDE_CONTRAST_WORDS = TIME_MOVE_NEGATIONS | {'yet', 'still', 'already', 'instead', 'nevertheless', 'nonetheless',
                                              'however', 'only', 'rarely', 'seldom', 'even', 'longer'}

def clause_time(verb):
    """'past' or 'present' for a clause's verb, read from its first helper when it has one ("has pushed" is
    present, "was built" past, "can save" present), or None."""
    head = min((c for c in verb.children if c.dep_ in ('aux', 'auxpass')), key=lambda c: c.i, default=verb)
    if head.tag_ == 'MD':
        return 'present' if head.lower_ in ('can', 'will', 'may', 'must', 'shall') else None
    return {'VBD': 'past', 'VBZ': 'present', 'VBP': 'present'}.get(head.tag_)

def aside_to_and(sent, pieces):
    """"The bridge, which was built in 1890, carries 2,000 cars a day." -> "The bridge was built in 1890 and carries
    2,000 cars a day." Only "which"/"who" as the aside's subject, right after the sentence's subject, an aside of at
    most twelve words with no comma, "and" or clause of its own ("which formed after a glacier retreated" would read
    "a glacier retreated and sits ..."), the main verb right after it with nothing joined to it ("opened in 1998 and
    now has 40 staff" would get a second "and"), the aside not in the present when the main verb is in the past, nor
    in the past perfect when the main verb isn't ("had climbed for weeks and fell sharply"), and no contrast between
    them (see ASIDE_EXPECTING_VERBS)."""
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    subject = next((c for c in root.children if c.dep_ in ('nsubj', 'nsubjpass') and c.i < root.i), None)
    if (subject is None or subject.left_edge.i != sent.start or subject.pos_ not in ('NOUN', 'PROPN')
            or uncertain_capital(sent[0]) or any(c.dep_ in ('conj', 'cc') for c in root.children)):
        return None
    clause = next((c for c in subject.children if c.dep_ == 'relcl' and c.i > subject.i), None)
    if clause is None:
        return None
    relative, close = clause.left_edge, clause.right_edge.i + 1
    if (relative.lower_ not in ('which', 'who') or relative.dep_ not in ('nsubj', 'nsubjpass') or relative.head.i != clause.i
            or relative.i < 2 or doc[relative.i - 1].text != ',' or close >= root.i or doc[close].text != ','
            or close - relative.i > 13):
        return None
    if any(t.text in MOVE_BREAK_MARKS or t.pos_ == 'CCONJ' for t in doc[relative.i + 1: close]):
        return None
    if any(t.pos_ in ('VERB', 'AUX') and t.i != clause.i and any(c.dep_ in ('nsubj', 'nsubjpass', 'expl') for c in t.children)
           for t in doc[relative.i + 1: close]):
        return None     # a clause inside the aside: "which formed after a glacier retreated"
    if any(t.dep_ not in ('aux', 'auxpass', 'neg', 'advmod') for t in doc[close + 1: root.i]):
        return None
    intent = clause.lemma_.lower() in ASIDE_INTENT_VERBS and any(
        c.dep_ in ('xcomp', 'oprd') or (c.dep_ == 'prep' and c.lower_ == 'as') for c in clause.children)
    if intent or clause.lemma_.lower() in ASIDE_EXPECTING_VERBS or any(
            c.dep_ in ('neg', 'advmod') and (c.lower_ in ASIDE_CONTRAST_WORDS or any(g.lower_ in ASIDE_CONTRAST_WORDS
                                                                                for g in c.children))
            for c in root.children):
        return None     # "The plan, which seemed sensible, did not work": a contrast
    helper = min((c for c in clause.children if c.dep_ in ('aux', 'auxpass')), key=lambda c: c.i, default=None)
    own = min((c for c in root.children if c.dep_ in ('aux', 'auxpass')), key=lambda c: c.i, default=None)
    if helper is not None and helper.lower_ in ('had', "'d", '’d') and (own is None or own.lower_ != helper.lower_):
        return None     # "Bond yields, which had climbed for weeks, fell sharply": "had climbed and fell"
    # "The bridge, which was built in 1890, carries 2,000 cars a day" -> "... was built in 1890 and carries ...", but
    # not "The lanes, which make riding safer, opened in March" -> "... make riding safer and opened in March"
    when, now = clause_time(clause), clause_time(root)
    if when is None or now is None or (when, now) == ('present', 'past') or has_quoted(doc, range(sent.start, stop)):
        return None
    first = piece_run(pieces, doc, sent.start, relative.i - 2)
    aside = piece_run(pieces, doc, relative.i + 1, close - 1)
    main = piece_run(pieces, doc, close + 1, stop - 1)
    if first is None or aside is None or main is None or not first[1] < aside[0] < aside[1] < main[0]:
        return None
    return ([list(p) for p in pieces[first[0]: first[1] + 1]] + [list(p) for p in pieces[aside[0]: aside[1] + 1]]
            + [[None, 'and']] + [list(p) for p in pieces[main[0]: main[1] + 1]] + take(pieces, stop, stop))

# --- "The legacy of the Industrial Revolution" -> "The Industrial Revolution's legacy" ---
# Only nouns that read naturally with an owner in front, and only a short, definite owner: a name or "the/this/our
# ..." with at most four words (two after "your"/"our": "your job's stability", not "your available credit's"), not
# a plural or a name ending in "s" ("the process's"), a time, an amount or a list. Never a quantity, a measure or a
# part ("the number of", "the size of", "the quality of", "the end of", "the rest of"), a material, people and
# goods of a place ("the people of Paris", "the wines of France"), or the head of an idiom ("the benefit of the doubt",
# "the weight of the world", "scratches the surface of"): those nouns aren't listed.

POSSESSED_NOUNS = {'legacy', 'history', 'future', 'impact', 'influence', 'success', 'failure', 'growth', 'decline',
                   'role', 'reputation', 'popularity', 'design', 'architecture', 'economy', 'culture', 'identity',
                   'heritage', 'origin', 'founder', 'leader', 'leadership', 'owner', 'mission', 'vision', 'goal',
                   'strategy', 'approach', 'decision', 'response', 'performance', 'revenue', 'profit', 'budget',
                   'efficiency', 'beauty', 'charm', 'appeal', 'strength', 'weakness', 'advantage',
                   'feature', 'construction', 'development', 'collapse', 'launch', 'structure', 'capacity',
                   'capital', 'location', 'layout', 'interior', 'exterior', 'achievement', 'contribution', 'career',
                   'death', 'birth', 'member', 'customer', 'user', 'result', 'finding', 'aim', 'purpose', 'effect',
                   'safety', 'security', 'health', 'condition', 'policy', 'rule', 'president',
                   'director', 'chief', 'staff', 'workforce', 'shape', 'colour', 'color',
                   'author', 'creator', 'inventor', 'designer', 'architect', 'campus', 'website', 'menu'}
OWNER_DETERMINERS = {'the', 'this', 'that', 'our', 'your', 'my', 'his', 'her', 'their', 'its'}
# Words that single the noun out together with what follows it ("the only member of the team to finish the race",
# "the first member of his family to attend college"): "the team's only member to finish" changes what is singled out
OWNER_SCOPE_WORDS = {'only', 'first', 'last', 'sole', 'lone', 'single', 'next', 'same', 'second', 'third', 'final',
                     'latest', 'previous'}
# Abstract owners, which read oddly with "'s" ("the change's effect", "the evidence's weight"), and titles and fixed
# phrases ("the leader of the pack", "the death of the author")
OWNER_ABSTRACT_NOUNS = {'doubt', 'evidence', 'change', 'effect', 'matter', 'truth', 'fact', 'idea', 'issue', 'question',
                        'problem', 'argument', 'situation', 'case', 'concept'}
OWNER_FIXED_PHRASES = {('leader', 'pack'), ('death', 'author'), ('birth', 'nation')}

def of_to_possessive(sent, pieces):
    """"The legacy of the Industrial Revolution can still be seen" -> "The Industrial Revolution's legacy can still be
    seen"; "the history of the region" -> "the region's history". The owner's words move in front of the noun, and
    the dropped "the" and "of" stay as empty pieces, so the templates that run after it still find every word. Not
    in a title ("We watched The Death of Stalin"), not when spaCy took a noun after the owner for the verb ("The
    success of the vaccine rollout surprised ..."), and a line in lower case stays so ("the leader ..." in a poem)."""
    doc = sent.doc
    texts = {p[0]: p[1] for p in pieces if p[0] is not None}
    for token in sent:
        if (token.tag_ not in ('NN', 'NNS') or token.lemma_.lower() not in POSSESSED_NOUNS or token.i + 3 >= sent.end
                or (token.dep_ == 'pobj' and token.head.lower_ == 'of') or any(c.dep_ == 'conj' for c in token.children)):
            continue
        det, of = doc[token.left_edge.i], doc[token.i + 1]
        if (det.lower_ != 'the' or det.head.i != token.i or of.lower_ != 'of' or of.head.i != token.i
                or any(c.dep_ not in ('amod', 'compound') for c in token.lefts if c.i != det.i) or token.i - det.i > 3):
            continue
        if (any(c.lower_ in OWNER_SCOPE_WORDS or c.tag_ == 'JJS' or c.ent_type_ == 'ORDINAL' for c in token.lefts)
                or any(c.dep_ in ('acl', 'relcl') for c in token.rights)):
            continue    # "the only member of the team to finish the race"
        if ((det.i != sent.start and det.text[:1].isupper()) or token.text[:1].isupper()
                or any(t.ent_type_ == 'WORK_OF_ART' for t in doc[det.i: token.i + 1])):
            continue    # a title ("We watched The Death of Stalin") or a line in capitals ("THE LEGACY OF ...")
        owner = next((c for c in of.children if c.dep_ == 'pobj'), None)
        if (owner is None or owner.tag_ not in ('NN', 'NNP') or owner.lower_.endswith('s') or owner.left_edge.i != of.i + 1
                or owner.right_edge.i != of.right_edge.i or owner.i != owner.right_edge.i or owner.i - of.i > 4
                or owner.lemma_.lower() in TIME_NOUNS or owner.ent_type_ in MOVE_NUMBER_ENTITIES
                or owner.lemma_.lower() in OWNER_ABSTRACT_NOUNS
                or (token.lemma_.lower(), owner.lemma_.lower()) in OWNER_FIXED_PHRASES
                or any(t.ent_type_ == 'WORK_OF_ART' for t in doc[of.i: owner.i + 1])):
            continue
        words = doc[of.i + 1: owner.i + 1]
        first = words[0]
        if not ((first.lower_ in OWNER_DETERMINERS and first.dep_ in ('det', 'poss')) or (first.pos_ == 'PROPN' and owner.pos_ == 'PROPN')):
            continue    # "the price of oil": a bare owner reads oddly ("oil's price")
        if first.lower_ not in ('the', 'this', 'that') and first.pos_ != 'PROPN' and owner.i - first.i > 1:
            continue    # "the size of your available credit" isn't "your available credit's size"
        if any(t.pos_ not in ('DET', 'PRON', 'ADJ', 'NOUN', 'PROPN') or t.like_num for t in words):
            continue
        after = doc[owner.i + 1] if owner.i + 1 < len(doc) else None
        if after is not None and (after.lower_ in ('that', 'which', 'who', 'whose', 'where', "'s", '’s') or after.tag_ == 'POS'):
            continue    # "the legacy of the war that shaped Europe": the clause may be the war's
        readings = getAllLemmas(after.lower_) if after is not None else {}
        if after is not None and after.pos_ == 'VERB' and ('VERB' not in readings or (
                after.tag_ != 'VBZ' and after.lower_ in readings.get('NOUN', ()))):
            continue    # "The success of the vaccine rollout surprised many", "... of the price cut surprised ...": a
                        # noun of the owner's ("vaccine rollout", "price cut") may have been read as the verb
        span = range(det.i, owner.i + 1)
        if any(texts.get(i) != doc[i].text for i in span) or has_quoted(doc, span):
            continue    # a swap or another template already changed these words
        if piece_run(pieces, doc, det.i, owner.i) is None:
            continue
        owner_pieces = [[i, texts[i]] for i in range(of.i + 1, owner.i + 1)]
        if 'curly' not in doc.user_data:
            doc.user_data['curly'] = '’' in doc.text      # the writer's own apostrophes, looked up once per text
        owner_pieces[-1][1] += "’s" if doc.user_data['curly'] else "'s"
        if det.i == sent.start and det.text[:1].isupper():
            capitalize_first(owner_pieces)
        new = []
        for p in pieces:
            if p[0] == det.i:
                new += [[det.i, '']] + owner_pieces + [[i, texts[i]] for i in range(det.i + 1, token.i + 1)]
            elif p[0] == of.i:
                new.append([of.i, ''])
            elif p[0] is None or not (det.i < p[0] <= token.i or of.i < p[0] <= owner.i):
                new.append(list(p))
        return new
    return None

# --- "allows us to stay in touch" -> "lets us stay in touch" ---

# "let" for each form of "allow"/"enable". A bare present has none: "Flexible hours let parents work" reads as the
# past as easily as the present
LET_TAG_FORMS = {'VBZ': 'lets', 'VBD': 'let', 'VB': 'let', 'VBN': 'let', 'VBG': 'letting'}
# "Allow 20 minutes to cook", "allow enough room to open the door": giving time or room, not letting someone do it
ALLOWANCE_NOUNS = {'time', 'room', 'space', 'money', 'budget', 'margin', 'enough', 'plenty', 'extra', 'minute', 'hour',
                   'day', 'week', 'month', 'year', 'second', 'fund', 'resource', 'access'}
# What is left to settle by itself, which a person may "allow" to happen ("Allow the dough to rest", "Allow the paint
# to dry"). A person who allows or enables anything else switches it on for a purpose of their own: "Enable location
# services to see nearby stores" isn't "Let location services see nearby stores"
LET_SETTLING_VERBS = {'rise', 'rest', 'dry', 'cool', 'set', 'soak', 'settle', 'sit', 'drain', 'melt', 'thaw', 'steep',
                      'simmer', 'chill', 'harden', 'firm', 'cure', 'prove', 'proof', 'marinate', 'dissolve', 'evaporate',
                      'stand', 'breathe', 'warm', 'heal', 'recover', 'soften', 'thicken', 'reduce', 'bubble', 'infuse',
                      'ferment', 'sink', 'flow', 'drip', 'cook', 'bake', 'boil', 'brown'}
# Subjects that are people: one who allows or enables things is giving a setting or a permission
LET_PERSON_SUBJECTS = {'i', 'you', 'we', 'they', 'he', 'she', 'one', 'everyone', 'everybody', 'someone', 'somebody',
                       'anyone', 'anybody'}

def allows_to_lets(sent, pieces):
    """"While it allows us to stay connected ..." -> "While it lets us stay connected ..."; "Allow the dough to rest."
    -> "Let the dough rest." Only "allow"/"enable" with its doer right after it and "to" + a verb (every "to" of the
    verbs joined to it goes too: "to spend ... and to meet" -> "spend ... and meet"); not a passive or a negation
    ("is allowed to", "doesn't allow"), nor time, room or money given for something ("allow 20 minutes to cook"). A
    person or an order allowing a thing is a setting switched on for a purpose ("Enable JavaScript to view this
    page", "You must allow cookies to use this site"), unless the thing just settles (LET_SETTLING_VERBS); "enable"
    lets only people do things ("The grant enabled researchers to hire staff"). "let" must keep the tense: "allowed"
    only after a singular ("The new law let workers strike"), never a bare present (see LET_TAG_FORMS). Not "allow X to
    be" + a complement, which grants that X is so ("He allowed them to be right"). The verb's and the dropped words'
    slots stay in `pieces`."""
    doc = sent.doc
    texts = {p[0]: p[1] for p in pieces if p[0] is not None}
    for verb in sent:
        if (verb.lemma_.lower() not in ('allow', 'enable') or verb.pos_ != 'VERB' or verb.tag_ not in LET_TAG_FORMS
                or texts.get(verb.i) != verb.text or any(c.dep_ in ('auxpass', 'neg', 'nsubjpass') for c in verb.children)):
            continue
        action = next((c for c in verb.children if c.dep_ in ('xcomp', 'ccomp') and c.tag_ == 'VB' and c.i > verb.i), None)
        if action is None or action.i < 2:
            continue
        if action.lemma_ == 'be' and any(c.dep_ in ('acomp', 'attr') for c in action.children):
            continue    # "He allowed them to be right about the cost": granting that something is so ("let them be
                        # right" would say he made them so)
        to = doc[action.i - 1]
        doer = (next((c for c in verb.children if c.dep_ == 'dobj' and c.i < action.i), None)
                or next((c for c in action.children if c.dep_ == 'nsubj' and c.i < action.i), None))
        if (to.lower_ != 'to' or to.head.i != action.i or doer is None or doer.left_edge.i != verb.i + 1
                or doer.right_edge.i + 1 != to.i or to.i - verb.i > 6 or texts.get(to.i) != to.text):
            continue
        if any(t.like_num or t.lemma_.lower() in ALLOWANCE_NOUNS or t.ent_type_ in MOVE_NUMBER_ENTITIES
               for t in doc[verb.i + 1: to.i]) or has_quoted(doc, range(verb.i, action.i + 1)):
            continue
        subject = next((c for c in verb.children if c.dep_ in ('nsubj', 'csubj')), None)
        by_person = ((subject is None and verb.tag_ == 'VB' and verb.dep_ in ('ROOT', 'conj'))
                     or (subject is not None and (names_people(subject) or subject.lower_ in LET_PERSON_SUBJECTS)))
        person_doer = names_people(doer) or (doer.tag_ == 'PRP' and doer.lower_ not in ('it', 'itself'))
        if verb.lemma_.lower() == 'enable':
            if by_person or not person_doer:
                continue    # "Please enable location services to find nearby stores"
        elif by_person and not (person_doer or doer.lower_ in ('it', 'itself')
                                or action.lemma_.lower() in LET_SETTLING_VERBS):
            continue        # "Allow pop-ups to view the full document"
        if verb.tag_ == 'VBD':
            relative = subject is not None and subject.lower_ in ('that', 'which', 'who') and verb.dep_ == 'relcl'
            owner = verb.head if relative else subject
            if owner is None or not (owner.tag_ in ('NN', 'NNP') or owner.lower_ in ('he', 'she', 'it', 'this', 'that')):
                continue    # "The new rules allowed tenants to keep pets": "The new rules let tenants" reads as now
        drop = {to.i} | repeated_infinitive_marks(action, sent.end)
        new = [list(p) for p in pieces]
        for p in new:
            if p[0] == verb.i:
                p[1] = match_case(verb.text, LET_TAG_FORMS[verb.tag_])
            elif p[0] in drop:
                p[1] = ''
        return new
    return None

# --- Coverage: no longer sentence is left exactly as the AI wrote it ---

COVERAGE_MIN_WORDS = 8
CHAOS_COVERAGE_MIN_WORDS = 6    # Chaos gives shorter sentences a change too
# The constructions most typical of AI text, rewritten whenever they appear
SIGNATURE_TRANSFORMS = (existential_to_direct, extraposed_to_gerund, essential_for_to, importance_of, not_only_inverted,
                        category_to_direct, goal_to_gerund, claim_to_end, aside_to_and)
# Phrases moved from one end of a sentence to the other
PHRASE_MOVES = (front_trailing_adjunct, back_fronted_adjunct, enabling_to_with, front_subject_place, time_to_front,
                time_to_end, source_to_end, source_to_front)
# ...of which these put nothing in front of the sentence, so they may follow a sentence that was reordered
MOVES_TO_END = (back_fronted_adjunct, time_to_end, source_to_end)
REORDERING = SIGNATURE_TRANSFORMS + (passive_to_active,) + PHRASE_MOVES
# A date or a source can move on top of a passive turned active in Chaos (each checks the pieces as they now stand).
# Not after the templates that join two statements ("X, which A, B" -> "X A and B"; "Not only ..., but ..."), where a
# date moved to either end would reach a statement it didn't belong to ("In June, the stadium seats 60,000 fans and
# will host the final")
FOLLOW_UP_MOVES = (time_to_front, time_to_end, source_to_end, source_to_front)

def in_place_transforms(tone):
    """The rewrites that work inside a phrase and keep a slot in the pieces for every word, moved within its phrase
    ("the region's history", a list in a new order) or dropped (an empty slot), so a template or a move can still
    follow them; "lets" and "is how" don't suit Academic."""
    plain = [allows_to_lets, process_to_how] if tone != 'academic' else []
    return [denominalize, the_way_to_how, reorder_list] + plain + [of_to_possessive]

def untouched(pieces, doc):
    indices = [p[0] for p in pieces]
    return None not in indices and indices == sorted(indices) and all(p[1] == doc[p[0]].text for p in pieces)

def force_word_edit(sent, pieces, tone, deep_mode, vet=None, nearby=frozenset()):
    """One safe word swap: the same rules as plan_word_edits, without the dice. WordNet guesses are only
    tried when `vet` can check them with the model. `nearby` holds the neighbouring sentences' words as they
    now read, so the swap doesn't echo them. The swaps of AI_FAVOURITES and EVERYDAY_SWAPS read their own frames
    ("benefited the most from", "remained stable") and amounts ("roughly one third"), so those words are tried
    there too."""
    doc = sent.doc
    floor = CHAOS_FLOOR_VETTED if deep_mode else SYNONYM_FLOOR[tone]
    avoid = {t.lower_ for t in sent} | {t.lemma_.lower() for t in sent} | set(nearby)
    candidates = [t for t in sent if t.i not in phrase_tokens(doc)
                  and (swap_key(t) or (can_rewrite(t) and not t.ent_type_ and not fixed_frame(t)))]
    random.shuffle(candidates)
    guesses = 0
    for token in candidates:
        key, extra = curated_key(token), swap_key(token)
        word, more = None, {}
        if key in AI_WORDS or key in PLAIN_WORDS:
            plain = ACADEMIC_PLAIN if tone == 'academic' else PLAIN_WORDS
            if (not curated_fits(token, key) or key in TONE_KEEPS.get(tone, ())
                    or (key not in AI_WORDS and key not in plain)):
                continue
            choices = (curated_choices(AI_WORDS[key], tone, token) if key in AI_WORDS
                       else context_choices(plain[key], token))
            choice = fresh_choice(choices, avoid) or echo_fallback(key, token, avoid)
            word = inflect(choice, token.tag_) if choice else None
        elif extra:
            choice = fresh_choice(swap_choices(token, extra, tone), avoid)
            more = swap_edits(token, extra, choice) if choice is not None else None
            if not more:
                continue
            word = more.pop(token.i)
        elif (vet is not None and guesses < 3 and not linking_adverb(token) and not summary_noun(token)
              and not adjectival_participle(token)):
            entry = SYNONYMS.get(f'{token.lemma_.lower()}|{UPOS_TO_WORDNET[token.pos_]}')
            if entry:
                _, word = choose_synonym(token, entry, tone, floor, avoid, token.pos_ == 'NOUN',
                                         vetted=vet is not None and deep_mode)
                if word and vet is not None:
                    guesses += 1
                    if not vet(sent, {token.i: match_case(token.text, word)}):
                        continue
        if not word or word.lower() == token.lower_:
            continue
        split = particle_split(token, word)     # "speed them up"
        if split:
            word = split[0]
        new = [list(p) for p in pieces]
        for piece in new:
            if piece[0] == token.i:
                piece[1] = match_case(token.text, word)
            elif piece[0] in more:
                piece[1] = more[piece[0]]       # "carried it out"
            elif split and piece[0] == split[1].i:
                piece[1] = split[2]
            elif piece[0] == token.i - 1 and doc[token.i - 1].lower_ in ('a', 'an'):
                piece[1] = match_case(piece[1], 'an' if starts_with_vowel_sound(word) else 'a')
            elif (piece[0] == token.i + 1 and doc[token.i + 1].tag_ == 'POS' and piece[1] in ("'", "’")
                  and not word.lower().endswith('s')):
                piece[1] += 's'     # "individuals' risk" -> "people's risk"
        return new
    return None

def ensure_change(sentence, tone, deep_mode, vet=None, previous='', nearby=frozenset()):
    """Give an untouched sentence one change: a template, a move or an in-place rewrite where one fits, or a word swap.
    Outside Chaos they are tried in a random order, so a word swap may win where a template would also fit; Chaos tries
    the structural ones first and puts a word swap on top where one fits."""
    sent, pieces = sentence.span, sentence.pieces
    options = [force_word_edit]
    if restructurable(sent):
        options += in_place_transforms(tone) + list(REORDERING)
    random.shuffle(options)
    if deep_mode:
        options.sort(key=lambda option: option is force_word_edit)
    for option in options:
        new = (option(sent, pieces, tone, deep_mode, vet, nearby) if option is force_word_edit else option(sent, pieces))
        if new and option in REORDERING and opens_like(new, previous):
            continue
        if new:
            if deep_mode and option is not force_word_edit:
                new = swap_on_top(sent, new, force_word_edit(sent, new, tone, deep_mode, vet, nearby))
            sentence.pieces = new
            sentence.reordered = option in REORDERING
            return
    # Nothing else fits: split it at a safe point, if it has one (done in the rhythm layer)
    sentence.must_split = True

def swap_on_top(sent, new, swapped):
    """The pieces with a word swap made after a template (`swapped`), if the swap only touched words the template
    left as written: a swap inflects its word as the original was ("reduces" -> "lowers"), so on a verb the
    template made agree ("Cover crops improve soil health and reduce erosion") it would undo the agreement; and an
    "a"/"an" it changes must still stand right before its word. Else `new` as it was."""
    if not swapped:
        return new
    doc = sent.doc
    changed = [k for k, (a, b) in enumerate(zip(new, swapped)) if a != b]
    if len(swapped) != len(new) or any(new[k][0] is None or new[k][1] != doc[new[k][0]].text for k in changed):
        return new
    for k in changed:
        if doc[new[k][0]].lower_ in ('a', 'an'):
            after = live_neighbour(swapped, k, 1)
            if after is None or after[0] != new[k][0] + 1:
                return new
    return swapped

# Chance of changing a sentence's structure, by tone (Academic keeps more of its passive voice)
PASSIVE_RATE = {'standard': 0.7, 'casual': 0.85, 'professional': 0.55, 'academic': 0.2}
MOVE_RATE = {'standard': 0.4, 'casual': 0.45, 'professional': 0.3, 'academic': 0.25}
# The constructions AI text is known for are rewritten more often than plain reordering
SIGNATURE_RATE = {'standard': 0.8, 'casual': 0.9, 'professional': 0.6, 'academic': 0.4}
CHAOS_STRUCTURE_BOOST = 0.3
# Chaos moves phrases and reorders lists and possessives wherever they fit (every rate reaches 1): these keep the
# words and their sense, only rearranged (a possessive drops "the ... of", "lets" its "to"). Academic still never
# gets "lets" (LET_RATE)
CHAOS_REORDER_BOOST = 1.0

def opens_like(new, previous):
    """True if rewritten pieces would open with the same word as the sentence before ("With just a laptop,
    ... With this flexibility, ...")."""
    piece = first_piece(new) if new else None
    return bool(previous) and piece is not None and piece[1].split()[0].lower() == previous.split()[0]

def restructure(sent, pieces, tone, deep_mode, after_move, previous=''):
    """One template or move, by the tone's rates. After a sentence that was reordered only the moves to the end
    apply, so two sentences in a row don't both open with a moved phrase. In Chaos a date or a source may move on top
    of a passive turned active (see follow_up_move)."""
    boost = CHAOS_STRUCTURE_BOOST if deep_mode else 0.0
    if random.random() < SIGNATURE_RATE[tone] + boost:
        for transform in SIGNATURE_TRANSFORMS:
            new = transform(sent, pieces)
            if new and not opens_like(new, previous):
                return new
    if random.random() < PASSIVE_RATE[tone] + boost:
        new = passive_to_active(sent, pieces)
        if new and not opens_like(new, previous):
            return follow_up_move(sent, pieces, new, after_move, previous) if deep_mode else new
    if random.random() < MOVE_RATE[tone] + (CHAOS_REORDER_BOOST if deep_mode else 0.0):
        moves = list(MOVES_TO_END if after_move else PHRASE_MOVES)
        random.shuffle(moves)
        for move in moves:
            new = move(sent, pieces)
            if new and not opens_like(new, previous):
                return new
    return None

def follow_up_move(sent, pieces, new, after_move, previous):
    """Chaos: a date or a source moved too, after a passive turned active ("The cornerstone was laid in 1902 by the
    mayor." -> "The mayor laid the cornerstone in 1902." -> "In 1902, the mayor laid the cornerstone."). A phrase only
    comes to the front if nothing was put there ("With X, ..." would add a comma) and the sentence before wasn't
    reordered."""
    added_comma = sum(1 for p in new if p[1] == ',') > sum(1 for p in pieces if p[1] == ',')
    moves = [m for m in FOLLOW_UP_MOVES if m in MOVES_TO_END or not (after_move or added_comma)]
    random.shuffle(moves)
    for move in moves:
        again = move(sent, new)
        if again and not opens_like(again, previous):
            return again
    return new

# --- Where a sentence could become two ---

def main_tense(span):
    """'past' or 'present' for a sentence's main verb, read from its first auxiliary when it has one ("has
    posed" is present, "was marked" and "had replaced" are past), or None (a modal, or no finite verb)."""
    root = span.root
    head = min((c for c in root.children if c.dep_ in ('aux', 'auxpass')), key=lambda c: c.i, default=root)
    return {'VBD': 'past', 'VBZ': 'present', 'VBP': 'present'}.get(head.tag_)

# Each part of a split keeps at least this many of its own words ("The room was small. But it was clean." is the
# shortest), and a command before its next step at least three ("Add the onions. Cook them until soft.")
SPLIT_MIN_LEFT_WORDS, SPLIT_MIN_RIGHT_WORDS, SPLIT_MIN_COMMAND_WORDS = 4, 3, 3
# What may come before the subject of a sentence that is split, and stay with its first part: a link to what came
# before ("However, ...") or a time ("In 1998, ...", "Last year, ..."). Anything else - a condition, a hedge, a
# frame - may cover the whole sentence ("If you cancel early, you get a refund, but the fee is kept")
FRONTED_CONNECTIVES = {'however', 'moreover', 'furthermore', 'additionally', 'also', 'still', 'yet', 'meanwhile',
                       'finally', 'first', 'firstly', 'second', 'secondly', 'third', 'next', 'then', 'today', 'now',
                       'later', 'afterwards', 'eventually', 'instead', 'indeed', 'similarly', 'likewise',
                       'nevertheless', 'nonetheless', 'besides'}
# A personal pronoun subject is simply repeated by a split ("She won the race and broke the record." -> "... She
# broke the record.")
KEPT_PRONOUNS = {'i': 'I', 'we': 'We', 'you': 'You', 'he': 'He', 'she': 'She', 'it': 'It', 'they': 'They'}
# ...but not the "it" of the weather, which stands for nothing ("It rained all night. It flooded the streets.")
WEATHER_VERBS = {'rain', 'snow', 'pour', 'hail', 'drizzle', 'sleet', 'thunder', 'freeze', 'storm', 'shower', 'blow'}
# Bodies that names_people counts as people but a split names again as "it" ("The council approved the plan. It
# also ..."); a team, a crowd or a family stay "they" or "he/she", so their sentences are not split
INSTITUTION_NOUNS = {'company', 'firm', 'board', 'council', 'government', 'agency', 'bank', 'university', 'school',
                     'hospital', 'court', 'parliament', 'congress', 'senate', 'committee', 'department', 'business',
                     'startup', 'organization', 'organisation', 'charity', 'institute', 'lab', 'laboratory', 'store',
                     'shop', 'restaurant', 'studio', 'brand', 'city', 'town', 'village', 'country', 'nation', 'state',
                     'retailer'}
# Subjects no pronoun can stand for in a sentence of their own ("Every student wrote ... . They ..." reads as all of
# them together, "No company paid ... . It ..." says the opposite, and "Half the village flooded ... . It ..." means
# the whole village)
QUANTIFIED_SUBJECTS = {'no', 'every', 'each', 'any', 'all', 'some', 'many', 'most', 'few', 'several', 'both', 'either',
                       'neither', 'another', 'much', 'little', 'none', 'one', 'everyone', 'everybody', 'everything',
                       'nobody', 'nothing', 'someone', 'somebody', 'something', 'anyone', 'anybody', 'anything',
                       'others', 'this', 'that', 'these', 'those', 'such', 'half'}
# Words after "and"/"but" that tie the second part to the first, so it can't open a sentence with a pronoun ("and
# so collapsed", "and yet survived", "and even ...")
REFUSED_AFTER_CONJUNCTION = {'so', 'yet', 'thus', 'hence', 'therefore', 'even', 'not', 'never', 'only', 'just'}

def stretch_head(doc, start, end):
    """The one word of doc[start:end] that the others hang from (its own head lies outside the stretch, or it is
    the root), punctuation aside; None when the stretch is not one phrase."""
    heads = [t for t in doc[start:end] if not t.is_punct and (t.head.i == t.i or not start <= t.head.i < end)]
    return heads[0] if len(heads) == 1 else None

def clause_subject(head):
    """The subject of a verb, or None; spaCy sometimes hangs it on the auxiliary ("The fort was closed ...")."""
    near = [head] + [c for c in head.children if c.dep_ in ('aux', 'auxpass')]
    return next((c for w in near for c in w.children if c.dep_ in ('nsubj', 'nsubjpass', 'expl', 'csubj')), None)

def standalone_clause(doc, start, end):
    """The verb of doc[start:end] if the stretch could be a sentence of its own: one phrase, headed by a finite verb
    with its own subject ("the battery lasts up to 14 days", "you can leave the charger at home"); else None."""
    head = stretch_head(doc, start, end)
    if head is None or head.pos_ not in ('VERB', 'AUX'):
        return None
    inside = [c for c in head.children if start <= c.i < end]
    finite = head.tag_ in ('VBD', 'VBZ', 'VBP') or any(c.dep_ in ('aux', 'auxpass') and c.tag_ in ('MD', 'VBD', 'VBZ', 'VBP')
                                                      for c in inside)
    subject = clause_subject(head)
    return head if finite and subject is not None and start <= subject.i < end else None

def command_clause(doc, start, end):
    """The verb of doc[start:end] if the stretch is a command: one phrase headed by a bare verb with no subject or
    auxiliary, and only adverbs before it ("save the file", "First, clone the repository"); else None."""
    head = stretch_head(doc, start, end)
    if (head is None or head.tag_ != 'VB' or head.pos_ != 'VERB'
            or any(c.dep_ in ('nsubj', 'nsubjpass', 'expl', 'aux', 'auxpass') for c in head.children)):
        return None
    return head if all(t.pos_ == 'ADV' or t.is_punct for t in doc[start:head.i]) else None

def words_in(doc, start, end):
    return sum(1 for t in doc[start:end] if t.text[:1].isalnum())

def framed_opening(sent, start):
    """True if something before token `start` (the subject of the first part, or a command's verb) may cover the
    whole sentence: a clause ("If you cancel early, ...", "While prices rose, ..."), a hedge or a frame ("In most
    cases, ...", "According to the report, ..."). A link to what came before ("However, ...") or a time ("In 1998,
    ...", "Last year, ...", "Eight years later, ...") stays with the first part without changing what the second
    says."""
    doc = sent.doc
    for t in doc[sent.start:start]:
        if t.is_punct or (sent.start <= t.head.i < start and t.head.i != t.i):
            continue        # a word inside a fronted phrase: the phrase's head decides
        if t.lower_ in FRONTED_CONNECTIVES and t.dep_ in ('advmod', 'cc') and not list(t.children):
            continue
        if t.dep_ in ('prep', 'npadvmod', 'advmod') and (heads_time_phrase(t) if t.dep_ == 'prep' else any(
                w.ent_type_ in ('DATE', 'TIME') for w in t.subtree)):
            continue
        return True
    return False

def claims_in(doc, head, start, end):
    """True if the clause headed by `head` in doc[start:end] reports a claim ("Experts say the battery lasts ...",
    "According to the survey, ...", "The company, analysts say, is losing money"): what follows it may be part of the
    claim, or the writer's own words."""
    return (head.lemma_.lower() in CLAIM_FRAMES or any(c.dep_ == 'ccomp' and start <= c.i < end for c in head.children)
            or any(t.dep_ == 'parataxis' and t.lemma_.lower() in CLAIM_FRAMES for t in doc[start:end])
            or bool(CITED_SOURCE.search(doc[start:end].text)) or bool(CITED_OPENING.search(doc[start:end].text)))

def verb_number(verb):
    """'singular' or 'plural' when a verb or its first auxiliary shows the number of its subject ("crushes",
    "were", "has been"); None for a modal or a plain past ("will open", "opened")."""
    first = min([verb] + [c for c in verb.children if c.dep_ in ('aux', 'auxpass')], key=lambda w: w.i)
    if first.tag_ == 'VBZ' or first.lower_ in ('was', 'is', 'has', 'does'):
        return 'singular'
    if first.tag_ == 'VBP' or first.lower_ in ('were', 'are', 'have', 'do'):
        return 'plural'
    return None

def subject_pronoun(subject, verb):
    """The pronoun that can open a new sentence in place of a subject, agreeing with `verb`: a personal pronoun is
    repeated; two or more subjects and a plural become "They"; one thing, place or institution "It", a country with
    a plural name too ("The Philippines ... It became a republic."). None for a person or a group of people (he, she
    or they?), a name that may be a person's, a quantified subject ("every student", "most people", "half the
    village"), and a subject the verb doesn't agree with (a misparse)."""
    number = verb_number(verb)
    if subject.tag_ == 'PRP':
        return KEPT_PRONOUNS.get(subject.lower_)
    if subject.lower_ in QUANTIFIED_SUBJECTS or any(c.dep_ in ('det', 'amod', 'quantmod', 'nummod', 'advmod', 'predet')
                                                    and c.lower_ in QUANTIFIED_SUBJECTS for c in subject.children):
        return None
    if subject.conjuncts:
        return 'They' if number != 'singular' and not any(c.tag_ == 'PRP' for c in subject.conjuncts) else None
    if subject.tag_ == 'NNPS' and subject.ent_type_ == 'GPE':
        return 'It' if number != 'plural' else None
    if subject.tag_ in ('NNS', 'NNPS'):
        return 'They' if number != 'singular' else None
    if subject.tag_ not in ('NN', 'NNP') or number == 'plural':
        return None
    if subject.lemma_.lower() in INSTITUTION_NOUNS or subject.ent_type_ in ('ORG', 'GPE', 'FAC', 'LOC', 'PRODUCT', 'EVENT',
                                                                        'WORK_OF_ART', 'LAW'):
        return 'It'
    if names_people(subject) or subject.tag_ == 'NNP' or any(t.ent_type_ in ('PERSON', 'NORP') for t in subject.subtree):
        return None
    return 'It'

# A sum, a share or a measure is one thing, however plural its noun ("40 million euros")
AMOUNT_ENTITIES = {'MONEY', 'PERCENT', 'QUANTITY'}
AMOUNT_UNITS = {'dollar', 'euro', 'pound', 'yen', 'yuan', 'rupee', 'franc', 'cent', 'percent', 'ton', 'tonne', 'kilogram',
                'gram', 'litre', 'liter', 'gallon', 'barrel', 'mile', 'kilometre', 'kilometer', 'metre', 'meter', 'acre',
                'hectare'}

def names_amount(noun):
    return noun.ent_type_ in AMOUNT_ENTITIES or (noun.lemma_.lower() in AMOUNT_UNITS
                                                 and any(c.like_num or c.dep_ == 'nummod' for c in noun.children))

def names_time(token):
    return token.ent_type_ in ('DATE', 'TIME') or token.lemma_.lower() in TIME_NOUNS

# Prepositions of a setting: their noun ("in Britain", "in human history", "across the region") is rarely taken for
# what the next sentence's subject pronoun stands for
SETTING_PREPOSITIONS = {'in', 'at', 'on', 'across', 'around', 'throughout', 'near', 'within', 'along', 'during',
                        'inside', 'outside'}

def pronoun_clash(doc, start, end, pronoun):
    """True if the last noun of doc[start:end] could be taken for what a new sentence's opening "It" or "They"
    stands for ("The tool syncs your files with the cloud. It keeps a history ..." may mean the cloud, and "The
    council approved the budget on Monday. It will ..." the budget). A time or a setting doesn't compete ("The
    branch opened in 1998. It now has 40 staff.", "... began in Britain. It marked ...")."""
    def competes(noun):
        head = noun
        while head.dep_ in ('compound', 'nmod', 'poss') and head.head.pos_ in ('NOUN', 'PROPN'):
            head = head.head        # "town" in "the town centre" goes with "centre"
        return not (names_time(noun) or names_time(head) or (head.dep_ == 'pobj' and head.head.lower_ in SETTING_PREPOSITIONS))
    noun = next((t for t in reversed(doc[start:end]) if t.pos_ in ('NOUN', 'PROPN') and competes(t)), None)
    return pronoun in ('It', 'They') and noun is not None and (noun.tag_ in ('NNS', 'NNPS')) == (pronoun == 'They')

def split_points(sent, pieces):
    """", and"/", but" before a clause with its own subject, "; " before a clause, ", which ..." at the end, and
    the shapes of token_point, while_point, aside_point, shared_subject_point and command_pair_point. A point is
    only offered where the words from it to the end are still the sentence's own, in their own stretch: a template
    may have dropped or moved words, and the split would then cut in the wrong place. Nothing after "not only",
    "not just", "not merely" or "not simply" is cut ("The app is not only fast, but it is also cheap" is one
    statement), and a sentence in capitals is left whole (its new openers would be in lower case)."""
    doc, stop = sent.doc, sent[-1].i
    position = {p[0]: k for k, p in enumerate(pieces) if p[0] is not None}
    points = []
    letters = [t for t in sent if t.is_alpha]
    if len(letters) > 1 and all(t.is_upper for t in letters):
        return points
    correlative = next((t.i for t in sent if t.lower_ in ('not', "n't", 'n’t') and t.i < stop
                        and doc[t.i + 1].lower_ in ('only', 'just', 'merely', 'simply')), stop)

    def at(start, dropped=None):
        """The piece position of token `start` if the pieces from there on are tokens start..stop (in any
        order: a reordered list is fine) and, when given, the piece right after it is token `dropped` (the
        word split_sentence leaves out); else None."""
        k = position.get(start)
        if k is None:
            return None
        tail = [p[0] for p in pieces[k:]]
        if None in tail or sorted(tail) != [t.i for t in doc[start:stop + 1] if not t.is_space]:
            return None
        if dropped is not None and (k + 1 >= len(pieces) or pieces[k + 1][0] != dropped):
            return None
        return k

    def offer(found):
        """Adds a point a shape rule found, (token cut at, word left out or None, kind), if both parts keep
        enough words of their own and the pieces still match."""
        if found is not None:
            cut, dropped, kind = found
            k = at(cut, dropped)
            least = SPLIT_MIN_COMMAND_WORDS if kind in ('command', 'then') else SPLIT_MIN_LEFT_WORDS
            if (k is not None and words_in(doc, sent.start, cut) >= least
                    and words_in(doc, (cut if dropped is None else dropped) + 1, stop) >= SPLIT_MIN_RIGHT_WORDS):
                points.append((k, kind))

    past = main_tense(sent) == 'past'
    for token in sent:
        i = token.i
        if i - sent.start < 6 or stop - i < 6:
            if i > sent.start:
                offer(token_point(token, sent))
            continue
        before = doc[i - 1]
        point = None
        if token.dep_ == 'cc' and token.lower_ in ('and', 'but') and before.text == ',' and token.head.dep_ == 'ROOT':
            clause = next((c for c in token.head.children if c.dep_ == 'conj' and c.i > i), None)
            first = clause_subject(token.head)
            if clause is not None and clause.pos_ in ('VERB', 'AUX') and any(
                    c.dep_ in ('nsubj', 'nsubjpass', 'expl') for c in clause.children) and not framed_opening(
                    sent, first.left_edge.i if first is not None and first.i < i else token.head.i):
                # not after an opening condition, hedge or frame: "If you cancel early, you get a refund, and
                # the fee is kept" says the fee is kept if you cancel early
                point = (at(i - 1, i), token.lower_)
        elif token.text == ';':
            if any(t.dep_ in ('nsubj', 'nsubjpass', 'expl') and t.head.i > i for t in doc[i + 1: stop]):
                point = (at(i), ';')
        elif (token.lower_ == 'which' and before.text == ',' and token.dep_ in ('nsubj', 'nsubjpass')
              and token.head.dep_ == 'relcl' and token.head.right_edge.i == stop - 1):
            # "They" after a plural verb; "This" only where the clause can be said of the whole event (not "She
            # drives an old Volvo. This is very reliable.", nor "small purchases, which can add up": see
            # comments_on_event)
            verb = token.head
            plural = verb.tag_ == 'VBP' or any(c.lower_ in ('are', 'were', 'have') for c in verb.children)
            if plural or comments_on_event(doc[i - 2], verb):
                point = (at(i - 1, i), 'which-plural' if plural else 'which')
        elif (token.lower_ == 'because' and token.dep_ == 'mark' and before.text == ','
              and token.head.dep_ == 'advcl' and token.head.head.dep_ == 'ROOT' and token.head.right_edge.i == stop - 1
              and any(c.dep_ in ('nsubj', 'nsubjpass', 'expl') for c in token.head.children)):
            # "..., because the country gained freedom" -> ". That's because the country gained freedom". Not
            # "as": it may be causal, temporal or a comment ("as the data show"), and dropping it loses the link
            point = (at(i - 1, i), 'because')
        elif (token.lower_ in ('although', 'though') and token.dep_ == 'mark' and before.text == ','
              and token.head.dep_ == 'advcl' and token.head.head == sent.root and token.head.right_edge.i == stop - 1
              and any(c.dep_ in ('nsubj', 'nsubjpass', 'expl') for c in token.head.children)
              and not reports_claim(sent) and not opens_with_contrast(sent)):
            # "..., although it also has limits." -> ". However, it also has limits." (the contrast stays). Not
            # inside a reported claim ("The findings indicate that X, although Y": Y is part of what they
            # indicate), nor after an opening contrast ("However, X. However, Y.")
            point = (at(i - 1, i), 'although')
        elif (token.lower_ == 'who' and token.dep_ == 'nsubj' and token.head.dep_ == 'relcl'
              and token.head.right_edge.i == stop - 1 and stop - i >= 5
              and (token.head.head.tag_ in ('NNS', 'NNPS') or token.head.head.lower_ in ('others', 'people'))
              and (before.text == ',' or token.head.head.lower_ == 'others')):
            # "... and many others who fought bravely" -> "... and many others. They fought bravely". Without a
            # comma the clause says which ones ("free childcare to parents who work night shifts"); as a
            # sentence of its own it would say all of them do
            point = (at(i - 1, i) if before.text == ',' else at(i), 'who')
        elif (token.tag_ == 'VBG' and before.text == ',' and token.lower_ in RESULT_PARTICIPLES and token.head == sent.root
              and token.dep_ in ('advcl', 'xcomp', 'conj', 'dep') and token.right_edge.i == stop - 1
              and i in position and pieces[position[i]][1] == token.text
              and not any(c.dep_ == 'aux' and c.tag_ == 'MD' for c in sent.root.children)
              and states_result(token, sent)):
            # "..., making it easy to get around." -> ". This makes it easy to get around." (not after a modal:
            # "could reduce costs, making X" doesn't say that it makes X)
            forms = getInflection(token.lemma_, tag='VBD' if past else 'VBZ')
            if forms:
                point = (at(i - 1, i), 'result:' + forms[0])
        elif (token.lower_ == 'including' and before.text == ',' and token.dep_ == 'prep' and token.head.pos_ == 'NOUN'
              and token.right_edge.i == stop - 1 and i in position and pieces[position[i]][1] == token.text):
            # "..., including X, Y and Z." -> ". These include X, Y and Z."
            plural = token.head.tag_ in ('NNS', 'NNPS')
            verb = 'included' if past else ('include' if plural else 'includes')
            point = (at(i - 1, i), 'including:' + ('These ' if plural else 'This ') + verb)
        if point is not None and point[0] is not None:
            points.append(point)
        else:
            offer(token_point(token, sent))
    if [p[0] for p in pieces] == [t.i for t in sent if not t.is_space]:
        # these two rework the first part too, so every word must still be in its own place
        offer(while_point(sent))
        offer(aside_point(sent))
    offer(shared_subject_point(sent))
    offer(command_pair_point(sent))
    return [p for p in points if pieces[p[0]][0] < correlative]

# Second commands that say what the first one gets you ("Sign up today and get 20% off" means "if you sign up",
# "Spend over fifty pounds and qualify for free delivery"), so the two can't be separate steps
CONSEQUENCE_VERBS = {'get', 'receive', 'save', 'enjoy', 'win', 'earn', 'see', 'find', 'start', 'discover', 'unlock', 'gain',
                     'watch', 'feel', 'notice', 'join', 'grab', 'claim', 'take', 'be', 'have', 'keep', 'make', 'let',
                     'qualify', 'enter', 'become', 'access', 'stay', 'avoid', 'skip', 'learn', 'benefit', 'collect',
                     'redeem', 'score', 'experience', 'relax'}

def command_pair_point(sent):
    """"Add the onions and cook them until soft." -> "Add the onions. Cook them until soft.": two commands joined by
    "and" become two steps. Only where the first command has its own object or phrase ("Wash and dry the
    leaves" share theirs), nothing after the second belongs to both, nothing before the first covers both ("Before
    you paint, move the furniture and cover the floor"), and the second doesn't say what the first gets you ("Sign
    up today and get 20% off", "Go and see")."""
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    if command_clause(doc, sent.start, stop) is None:
        return None
    conjuncts = [c for c in root.children if c.dep_ == 'conj']
    links = [c for c in root.children if c.dep_ == 'cc']
    if len(conjuncts) != 1 or len(links) != 1:
        return None
    second, link = conjuncts[0], links[0]
    if (link.lower_ != 'and' or not root.i < link.i < second.i or second.tag_ != 'VB' or second.right_edge.i != stop - 1
            or second.lemma_.lower() in CONSEQUENCE_VERBS or root.lemma_.lower() in ('go', 'come', 'try', 'wait')
            or doc[link.i + 1].lower_ in REFUSED_AFTER_CONJUNCTION
            or any(c.dep_ in ('nsubj', 'nsubjpass', 'expl', 'aux', 'auxpass') for c in second.children)
            or any(t.dep_ in ('nsubj', 'nsubjpass', 'expl', 'csubj') for t in doc[link.i + 1:second.i])):
        return None
    own = [c for c in root.children if c.dep_ not in ('conj', 'cc', 'punct')]
    if (any(c.right_edge.i >= link.i for c in own) or not any(c.i > root.i for c in own)
            or any(c.dep_ == 'punct' and link.i < c.i < stop for c in root.children) or framed_opening(sent, root.i)):
        return None
    return (link.i - 1, link.i, 'command') if doc[link.i - 1].text == ',' else (link.i, None, 'command')

def token_point(token, sent):
    """The split at one word for the shorter shapes below, as (token cut at, word left out or None, kind), or None."""
    word = token.lower_
    if word == 'so':
        return so_point(token, sent)
    if word == 'then':
        return then_point(token, sent)
    if word == 'which':
        return comment_which_point(token, sent)
    if word in OF_WHICH_QUANTIFIERS:
        return of_which_point(token, sent)
    if word == 'but' or token.text == ';':
        return clause_pair_point(token, sent)
    if word in CAUSE_PARTICIPLES:
        return cause_point(token, sent)
    return None

# Verbs of the writer's own advice or request: after "so", they answer the fact rather than follow from it ("Spaces
# are limited, so we recommend booking early" keeps "So", not "This means we recommend ...")
ADVICE_VERBS = {'recommend', 'suggest', 'advise', 'ask', 'urge', 'encourage', 'invite', 'request', 'propose', 'insist',
                'remind'}

def so_point(token, sent):
    """", so you can leave the charger at home." -> ". So you can ..." or ". This means you can ...": "so" after a
    comma, between two clauses that can each be a sentence (the second may be a command: ", so book early"). Not
    "so that", "so far", "so much", "so few" ("So few people saw it" reads as "very few"), not an inversion (", so do
    I"), not before a past modal ("so he could catch the train" may give a purpose, not a result), and not when an
    opening phrase or a reported claim may cover both. The result is said outright ("This means ..."; "As a result,
    ..." for a past one) only where both parts share a tense and the second opens with its own subject; "So" stays
    before anything else ("So of course the bus was late", "So this phone is very light"), before the writer's own
    advice ("So we recommend booking early"), before what people then did or now do ("So they kept walking", "So
    the council is adding bike lanes") and after a duty, where "so" gives the purpose ("Requests should be made
    two weeks ahead, so managers can plan cover")."""
    doc, i, stop = sent.doc, token.i, sent[-1].i
    if (i - 1 <= sent.start or doc[i - 1].text != ',' or i + 1 >= stop or doc[i + 1].is_punct
            or doc[i + 1].lower_ in ('that', 'as', 'far', 'much', 'many', 'long', 'to', 'too', 'called', 'few', 'little')):
        return None
    left = standalone_clause(doc, sent.start, i - 1)
    if (left is None or framed_opening(sent, clause_subject(left).left_edge.i)
            or claims_in(doc, left, sent.start, i - 1)):
        return None
    right = standalone_clause(doc, i + 1, stop)
    if right is None:
        return (i - 1, i, 'so:command') if command_clause(doc, i + 1, stop) is not None else None
    if doc[i + 1].pos_ in ('VERB', 'AUX') or any(c.dep_ == 'aux' and c.lower_ in ('could', 'would', 'might', 'should')
                                                 for c in right.children):
        return None
    tenses = {'past' if min([v] + [c for c in v.children if c.dep_ in ('aux', 'auxpass')], key=lambda w: w.i).tag_ == 'VBD'
              else 'present' for v in (left, right)}
    subject = clause_subject(right)
    if (len(tenses) > 1 or subject.left_edge.i != i + 1 or doc[i + 1].lower_ in ('this', 'that', 'these', 'those')
            or (subject.lower_ in ('i', 'we') and right.lemma_.lower() in ADVICE_VERBS)
            or ('past' in tenses and subject.lower_ in ('i', 'we', 'you', 'he', 'she', 'they'))
            or ('present' in tenses and (names_people(subject) or subject.lower_ in ('i', 'we', 'he', 'she', 'they'))
                and right.lemma_.lower() not in ('be', 'have', 'need', 'feel', 'seem', 'remain', 'stay')
                and not any(c.dep_ == 'aux' and c.tag_ == 'MD' for c in right.children))
            or any(c.dep_ == 'aux' and c.lower_ in ('should', 'must', 'ought', 'shall') for c in left.children)
            or (left.lemma_.lower() in ('need', 'have') and any(c.dep_ == 'xcomp' for c in left.children))):
        return i - 1, i, 'so:plain'
    return i - 1, i, 'so:' + tenses.pop()

def then_point(token, sent):
    """", then season it well." -> ". Then season it well.": steps in order, both commands ("Replace the key,
    then save the file") or both clauses with a subject of their own; not when an opening phrase may cover both
    steps ("If the sauce is too thick, add water, then stir")."""
    doc, i, stop = sent.doc, token.i, sent[-1].i
    if i - 1 <= sent.start or doc[i - 1].text != ',' or i + 1 >= stop:
        return None
    left = command_clause(doc, sent.start, i - 1)
    if left is not None:
        if command_clause(doc, i + 1, stop) is None or framed_opening(sent, left.i):
            return None
    else:
        left = standalone_clause(doc, sent.start, i - 1)
        if (left is None or standalone_clause(doc, i + 1, stop) is None
                or framed_opening(sent, clause_subject(left).left_edge.i) or claims_in(doc, left, sent.start, i - 1)):
            return None
    return i - 1, None, 'then'

# "..., most of which ..." / "..., many of whom ..." -> ". Most of it ..." / ". Many of them ..."
OF_WHICH_QUANTIFIERS = {'most', 'many', 'some', 'all', 'none', 'much', 'half', 'each', 'one', 'both', 'several', 'few',
                        'two', 'three', 'four'}
# ...of which these count things one by one, so their verb is singular whatever they count ("each of which has")
DISTRIBUTIVE_QUANTIFIERS = {'each', 'one', 'none'}
# ...and these only count separate things, never part of an amount ("many of them", not "many of it")
COUNTING_QUANTIFIERS = {'many', 'several', 'few', 'both', 'each', 'one', 'two', 'three', 'four'}

def of_which_point(token, sent):
    """", most of which will come from a federal grant." -> ". Most of it will come from a federal grant.": a
    closing "<quantity> of which/whom" clause becomes a sentence of its own, "which" becoming "it" for one thing or
    an amount and "them" for several, the verb deciding where it shows its number ("200 songs, most of which is
    lost" is refused as a clash). Only where the noun counted comes right before the comma, is no list ("coal and
    gas, most of which" may count the gas alone) and is not inside a phrase ("$50 million for schools, most of
    which" counts the money), and not inside a reported claim."""
    doc, i, stop = sent.doc, token.i, sent[-1].i
    if not (i - 1 > sent.start and doc[i - 1].text == ',' and i + 3 < stop and doc[i + 1].lower_ == 'of'
            and doc[i + 2].lower_ in ('which', 'whom') and doc[i + 1].head.i == i and doc[i + 2].head.i == i + 1
            and token.dep_ in ('nsubj', 'nsubjpass')):
        return None
    verb, noun = token.head, token.head.head
    if (verb.dep_ != 'relcl' or verb.right_edge.i != stop - 1 or noun.i != i - 2 or noun.conjuncts
            or noun.dep_ in ('conj', 'pobj') or reports_claim(sent)):
        return None
    number = verb_number(verb)
    if doc[i + 2].lower_ == 'which' and (names_amount(noun) or (noun.tag_ == 'NN' and not noun.ent_type_
                                                                and not names_people(noun))):
        pronoun = 'it'      # an amount ("$12 million", "40 million euros") or one thing
        clash = number == 'plural' or token.lower_ in COUNTING_QUANTIFIERS
    elif noun.tag_ in ('NNS', 'NNPS'):
        pronoun = 'them'
        clash = (number == 'singular' and token.lower_ not in DISTRIBUTIVE_QUANTIFIERS) or token.lower_ == 'much'
    else:
        return None
    return None if clash else (i - 1, None, 'of:' + pronoun)

# Verbs whose subject, in a closing ", which" clause, is the whole event rather than the noun before the comma
# ("hired a new chef, which means ...", "connects over Bluetooth, which lets you start it from bed"); "make" and
# "help" only with what they make or help happen ("which makes it popular", not "which makes great coffee")
EVENT_VERBS = {'mean', 'cause', 'lead', 'explain', 'suggest', 'allow', 'let', 'enable', 'make', 'help'}
# Verbs that remove, block or fix a thing: a ", which" clause after it is about the thing, not the event ("Brushing
# removes plaque, which can cause gum disease" -> "This can cause gum disease" would blame the brushing)
REMOVAL_VERBS = {'remove', 'fix', 'patch', 'block', 'filter', 'prevent', 'close', 'eliminate', 'kill', 'stop', 'ban',
                 'replace', 'repair', 'treat', 'cure', 'fight', 'combat', 'avoid', 'resolve', 'tackle', 'neutralize',
                 'neutralise', 'destroy', 'end', 'abolish', 'scrap', 'delete', 'disable', 'reduce', 'cut', 'lower',
                 'limit', 'minimize', 'minimise', 'trap', 'absorb', 'protect', 'shield', 'counter', 'address', 'clear',
                 'drop', 'retire', 'withdraw', 'strip', 'flush', 'repel', 'deter', 'suppress', 'curb', 'restrict',
                 'outlaw', 'prohibit', 'shut', 'plug', 'seal', 'detect', 'catch', 'relieve', 'ease'}

def event_verb(verb):
    """True if a verb is one of EVENT_VERBS, with what it makes or helps happen after "make" and "help"."""
    lemma = verb.lemma_.lower()
    return lemma in EVENT_VERBS and (lemma not in ('make', 'help') or any(
        c.dep_ in ('ccomp', 'xcomp', 'oprd', 'acomp') for c in verb.children))

def removal_object(noun):
    """True if a noun is what a verb removes, blocks or fixes, or what is kept away ("removes plaque", "helps remove
    plaque", "fixes a rare memory leak", "protection against measles")."""
    head = noun
    while head.dep_ in ('dobj', 'pobj', 'prep', 'conj', 'appos', 'compound') and head.head.i != head.i:
        if (head.dep_ == 'dobj' and head.head.lemma_.lower() in REMOVAL_VERBS) or (
                head.dep_ == 'pobj' and head.head.lower_ in ('against', 'without', 'from')):
            return True
        head = head.head
    return False

def comments_on_event(noun, verb):
    """True if a closing ", which" clause after `noun` (the word before the comma) can become a sentence of its own
    that opens with "This", the whole event. After a plural noun, only with a singular verb ("all city buses and
    the subway, which makes getting around easy") or an event's verb ("new speed cameras, which could lead to fewer
    accidents"); after a singular noun, only with an event's verb ("hired a new chef, which means ...", not "drives
    an old Volvo, which is very reliable"); never after what is removed or blocked ("removes plaque, which can cause
    gum disease": the plaque does). After a time, an amount, an adverb or an adjective ("crashes occasionally, which
    can be frustrating"), yes, unless the clause names the time ("in 2015, which was the year ...") or the verb
    before has an object the clause may describe ("bought a car last week, which broke down": then only an event's
    verb)."""
    nominal = noun.pos_ in ('NOUN', 'PROPN', 'PRON', 'NUM')
    if nominal and removal_object(noun):
        return False
    if nominal and not (names_time(noun) or names_amount(noun)):
        if noun.tag_ in ('NNS', 'NNPS'):
            number = verb_number(verb)
            return number == 'singular' or (number is None and event_verb(verb))
        if noun.dep_ not in ('npadvmod', 'advmod'):
            return event_verb(verb)
    if names_time(noun) and verb.lemma_ == 'be' and any(c.dep_ == 'attr' for c in verb.children):
        return False
    governor = noun
    while governor.pos_ not in ('VERB', 'AUX') and governor.head.i != governor.i:
        governor = governor.head
    return event_verb(verb) or not any(c.dep_ in ('dobj', 'attr', 'dative') and c.i < noun.i
                                       and not (names_time(c) or names_amount(c)) for c in governor.children)

def comment_which_point(token, sent):
    """", which can be frustrating." -> ". This can be frustrating.": a closing "which" clause that comments on the
    event, however short (the plain ", which" rule wants six words on each side), or that spaCy hangs on the verb.
    "This" must fit what "which" stands for (see comments_on_event), and a plural verb is left to the plain rule."""
    doc, i, stop = sent.doc, token.i, sent[-1].i
    if not (i - 1 > sent.start and doc[i - 1].text == ',' and token.dep_ in ('nsubj', 'nsubjpass')):
        return None
    verb = token.head
    if (verb.dep_ not in ('relcl', 'advcl') or verb.right_edge.i != stop - 1 or verb_number(verb) == 'plural'
            or not comments_on_event(doc[i - 2], verb)):
        return None
    return i - 1, i, 'which'

def clause_pair_point(token, sent):
    """", but she declined it." -> ". But she declined it." and "X; Y." -> "X. Y." where the plain rules' six words
    on each side are missing: both parts must be clauses that can stand alone, and no opening phrase or reported
    claim may cover both ("If you cancel early, you get a refund, but ..." keeps its condition on both)."""
    doc, i, stop = sent.doc, token.i, sent[-1].i
    if token.text == ';':
        cut, dropped = i, None
    elif token.dep_ == 'cc' and i - 1 > sent.start and doc[i - 1].text == ',':
        cut, dropped = i - 1, i
    else:
        return None
    left = standalone_clause(doc, sent.start, cut)
    if (left is None or standalone_clause(doc, i + 1, stop) is None
            or framed_opening(sent, clause_subject(left).left_edge.i) or claims_in(doc, left, sent.start, cut)):
        return None
    return cut, dropped, ';' if dropped is None else 'but'

# "..., driven mainly by strong demand" -> ". This was driven mainly by strong demand": participles that name the
# cause of a change, said of the change itself (not "led by": "Stocks fell, led by tech shares" names what fell most)
CAUSE_PARTICIPLES = {'driven', 'fueled', 'fuelled', 'boosted', 'spurred', 'propelled', 'powered', 'buoyed', 'offset',
                     'supported'}
# ...when the main verb is one of change
CHANGE_VERBS = {'rise', 'fall', 'grow', 'increase', 'decrease', 'decline', 'jump', 'climb', 'surge', 'soar', 'drop', 'slide',
                'slip', 'rebound', 'recover', 'improve', 'expand', 'shrink', 'contract', 'double', 'triple', 'slow',
                'accelerate', 'gain', 'advance', 'retreat', 'plunge', 'tumble', 'rally', 'widen', 'narrow', 'edge', 'dip'}
# ...and its subject a measure, whose change has causes ("Revenue rose, driven by ..."); a thing that moves has a
# driver ("The car accelerated away, driven by a man in a mask")
CHANGE_MEASURES = {'revenue', 'sale', 'profit', 'earning', 'income', 'price', 'cost', 'export', 'import', 'index', 'share',
                   'stock', 'rate', 'growth', 'demand', 'supply', 'output', 'production', 'productivity', 'spending',
                   'investment', 'inflation', 'wage', 'salary', 'value', 'market', 'economy', 'gdp', 'traffic',
                   'attendance', 'enrolment', 'enrollment', 'membership', 'population', 'number', 'figure', 'total',
                   'level', 'volume', 'usage', 'consumption', 'emission', 'employment', 'unemployment', 'debt',
                   'deficit', 'yield', 'return', 'margin', 'turnover', 'budget', 'fee', 'fare', 'rent', 'tax', 'loss',
                   'rating', 'borrowing', 'lending', 'saving', 'currency', 'dollar', 'euro', 'pound', 'yen', 'bond',
                   'equity', 'commodity', 'oil', 'gold', 'ridership', 'booking', 'order', 'subscription', 'download'}

def cause_point(token, sent):
    """"Revenue rose 12 percent, driven mainly by cloud sales." -> "Revenue rose 12 percent. This was driven mainly
    by cloud sales.": a closing "driven by"/"fueled by" phrase gives the cause of a rise or fall, so "This" (the
    change) can carry it. Only after a verb of change whose subject is a measure (CHANGE_MEASURES, or an amount),
    not a person, a group or a moving thing ("She kept going, driven by ambition" is about her, "The car
    accelerated away, driven by a man" about the car), with the agent ("by ...") in the phrase, and no modal or
    negation ("could rise, driven by" is no change that happened)."""
    doc, i, stop, root = sent.doc, token.i, sent[-1].i, sent.root
    if not (token.tag_ == 'VBN' and i - 1 > sent.start and doc[i - 1].text == ',' and token.head.i == root.i
            and token.dep_ in ('advcl', 'conj', 'dep', 'xcomp') and token.right_edge.i == stop - 1
            and root.lemma_.lower() in CHANGE_VERBS and any(c.dep_ == 'agent' for c in token.children)):
        return None
    aux = [c for c in root.children if c.dep_ in ('aux', 'auxpass')]
    subject = clause_subject(root)
    if (subject is None or any(a.tag_ == 'MD' for a in aux) or any(c.dep_ == 'neg' for c in root.children)
            or subject.tag_ == 'PRP' or names_people(subject) or reports_claim(sent)
            or not (names_amount(subject) or any(w.lemma_.lower() in CHANGE_MEASURES or w.lower_ in CHANGE_MEASURES
                                                 for w in [subject] + [c for c in subject.children if c.dep_ == 'compound']))):
        return None
    if any(a.lower_ in ('has', 'have') for a in aux):
        be = 'has been'
    elif root.tag_ == 'VBD' or any(a.tag_ == 'VBD' for a in aux):
        be = 'was'
    elif root.tag_ in ('VBZ', 'VBP') or any(a.tag_ in ('VBZ', 'VBP') for a in aux):
        be = 'is'
    else:
        return None
    return i - 1, None, 'cause:' + be

# Verbs of a "while" clause that state a fact the main clause sets against ("While it allows us to stay in touch, it
# can also ..."), unlike the activities of a "while" of time ("While you wait, ...")
STATIVE_WHILE_VERBS = {'be', 'have', 'offer', 'provide', 'allow', 'enable', 'bring', 'seem', 'appear', 'remain', 'contain',
                       'include', 'show', 'promise', 'mean', 'cost', 'exist', 'create', 'give', 'improve', 'help', 'save',
                       'support', 'deliver', 'add', 'reduce', 'increase', 'require', 'need', 'lack'}
THIRD_PERSON_PRONOUNS = {'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'he', 'him', 'his',
                         'himself', 'she', 'her', 'hers', 'herself'}

def while_point(sent):
    """"While it allows us to stay in touch, it can also lead to isolation." -> "It allows us to stay in touch. But
    it can also lead to isolation." Only a "while" of contrast: the "while" clause hedges ("may", "might", "can",
    "could"), or states a fact (STATIVE_WHILE_VERBS) that the main clause answers with "also", "still" or a "not",
    or it is "some ..., others ...". A state with "be" or "remain" is as often a time ("While she was in Paris,
    she also visited the Louvre", "While the roads are icy, the buses will not run", "While the kids are young,
    they still need ..."), so only a hedge makes it a contrast; one with "have" needs "still" ("While he had the
    flu, he still went to work") or a "not" on the same verb ("While the hotel has a pool, it does not have a
    gym"). Not a "while" in progress ("While I was cooking"), nor one whose pronoun stands for something the main
    clause names first ("While its price is high, the phone ...": as a sentence of its own, "Its price" would
    point back to the sentence before)."""
    doc, mark, root, stop = sent.doc, sent[0], sent.root, sent[-1].i
    clause = mark.head
    if mark.lower_ not in ('while', 'whilst') or mark.dep_ != 'mark' or clause.dep_ != 'advcl' or clause.head.i != root.i:
        return None
    comma = clause.right_edge.i + 1
    if comma >= stop or doc[comma].text != ',':
        return None
    first, main = standalone_clause(doc, sent.start + 1, comma), standalone_clause(doc, comma + 1, stop)
    if first is None or main is None or first.i != clause.i or main.i != root.i:
        return None
    if clause.tag_ == 'VBG' or any(c.dep_ == 'aux' and c.tag_ == 'VBG' for c in clause.children):
        return None
    hedged = any(c.dep_ == 'aux' and c.lower_ in ('may', 'might', 'can', 'could') for c in clause.children)
    lemma = clause.lemma_.lower()
    if lemma in ('be', 'remain'):
        answered = False
    elif lemma == 'have':
        answered = any(c.dep_ == 'dobj' for c in clause.children) and (
            any(c.dep_ == 'advmod' and c.lower_ == 'still' for c in root.children)
            or (root.lemma_.lower() == 'have' and any(c.dep_ == 'neg' for c in root.children)))
    else:
        answered = lemma in STATIVE_WHILE_VERBS and any(
            (c.dep_ == 'advmod' and c.lower_ in ('also', 'still')) or c.dep_ == 'neg' for c in root.children)
    subject, theirs = clause_subject(clause), clause_subject(root)
    others = (theirs.lower_ == 'others' and any(t.lower_ in ('some', 'many') and t.dep_ in ('det', 'amod', 'nsubj')
                                               for t in subject.subtree))
    if not (hedged or answered or others):
        return None
    if theirs.tag_ != 'PRP' and any(t.lower_ in THIRD_PERSON_PRONOUNS for t in doc[sent.start:comma]):
        return None
    return comma, None, 'while'

def aside_point(sent):
    """"Electric buses, which produce no exhaust fumes, now run on every route." -> "Electric buses produce no
    exhaust fumes. They now run on every route.": a clause between commas that describes the subject becomes the
    first sentence, and the main clause the second, opening with a pronoun for the subject. Only where the subject
    opens the sentence (a fronted phrase belongs to the main clause), the noun right before the comma is the
    subject's own head ("The rise in fuel prices, which ..." may describe the prices), the clause's "which" or
    "who" is its subject, nothing but adverbs and auxiliaries comes between the second comma and the main verb,
    and the pronoun is certain (no single person: he or she?)."""
    doc, root = sent.doc, sent.root
    subject = clause_subject(root)
    if subject is None or subject.dep_ not in ('nsubj', 'nsubjpass') or subject.left_edge.i != sent.start or subject.i > root.i:
        return None
    clause = next((c for c in subject.children if c.dep_ == 'relcl'), None)
    if clause is None:
        return None
    wh = clause.left_edge
    first, second = wh.i - 1, clause.right_edge.i + 1
    if not (wh.lower_ in ('which', 'who') and wh.dep_ in ('nsubj', 'nsubjpass') and wh.head.i == clause.i
            and first == subject.i + 1 and doc[first].text == ',' and second < root.i and doc[second].text == ','
            and not any(t.text in (',', ';', ':', '(', ')', '—', '–') for t in doc[wh.i:second])
            and all(t.head.i == root.i and t.dep_ in ('aux', 'auxpass', 'advmod', 'neg') for t in doc[second + 1:root.i])):
        return None
    if wh.lower_ == 'who' and subject.tag_ not in ('NNS', 'NNPS'):
        return None
    pronoun = subject_pronoun(subject, root)
    if pronoun is None or words_in(doc, wh.i + 1, second) < SPLIT_MIN_RIGHT_WORDS:
        return None
    # The pronoun opens the second sentence; where the first ends with a noun it could be taken for, the main
    # clause goes first instead ("Missing data were handled using imputation. They comprised under 3% of responses.")
    if not pronoun_clash(doc, wh.i + 1, second, pronoun):
        return second, None, f'aside:{pronoun}:{first}'
    if not pronoun_clash(doc, second + 1, sent[-1].i, pronoun):
        return second, None, f'aside-after:{pronoun}:{first}'
    return None

def shared_subject_point(sent):
    """"The Tokyo branch opened in 1998 and now has 40 staff." -> "The Tokyo branch opened in 1998. It now has 40
    staff.": a second verb that shares the subject starts a sentence of its own, with a pronoun for the subject
    ("But it ..." after "but"). Only where the pronoun is certain (see subject_pronoun), the subject opens the
    sentence (a fronted phrase would cover both parts), nothing between the subject and the first verb could cover
    both ("later", "never", "also"), the first verb has its own complement ("makes and sells chargers" shares its
    object), nothing after the second verb belongs to both, and the second verb is finite on its own: "will open a
    store and hire staff" shares its "will", and "was built and repaired" its "was". The "it" of the weather is no
    subject to repeat ("It rained all night and flooded the streets")."""
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    subject = clause_subject(root)
    conjuncts = [c for c in root.children if c.dep_ == 'conj']
    links = [c for c in root.children if c.dep_ == 'cc']
    if (subject is None or subject.dep_ not in ('nsubj', 'nsubjpass') or subject.left_edge.i != sent.start
            or len(conjuncts) != 1 or len(links) != 1
            or (subject.lower_ == 'it' and root.lemma_.lower() in WEATHER_VERBS)):
        return None
    second, link = conjuncts[0], links[0]
    if (link.lower_ not in ('and', 'but') or not subject.i < root.i < link.i < second.i or second.pos_ not in ('VERB', 'AUX')
            or second.right_edge.i != stop - 1 or doc[link.i + 1].lower_ in REFUSED_AFTER_CONJUNCTION
            or clause_subject(second) is not None
            or any(t.lower_ in QUANTIFIED_SUBJECTS or t.dep_ in ('nsubj', 'nsubjpass', 'expl', 'csubj')
                   for t in doc[link.i + 1:second.i])):
        # a subject of its own, maybe hung on its auxiliary ("The streets are clean and the parks are well kept"), or
        # one spaCy took for an adverb ("Drivers were sceptical, but most now accept it")
        return None
    own = [c for c in root.children if c.dep_ not in ('conj', 'cc', 'punct')]
    own_aux = any(c.dep_ in ('aux', 'auxpass') and c.tag_ in ('MD', 'VBZ', 'VBP', 'VBD') and c.i > link.i
                  for c in second.children)
    if (any(c.right_edge.i >= link.i for c in own) or any(c.dep_ == 'punct' and link.i < c.i < stop for c in root.children)
            or not (own_aux or any(c.i > root.i for c in own))
            or any(c.dep_ in ('advmod', 'neg', 'npadvmod') and subject.i < c.i < root.i for c in root.children)):
        return None
    shared = ''
    if not own_aux:
        shared = shared_auxiliaries(root, second)
        # no auxiliary of its own and none to share: it must be finite on its own ("monitor volcanoes and warn
        # communities": spaCy may tag the second present verb as a bare one)
        bare_present = second.tag_ == 'VB' and root.tag_ == 'VBP'
        if shared is None or (not shared and ((second.tag_ not in ('VBZ', 'VBP', 'VBD') and not bare_present)
                                              or any(c.dep_ in ('aux', 'auxpass') for c in second.children))):
            return None
    pronoun = subject_pronoun(subject, root if shared else second)
    if pronoun is None or (subject.tag_ != 'PRP' and pronoun_clash(doc, subject.i + 1, link.i, pronoun)):
        return None
    kind = f'vp:{link.lower_}:{pronoun}' + (':' + shared if shared else '')
    return (link.i - 1, link.i, kind) if doc[link.i - 1].text == ',' else (link.i, None, kind)

def shared_auxiliaries(root, second):
    """The auxiliaries a second verb shares with the first, to be said again before it ("was built in 1890 and
    restored in 2004" -> "It was restored in 2004", "will open a store and hire staff" -> "It will hire staff"): ''
    when the first has none, None when sharing can't be told or can't be repeated. A second verb in the same form
    as the first shares them, but a past that is no participle is finite on its own ("was founded in 1901 and grew
    steadily" -> "It grew steadily", not "It was grew"). After a passive, a form that may be a past or a participle
    shares it only with an agent ("and restored by volunteers in 2004"): many such verbs are active or can't be
    passive at all ("was built in 1902 and stood for a century" -> "It was stood", "was published in March and
    found that ..." -> "It was found that ..."); a form that is only a participle ("torn down", "given a charter")
    shares it. "Has" is not said again before a time that dates the second verb ("has grown every year and
    attracted 40,000 visitors last summer"). Not "do" ("does track and store" would become "It does store"), a
    negation ("was not built and opened" may deny both) or a contraction ("'ll")."""
    chain = sorted((c for c in root.children if c.dep_ in ('aux', 'auxpass')), key=lambda c: c.i)
    if not chain:
        return '' if second.tag_ != 'VBN' else None
    if (any(a.lemma_.lower() == 'do' or not a.text[:1].isalpha() for a in chain) or any(c.dep_ == 'neg' for c in root.children)
            or any(a.i > root.i for a in chain)):
        return None
    passive = any(a.dep_ == 'auxpass' for a in chain)
    if root.tag_ == 'VB' and second.tag_ == 'VB':
        return ' '.join(a.lower_ for a in chain)
    if root.tag_ == 'VBN' and second.tag_ in ('VBN', 'VBD'):
        word = second.lower_
        if word not in {form.lower() for form in getInflection(second.lemma_, tag='VBN') or ()}:
            return ''
        past = word in {form.lower() for form in getInflection(second.lemma_, tag='VBD') or ()}
        if ((passive and past and not any(c.dep_ == 'agent' for c in second.children))
                or (any(a.lower_ in ('has', 'have') for a in chain)
                    and any(t.ent_type_ in ('DATE', 'TIME') for t in second.subtree))):
            return None
        return ' '.join(a.lower_ for a in chain)
    if root.tag_ == 'VBG' and second.tag_ == 'VBG':
        return ' '.join(a.lower_ for a in chain)
    return '' if second.tag_ in ('VBZ', 'VBP') else None     # finite on its own ("is made of glass and holds two litres")

# --- Chaos only: "... proposed a new scheme that would offer loans." -> "... proposed a new scheme. The new scheme
# would offer loans." ---
# Verbs whose object may not exist or not be one particular thing ("needed a car that could tow a caravan", "chose a
# dress that matched her shoes"): the clause says which kind, and a sentence of its own would make it one real thing
OPEN_OBJECT_VERBS = {'need', 'want', 'require', 'seek', 'look', 'search', 'choose', 'find', 'pick', 'select', 'prefer',
                     'expect', 'hope', 'wish', 'demand', 'ask', 'plan', 'intend', 'consider', 'imagine', 'avoid',
                     'prevent', 'lack', 'recommend', 'suggest', 'try', 'aim', 'deserve', 'await', 'wait', 'miss',
                     'forbid', 'ban', 'reject', 'refuse', 'oppose', 'fear', 'doubt', 'deny', 'resist', 'apply', 'vote',
                     'mean', 'like', 'love', 'hate', 'favour', 'favor', 'dream', 'crave', 'desire'}
# Verbs that may present something not (yet) real: the clause must keep its modal ("proposed a scheme that would offer
# loans" -> "The scheme would offer loans"; not "described a world that has no money" -> "The world has no money")
PICTURING_VERBS = {'envisage', 'envision', 'predict', 'describe', 'depict', 'propose', 'promise', 'picture', 'foresee',
                   'forecast', 'outline', 'sketch', 'call', 'advocate', 'urge', 'portray', 'model', 'simulate',
                   'invent', 'fantasize', 'fantasise', 'pitch', 'draft', 'design', 'unveil', 'announce'}
# Present verbs whose object is one real thing when their subject is one ("The hotel has a rooftop pool that stays
# open until midnight", "The update introduces a dark mode that ..."), unlike verbs of habit or rule ("The policy
# covers a repair that costs under 500 pounds" means any such repair) or of a range on sale ("The airline offers a
# fare that lets you change dates" leaves "The airline offers a fare." saying nothing)
INTRODUCING_VERBS = {'have', 'include', 'feature', 'contain', 'introduce', 'launch', 'provide', 'run', 'operate', 'own',
                     'boast', 'house', 'host', 'add', 'use'}
# Nouns that "the" would make the one known thing rather than the one just mentioned ("a world that could exist" ->
# "The world could exist")
ONE_OF_A_KIND_NOUNS = {'world', 'earth', 'sun', 'moon', 'universe', 'internet', 'web', 'sky', 'future', 'past', 'present',
                       'public', 'economy', 'environment', 'weather', 'climate', 'government', 'state', 'press', 'media',
                       'market', 'law', 'church', 'crown', 'army', 'police', 'planet', 'nation', 'country', 'city',
                       'town', 'village', 'sea', 'ocean', 'nature', 'society', 'industry', 'sector'}
# Nouns that only stand in for what the clause says ("found a way that works", "one that lasts", "a kind that grows")
STAND_IN_NOUNS = {'one', 'thing', 'way', 'kind', 'type', 'sort', 'lot', 'bit', 'number', 'amount', 'range', 'variety',
                  'series', 'set', 'couple', 'pair', 'part', 'piece', 'place', 'point', 'case', 'situation', 'level',
                  'degree', 'form', 'manner', 'means', 'side', 'aspect', 'element', 'factor', 'feature', 'example'}
# Modifiers that make "the" name a kind ("The average family owns a car that ...")
GENERIC_SUBJECT_MODIFIERS = {'average', 'typical', 'ideal', 'perfect', 'best', 'usual', 'standard', 'normal', 'modern',
                             'good', 'ordinary', 'traditional', 'classic', 'right'}
# Words of habit or of a long period before the clause: the noun is then a kind ("In the 1800s, farmers used a plough
# that was pulled by horses", "Usually the shop offers a discount that ...")
HABIT_WORDS = {'usually', 'often', 'generally', 'normally', 'typically', 'sometimes', 'occasionally', 'rarely', 'seldom',
               'frequently', 'traditionally', 'historically', 'mostly', 'always', 'ever', 'every', 'each', 'whenever',
               'regularly', 'daily', 'weekly', 'monthly', 'yearly', 'annually', 'century', 'centuries', 'decade',
               'decades', 'era', 'age', 'ages', 'generation', 'generations', 'years', 'tradition'}
# The prepositions after a main verb whose noun is a real thing the sentence is about ("replaced it with a system
# that ...", "bought it from a man who ...", "moved to a town that ...", "worked on a project that ..."); not "for"
# ("saved for a house that ...", "applied for a job that ...")
RELATIVE_CLAUSE_PREPOSITIONS = {'with', 'from', 'to', 'on'}

def one_particular_subject(subject):
    """True if a subject names one particular thing, person or group: "the hotel", "our app", "Leeds", "we" (not
    "Good teachers", "Farmers", "A typical school", "The average family", "They")."""
    if subject.tag_ == 'PRP':
        return subject.lower_ in ('i', 'we', 'he', 'she')
    if subject.tag_ in ('NNP', 'NNPS') or subject.ent_type_ in ('ORG', 'GPE', 'PERSON', 'FAC', 'PRODUCT'):
        return True
    return (any((c.dep_ in ('det', 'poss') and c.lower_ in ('the', 'this', 'that', 'these', 'those', 'our', 'my', 'his',
                                                              'her', 'its', 'their', 'your')) or c.tag_ in ('NNP', 'POS')
                for c in subject.children)
            and not any(c.lower_ in GENERIC_SUBJECT_MODIFIERS for c in subject.children))

def relative_clause_point(sent, pieces):
    """Chaos only. "The council has proposed a new scheme that would offer low-interest loans." -> "The council has
    proposed a new scheme. The new scheme would offer low-interest loans.": a closing "that", "which" or "who" clause
    that is the subject of its verb, right after a singular noun brought in with "a"/"an", becomes a sentence about
    "the" noun, named again (a pronoun could be taken for the subject). Only where the noun is one real thing: after one
    event in the past or present perfect, or a present INTRODUCING_VERBS verb, whose subject is one particular thing or
    group ("Good writers use a voice that suits the audience" names a kind of voice, "Each room has a desk that ..."
    would make one desk of them all), with no word of habit or long period before the clause (HABIT_WORDS), no negation,
    condition, modal or reported claim, no verb whose object may not exist (OPEN_OBJECT_VERBS) or picturing verb without
    a modal in the clause (PICTURING_VERBS), and the noun the object of the main verb or of with/from/to/on after it.
    Not a noun "the" would make a known thing ("a world that could exist") or one that stands in for another ("found a
    way that works"), nor one the sentence already names ("A man met a man who ..."), and no list or stop inside the
    clause ("a desk that faces the window and a lamp": the lamp may belong to the main clause), and no "it", "its",
    "they" or "her" in it, which the new subject could take over ("installed a heat pump that halved its energy bills"
    -> "The heat pump halved its energy bills" reads as the pump's bills). Returns (piece position of the relative word,
    'relcl:<first token of the noun's name>:<noun token>') or None."""
    doc, root, stop = sent.doc, sent.root, sent[-1].i
    # (a reported claim: "The council says ...", "According to ...", "As the report notes, ..."; a verb that can report
    # one but has the noun as its object reports none: "The council has proposed a new scheme that ...")
    if (root.pos_ != 'VERB' or root.lemma_.lower() in OPEN_OBJECT_VERBS
            or any(c.dep_ in ('ccomp', 'parataxis') or (c.dep_ == 'advcl' and c.lemma_.lower() in CLAIM_FRAMES)
                   for c in root.children)
            or CITED_SOURCE.search(sent.text) or CITED_OPENING.search(sent.text)
            or any(t.lower_ in ('not', "n't", 'n’t', 'never', 'no', 'nor', 'without', 'if', 'unless', 'whether')
                   for t in sent)):
        return None
    aux = [c.lower_ for c in root.children if c.dep_ in ('aux', 'auxpass')]
    subject = clause_subject(root)
    if (subject is None or subject.dep_ != 'nsubj' or subject.i > root.i or not one_particular_subject(subject)
            or any(t.lower_ in QUANTIFIED_SUBJECTS for t in subject.subtree) or framed_opening(sent, subject.left_edge.i)
            or not ((root.tag_ == 'VBD' and not aux) or (root.tag_ == 'VBN' and aux in (['has'], ['have'], ['had']))
                    or (root.tag_ in ('VBZ', 'VBP') and not aux and root.lemma_.lower() in INTRODUCING_VERBS))):
        return None
    clause = next((t for t in sent if t.dep_ == 'relcl' and t.right_edge.i == stop - 1), None)
    if clause is None:
        return None
    noun, wh = clause.head, clause.left_edge
    if not (wh.lower_ in ('that', 'which', 'who') and wh.dep_ in ('nsubj', 'nsubjpass') and wh.head.i == clause.i
            and wh.i == noun.i + 1 and noun.pos_ == 'NOUN' and noun.tag_ == 'NN' and not noun.ent_type_
            and clause.i - wh.i <= 3 and not any(c.dep_ in ('nsubj', 'nsubjpass', 'expl', 'csubj') and c.i != wh.i
                                                 for c in clause.children)
            and not any(t.lower_ in HABIT_WORDS or (t.ent_type_ == 'DATE' and t.like_num and t.text.endswith('s'))
                        for t in doc[sent.start:wh.i])):
        return None
    if not ((noun.dep_ == 'dobj' and noun.head.i == root.i) or (
            noun.dep_ == 'pobj' and noun.head.dep_ == 'prep' and noun.head.head.i == root.i
            and noun.head.lower_ in RELATIVE_CLAUSE_PREPOSITIONS)):
        return None
    lemma = noun.lemma_.lower()
    dets = [c for c in noun.children if c.dep_ in ('det', 'poss', 'predet', 'nummod', 'quantmod')]
    if (len(dets) != 1 or dets[0].lower_ not in ('a', 'an') or dets[0].i != noun.left_edge.i
            or lemma in ONE_OF_A_KIND_NOUNS or lemma in STAND_IN_NOUNS or names_time(noun)
            or any(t.lemma_.lower() == lemma for t in doc[sent.start:noun.i])
            or any(t.lower_ in QUANTIFIED_SUBJECTS | {'only', 'same', 'other', 'certain', 'single', 'particular',
                                                     'various', 'numerous', 'further', 'similar', 'different'}
                   for t in doc[dets[0].i + 1:noun.i])
            or (root.lemma_.lower() in PICTURING_VERBS and not any(t.tag_ == 'MD' for t in doc[wh.i:stop]))):
        return None
    if (any(t.text in (',', ';', ':', '—', '–') or (t.dep_ == 'conj' and t.pos_ in ('NOUN', 'PROPN'))
            or t.lower_ in THIRD_PERSON_PRONOUNS for t in doc[wh.i:stop])
            or words_in(doc, wh.i + 1, stop) < 4 or words_in(doc, sent.start, wh.i) < 5):
        return None     # (a pronoun in the clause could then be taken for the noun: "The heat pump halved its bills")
    # The words before the noun must all be its modifiers, with no comma or "and". The noun is named again with its
    # compound and the compound's own words ("a social media campaign" -> "The social media campaign", "a low-interest
    # loan scheme" -> "The low-interest loan scheme", "a fresh tomato salad" -> "The tomato salad"), or with its one
    # adjective ("a dark mode" -> "The dark mode", not "The mode"); after two adjectives, alone ("a rich cultural
    # tapestry" -> "The tapestry", not a heavy echo)
    words = doc[dets[0].i + 1:noun.i]
    if any(t.dep_ in ('cc', 'conj', 'prep', 'det', 'poss') or (t.is_punct and t.text not in ('-', '–'))
           or not dets[0].i < t.head.i <= noun.i for t in words):
        return None
    own = [t for t in words if t.head.i == noun.i]
    compounds = [t for t in own if t.dep_ == 'compound']
    first = (min(t.left_edge.i for t in compounds) if compounds else own[0].left_edge.i if len(own) == 1 else noun.i)
    if any(t.head.i == noun.i and t.dep_ != 'compound' for t in doc[first:noun.i]) and compounds:
        return None     # an adjective between the compound's words
    position = {p[0]: k for k, p in enumerate(pieces) if p[0] is not None}
    k = position.get(wh.i)
    if (k is None or [p[0] for p in pieces[k:]] != [t.i for t in doc[wh.i:stop + 1] if not t.is_space]
            or any(i not in position for i in range(first, noun.i + 1))):
        return None     # the clause and the name must still be the sentence's own
    return k, f'relcl:{first}:{noun.i}'


# Trailing "-ing" clauses that only state a result ("..., making it easy to ...") - a hallmark of AI text,
# which uses them several times as often as people do
RESULT_PARTICIPLES = {'making', 'leading', 'resulting', 'allowing', 'enabling', 'ensuring', 'causing', 'highlighting',
                      'underscoring', 'demonstrating', 'reflecting', 'contributing', 'paving', 'giving', 'creating',
                      'prompting', 'fueling', 'fuelling', 'improving', 'boosting', 'reducing', 'driving',
                      'fostering', 'showcasing', 'emphasizing', 'emphasising'}
# ...of which these always name an outcome of the event ("This led to ...")
OUTCOME_PARTICIPLES = {'leading', 'resulting', 'causing', 'paving'}
# People and groups: after "The artist spent ten years in Paris, creating dozens of paintings" the artist did the
# creating, and "This created dozens of paintings" would give it to the stay
AGENT_NOUNS = {'artist', 'chef', 'volunteer', 'teacher', 'doctor', 'nurse', 'scientist', 'writer', 'worker', 'farmer',
                'engineer', 'developer', 'designer', 'manager', 'leader', 'founder', 'owner', 'player', 'athlete',
                'musician', 'singer', 'actor', 'director', 'officer', 'official', 'employer', 'investor',
                'entrepreneur', 'president', 'minister', 'mayor', 'governor', 'king', 'queen', 'emperor', 'soldier',
                'army', 'police', 'parent', 'child', 'mother', 'father', 'family', 'friend', 'man', 'woman', 'boy',
                'girl', 'resident', 'client', 'patient', 'crew', 'group', 'organization', 'organisation', 'firm',
                'startup', 'agency', 'council', 'court', 'parliament', 'congress', 'senate', 'university', 'school',
                'hospital', 'charity', 'band', 'club', 'scholar', 'professor', 'historian', 'economist', 'analyst',
                'expert', 'journalist', 'reporter', 'editor', 'photographer', 'painter', 'poet', 'novelist',
                'composer', 'architect', 'builder', 'pilot', 'driver', 'coach', 'captain', 'explorer', 'inventor',
                'pioneer', 'immigrant', 'settler', 'tourist', 'traveler', 'traveller', 'visitor', 'guest', 'host',
                'fan', 'voter', 'crowd', 'nation', 'country', 'state', 'city', 'town', 'village', 'department', 'bank',
                'brand', 'business', 'store', 'shop', 'restaurant', 'studio', 'lab', 'laboratory', 'institute'}
# ...and people named by their tie to each other or their stage of life ("The couple ran a small bakery for twenty
# years, giving free bread to the poor")
PEOPLE_NOUNS = {'couple', 'pair', 'duo', 'trio', 'partner', 'kid', 'learner', 'pupil', 'newcomer', 'beginner',
                'trainee', 'intern', 'teenager', 'adult', 'youth', 'sibling', 'brother', 'sister', 'husband', 'wife',
                'mum', 'mam', 'granny', 'nana'}
# ...and every noun whose usual sense is a person ("grandparents", "neighbours", "newlyweds", "monks"; built
# offline with NLTK from WordNet by scripts/build_person_nouns.py)
with gzip.open(os.path.join(DATA_DIR, 'person_nouns.json.gz'), 'rt', encoding='utf-8') as _f:
    PERSON_WORDS = frozenset(json.load(_f))
PERSONAL_PRONOUNS = {'i', 'we', 'you', 'he', 'she', 'who'}     # not "they"/"it", which are as often things
# ...but "they" who spent, worked or lived somewhere, or ran or managed something, are people ("They spent ten
# years in Paris, creating ...", "They managed the farm for thirty years, creating ...")
PEOPLE_ACTIVITIES = {'spend', 'work', 'live', 'stay', 'travel', 'serve', 'run', 'teach', 'study', 'volunteer',
                     'train', 'play', 'perform', 'tour', 'move', 'settle', 'campaign', 'fight', 'write', 'paint',
                     'build', 'farm', 'manage', 'own', 'lead', 'host', 'organise', 'organize', 'found'}

def names_people(noun):
    """True if a noun names a person, people or a group of them ("my grandparents", "the council", "Oxfam"). A
    plural name is a family or a band ("The Smiths ran the corner shop ..."), unless it is a place ("the Alps")."""
    lemma = noun.lemma_.lower()
    return (lemma in GROUP_NOUNS or lemma in AGENT_NOUNS or lemma in PEOPLE_NOUNS
            or (noun.pos_ == 'NOUN' and lemma in PERSON_WORDS) or noun.ent_type_ in ('PERSON', 'ORG', 'NORP')
            or (noun.tag_ == 'NNPS' and noun.ent_type_ not in ('GPE', 'LOC', 'FAC')))

def states_result(participle, sent):
    """True if "This ..." would name the result of the whole event: "leading to"/"resulting in" always do;
    "making" only with a complement ("making it easy to ..."); the others only when the sentence's subject
    isn't a person or a group, who would have done that action themselves."""
    word = participle.lower_
    if word in OUTCOME_PARTICIPLES:
        return True
    if word == 'making':
        return any(c.dep_ in ('ccomp', 'oprd', 'acomp', 'attr', 'xcomp') for c in participle.children)
    subject = next((c for c in sent.root.children if c.dep_ in ('nsubj', 'nsubjpass')), None)
    if subject is None:
        return False
    # Every one of the subjects counts: "My mum and dad ran a small shop for forty years, giving credit to ..."
    return not (subject.lower_ in PERSONAL_PRONOUNS or any(names_people(s) for s in (subject, *subject.conjuncts))
                or (subject.lower_ == 'they' and sent.root.lemma_.lower() in PEOPLE_ACTIVITIES)
                or any(t.ent_type_ in ('PERSON', 'ORG', 'NORP') for t in subject.subtree))

# A main verb that reports a claim: what follows it, a concession included, is part of the claim
CLAIM_FRAMES = CLAIM_VERBS | {'say', 'report', 'find', 'note', 'state', 'show', 'claim', 'argue', 'conclude', 'reveal',
                              'confirm', 'warn', 'explain', 'believe', 'think', 'estimate', 'predict', 'expect',
                              'announce', 'insist', 'agree', 'admit', 'demonstrate', 'appear', 'seem', 'suggest',
                              'assert', 'maintain', 'acknowledge', 'contend', 'observe', 'hypothesize', 'propose'}
# ...and so does a source named in a phrase ("According to the survey, X, although Y": Y is what the survey found too)
CITED_SOURCE = re.compile(r"\baccording\s+to\b|\bas\s+(?:reported|stated|noted|shown|found|described)\s+by\b", re.I)
# ...or before the main verb ("Based on the survey, ...", "In the words of the inspectors, ...", "In their view, ...")
CITED_OPENING = re.compile(r"\bbased\s+on\b|\bper\s+(?:the|its|their|his|her)\b|\bin\s+(?:(?:the|his|her|their|its)\s+)?"
                           r"(?:[\w-]+['’]s?\s+)?(?:words|view|opinion|eyes|judge?ment|estimation)\b", re.I)
CONTRAST_OPENINGS = ('however', 'but ', 'but,', 'still,', 'yet ', 'yet,', 'nevertheless', 'nonetheless', 'that said',
                     'even so', 'then again', 'by contrast', 'in contrast', 'conversely', 'on the other hand', 'though',
                     'although', 'even though')

def reports_claim(sent):
    root = sent.root
    return (root.lemma_.lower() in CLAIM_FRAMES or any(c.dep_ == 'ccomp' for c in root.children)
            or bool(CITED_SOURCE.search(sent.text))
            or bool(CITED_OPENING.search(sent.doc.text, sent.start_char, root.idx))
            # "As the report notes, X, although Y", "As noted in the report, ..." (but not "As expected, ...")
            or any(c.dep_ == 'advcl' and c.lemma_.lower() in CLAIM_FRAMES
                   and any(m.dep_ == 'mark' and m.lower_ == 'as' for m in c.children)
                   and any(s.dep_ in ('nsubj', 'nsubjpass', 'agent')
                           or (s.dep_ == 'prep' and s.lower_ in ('in', 'by') and not heads_time_phrase(s))
                           for s in c.children) for c in root.children))

def opens_with_contrast(sent):
    return sent.text.lower().startswith(CONTRAST_OPENINGS)

def splittable(sent):
    """Plain prose sentences, up to a long length: a split keeps every word in place, so long or dashed
    sentences that are too risky to restructure may still be split."""
    texts = {t.text for t in sent}
    return (sent[-1].text == '.' and 6 <= len(sent) <= 80 and sent.root.pos_ in ('VERB', 'AUX')
            and not texts & {'"', '“', '”', '‘', '(', ')', '[', ']', ':'} and not any(t.like_url or t.like_email for t in sent)
            and not has_quoted(sent.doc, range(sent.start, sent.end)))

# How the second part of a ", so ..." split opens, by tone ("This means" becomes "As a result," after a past verb:
# "This meant engineers reinforced its piers" reads oddly before what someone then did)
SO_OPENERS = {'standard': ('So', 'This means'), 'casual': ('So', 'That means'), 'professional': ('This means',),
              'academic': ('This means',)}

def split_sentence(sentence, point, tone):
    position, kind = point
    left = [list(p) for p in sentence.pieces[:position]] + [[None, '.']]
    rest = [list(p) for p in sentence.pieces[position + 1:]]
    if kind == 'and' or (kind == 'but' and tone == 'academic') or (kind == 'although' and tone in ('academic', 'professional')):
        # ", and, as a result, X" -> ". As a result, X": the comma after the dropped word goes too
        after = first_piece(rest[1:])
        if after is not None and after[1] == ',':
            after[1] = ''
    if kind == 'and':
        rest = rest[1:]
    elif kind == 'but':
        rest = [[None, 'However,' if tone == 'academic' else 'But']] + rest[1:]
    elif kind == 'which':
        # "This", unless the sentence already opens with it ("This process ... This is ..."); not "That" before
        # a comma, where "That is, ..." would read as the phrase meaning "in other words"
        words = [p for p in rest[1:] if p[1]]
        before_comma = len(words) > 1 and words[1][1] == ','
        rest = [[None, 'That' if sentence.opener() == 'this' and not before_comma else 'This']] + rest[1:]
    elif kind == 'which-plural':
        rest = [[None, 'They']] + rest[1:]
    elif kind == 'although':
        rest = [[None, 'However,' if tone in ('academic', 'professional') else 'But']] + rest[1:]
    elif kind == 'because':
        opener = "That's because" if tone in ('standard', 'casual') else 'This is because'
        rest = [[None, opener]] + rest[1:]
    elif kind == 'who':
        if rest and rest[0][1].lower() == 'who':
            rest = rest[1:]
        rest = [[None, 'They']] + rest
    elif kind.startswith('result:'):
        rest = [[None, 'This'], [None, kind.split(':', 1)[1]]] + rest[1:]
    elif kind.startswith('including:'):
        rest = [[None, kind.split(':', 1)[1]]] + rest[1:]
    elif kind.startswith('so:'):
        # the "so" stays ("So you can ..."), or the result is said outright ("This means you can ...")
        opener = 'So' if kind in ('so:command', 'so:plain') else random.choice(SO_OPENERS[tone])
        if opener != 'So':
            rest = [[None, 'As a result,' if kind == 'so:past' else opener]] + rest[1:]
    elif kind.startswith('of:'):
        rest = rest[:2] + [[rest[2][0], kind.split(':', 1)[1]]] + rest[3:]     # "most of which" -> "most of it"
    elif kind == 'while':
        left = [list(p) for p in sentence.pieces[1:position]] + [[None, '.']]
        capitalize_first(left)
        rest = [[None, 'However,' if tone in ('academic', 'professional') else 'But']] + rest
    elif kind.startswith('aside'):
        # "Electric buses, which produce no fumes, now run ..." -> "Electric buses produce no fumes. They now run ..."
        # or, after a noun the pronoun could be taken for, "Electric buses now run ... . They produce no fumes."
        name, pronoun, comma = kind.split(':')
        first = next(k for k, p in enumerate(sentence.pieces) if p[0] == int(comma))
        subject = [list(p) for p in sentence.pieces[:first]]
        aside = [list(p) for p in sentence.pieces[first + 2:position]]
        if name == 'aside':
            left, rest = subject + aside + [[None, '.']], [[None, pronoun]] + rest
        else:
            left, rest = subject + rest, [[None, pronoun]] + aside + [[None, '.']]
    elif kind == 'command' and sentence.pieces[position][1] == ',':
        rest = rest[1:]     # the "and" after the comma
    elif kind.startswith('cause:'):
        rest = [[None, 'This'], [None, kind.split(':', 1)[1]]] + rest
    elif kind.startswith('relcl:'):
        # "... proposed a new scheme that would offer loans." -> "... a new scheme. The new scheme would offer loans."
        first, noun = (int(n) for n in kind.split(':')[1:])
        words = {p[0]: p[1] for p in sentence.pieces if p[0] is not None}
        name = ''.join(words[i] + sentence.doc[i].whitespace_ for i in range(first, noun + 1)).strip()
        rest = [[None, 'The'], [None, name]] + rest     # one piece, so "low-interest" keeps its spacing
    elif kind.startswith('vp:'):
        _, link, pronoun, *shared = kind.split(':')
        if sentence.pieces[position][1] == ',':
            rest = rest[1:]     # the "and"/"but" after the comma
        opener = [] if link == 'and' else [[None, 'However,' if tone in ('academic', 'professional') else 'But']]
        rest = (opener + [[None, pronoun if not opener or pronoun == 'I' else pronoun.lower()]]
                + [[None, word] for word in shared] + rest)
    for part in (left, rest):
        # one full stop after an abbreviation that ends a part ("arrived at 9 a.m.. So ..." -> "at 9 a.m. So ...")
        live = [p for p in part if p[1]]
        if len(live) > 1 and live[-1][1] == '.' and live[-2][1].endswith('.'):
            live[-1][1] = ''
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
CHAOS_RHYTHM_TARGET, CHAOS_RHYTHM_OPS = 0.7, 1.0
MERGE_MAX_WORDS = 14

# Openers that become the joining word when two sentences merge ("X. However, Y." -> "X, but Y.")
JOINING_OPENERS = {'also': 'and', 'plus': 'and', 'however': 'but', 'but': 'but', 'still': 'but', 'yet': 'but', 'so': 'so'}

def spread(lengths):
    mean = sum(lengths) / len(lengths)
    return (sum((x - mean) ** 2 for x in lengths) / len(lengths)) ** 0.5 / mean if mean else 0.0

def content_lemmas(sentence):
    """The nouns, verbs and adjectives still in a sentence (moved or not), each as (the writer's lemma, the word that
    replaced it or '' where it is as written). Two sentences share a word only where the writer used it in both and it
    now reads the same in both: "The city draws on a vibrant tapestry of traditions" -> "... a lively blend of
    traditions" is not about "the scenes on the tapestry" in the sentence before, and "vital" made "needed" in one and
    "necessary" in the other no longer ties two topics; "a profound effect" and "a profound change" that both became
    "major" still do."""
    doc = sentence.doc
    return {(doc[i].lemma_.lower(), '' if text.lower() == doc[i].lower_ else text.lower())
            for i, text in sentence.pieces
            if i is not None and text and doc[i].pos_ in ('NOUN', 'PROPN', 'VERB', 'ADJ') and not doc[i].is_stop}

def related(a, b):
    """Only sentences about the same thing are joined: the second opens with a joining word or
    "This"/"That", or the two share a content word. Otherwise the join reads as a run-on."""
    head = first_piece(b.pieces)
    opener = head[1].lower().rstrip(',') if head else ''
    return (opener in JOINING_OPENERS or opener in ('this', 'that', 'these', 'those')
            or bool(content_lemmas(a) & content_lemmas(b)))

# Openers that limit what their sentence says ("Usually, ...", "Ideally, ..."): a sentence joined after it would
# fall under them too
HEDGE_ADVERBS = {'perhaps', 'maybe', 'possibly', 'probably', 'likely', 'arguably', 'reportedly', 'allegedly', 'apparently',
                 'supposedly', 'presumably', 'ideally', 'typically', 'usually', 'often', 'generally', 'normally',
                 'sometimes', 'occasionally', 'rarely', 'seldom', 'theoretically', 'hopefully', 'potentially',
                 'traditionally', 'historically', 'mostly', 'mainly', 'largely', 'frequently', 'initially'}

# Links that take a sentence joined after theirs into their scope ("But a reference is required" + ", and repairs
# require written approval" sets the repairs against the sentence before too; "This meant the bank raised rates" +
# ", and the fund raised $1m" makes the fund's money a result)
SCOPE_OPENERS = ('so', 'but', 'however', 'yet', 'then', 'still', 'this means', 'this meant', 'that means', 'that meant',
                 'as a result', "that's because", 'this is because', 'nevertheless', 'nonetheless', 'therefore', 'thus',
                 'hence', 'instead')

def opens_with_frame(sentence):
    """True if a sentence, as it now reads, opens with something that would also cover a sentence joined after
    it: a clause ("If the swelling lasts longer than five days, you should see a doctor" + ", and patients over 65
    ..." would put them under the condition), a participle ("Using chlorophyll, ..."), a hedge ("Usually, ..."), a
    frame or a time ("In most cases, ...", "In 1998, the company moved ..." + ", and the company now employs ...",
    "Most of the time, ..."), an attitude ("Luckily, nobody was hurt" + ", and the fire destroyed the barn"), a
    link with a scope (SCOPE_OPENERS), or the "This" a split put before a comment on the sentence before ("This was
    fueled by low inventory" + ", and mortgage rates eased ..." runs on). A plain link to what came before ("Also,
    ...", "Meanwhile, ...") doesn't."""
    live = [p for p in sentence.pieces if p[1]]
    if not live:
        return False
    opening = ' '.join(p[1] for p in live[:4]).lower()
    if any(opening.startswith(link + ' ') or opening.startswith(link + ',') for link in SCOPE_OPENERS):
        return True
    if live[0][0] is None:
        return live[0][1].split()[0].lower() in ('this', 'that', 'these')
    token = sentence.doc[live[0][0]]
    if token.lower_ in FRONTED_CONNECTIVES and token.dep_ in ('advmod', 'cc'):
        return False
    if token.dep_ in ('mark', 'advcl', 'prep', 'npadvmod') or token.lower_ in HEDGE_ADVERBS:
        return True     # (by the dependency, not the tag: "Rising prices ..." opens with its subject)
    # a phrase before the subject, set off by a comma: "Last year, ...", "Luckily, ..."
    head = token
    while head.head.i != head.i and head.head.pos_ not in ('VERB', 'AUX') and head.head.left_edge.i == token.i:
        head = head.head
    end = next((k for k, p in enumerate(live) if p[0] == head.right_edge.i), None)
    return (head.dep_ in ('npadvmod', 'advmod', 'prep') and end is not None and end + 1 < len(live)
            and live[end + 1][1] == ',')

def reads_as_command(sentence):
    """True if a sentence, as it now reads, is a command: its main verb is a bare one with no subject ("Choose a
    reputable service.", "Then add the cream.", "If the light keeps flashing, turn the printer off.", "Do not use
    it on broken skin."), and no subject was put before it by a split ("You can get reminders.")."""
    doc, live = sentence.doc, [p for p in sentence.pieces if p[1]]
    for index, text in live:
        if index is not None:
            break
        if text.split()[-1].lower() in KEPT_PRONOUNS or text.split()[0].lower() in ('this', 'that', 'these', 'those'):
            return False
    indices = {index for index, _ in live if index is not None}
    verb = next((doc[i] for i in sorted(indices) if doc[i].pos_ in ('VERB', 'AUX')
                 and (doc[i].head.i == i or doc[i].head.i not in indices)), None)
    return verb is not None and verb.tag_ == 'VB' and not any(
        c.dep_ in ('nsubj', 'nsubjpass', 'expl') or (c.dep_ in ('aux', 'auxpass') and c.lower_ != 'do')
        for c in verb.children)

# Words that set a sentence against the one before from inside it ("Wages, however, rose.", "The hotel is actually
# quite run down."): joined with "and", the two would contradict each other
INNER_CONTRASTS = {'however', 'nevertheless', 'nonetheless', 'though', 'instead', 'meanwhile', 'conversely', 'actually'}

def coordinated(sentence):
    """True if a sentence already joins two parts with a comma and a conjunction ("X, and Y"), or two clauses with a
    bare "and"/"but" ("The rooms are large and the beds are comfortable"): one more join would make a run-on ("Data
    is valuable, and we never sell it, and their advice ...", "The rooms are large and the beds are comfortable, and
    the breakfast ..."). A second verb that shares the subject ("opened in 1998 and now has 40 staff") is no clause."""
    texts = [t.lower() for _, t in sentence.pieces if t]
    if any(t == ',' and n in ('and', 'but', 'or', 'so', 'yet') for t, n in zip(texts, texts[1:])):
        return True
    doc = sentence.doc
    live = {i for i, text in sentence.pieces if i is not None and text}
    return any(doc[i].dep_ == 'conj' and doc[i].pos_ in ('VERB', 'AUX') and doc[i].head.i in live
               and clause_subject(doc[i]) is not None and clause_subject(doc[i]).i in live for i in live)

def joins_as_which(sentence):
    """True if a sentence opens with "This" or "That" as the subject of its verb, so a join makes it a ", which"
    clause ("We added caching. This cut latency." -> "We added caching, which cut latency."; not "That said, ...")."""
    head = first_piece(sentence.pieces)
    if not head or head[0] is None or head[1].lower() not in ('this', 'that'):
        return False
    doc, i = sentence.doc, head[0]
    return (doc[i].dep_ == 'nsubj' and i + 1 < len(doc) and doc[i + 1].pos_ in ('VERB', 'AUX')
            and not (i + 2 < len(doc) and doc[i + 2].text == ','))

def stacks_link(sentence):
    """True if a sentence opens with a link or a frame of its own that a join would leave after "and" ("X, and
    overall, Y", "X, and on top of that, Y"); the links merge_sentences turns into the conjunction ("Also,",
    "However,", "That said,") are fine."""
    live = [p for p in sentence.pieces if p[1]]
    words = [p[1].lower() for p in live[:4]]
    if not words or words[0].rstrip(',') in JOINING_OPENERS or words[:2] in (['that', 'said'], ['this', 'said']) or (
            words[:3] in (['that', 'being', 'said'], ['having', 'said', 'that'])):
        return False
    return words[0].endswith(',') or opens_with_frame(sentence)

def joined_right(sentence):
    """A sentence's pieces as a join puts them after the sentence before, with the word that joins them: "and"; the
    conjunction an opening link becomes ("However, X" -> "but X", "That said, X" -> "but X", "Also, X" -> "and X");
    or "which", for a "This" or "That" that becomes a ", which" clause."""
    right = [list(p) for p in sentence.pieces]
    live = [k for k, p in enumerate(right) if p[1]]
    words = [right[k][1].lower() for k in live[:4]]
    for phrase in (['that', 'said'], ['this', 'said'], ['that', 'being', 'said'], ['having', 'said', 'that']):
        n = len(phrase)
        if len(words) > n and words[:n] == phrase and words[n] == ',':
            return 'but', right[live[n] + 1:]
    head = first_piece(right)
    if joins_as_which(sentence):
        return 'which', right[right.index(head) + 1:]
    opener = head[1].lower().rstrip(',') if head else ''
    if opener not in JOINING_OPENERS:
        return 'and', right
    right = right[right.index(head) + 1:]
    if right and right[0][1] == ',':
        right = right[1:]
    return JOINING_OPENERS[opener], right

# Common English words that are also names of places, people or firms ("Reading", "Will", "May", "Turkey"): where
# one opens a sentence nothing shows which the writer meant, and a join would have to choose between "and Reading
# ..." and "and reading ...", so the two sentences stay as written
JOIN_NAME_HOMOGRAPHS = {
    'reading', 'bath', 'nice', 'buffalo', 'phoenix', 'victoria', 'florence', 'derby', 'deal', 'sale', 'hull', 'cork',
    'wells', 'turkey', 'china', 'chad', 'jordan', 'orange', 'hope', 'will', 'bill', 'mark', 'rose', 'may', 'june',
    'april', 'august', 'march', 'grace', 'faith', 'joy', 'jack', 'dawn', 'sandy', 'frank', 'pat', 'sue', 'ray', 'guy',
    'gene', 'amber', 'ruby', 'holly', 'ivy', 'iris', 'lily', 'daisy', 'robin', 'drew', 'wade', 'chase', 'cliff', 'dean',
    'don', 'herb', 'max', 'miles', 'nick', 'pierce', 'reed', 'rod', 'penny', 'pearl', 'heather', 'hazel', 'olive',
    'violet', 'glen', 'dale', 'grant', 'lance', 'mason', 'hunter', 'carter', 'cooper', 'taylor', 'parker', 'porter',
    'baker', 'brook', 'banks', 'fisher', 'king', 'apple', 'amazon', 'shell', 'next', 'boots', 'virgin', 'sage', 'dove',
    'ford', 'target', 'gap', 'oracle', 'visa', 'go', 'can'}

def joined_case(pieces, doc):
    """How the first word of a sentence reads after a join's "and" or "but": 'lower' for an ordinary word ("Further
    updates will be posted" -> "..., and further updates will be posted"), 'keep' for "I", an acronym or a name the
    text shows is one (written with a capital inside a sentence, or a recognised name that is no English word:
    "Oxford", "Ben"), and None where the capital may be either: a word that is also a name (JOIN_NAME_HOMOGRAPHS;
    "Will brought the food", where a "modal" before a past verb is a name), or one spaCy takes for a name that is
    also an English word ("Insulin is made in the pancreas"). The sentences are then not joined."""
    piece = first_word_piece(pieces)
    if piece is None or not piece[1][:1].isupper():
        return 'lower'
    word = piece[1].split(' ')[0]
    if word == 'I' or (len(word) > 1 and word.isupper()) or any(c.isupper() for c in word[1:]):
        return 'keep'
    if piece[0] is None or piece is not first_piece(pieces) or word.lower() != doc[piece[0]].lower_:
        return 'lower'      # a word a template or the word layer put there, or one after an elision ("'Til")
    token = doc[piece[0]]
    if not token.text[:1].isupper():
        return 'lower'      # written in lower case; a template capitalised it
    if not token.is_sent_start or token.text in mid_sentence_words(doc):
        return 'keep'       # the writer capitalises it inside a sentence: a name
    if token.lower_ in JOIN_NAME_HOMOGRAPHS or (token.tag_ == 'MD' and token.head.tag_ != 'VB'):
        return None
    if not keeps_capital(token) or token.lower_ in NLP.Defaults.stop_words:
        return 'lower'      # a common word, or a function word spaCy took for a name ("Further updates")
    return None if uncertain_capital(token) else 'keep'

# Verbs that name a source after a comma at the end of a sentence (", scientists have found.", ", officials said.",
# ", she added.")
ATTRIBUTION_VERBS = CLAIM_FRAMES | {'add', 'allege', 'calculate', 'caution', 'concede', 'emphasise', 'emphasize', 'fear',
                                    'feel', 'forecast', 'reckon', 'stress', 'write', 'tell', 'recall', 'remark',
                                    'comment', 'reply', 'complain', 'suspect', 'hope', 'worry', 'speculate'}

def ends_in_source(sentence):
    """True if the part of a sentence after its last comma names who said it (", scientists have found.", ", as the
    report notes.", ", according to the council.") or was moved there from in front of the rest (claim_to_end,
    source_to_end, back_fronted_adjunct: "The broth is the most important part, many chefs say."). Joined to another
    sentence, it would sit between two statements or stretch over both ("Bees matter for crops, and bee numbers
    fell, scientists have found" says the scientists found both)."""
    live = [p for p in sentence.pieces if p[1] and p[1] not in ('.', '!', '?')]
    comma = next((k for k in range(len(live) - 1, 0, -1) if live[k][1] == ','), None)
    if comma is None or comma == len(live) - 1:
        return False
    doc = sentence.doc
    tail = [p[0] for p in live[comma + 1:] if p[0] is not None]
    rest = [p[0] for p in live[:comma] if p[0] is not None]
    inside = set(tail)
    if [p[1].lower() for p in live[comma + 1:comma + 3]] == ['according', 'to'] or (
            tail and rest and max(tail) < min(rest) and doc[tail[0]].sent.root.i not in inside):
        return True     # (not the main clause after a phrase moved to the front: "Last month, the council ...")
    return len(live) - comma <= 9 and any(
        doc[i].pos_ == 'VERB' and doc[i].lemma_.lower() in ATTRIBUTION_VERBS
        and any(c.dep_ in ('nsubj', 'nsubjpass') and c.i in inside for c in doc[i].children)
        and not any(c.dep_ in ('dobj', 'ccomp', 'xcomp', 'oprd', 'acomp', 'attr', 'dative') and c.i in inside
                    for c in doc[i].children) for i in tail)

def dummy_it(sentence):
    """True if a sentence has an "it" that stands for nothing but the clause it brings in ("It is important for the
    council to listen to residents", "It was found that ...", "It seems that ...")."""
    doc = sentence.doc
    for i, text in sentence.pieces:
        if i is None or not text or doc[i].lower_ != 'it' or doc[i].dep_ not in ('nsubj', 'nsubjpass', 'expl'):
            continue
        verb = doc[i].head
        if not (verb.lemma_.lower() in ('be', 'seem', 'appear', 'remain') or any(c.dep_ == 'auxpass' for c in verb.children)):
            continue
        if any(c.i > i and (c.dep_ in ('xcomp', 'ccomp', 'csubj') or (c.dep_ == 'advcl' and any(
                m.dep_ in ('mark', 'aux') and m.lower_ in ('for', 'to') for m in c.children))) for c in verb.children):
            return True
    return False

def opening_subject(pieces, doc):
    """The words, as they now read, of the subject a sentence opens with ("the hotel" in "The hotel has a pool"); None
    when it opens with something else, an inserted word or a personal pronoun."""
    piece = first_piece(pieces)
    if piece is None or piece[0] is None:
        return None
    token = doc[piece[0]]
    subject = next((t for t in [token, *token.ancestors] if t.dep_ in ('nsubj', 'nsubjpass')), None)
    if subject is None or subject.left_edge.i != token.i or subject.tag_ == 'PRP':
        return None
    inside = {t.i for t in subject.subtree}
    return [text.lower() for i, text in pieces if text and i in inside]

def mergeable(a, b):
    # A ", which" clause comments on the sentence before, however that one opens; any other join puts the second
    # sentence under the first one's opening frame or link
    comment = joins_as_which(b)
    if ((a.origin is not None and a.origin is b.origin) or (not comment and opens_with_frame(a)) or coordinated(a)
            or coordinated(b)):
        return False
    words = [t.lower() for _, t in b.pieces if t]
    if (any(w in INNER_CONTRASTS for w in words[1:]) or words[:2] == ['so', ',']
            or any(pair in (('in', 'fact'), ('in', 'reality')) for pair in zip(words, words[1:]))):
        return False    # "and wages, however, rose"; a summing-up "So, ..." is no result of the sentence before
    head = first_piece(b.pieces)
    if head is not None and head[0] is not None and b.doc[head[0]].dep_ in ('mark', 'advcl'):
        return False    # "..., and if the light keeps flashing, turn the printer off"
    if not comment and stacks_link(b):
        return False    # "..., and overall, it was the best holiday"
    for s in (a, b):
        last = s.pieces[-1][1] if s.pieces else ''
        # Dashes too: two dashed sentences joined would have their dashes paired into one bracket
        if (last != '.' or s.words() > MERGE_MAX_WORDS
                or any(t in (';', ':', '"', '“', '”', '‘', '(', '—', '–') for _, t in s.pieces)
                or has_quoted(s.doc, (p[0] for p in s.pieces))):
            return False
    if not related(a, b) or ends_in_source(a) or ends_in_source(b):
        return False
    conjunction, right = joined_right(b)
    if conjunction != 'which':
        first = first_piece(right)
        if joined_case(right, b.doc) is None:
            return False    # "..., and reading also suffered serious flooding": a town or a pastime?
        if first is not None and first[1].lower() in ('it', 'its') and dummy_it(a):
            return False    # "It is important for the council to listen, and it runs the library": which "it"?
        subject = opening_subject(right, b.doc)
        if subject is not None and subject == opening_subject(a.pieces, a.doc):
            return False    # "The hotel has a pool, and the hotel has a car park" says its subject twice
    # Not after a closing ", which", ", who" or ", -ing" clause ("X, which may reflect Y, and these results show Z",
    # "20,000 trees, most of which were native, and trees cool the streets", "built a well, giving clean water to
    # every home, and the new law ..." run on), nor a statement and a command ("Some providers keep logs, and choose a
    # reputable service")
    live = [p for p in a.pieces if p[1]]
    return (not any(text == ',' and ({'which', 'whom', 'who'} & {p[1].lower() for p in live[k + 1:k + 4]} or (
                        live[k + 1][0] is not None and a.doc[live[k + 1][0]].tag_ == 'VBG'))
                    for k, (_, text) in enumerate(live[:-1]))
            and (comment or reads_as_command(a) == reads_as_command(b)))

def merge_sentences(a, b, tone):
    left = [list(p) for p in a.pieces[:-1]]
    conjunction, right = joined_right(b)
    if conjunction == 'which':
        return Sentence(left + [[None, ','], [None, 'which']] + right, a.doc)
    if joined_case(right, a.doc) == 'lower':
        lowercase_first(right, a.doc)
        piece = first_word_piece(right)
        if piece is not None and piece[1][:1].isupper():
            piece[1] = piece[1][0].lower() + piece[1][1:]   # also a function word spaCy took for a name ("Further")
    joiner = [[None, ';']] if tone == 'academic' and conjunction == 'and' else [[None, ','], [None, conjunction]]
    return Sentence(left + joiner + right, a.doc)

def as_written(sentence):
    return sentence.span is not None and sentence.words() >= 6 and untouched(sentence.pieces, sentence.doc)

def usable_splits(sentence, tone):
    """A sentence's split points that suit the tone: Academic opens no sentence with "So" ("So book early.")."""
    return [p for p in sentence.splits if not (tone == 'academic' and p[1] in ('so:plain', 'so:command'))]

def vary_rhythm(sentences, tone, deep_mode):
    # The coverage pass splits or joins a sentence of eight words or more that nothing else could change. Chaos Mode
    # goes further: it splits every sentence with a safe split point, reworded or not (Academic only those still as
    # written), also before a closing relative clause where nothing else could split it (relative_clause_point), and
    # joins any other sentence still as written to a related neighbour where it can
    if deep_mode:
        for s in sentences:
            point = (relative_clause_point(s.span, s.pieces) if not s.splits and s.span is not None and not s.reordered
                     and splittable(s.span) else None)
            if point is not None:
                s.splits = [point]
    for k in range(len(sentences) - 1, -1, -1):
        s = sentences[k]
        points = usable_splits(s, tone)
        if points and (s.must_split or (deep_mode and (tone != 'academic' or as_written(s)))):
            sentences[k:k + 1] = split_sentence(s, random.choice(points), tone)
    # A sentence nothing else could change safely, and that has no split point, joins a related neighbour
    k = 0
    while k < len(sentences):
        s = sentences[k]
        if not usable_splits(s, tone) and (s.must_split or (deep_mode and as_written(s))):
            if k + 1 < len(sentences) and mergeable(s, sentences[k + 1]):
                sentences[k:k + 2] = [merge_sentences(s, sentences[k + 1], tone)]
            elif k > 0 and mergeable(sentences[k - 1], s):
                sentences[k - 1:k + 1] = [merge_sentences(sentences[k - 1], s, tone)]
                k -= 1
        k += 1
    target = CHAOS_RHYTHM_TARGET if deep_mode else RHYTHM_TARGET[tone]
    budget = max(1, round(len(sentences) * (CHAOS_RHYTHM_OPS if deep_mode else RHYTHM_OPS[tone])))
    for _ in range(budget):
        lengths = [s.words() for s in sentences]
        current = spread(lengths) if len(lengths) > 1 else 0.0
        if current >= target and random.random() < 0.7:
            break
        options = []
        for k, s in enumerate(sentences):
            for point in usable_splits(s, tone):
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
    vary_split_openers(sentences)
    return sentences

def vary_split_openers(sentences):
    """A split chooses its opener without looking at its neighbours, so one of the two sentences before may already
    open the same way. A split's "This" then becomes "That" ("This isn't nearly enough for a large house. The app also
    crashes from time to time. That can be frustrating."), but not before a comma, where "That is, ..." would mean "in
    other words"; a split's "But" or "However," becomes "Yet" ("However, the effect was not significant. Earlier
    studies have shown X. Yet these findings have not been replicated.")."""
    for k, s in enumerate(sentences):
        live = [p for p in s.pieces if p[1]]
        if s.origin is None or k == 0 or not live or live[0][0] is not None:
            continue
        head = live[0]
        before = {sentences[j].opener().split(' ')[0].rstrip(',') for j in range(max(0, k - 2), k)}
        first = head[1].split(' ')[0]
        if first == 'This' and 'this' in before and 'that' not in before and not (
                len(live) > 2 and ' ' not in head[1] and live[2][1] == ','):
            head[1] = 'That' + head[1][4:]
        elif head[1] in ('But', 'However,') and before & {'but', 'however'} and 'yet' not in before:
            head[1] = 'Yet'

def vary_openings(sentences):
    """Two sentences in a row that open with the same word: restructure the second if possible."""
    for k in range(1, len(sentences)):
        s = sentences[k]
        if (s.opener() != sentences[k - 1].opener() or s.span is None or s.reordered
                or has_quoted(s.doc, range(s.span.start, s.span.end))):
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
FINITE_TAGS = {'VBZ', 'VBP', 'VBD', 'MD'}

WH_WORDS = {'WDT', 'WP', 'WP$'}

def clause_follows(fragment):
    """True if the words after a dash form an independent clause: a subject with a finite main verb, and not a
    relative or subordinate clause ("— one that reshaped the industry", "— a place where ...", "— because ...")."""
    with _NLP_LOCK:
        doc = NLP(fragment)
    for token in doc:
        if token.dep_ in ('nsubj', 'nsubjpass', 'expl', 'csubj') and token.head.dep_ == 'ROOT':
            verb = token.head
            if not (verb.tag_ in FINITE_TAGS
                    or any(c.dep_ in ('aux', 'auxpass') and c.tag_ in FINITE_TAGS for c in verb.children)):
                continue
            return token.tag_ not in WH_WORDS and not any(
                c.dep_ == 'mark' or (c.tag_ in WH_WORDS | {'WRB'} and c.i < verb.i) for c in verb.children)
    return False

def capitalize_start(text):
    """Text with its first word capitalised ("iPhone" and "eBay" keep their spelling), past any opening mark."""
    m = re.match(r'(\W*)(\w+)', text)
    return m.group(1) + capitalized(m.group(2)) + text[m.end():] if m else text

# "most offer tastings", "some date back to the 1500s": a quantifier and a word that can be a verb, which the
# parser may not read as a clause
QUANTIFIER_DASH = re.compile(r'(?:most|some|many|several|few|all|both|each|none|others|half)\s+([a-z]+)\b')

def soften_dashes(text, tone, deep_mode=False):
    """AI text leans on em dashes; most become parentheses (or commas, in Casual). A single dash before a
    clause becomes a full stop ("; " in Academic), before a list a colon, otherwise a comma. A dash before what
    may be a clause the parser missed ("— most offer tastings") stays: a comma would splice two sentences, and a
    full stop could leave a fragment ("— all made by hand")."""
    if '—' not in text or (not deep_mode and random.random() < 0.3):
        return text
    aside = (lambda m: ', ' + m.group(1) + ', ') if tone == 'casual' else (lambda m: ' (' + m.group(1) + ') ')
    text = EM_DASH_PAIR.sub(aside, text)
    if NLP is None:
        return EM_DASH_ONE.sub(', ', text)
    position = 0    # where the search goes on: past any dash that stays
    while (match := EM_DASH_ONE.search(text, position)):
        left, right = text[:match.start()], text[match.end():]
        end = left[-1:]
        fragment = re.split(r'(?<=[.!?])\s', right, maxsplit=1)[0]
        clause = None if end in (',', ';', ':') else clause_follows(fragment)
        if end in (',', ';', ':'):
            text = left + ' ' + right       # the writer's own mark already separates the parts: the dash just goes
        elif not left or end in ('!', '?') or (end == '.' and clause):
            # the dash opens a sentence: it goes, and the sentence gets its capital. A full stop with no clause
            # after it is an abbreviation's ("Washington, D.C. — near the river"): the cases below apply
            text = left + (' ' if left else '') + capitalize_start(right)
        elif clause:
            text = left + ('; ' + right if tone == 'academic' else '. ' + capitalize_start(right))
        elif ',' in fragment and re.search(r'\b(?:and|or)\b', fragment):
            text = left + ': ' + right
        elif (lead := QUANTIFIER_DASH.match(fragment)) and 'VERB' in getAllLemmas(lead.group(1)):
            position = match.end()
        else:
            text = left + ', ' + right
    return text

TIDY_SPACING = re.compile(r'[ \t]+([,.;:!?)])(?=\s|$)|(\()[ \t]+')

def tidy(text):
    """Only spacing the layers can introduce ("word ,", "( word"). The writer's own spacing is left alone, and so
    are quotations and code ("git add .", "`foo( x , y )`")."""
    keep = quoted_ranges(text)         # sorted by start, and the matches come in order: one pointer will do
    state = {'next': 0, 'reach': -1}

    def close_up(m):
        while state['next'] < len(keep) and keep[state['next']][0] < m.end():
            state['reach'] = max(state['reach'], keep[state['next']][1])
            state['next'] += 1
        if state['reach'] > m.start():
            return m.group(0)
        return m.group(1) or m.group(2)
    return TIDY_SPACING.sub(close_up, text)

# ==========================================
#   PUTTING THE LAYERS TOGETHER
# ==========================================

BULLET = re.compile(r'^\s*(?:[-*•▪◦]|\d+[.)]|[A-Za-z][.)])\s')
PROSE_END = re.compile(r'[.!?]["”’)\]]?\s*$')
LINE_BREAKS = re.compile(r'(\s*\n\s*)')

def is_prose(block):
    """Running text, as opposed to a heading or list item (those only get word-level edits)."""
    return not BULLET.match(block) and bool(PROSE_END.search(block)) and len(block.split()) >= 6

ENGLISH_SHARE = 0.6

def looks_english(tokens):
    """True unless the words are mostly not English: the share of words (names aside) that are English words
    or common function words is at least ENGLISH_SHARE. Every layer is built for English, so a paragraph in
    German or Dutch ("..., und die Kosten ...") is left as written, not given an English "and"."""
    words = [t for t in tokens if t.is_alpha and t.ent_type_ not in NAME_ENTITIES]
    if len(words) < 4:
        return True
    known = sum(1 for t in words if t.is_stop or getAllLemmas(t.lower_))
    return known / len(words) >= ENGLISH_SHARE

def content_words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3 and (NLP is None or w not in NLP.Defaults.stop_words)}

def written_by_model(sent, model_written):
    """True if Deep Rewrite wrote this sentence. It is recognised by its content words, which the phrase
    rules and contractions applied since then barely change."""
    words = content_words(sent.text)
    return bool(words) and any(len(words & w) / len(words | w) >= 0.6 for w in model_written if w)

# Sentence templates safe to apply to a sentence a model wrote (Chaos Mode): they rework one construction
# in place. Phrase moves are not used on model sentences, as they compound with the model's own changes.
MODEL_SAFE_TRANSFORMS = (existential_to_direct, extraposed_to_gerund, essential_for_to, importance_of, not_only_inverted,
                         category_to_direct, goal_to_gerund, passive_to_active, denominalize)
# Chance of "allows X to" -> "lets X" and of "the legacy of the war" -> "the war's legacy", by tone
LET_RATE = {'standard': 0.6, 'casual': 0.7, 'professional': 0.4, 'academic': 0.0}
POSSESSIVE_RATE = {'standard': 0.5, 'casual': 0.6, 'professional': 0.4, 'academic': 0.3}

def drop_doubled_stop(pieces, doc):
    """Blanks a sentence's full stop that a move put right after a word ending in one ("..., according to Acme Inc.",
    "... opened to the public at 9 a.m.", which would read "Inc.." and "a.m.."); a stop the writer put there stays."""
    live = [p for p in pieces if p[1]]
    if len(live) > 1 and live[-1][1] == '.' and live[-2][1].endswith('.') and not (
            live[-1][0] is not None and live[-2][0] is not None and live[-1][0] == live[-2][0] + 1):
        live[-1][1] = ''

def sentence_gap(before, after, doc):
    """The space between two sentences as rendered: the writer's own when they still meet as written (the last word
    of one and the first of the next are neighbours in the text, unchanged: "Then utilize `git add .` to stage ...",
    where spaCy ends a sentence inside the code), else one space."""
    last = next((p for p in reversed(before.pieces) if p[1]), None)
    first = first_piece(after.pieces)
    if (last is not None and first is not None and last[0] is not None and first[0] is not None
            and first[0] == last[0] + 1 and last[1] == doc[last[0]].text and first[1] == doc[first[0]].text):
        return doc[last[0]].whitespace_
    return ' '

def rewrite_block(doc, tone, deep_mode, prose, model_written=(), vet=None):
    model_starts = {sent.start for sent in doc.sents if model_written and written_by_model(sent, model_written)}
    edits = plan_word_edits(doc, tone, deep_mode, model_starts, unlock_model=deep_mode, vet=usable(vet))
    if not prose:
        return ''.join(edits.get(t.i, t.text) + t.whitespace_ for t in doc)
    sentences = []
    sents = list(doc.sents)
    for k, sent in enumerate(sents):
        pieces = [[t.i, edits.get(t.i, t.text)] for t in sent if not t.is_space]
        if not pieces:
            continue
        sentence = Sentence(pieces, doc, sent)
        from_model = sent.start in model_starts
        previous = sentences[-1].opener() if sentences else ''
        boost = CHAOS_STRUCTURE_BOOST if deep_mode else 0.0
        reworked = False        # a template dropped or moved words, so the original split points no longer hold
        if from_model:
            # Deep Rewrite already restructured it; moving its phrases again risks the meaning
            sentence.reordered = True
            if deep_mode and restructurable(sent):
                for transform in MODEL_SAFE_TRANSFORMS:
                    new = transform(sent, sentence.pieces)
                    if new:
                        sentence.pieces = new
                        reworked = transform is not denominalize
                        break
        elif restructurable(sent):
            if random.random() < NOMINAL_RATE[tone] + boost:
                pieces = denominalize(sent, pieces) or pieces
            if random.random() < NOMINAL_RATE[tone] + boost:
                pieces = the_way_to_how(sent, pieces) or pieces
            if tone != 'academic' and random.random() < NOMINAL_RATE[tone] + boost:
                pieces = process_to_how(sent, pieces) or pieces
            reorder_boost = CHAOS_REORDER_BOOST if deep_mode else 0.0
            if random.random() < LIST_RATE[tone] + reorder_boost:
                pieces = reorder_list(sent, pieces) or pieces
            if LET_RATE[tone] and random.random() < LET_RATE[tone] + reorder_boost:
                pieces = allows_to_lets(sent, pieces) or pieces
            if random.random() < POSSESSIVE_RATE[tone] + reorder_boost:
                pieces = of_to_possessive(sent, pieces) or pieces
            sentence.pieces = pieces
            after_move = bool(sentences and sentences[-1].reordered)
            new = restructure(sent, pieces, tone, deep_mode, after_move=after_move, previous=previous)
            if new:
                sentence.pieces, sentence.reordered = new, True
        shortest = CHAOS_COVERAGE_MIN_WORDS if deep_mode else COVERAGE_MIN_WORDS
        if not from_model and untouched(sentence.pieces, doc) and sentence.words() >= shortest:
            # the neighbours as they now read, so a forced swap doesn't echo them ("looks into ... looked at")
            nearby = word_forms(' '.join(text for _, text in sentences[-1].pieces)) if sentences else set()
            if k + 1 < len(sents):
                nearby |= word_forms(' '.join(edits.get(t.i, t.text) for t in sents[k + 1]))
                nearby |= {t.lemma_.lower() for t in sents[k + 1]}
            ensure_change(sentence, tone, deep_mode, usable(vet), previous, nearby)
        if splittable(sent) and ((from_model and not reworked) or not sentence.reordered):
            sentence.splits = split_points(sent, sentence.pieces)
        sentences.append(sentence)
    sentences = vary_rhythm(sentences, tone, deep_mode)
    vary_openings(sentences)
    out = []
    for k, s in enumerate(sentences):
        drop_doubled_stop(s.pieces, doc)
        text = s.text() if has_quoted(doc, (p[0] for p in s.pieces)) else soften_dashes(s.text(), tone, deep_mode)
        before = sentences[k - 1].span if k else None
        # A mark cut off its word is written back against it ("'Til midnight ..."); sentences are a space apart
        glued = before is not None and len(before) == 1 and cut_off_opening_mark(before[0])
        out.append(text if k == 0 or glued else sentence_gap(sentences[k - 1], s, doc) + text)
    return ''.join(out)

# A line with a longer run of characters without a space is no prose (a hash, a data dump, a run of brackets), and spaCy's
# tokenizer takes time that grows with the square of such a run
UNBROKEN_CHUNK_LIMIT = 1000

def rewrite_with_spacy(text, tone, deep_mode, model_written=(), vet=None):
    parts = LINE_BREAKS.split(text)      # paragraphs and lines at even positions, the breaks between them at odd ones
    quotes = quoted_ranges(text)         # found in the whole text: a quotation may be hard-wrapped over lines
    code = [(m.start(), m.end()) for m in CODE_SPANS.finditer(text)]
    jobs, offset = [], 0
    for k, block in enumerate(parts):
        core = block.strip()
        if k % 2 == 0 and core:
            lead = block[:len(block) - len(block.lstrip())]
            start = offset + len(lead)
            # a line of a code block stays, and so does a line with a run of more than UNBROKEN_CHUNK_LIMIT characters
            if (not any(s <= start and start + len(core) <= e for s, e in code)
                    and max(len(chunk) for chunk in core.split()) <= UNBROKEN_CHUNK_LIMIT):
                jobs.append((k, lead, core, block[len(block.rstrip()):], start))
        offset += len(block)
    with _NLP_LOCK:
        docs = list(NLP.pipe([core for _, _, core, _, _ in jobs]))
    for (k, lead, core, trail, start), doc in zip(jobs, docs):
        doc.user_data['quoted'] = frozenset(t.i for t in doc if overlaps_quote(text, start + t.idx, start + t.idx + 1))
        if not looks_english(doc):
            continue
        parts[k] = lead + tidy(rewrite_block(doc, tone, deep_mode, is_prose(core), model_written, vet)) + trail
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
    quotes = quoted_ranges(text)
    tokens = re.split(r'(\s+)', text)
    end = 0
    for i, token in enumerate(tokens):
        start, end = end, end + len(token)
        match = WORD_PATTERN.match(token)
        if not match or any(s < end and start < e for s, e in quotes):
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
#   DEEP REWRITE (OPTIONAL PARAPHRASING MODELS)
# ==========================================
# Small local models rewrite whole sentences (see paraphraser.py): a T5 paraphraser and, when it is
# installed, back-translation through another language. The rule-based layers then run on the result;
# a sentence a model wrote gets word-level edits (in Chaos Mode also the sentence templates) but no phrase
# moves. A sentence keeps its own wording unless one of the candidates passes every check below.

NUMBER_ENTITIES = {'DATE', 'TIME', 'MONEY', 'PERCENT', 'QUANTITY', 'CARDINAL', 'ORDINAL'}
DEEP_MIN_WORDS = 5
# Seconds a Deep Rewrite aims to stay within: back-translation is skipped once 40% of it has gone, and Chaos
# Mode's second round when it wouldn't finish in time. Generating and checking the first round always run.
DEEP_TIME_BUDGET = paraphraser.env_number('DEEP_REWRITE_SECONDS', 45.0, float, minimum=1)

def deep_rewrite_available():
    return NLP is not None and paraphraser.available()

def plain_words(text):
    return re.findall(r"[\w'’-]+", text)

NOUN_RETENTION = 0.6    # share of the sentence's topic nouns a rewrite must keep
LENGTH_RANGE = (0.7, 1.4)    # rewrite length / original length; longer ones tend to add details
# Words of four letters or more, leaving out the "isn" of "isn't"
LONG_WORDS = re.compile(r"(?<!['’])\b[a-z]{4,}\b(?!['’]t\b)")
# Nouns of stock frames rather than of the topic ("plays a crucial role", "in today's world"): a rewrite
# that drops the frame keeps the meaning
LIGHT_NOUNS = {'role', 'part', 'way', 'aspect', 'factor', 'manner', 'process', 'nature', 'kind', 'sort', 'type',
               'thing', 'lot', 'range', 'variety', 'number', 'amount', 'level', 'world', 'area', 'field', 'realm',
               'landscape', 'importance', 'need', 'impact', 'conclusion', 'addition', 'fact', 'end', 'time', 'today',
               'force', 'case', 'example', 'instance', 'term', 'regard', 'respect', 'order', 'sense', 'means', 'use'}
# Generic nouns for a way of doing something, which a faithful rewrite may trade for one another
GENERIC_NOUNS = {'approach', 'method', 'technique', 'strategy', 'tactic', 'tool', 'option', 'step', 'measure', 'practice',
                 'solution', 'mechanism', 'procedure', 'idea', 'plan', 'thing', 'option', 'choice', 'moment', 'occasion'}
# "unable to" says "cannot", so it counts as a negator of its own ("unable to afford" = "cannot afford")
NEGATORS = {'not', 'no', 'never', 'without', 'nor', 'neither', 'none', 'cannot', 'nothing', 'nobody', 'nowhere',
            'unable', 'incapable', 'unwilling'}
# Words that look like a negating prefix plus a word but don't negate it ("incredibly", "invaluable", "insure")
NOT_NEGATED = {'incredible', 'incredibly', 'intense', 'intensely', 'intensive', 'invaluable', 'infamous', 'inflammable',
               'inside', 'insure', 'insured', 'immediate', 'immediately', 'impending', 'incorporate', 'disclose',
               'disclosed', 'inborn', 'inbound', 'imaging', 'uncanny', 'unravel', 'unloosen', 'irradiate', 'impassioned',
               'income', 'indent', 'inflow', 'input', 'insight', 'install', 'invest', 'involve', 'impart', 'imprint',
               'unable', 'incapable', 'unwilling'}
# Verbs that negate what follows without a "not": "prevents landlords from raising rents", "avoid taking", "bans
# smoking", "failed to address", "stopped working". They count with the explicit negators, so "does not allow"
# may become "prevents ... from", but "prevents ... from" may not become "allows".
FROM_NEGATORS = {'prevent', 'stop', 'keep', 'bar', 'block', 'deter', 'restrict', 'discourage', 'exclude', 'exempt'}
ALWAYS_NEGATING = {'avoid', 'refrain', 'abstain', 'lack', 'ban', 'forbid', 'prohibit'}
# ...with a verb after them ("should not discontinue taking it", but not "ended up using it"), or with the use of
# something as their object ("halt their use of it", "discontinue the medication")
CEASING = {'fail', 'stop', 'cease', 'quit', 'neglect', 'refuse', 'decline', 'discontinue', 'halt', 'suspend', 'abandon',
           'terminate', 'interrupt', 'end'}
USE_NOUNS = {'use', 'usage', 'intake', 'consumption', 'treatment', 'medication', 'medicine', 'therapy', 'dose', 'dosage',
             'drug', 'cream', 'pill', 'tablet', 'supplement'}
# What a sentence claims: "must" is not "should", "can" (ability) is not "may" (possibility), "will" is a
# prediction, and "purportedly" casts doubt
MODALITY = {'must': 'must', 'should': 'should', 'ought': 'should', 'may': 'may', 'might': 'may', 'could': 'may',
            'can': 'can', 'cannot': 'can', 'will': 'will', 'would': 'will', 'shall': 'will'}
# "is not required to", "aren't obliged to" state an obligation as much as "have to" ("the app does not collect
# your data" is not "you are not required to give it your data")
OBLIGATION = re.compile(r"\b(?:need|needs|needed|have|has|had)\s+to\b|\b(?:is|are|was|were|be|been|being|am)(?:n['’]t)?\s+"
                        r"(?:not\s+)?(?:\w+ly\s+)?(?:required|obliged|obligated|forced|compelled)\s+to\b", re.I)
# ...and so do these verbs in either voice ("the law requires employers to give", "mandates that") and these words
# as a predicate ("is mandatory", "made compulsory"; not "the mandatory retirement age")
OBLIGING_VERBS = {'require', 'oblige', 'obligate', 'mandate', 'stipulate'}
OBLIGING_PREDICATES = {'mandatory', 'compulsory', 'obligatory'}
# The active "The storm forced residents to leave" states a cause as much as an obligation, so it is no "must" of
# its own ("Residents left their homes because of the storm" says the same); the passive "were forced to" is one
FORCING_VERBS = {'force', 'compel'}
# A need the main clause states without laying it on anyone ("Fishing requires a permit", "A permit is needed",
# "is necessary", "is a prerequisite"): a rewrite may state it as an obligation ("A permit is required to fish"),
# and an obligation this way, if it keeps what is needed and who needs it. A demand is no need ("Parents demand
# more information from schools" is not "Schools must give parents more information").
NEED_STATING_VERBS = {'require', 'need', 'necessitate', 'mandate', 'stipulate'}
NEED_STATING_WORDS = {'necessary', 'needed', 'required', 'mandatory', 'compulsory', 'obligatory', 'essential',
                      'imperative', 'indispensable', 'requirement', 'prerequisite', 'necessity', 'obligation'}
# "can be seen" says only that something is there; the modal adds no claim
SEEN = re.compile(r"\b(?:can|could)\s+(?:still\s+|also\s+)?be\s+(?:seen|felt|observed|found)\b", re.I)
ENABLING = re.compile(r"\b(?:allow|allows|allowed|allowing|enable|enables|enabled|enabling|let|lets|letting|permit|"
                      r"permits|permitted|permitting)\b", re.I)
HEDGES = {'possibly': 'maybe', 'perhaps': 'maybe', 'potentially': 'maybe', 'likely': 'maybe', 'probably': 'maybe',
          'arguably': 'maybe', 'purportedly': 'doubt', 'allegedly': 'doubt', 'supposedly': 'doubt',
          'reportedly': 'doubt', 'apparently': 'doubt', 'seemingly': 'doubt'}
# How often a claim holds: "tended to score higher" is not "scored higher", nor "often described as" "considered"
FREQUENCY = {**dict.fromkeys(('often', 'usually', 'generally', 'typically', 'commonly', 'tend', 'tends', 'tended',
                              'tending'), 'often'),
             **dict.fromkeys(('sometimes', 'occasionally'), 'sometimes'),
             **dict.fromkeys(('rarely', 'seldom'), 'rarely')}
FREQUENCY_PHRASES = [('often', re.compile(r"\b(?:in\s+most\s+cases|more\s+often\s+than\s+not|as\s+a\s+rule)\b", re.I)),
                     ('sometimes', re.compile(r"\b(?:in\s+some\s+(?:cases|instances)|at\s+times|now\s+and\s+then|from\s+time\s+to\s+time)\b", re.I)),
                     ('rarely', re.compile(r"\b(?:in\s+rare\s+(?:cases|instances)|hardly\s+ever|in\s+few\s+cases)\b", re.I))]
FREQUENCY_LICENSES = {**dict.fromkeys(('common', 'usual', 'typical', 'frequent', 'regular', 'normal', 'general', 'routine',
                                        'likely', 'likelihood', 'probability', 'chance', 'chances', 'frequently',
                                        'regularly', 'normally', 'popular', 'standard', 'customary'), 'often'),
                      **dict.fromkeys(('occasional', 'periodic', 'intermittent', 'occasions'), 'sometimes'),
                      **dict.fromkeys(('rare', 'uncommon', 'infrequent', 'unusual', 'exceptional'), 'rarely')}
# Doing something unaided: "colds go away on their own" is not "colds go away"
UNAIDED = re.compile(r"\b(?:on\s+(?:its|their|his|her|my|our|your)\s+own|by\s+(?:itself|themselves|himself|herself|myself|"
                     r"ourselves|yourself|yourselves)|alone)\b", re.I)
# Whose words a sentence reports: "residents said the detour would add 20 minutes" is not a fact the writer states
REPORTING_VERBS = {'say', 'tell', 'report', 'state', 'claim', 'warn', 'explain', 'announce', 'insist', 'argue', 'write',
                   'admit', 'caution', 'declare', 'contend', 'add', 'reply', 'complain', 'predict', 'believe', 'think',
                   'expect', 'assume', 'estimate', 'anticipate', 'maintain', 'allege', 'testify', 'recall', 'promise',
                   'mention', 'note', 'stress', 'emphasize', 'emphasise', 'confirm', 'reveal', 'disclose', 'acknowledge',
                   'point', 'remark', 'observe', 'comment', 'assert', 'concede', 'reckon'}
AS_REPORTED = re.compile(r"\bas\s+(?:stated|reported|noted|said|claimed|explained|announced|argued|told)\b", re.I)
# How strong the evidence for a claim is: "studies suggest" is not "studies show", nor "experts believe", and
# "is linked to" is not "causes". Counted only where the verb reports a claim ("suggest that ...", "shown to").
EVIDENCE = {**dict.fromkeys(('suggest', 'indicate', 'imply', 'appear', 'seem'), 'tentative'),
            **dict.fromkeys(('show', 'demonstrate', 'prove', 'confirm', 'establish', 'find', 'reveal'), 'shown'),
            **dict.fromkeys(('believe', 'think', 'expect', 'assume', 'argue', 'claim', 'contend', 'anticipate',
                             'predict', 'estimate', 'project', 'forecast'), 'opinion'),
            **dict.fromkeys(('link', 'associate', 'correlate', 'relate'), 'correlation')}
# A "relationship between X and Y" is a correlation; "the impact of X on Y" would make it a cause
CORRELATION_NOUNS = {'association', 'correlation', 'link', 'relationship', 'relation', 'connection'}
ATTRIBUTION = re.compile(r"\b(?:according\s+to|as\s+per)\b", re.I)
# How much of something: "many others" is not "hundreds of others", "some" is not "all", "most" is not "many",
# "some patients" is not "patients", "partly" is not "wholly"
QUANTIFIERS = {'all': 'all', 'every': 'all', 'each': 'all', 'always': 'all', 'everyone': 'all', 'everything': 'all',
               'everybody': 'all', 'universally': 'all', 'unanimously': 'all', 'entirety': 'all', 'population': 'all',
               'only': 'only', 'solely': 'only', 'just': 'only', 'merely': 'only', 'mere': 'only', 'sole': 'only',
               'exclusively': 'only',
               'many': 'many', 'numerous': 'many', 'countless': 'many', 'lot': 'many', 'lots': 'many', 'plenty': 'many',
               'majority': 'most', 'few': 'few', 'partly': 'part', 'partially': 'part',
               'several': 'some',
               'hundreds': 'specific', 'thousands': 'specific', 'millions': 'specific', 'billions': 'specific',
               'dozens': 'specific'}
# A whole amount: "its entire database" is "all of its database"
WHOLE = {'entire', 'whole', 'complete'}
# "all" in an idiom says nothing about how many ("all over the world", "after all")
IDIOM_ALL = re.compile(r"\ball\s+(?:over|around|across|along|through)\s+the\b|\ball\s+without\b|\ball\s+in\s+all\b|"
                       r"\bonce\s+and\s+for\s+all\b|\b(?:at|above|after)\s+all\b", re.I)
# Bounds on a number: "up to 15 percent" is not "15 percent", "at least three" is not "three", "longer than three
# days" is not "shorter than three days", "approx. 3 kg" is not "3 kg". "no more than" is an upper bound only, and
# "over 12 weeks" after a verb is a span of time unless it says how long something lasted ("for over 12 weeks").
TIME_UNIT = r"(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?|decades?|centuries|century)"
_COUNT = r"(?:\d[\d,.]*|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty)"
_NUM = (r"(?:" + _COUNT + r"|hundred|thousand|million|billion|a\s+(?:third|quarter|half|dozen|hundred|thousand|million)|"
        r"an?\s+" + TIME_UNIT + r")\b")
BOUNDS = [
    ('max', re.compile(r"\b(?:up\s+to|as\s+(?:much|many|high|low|long|little|few)\s+as|a\s+(?:maximum|max\.)\s+of|at\s+most|"
                       r"no\s+(?:more|longer|later|greater|higher)\s+than|(?:limited|capped)\s+(?:to|at))\s+" + _NUM
                       + r"|\bor\s+(?:less|fewer|younger|under|below)\b", re.I)),
    ('min', re.compile(r"\b(?:at\s+least|a\s+(?:minimum|min\.)\s+of|no\s+(?:less|fewer|shorter|earlier|lower)\s+than|"
                       r"(?<!\bno\s)(?:more|longer|later|older|greater|higher|larger)\s+than|upwards\s+of|in\s+excess\s+of|"
                       r"above|beyond)\s+" + _NUM
                       + r"|\bover\s+(?!" + _COUNT + r"\s+" + TIME_UNIT + r")" + _NUM
                       + r"|\b" + _NUM + r"\s+(?:or|and)\s+(?:more|older|over|above|up|upwards|longer|later)\b"
                       + r"|\bor\s+more\b", re.I)),
    ('under', re.compile(r"\b(?:(?<!\bno\s)(?:less|fewer|shorter|earlier|younger|smaller|lower)\s+than|under|below)\s+" + _NUM,
                         re.I)),
    ('approx', re.compile(r"\b(?:nearly|almost|approximately|approx\.|roughly|about|around|ca\.|circa|close\s+to|some|"
                          r"an\s+estimated)\s+" + _NUM, re.I)),
    ('only', re.compile(r"(?<!\bnot\s)\b(?:only|just|merely|a\s+mere)\s+" + _NUM, re.I)),
]
# "over 12 weeks" bounds a number only when nothing but a function word or a duration verb comes before it
# ("for over 12 weeks", "took over 12 weeks", "and over ten years had passed"); after a content word it is the
# span something happened in ("lost 4.2 kilograms over 12 weeks")
OVER_TIME = re.compile(r"\bover\s+" + _COUNT + r"\s+" + TIME_UNIT + r"\b", re.I)
DURATION_WORDS = {'last', 'lasts', 'lasted', 'lasting', 'take', 'takes', 'took', 'taking', 'taken', 'span', 'spans',
                  'spanned', 'spanning', 'run', 'runs', 'ran', 'running', 'need', 'needs', 'needed', 'require', 'requires',
                  'required', 'wait', 'waited', 'stay', 'stayed', 'persist', 'persists', 'persisted'}
# Stronger claims a rewrite may not add: absolutes, perfection, fame, certainty
STRONGER = {**dict.fromkeys(('completely', 'totally', 'entirely', 'fully', 'wholly', 'utterly', 'absolutely'), 'all'),
            **dict.fromkeys(('perfect', 'perfectly', 'ideal', 'ideally', 'flawless', 'flawlessly'), 'perfect'),
            **dict.fromkeys(('renowned', 'famous', 'famed', 'acclaimed', 'celebrated', 'well-known', 'prestigious',
                             'world-class', 'world-famous', 'world-renowned', 'legendary', 'iconic', 'recognized',
                             'recognised'), 'fame'),
            **dict.fromkeys(('undoubtedly', 'definitely', 'proven', 'certainly', 'surely'), 'certain')}
# A definition or example the original doesn't give ("Green algae, also known as chlorophyll", "Green pigment,
# or chlorophyll, is ...")
GLOSS = re.compile(r"\b(?:(?:also\s+)?(?:known|referred\s+to|described)\s+as|also\s+called|specifically|namely|i\.e\.|"
                   r"in\s+other\s+words|a\s+(?:type|kind|form|sort)\s+of)\b", re.I)
OR_GLOSS = re.compile(r"\b\w+,\s+or\s+\w+(?:\s+\w+)?,", re.I)
# How strongly something is recommended: "is essential" is not "is advisable", nor "is optional"
STRENGTH = {**dict.fromkeys(('essential', 'necessary', 'crucial', 'vital', 'critical', 'imperative', 'indispensable',
                             'required', 'key', 'important', 'central', 'fundamental', 'paramount', 'needed',
                             'integral', 'pivotal', 'pressing', 'urgent', 'serious', 'significant', 'major'), 'needed'),
            **dict.fromkeys(('advisable', 'recommended', 'helpful', 'useful', 'beneficial', 'wise', 'worthwhile',
                             'valuable', 'sensible', 'desirable', 'preferable'), 'helpful'),
            **dict.fromkeys(('optional', 'unnecessary', 'unimportant', 'minor', 'trivial', 'negligible', 'insignificant',
                             'irrelevant', 'secondary'), 'optional')}
# Words and their opposites. A rewrite may not bring in a word whose opposite the original uses, unless the
# original uses both ("absorb" -> "release", "began" -> "ended", "frequent" -> "rare", "rising" -> "falling")
GOING_UP = {'increase', 'raise', 'boost', 'grow', 'growth', 'rise', 'expand', 'enlarge', 'extend', 'heighten', 'intensify',
            'maximize', 'maximise', 'gain', 'accelerate', 'climb', 'soar', 'surge', 'widen', 'double', 'triple', 'multiply',
            'swell', 'spike', 'jump', 'hike', 'escalate', 'amplify', 'inflate', 'balloon', 'skyrocket', 'hoist', 'lift'}
GOING_DOWN = {'decrease', 'reduce', 'reduction', 'lower', 'cut', 'shrink', 'fall', 'decline', 'drop', 'diminish', 'lessen',
              'minimize', 'minimise', 'shorten', 'slow', 'plunge', 'plummet', 'loss', 'narrow', 'squeeze', 'erode', 'trim',
              'curb', 'slash', 'halve', 'dwindle', 'sink', 'tumble', 'dip', 'contract'}
ENABLE_WORDS = {'allow', 'permit', 'enable', 'let', 'help', 'facilitate'}
RESTRICT_WORDS = {'prevent', 'forbid', 'prohibit', 'ban', 'block', 'restrict', 'restricted', 'limit', 'limited', 'hinder',
                  'impede', 'constrain', 'stop', 'bar'}
OPPOSITES = [
    ({'begin', 'start', 'commence', 'initiate', 'launch'}, {'end', 'finish', 'conclude', 'cease', 'terminate'}, 'VERB'),
    ({'use', 'apply', 'adopt', 'employ', 'keep'}, {'dispose', 'discard', 'abandon', 'scrap', 'drop'}, 'VERB'),
    ({'absorb'}, {'release', 'emit'}),
    ({'connect', 'link', 'join', 'unite', 'attach'}, {'separate', 'divide', 'split', 'disconnect', 'detach'}),
    (GOING_UP, GOING_DOWN),
    ({'improve', 'improvement'}, {'worsen', 'deteriorate', 'deterioration'}),
    ({'strengthen'}, {'weaken'}),
    ({'win', 'victory'}, {'lose', 'defeat'}),
    ({'succeed', 'success', 'successful', 'pass'}, {'fail', 'failure', 'unsuccessful'}),
    ({'accept', 'approve', 'adopt', 'admit'}, {'reject', 'refuse', 'decline', 'deny', 'veto', 'oppose'}),
    ({'include', 'contain'}, {'exclude', 'omit'}),
    (ENABLE_WORDS, RESTRICT_WORDS),
    ({'add', 'insert'}, {'remove', 'delete', 'subtract', 'withdraw'}),
    ({'open'}, {'close', 'shut'}),
    ({'arrive', 'enter'}, {'leave', 'depart', 'exit'}),
    ({'buy', 'purchase'}, {'sell'}),
    ({'borrow'}, {'lend'}),
    ({'import'}, {'export'}),
    ({'remember'}, {'forget'}),
    ({'agree', 'support', 'favor', 'favour', 'endorse'}, {'disagree', 'oppose', 'object'}),
    ({'appear'}, {'disappear', 'vanish'}),
    ({'heat', 'warm', 'hot'}, {'cool', 'chill', 'cold'}),
    ({'high', 'higher', 'highest'}, {'low', 'lower', 'lowest'}),
    ({'more', 'most', 'greater', 'greatest'}, {'less', 'least', 'fewer', 'fewest'}),
    ({'better', 'best'}, {'worse', 'worst'}),
    ({'strong', 'stronger', 'strongest'}, {'weak', 'weaker', 'weakest'}),
    ({'fast', 'quick', 'rapid', 'rapidly', 'quickly', 'swift', 'swiftly'}, {'slow', 'slowly', 'gradual', 'gradually'}),
    ({'early', 'earlier', 'earliest'}, {'late', 'later', 'latest'}),
    ({'large', 'big', 'huge', 'vast', 'enormous', 'massive', 'substantial', 'considerable', 'sizable', 'sizeable'},
     {'small', 'little', 'tiny', 'minor', 'modest', 'slight', 'negligible'}),
    ({'long', 'longer', 'lengthy'}, {'short', 'shorter', 'brief'}),
    ({'wide', 'widen', 'broad', 'broaden'}, {'narrow'}),
    ({'rich', 'wealthy'}, {'poor'}),
    ({'easy', 'easily', 'simple'}, {'hard', 'difficult', 'difficulty'}),
    ({'safe'}, {'dangerous', 'unsafe', 'risky', 'hazardous'}),
    ({'positive'}, {'negative'}),
    ({'frequent', 'frequently', 'common', 'commonly', 'often', 'usual'}, {'rare', 'rarely', 'uncommon', 'infrequent',
                                                                          'seldom', 'unusual'}),
    ({'consistent'}, {'inconsistent', 'mixed'}),
    ({'true', 'correct', 'accurate'}, {'false', 'incorrect', 'wrong', 'inaccurate'}),
    ({'inside', 'indoor', 'indoors', 'internal', 'domestic'}, {'outside', 'outdoor', 'outdoors', 'external', 'foreign'}),
    ({'beneficial', 'helpful', 'useful'}, {'harmful', 'damaging', 'detrimental'}),
    ({'cheap', 'inexpensive', 'affordable'}, {'expensive', 'costly', 'pricey'}),
    ({'profit', 'profitable'}, {'loss', 'unprofitable'}),
    ({'majority'}, {'minority'}),
    ({'maximum'}, {'minimum'}),
    ({'first'}, {'last', 'final'}),
    ({'north', 'northern'}, {'south', 'southern'}),
    ({'east', 'eastern'}, {'west', 'western'}),
    ({'top'}, {'bottom'}),
    ({'together', 'jointly'}, {'apart', 'separately'}),
    ({'wet', 'humid'}, {'dry'}),
    ({'old', 'older', 'ancient', 'elderly', 'aged'}, {'young', 'younger', 'new', 'modern', 'youthful', 'recent',
                                                      'contemporary', 'cutting-edge', 'state-of-the-art', 'up-to-date'}),
    ({'public'}, {'private'}),
    ({'full'}, {'empty'}),
    ({'urban'}, {'rural'}),
    ({'alive', 'living'}, {'dead', 'deceased'}),
]
OPPOSITE_WORDS = set().union(*(group[0] | group[1] for group in OPPOSITES))
# A court, a parliament or a museum "in session" is open ("is closed on Mondays" -> "Mondays are the days when the
# museum is in session")
IN_SESSION = re.compile(r"\bin\s+session\b", re.I)
# Where something stands: "two points behind Uruguay" is not "a two-point lead over Uruguay", nor "below
# expectations" "above" them. The noun "lead" and the verb "trail" always say where something stands; the verb
# "lead" (which also means "cause" or "guide") and a "trail" the tagger took for a noun only with a margin ("by
# three points"), a score ("led 2-1"), a table ("have led the league", "trailed in the polls") or, for the verb,
# no object at all ("Brazil led at half-time"); "clear" only with a margin ("five points clear of both clubs"),
# and "top" or "head" only with a table or a margin ("Arsenal top the table").
POSITION_BEHIND = {'behind', 'deficit', 'adrift', 'below'}
POSITION_AHEAD = {'ahead', 'above', 'surplus'}
STANDINGS_NOUNS = {'table', 'standings', 'league', 'group', 'division', 'conference', 'race', 'field', 'pack', 'poll',
                   'ranking', 'chart', 'leaderboard', 'market', 'series'}
POSITION_MARGINS = {'point', 'goal', 'margin', 'run', 'vote', 'length', 'stroke', 'seat', 'game', 'lap', 'shot'}
TABLE_TOPPING_VERBS = {'top', 'head'}
# Prepositions that give the verb "lead" the sense of "go" or "cause" ("leads to the river", "led into a recession")
LEAD_ROUTES = {'to', 'into', 'toward', 'towards', 'through', 'from', 'onto', 'up', 'down', 'away', 'off'}
# Words that add a claim the original doesn't make ("optimal health", "had successfully killed", "a preliminary
# study", "a different value", "the late Henri Becquerel")
ADDED_CLAIMS = {'optimal', 'optimally', 'unique', 'uniquely', 'definite', 'definitively', 'specific', 'initial',
                'actual', 'preliminary', 'relatively', 'effective', 'effectively', 'successful', 'successfully',
                'ultimate', 'officially', 'official', 'deliberately', 'intentionally', 'directly', 'permanently',
                'permanent', 'forever', 'guaranteed'}
# ...unless the original says as much in other words
CLAIM_EQUIVALENTS = {'specific': {'particular', 'certain', 'precise', 'exact', 'given', 'specified', 'set', 'fixed',
                                  'defined', 'individual', 'each'},
                     'initial': {'first', 'early', 'original', 'opening', 'start', 'begin', 'beginning'},
                     'effective': {'efficient', 'work', 'successful', 'useful', 'helpful', 'help', 'well', 'effect',
                                   'powerful', 'efficiently'},
                     'effectively': {'efficiently', 'work', 'successfully', 'well', 'help', 'effect'},
                     'successful': {'success', 'succeed', 'manage', 'achieve', 'effective'},
                     'successfully': {'success', 'succeed', 'manage', 'achieve', 'effectively'},
                     'directly': {'direct', 'straight', 'immediately', 'right'},
                     'optimal': {'best', 'ideal', 'optimum'}, 'ultimate': {'final', 'eventual', 'end'},
                     'relatively': {'fairly', 'rather', 'somewhat', 'comparatively', 'quite'}}
# Words that say how much: a rewrite may trade one for another, but not bring one in ("significantly lessened",
# "an enormous safety net") where the original has none
INTENSITY_WORDS = {'significantly', 'substantially', 'considerably', 'dramatically', 'drastically', 'greatly', 'markedly',
                'vastly', 'hugely', 'enormously', 'tremendously', 'sharply', 'strongly', 'highly', 'extremely',
                'incredibly', 'remarkably', 'exceptionally', 'profoundly', 'severely', 'massively', 'immensely',
                'heavily', 'seriously', 'deeply', 'radically', 'fundamentally', 'exponentially', 'very', 'really',
                'so', 'much', 'far', 'lot', 'significant', 'substantial', 'considerable', 'dramatic', 'drastic', 'huge',
                'enormous', 'massive', 'vast', 'tremendous', 'immense', 'profound', 'severe', 'sharp', 'marked',
                'major', 'great', 'large', 'big', 'high', 'serious', 'heavy', 'strong', 'deep', 'intense',
                'extensive', 'widespread', 'rapid', 'rapidly', 'steep', 'steeply', 'notable', 'notably', 'remarkable',
                'powerful', 'especially', 'particularly', 'vital', 'crucial', 'essential', 'critical', 'pivotal',
                'important', 'key', 'pressing', 'urgent', 'impressive', 'exceptional', 'excellent', 'outstanding'}
INTENSIFIERS = {'significantly', 'substantially', 'considerably', 'dramatically', 'drastically', 'greatly', 'markedly',
                'vastly', 'hugely', 'enormously', 'tremendously', 'sharply', 'strongly', 'extremely', 'incredibly',
                'remarkably', 'exceptionally', 'profoundly', 'severely', 'massively', 'immensely', 'heavily',
                'radically', 'exponentially', 'significant', 'substantial', 'considerable', 'dramatic', 'drastic',
                'huge', 'enormous', 'massive', 'vast', 'tremendous', 'immense', 'profound', 'severe'}
# A direction ("reduce the risk") that a rewrite turns into a mere effect ("have an impact on the risk")
DIRECTED = GOING_UP | GOING_DOWN
UNDIRECTED = {'affect', 'impact', 'influence', 'effect', 'alter', 'change', 'shape', 'modify'}
# An expectation is not its outcome: "analysts had expected higher earnings" says nothing of their being met
EXPECTING = {'expect', 'anticipate', 'forecast', 'predict', 'project', 'hope', 'estimate', 'expectation', 'forecast',
             'prediction', 'projection', 'estimate'}
OUTCOMES = {'meet', 'exceed', 'surpass', 'beat', 'fulfil', 'fulfill', 'match', 'miss', 'outperform', 'underperform',
            'top', 'disappoint', 'confirm'}
# Cause and effect a rewrite may not bring in ("..., resulting in a drop", "By approving a buyback, the board
# raised ...") or turn around ("costs rose, which narrowed the margin" -> "the margin narrowed, resulting in
# higher costs")
CAUSAL_LINK = re.compile(r"\b(?:result(?:s|ed|ing)?\s+in|lead(?:s|ing)?\s+to|led\s+to|due\s+to|because(?:\s+of)?|"
                         r"attributed\s+to|thanks\s+to|owing\s+to|as\s+a\s+(?:result|consequence)|consequently|"
                         r"therefore|thus|hence|caus(?:e|es|ed|ing)|so\s+that|which\s+(?:means|meant)|forc(?:e|es|ed|ing)|"
                         r"prompt(?:s|ed|ing)?|spur(?:s|red|ring)?|trigger(?:s|ed|ing)?|fuel(?:s|ed|led|ing|ling)?|"
                         r"driv(?:e|es|en|ing)|drove)\b", re.I)
CAUSE_NP = re.compile(r"\b(?:due\s+to|because\s+of|thanks\s+to|owing\s+to|attributed\s+to)\s+", re.I)
EFFECT_AFTER = re.compile(r"\b(?:result(?:s|ed|ing)?\s+in|lead(?:s|ing)?\s+to|led\s+to|caus(?:e|es|ed|ing)(?!\s+by\b))\s+",
                          re.I)
# A link that names the cause after the effect ("rose because of the new machines", "driven by demand")
CAUSE_AFTER = re.compile(r"(?:due\s+to|because|thanks\s+to|owing\s+to|attributed\s+to|"
                         r"as\s+a\s+(?:result|consequence)\s+of)\b|\w+\s+by\b", re.I)
BY_DOING = re.compile(r"^\W*by\s+\w+ing\b", re.I)
# The original already relates cause and effect: a causal word, ", which" + verb, or a participle clause
CAUSAL_WORDS = {'because', 'due', 'thanks', 'therefore', 'thus', 'hence', 'consequently', 'owing', 'reason', 'why',
                'effect', 'impact', 'consequence', 'outcome', 'result', 'responsible', 'so', 'since'}
CAUSAL_VERBS = {'cause', 'lead', 'result', 'drive', 'prompt', 'trigger', 'spark', 'fuel', 'contribute', 'produce',
                'generate', 'bring', 'make', 'help', 'allow', 'enable', 'let', 'force', 'mean', 'explain', 'stem',
                'arise', 'affect', 'impact', 'influence', 'create', 'attribute', 'determine', 'leave', 'push', 'keep'}
# A participle that states an effect of what comes before ("The bridge was inaugurated in 1883, making it the longest")
PARTICIPLE_EFFECT = re.compile(r",\s+(?:making|leaving|turning|causing|leading|resulting)\b", re.I)
# Linking words that relate the sentence to the one before: a rewrite may not bring in a contrast, an example or
# an addition the original doesn't make ("However, the authors warn ...", "For example, a typical approach ...")
CONTRAST_OPENERS = re.compile(r"^\W*(?:however|but|yet|nevertheless|nonetheless|still|conversely|even\s+so|that\s+said|"
                              r"on\s+the\s+other\s+hand|in\s+contrast|by\s+contrast|by\s+comparison|instead|"
                              r"despite\s+this|in\s+spite\s+of\s+this)\b", re.I)
CONTRAST_INSIDE = re.compile(r"\b(?:however|but|yet|nevertheless|nonetheless|conversely|although|though|whereas|while|"
                             r"despite|unlike|instead|rather\s+than|on\s+the\s+other\s+hand|in\s+contrast|by\s+contrast|"
                             r"even\s+so|that\s+said|still|except|whilst|albeit|only\s+to)\b", re.I)
# "Operating costs, however, increased ..." contrasts with the sentence before as much as an opening "However,"
CONTRAST_ASIDE = re.compile(r",\s*(?:however|though|nevertheless|nonetheless|by\s+contrast|in\s+contrast|on\s+the\s+other\s+hand)\s*,",
                            re.I)
# a contrast the rewrite brings in (not "while", "yet" or "still", which are often about time)
CONTRAST_ADDED = re.compile(r"\b(?:however|but|nevertheless|nonetheless|conversely|although|though|whereas|despite|"
                            r"in\s+spite\s+of|on\s+the\s+other\s+hand|in\s+contrast|by\s+contrast|even\s+so|that\s+said)\b",
                            re.I)
EXAMPLE_OPENERS = re.compile(r"^\W*(?:for\s+example|for\s+instance|e\.g\.|as\s+an\s+example|to\s+illustrate)\b", re.I)
EXAMPLE_INSIDE = re.compile(r"\b(?:for\s+example|for\s+instance|e\.g\.|such\s+as|like|including|as\s+an\s+example)\b", re.I)
ADDING_OPENERS = re.compile(r"^\W*(?:also|in\s+addition|additionally|moreover|furthermore|besides|plus|what['’]s\s+more|"
                            r"as\s+part\s+of|on\s+top\s+of|beyond\s+(?:that|this)|further)\b", re.I)
ADDING_INSIDE = re.compile(r"\b(?:also|in\s+addition|additionally|moreover|furthermore|besides|plus|as\s+well|too|"
                           r"what['’]s\s+more|on\s+top\s+of|beyond\s+(?:that|this)|further|another|again)\b", re.I)
AFTER_THAT = re.compile(r"\bthat\s+(?:however|moreover|furthermore|therefore|thus|hence|nevertheless|nonetheless|"
                        r"additionally|consequently|meanwhile|also|still|besides)\s*,", re.I)
# A time limit a rewrite may not bring into a sentence that orders nothing in time ("When the bridge opened in
# 1883, it was the longest" -> "Until its opening in 1883, the bridge was the longest")
TIME_LIMITS = {'until', 'till', 'before'}
TIME_ORDER_WORDS = {'until', 'till', 'before', 'after', 'then', 'later', 'first', 'once', 'prior', 'since', 'previously',
                    'earlier', 'subsequently', 'afterwards', 'afterward', 'precede', 'precedes', 'preceded', 'preceding'}
# ...and phrases that may, but only by the thing they name ("Ahead of the royal visit" -> "Before the royal visit",
# "In the years leading up to the war", "in the run-up to")
TIME_ORDER_PHRASES = re.compile(r"\b(?:ahead\s+of|(?:lead|leads|leading|led)\s+up\s+to|(?:run|lead|build)-up\s+to|"
                                r"on\s+the\s+eve\s+of|in\s+(?:advance|anticipation)\s+of|in\s+preparation\s+for)\b", re.I)
# Who gives and who receives: "the app does not collect data from its users" is not "users are not given data by
# the app", and "Fleming, Florey and Chain shared the prize" is not "the prize was awarded by Fleming ..."
GIVING = {'give', 'provide', 'supply', 'offer', 'send', 'award', 'grant', 'present', 'bestow', 'confer', 'donate',
          'hand', 'deliver', 'lend', 'sell', 'issue', 'transfer', 'pass', 'distribute', 'allocate', 'assign'}
TAKING = {'receive', 'get', 'obtain', 'win', 'earn', 'gain', 'accept', 'collect', 'gather', 'take', 'request', 'acquire',
          'buy', 'borrow', 'share', 'secure', 'inherit', 'purchase', 'extract', 'harvest', 'afford'}
# Being unable to afford something is not being unable to get it ("unable to afford treatment" -> "unable to
# receive treatment")
AFFORD_SWAPS = {'receive', 'get', 'obtain', 'access', 'gain'}
NUMBER_RUN = re.compile(r'\d+(?:[.,]\d+)*')
# The T5 attaches the full stop to the last word; a stop on its own marks a sample that went wrong ("the new .")
END_GAP = re.compile(r'\s[.!?]["”’)\]]?\s*$')
# Who does what to whom: an argument's role for a verb both versions use ("Japan offers something for everyone"
# is not "Everyone has something to offer in Japan")
ROLE_PREPS = {'to': 'R', 'into': 'R', 'onto': 'R', 'toward': 'R', 'towards': 'R', 'for': 'R', 'from': 'S'}
ROLE_PRONOUNS = {'everyone', 'everybody', 'someone', 'somebody', 'anyone', 'anybody', 'nobody', 'everything',
                 'something', 'anything', 'nothing'}
ROLE_SKIP = {'be', 'have', 'do', 'get', 'make', 'let', 'become', 'seem', 'meet', 'marry', 'resemble', 'border', 'match',
             'equal', 'join', 'combine', 'merge', 'mix', 'connect', 'link', 'share', 'exchange', 'trade', 'compare'}
# Who is speaking or addressed: "our duties" may become "the duties we have", but not "your duties"
PERSONAL = {'i': 'i', 'me': 'i', 'my': 'i', 'mine': 'i', 'we': 'we', 'us': 'we', 'our': 'we', 'ours': 'we',
            'you': 'you', 'your': 'you', 'yours': 'you'}
# Describing words that don't pin the meaning down, so a rewrite may drop or vary them ("a common approach"
# -> "a frequently used technique"); others ("remote work", "chronic diseases", "choose carefully") must stay
LIGHT_DESCRIBERS = {
    'common', 'usual', 'typical', 'popular', 'standard', 'general', 'main', 'key', 'good', 'great', 'new', 'different',
    'various', 'several', 'certain', 'specific', 'particular', 'other', 'such', 'whole', 'entire', 'simple',
    'basic', 'overall', 'real', 'true', 'many', 'much', 'only', 'very', 'really', 'truly', 'actually', 'simply',
    'clearly', 'certainly', 'significantly', 'greatly', 'highly', 'particularly', 'especially', 'generally', 'usually',
    'typically', 'increasingly', 'effectively', 'successfully', 'ultimately',
    'additionally', 'finally', 'currently', 'recently', 'modern', 'current', 'recent', 'meaningful', 'powerful',
    'innovative', 'engaging', 'strong', 'important', 'significant', 'major', 'notable', 'remarkable', 'incredible',
    'incredibly', 'extremely', 'truly', 'deeply', 'fully', 'completely', 'totally', 'quite', 'rather', 'fairly'}

MONTHS = {'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
          'november', 'december', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec'}
# A rewrite the model stopped mid-sentence: it ends on a comma, colon, joining word or article
TRUNCATED = re.compile(r"(?:[,;:]|\b(?:and|or|but|nor|the|a|an))\s*[.!?]?\s*$", re.I)

def odd_marks(text):
    """Punctuation beyond commas, full stops, apostrophes and hyphens, including "--", "..." and a full stop
    standing alone ("within a 14 . days")."""
    marks = {c for c in text if not (c.isalnum() or c.isspace() or c in ",.'’-")}
    return marks | {seq for seq in ('--', '..', ' . ') if seq in text}

# "not only X but also Y" adds rather than negates, and so does "not solely"
ADDITIVE_NOT = ('only', 'just', 'merely', 'simply', 'solely', 'purely', 'exclusively')
# "no more than two hours" is a bound, not a negation
BOUND_AFTER_NO = ('more', 'less', 'fewer', 'later', 'earlier', 'greater', 'higher', 'longer', 'shorter', 'lower')

def negations(words):
    """Explicit negators in a list of lowercase words. "unless" is one too ("if not"): "do not take more unless a
    doctor tells you to" can't become "if a doctor tells you to, do not take more"."""
    return sum(w == 'unless' or ((w in NEGATORS or w.endswith(("n't", "n’t")))
                                 and not (w == 'not' and k + 1 < len(words) and words[k + 1] in ADDITIVE_NOT)
                                 and not (w == 'no' and k + 2 < len(words) and words[k + 1] in BOUND_AFTER_NO
                                          and words[k + 2] == 'than'))
               for k, w in enumerate(words))

def soft_negations(tokens):
    """Verbs that negate what follows without a "not" ("prevents X from rising", "avoid taking", "bans smoking",
    "failed to address", "refuse to speak", "discontinue the medication")."""
    n = 0
    for t in tokens:
        if t.pos_ != 'VERB':
            continue
        lemma = t.lemma_.lower()
        if lemma in ALWAYS_NEGATING:
            n += 1
        elif lemma in FROM_NEGATORS and (any(c.dep_ == 'prep' and c.lower_ == 'from' for c in t.children) or any(
                w.lower_ == 'from' and w.i + 1 < len(t.doc) and t.doc[w.i + 1].tag_ == 'VBG' for w in t.doc[t.i + 1: t.i + 8])):
            n += 1
        elif lemma in CEASING and not (lemma == 'end' and any(c.dep_ == 'prt' for c in t.children)) and any(
                (c.dep_ == 'xcomp' and c.pos_ in ('VERB', 'AUX')) or (c.dep_ == 'dobj' and c.lemma_.lower() in USE_NOUNS)
                for c in t.children):
            n += 1
    return n

def negating_verbs(tokens):
    """The verbs soft_negations() counts."""
    return [t for t in tokens if t.pos_ == 'VERB' and soft_negations([t])]

def negated_stem(word):
    """The word a negating prefix is attached to ("impossible" -> "possible", "unchanged" -> "changed"), or
    None if the word isn't a negated form."""
    if word in NOT_NEGATED or word == 'unless':
        return None
    for prefix in ('non', 'dis', 'un', 'in', 'im', 'il', 'ir'):
        if word.startswith(prefix) and len(word) - len(prefix) >= 4:
            stem = word[len(prefix):]
            lemmas = getAllLemmas(stem)
            if {'ADJ', 'ADV'} & set(lemmas) or (prefix in ('un', 'non') and stem.endswith('ed') and 'VERB' in lemmas):
                return stem
    return None

def modal_class(word):
    word = word.replace('’', "'")
    if word in ('unable', 'incapable'):
        return 'can'        # "unable to afford" is "cannot afford"
    if word.endswith("n't"):
        word = {"can't": 'can', "won't": 'will', "shan't": 'shall'}.get(word, word[:-3])
    return MODALITY.get(word)

# "so that they could run on the new platform" states a purpose, not a possibility: the modal makes no claim
PURPOSE_MODAL = re.compile(r"\b((?:so(?:\s+that)?|in\s+order\s+that|to\s+(?:ensure|make\s+sure)(?:\s+that)?)\s+\w+)\s+"
                           r"(?:could|can|would|might|may)\b", re.I)

def acts_as_verb(token):
    """True for a verb, or for a "noun" at the root with a subject or a clause of its own, which is a verb the
    tagger misread ("The government's data show that ...", "The council installs and repairs the lights")."""
    return token.pos_ in ('VERB', 'AUX') or (token.pos_ == 'NOUN' and token.dep_ == 'ROOT' and any(
        c.dep_ in ('nsubj', 'ccomp') for c in token.children))

def obliging(token):
    """True if a word lays an obligation on someone: "requires employers to give", "obliges firms to report",
    "mandates that", "is mandatory", "made compulsory". "require ... to" does so only when the action has an object
    of its own ("requires landlords to fix faults"); in "requires a permit to fish" and "require sunlight to grow"
    the object is what is needed."""
    lemma = token.lemma_.lower()
    if lemma in OBLIGING_PREDICATES:
        return token.dep_ in ('acomp', 'oprd', 'attr') or (token.dep_ == 'conj' and token.head.dep_ in ('acomp', 'oprd'))
    return lemma in OBLIGING_VERBS and acts_as_verb(token) and token.dep_ not in ('acl', 'amod') and any(
        (c.dep_ in ('xcomp', 'advcl') and c.pos_ in ('VERB', 'AUX') and any(g.tag_ == 'TO' for g in c.children)
         and (lemma != 'require' or any(g.dep_ in ('dobj', 'dative', 'ccomp') for g in c.children)))
        or (c.dep_ == 'ccomp' and any(g.lower_ == 'that' for g in c.children)) for c in token.children)

def modality(text, tokens):
    """(modal classes, hedge classes) of a sentence and its parsed `tokens`; "can be seen" and "so that they could"
    make no modal claim."""
    text = PURPOSE_MODAL.sub(r'\1', SEEN.sub('be seen', text))
    text = re.sub(r"\b(?:not|never|n['’]t)\s+(?:be\s+)?(?:able|capable)\b", ' unable', text, flags=re.I)
    words = [w.lower() for w in plain_words(text)]
    obliged = OBLIGATION.search(text) or any(map(obliging, tokens))
    return ({c for c in map(modal_class, words) if c} | ({'must'} if obliged else set()),
            {HEDGES[w] for w in words if w in HEDGES})

def needs_stated(doc):
    """(who or what needs it, what is needed) for each need the main clause states: "Fishing requires a permit" ->
    ({fish, river, ...}, {permit}), "A permit is needed / necessary / a prerequisite" -> (set(), {permit}), "It is
    necessary to have a permit" -> (set(), {permit}), "Drivers are required to carry their license" -> (set(),
    {driver, license}). Not a need in a clause of its own ("Employees wear a badge when needed") or a noun ("meet
    the needs of every student")."""
    def nouns(token):           # a noun and those joined to it, or an action and its object ("carrying a license")
        out = set()
        for t in (token, *token.conjuncts):
            if t.pos_ in ('NOUN', 'PROPN') or t.tag_ == 'VBG':
                out.add(t.lemma_.lower())
            out |= {c.lemma_.lower() for c in t.children if c.dep_ in ('dobj', 'dative') and c.pos_ in ('NOUN', 'PROPN')}
        return out
    def actions(token):         # who does what is to be done, and to what ("to have a permit", "that airlines pay")
        return {n for c in token.children if c.dep_ in ('xcomp', 'ccomp', 'acl', 'advcl') and c.pos_ in ('VERB', 'AUX')
                for g in c.children if g.dep_ in ('nsubj', 'dobj', 'dative') for n in nouns(g)}
    out = []
    for t in doc:
        lemma = t.lemma_.lower()
        if lemma in NEED_STATING_VERBS and t.pos_ != 'AUX' and acts_as_verb(t):
            clause = t
        elif lemma in NEED_STATING_WORDS and t.dep_ in ('acomp', 'attr', 'oprd') and t.head.lemma_ == 'be':
            clause = t.head
        else:
            continue
        if not (clause.dep_ == 'ROOT' or (clause.dep_ == 'conj' and clause.head.dep_ == 'ROOT')):
            continue
        subject = next((c for c in clause.children if c.dep_ in ('nsubj', 'nsubjpass', 'csubj', 'csubjpass')), None)
        needed = actions(t) | (actions(clause) if clause.i != t.i else set())
        if clause.i == t.i and subject is not None and subject.dep_ in ('nsubj', 'csubj'):
            holders = {g.lemma_.lower() for g in subject.subtree if g.pos_ in ('NOUN', 'PROPN', 'VERB', 'ADJ')}
            needed |= {n for c in t.children if c.dep_ == 'dobj' for n in nouns(c)}
        else:
            holders = {g.lemma_.lower() for c in clause.children if c.dep_ == 'agent' for g in c.subtree
                       if g.pos_ in ('NOUN', 'PROPN')}
            needed |= nouns(subject) if subject is not None else set()
        out.append((holders, needed))
    return out

def told_in_past(doc):
    """True if the main clause of a sentence (a rewrite's first) is told in the past ("needed", "was required")."""
    root = next(iter(doc.sents)).root
    return any(v.tag_ == 'VBD' for v in [root] + [c for c in root.children if c.dep_ in ('aux', 'auxpass')])

def forcing_nouns(doc):
    """Lemmas of the nouns that make someone act in an active "X forced/compelled Y to ..." ("The storm forced
    thousands of residents to leave" -> {storm}); none for the passive ("Residents were forced to leave"). The tagger
    may call the subject's own word an adjective ("The pandemic forced ...")."""
    return {g.lemma_.lower() for t in doc if t.lemma_.lower() in FORCING_VERBS and t.pos_ == 'VERB'
            and t.dep_ not in ('acl', 'amod') and not any(c.dep_ in ('auxpass', 'nsubjpass') for c in t.children)
            and any(c.dep_ in ('xcomp', 'advcl') and any(g.tag_ == 'TO' for g in c.children) for c in t.children)
            for c in t.children if c.dep_ == 'nsubj' for g in c.subtree
            if g.pos_ in ('NOUN', 'PROPN') or (g.i == c.i and g.pos_ == 'ADJ')}

# "as a result of the storm" (or "was a result of the storm") names a cause as "because of the storm" does
RESULT_OF = re.compile(r"as\s+a\s+(?:result|consequence)\s+of\s+", re.I)

def stated_causes(doc, sources=True):
    """Lemmas of the words a sentence gives as a cause, an agent or a source: after "because of", "due to" or "as a
    result of" (up to the next comma), before "led to" or "resulted in", in a "by" or "under" phrase, or as the
    subject of a causal or obliging verb ("Because of the storm, ...", "displaced by the storm", "Under the new
    rules, ...", "The storm drove ..."). With `sources` off an "under" phrase doesn't count: it says where an
    obligation comes from ("Under the law, companies must publish ..."), not why someone acted."""
    text, start = doc.text, doc[0].idx if len(doc) else 0
    out = set()
    for m in [*CAUSE_NP.finditer(text), *RESULT_OF.finditer(text)]:
        for t in doc:
            if t.idx - start < m.end():
                continue
            if t.text in (',', ';', '.'):
                break
            out.add(t.lemma_.lower())
    m = EFFECT_AFTER.search(text)
    if m:
        out |= {t.lemma_.lower() for t in doc if t.idx - start < m.start()}
    causing = CAUSAL_VERBS | OBLIGING_VERBS | FORCING_VERBS
    for t in doc:
        if t.dep_ == 'agent' or (t.dep_ == 'prep' and (t.lower_ == 'by' or (sources and t.lower_ == 'under'))):
            out |= {g.lemma_.lower() for g in t.subtree}
        if t.dep_ == 'nsubj' and t.head.lemma_.lower() in causing:
            out |= {g.lemma_.lower() for g in t.subtree}
    return out

# Words by which a rewrite turns a compulsion into permission, help or a choice ("The law forces companies to
# publish ..." -> "The law allows companies to publish ...", "Poor health compelled the mayor to step down" -> "Due
# to poor health, the mayor decided to step down")
PERMITTING = {'allow', 'let', 'enable', 'permit', 'help', 'encourage', 'invite', 'choose', 'decide', 'opt', 'persuade',
              'urge', 'able', 'free'}
# More ways to name a cause, read like CAUSE_NP up to the next comma: "on account of rising rents", "in response to
# falling sales", "in the wake of the war", "Faced with heavy snow, ...", "Given rising rents, ..."
CAUSE_FRAMES = re.compile(r"\b(?:on\s+account\s+of|in\s+response\s+to|in\s+the\s+(?:face|wake)\s+of|faced\s+with|"
                          r"confronted\s+with|given)\s+", re.I)

def forcing_dropped(sent, cand_doc):
    """True if the original's active "X forced Y to ..." comes back with X neither as a cause nor as the one who acts
    (the caller sees to a rewrite with a "must"): "The war forced many families to move to the cities" -> "Many
    families moved to the cities after the war" or "... were also forced to move ... after the war", "The law forces
    companies to publish ..." -> "Under the law, companies publish ..."; not "... because of the war", "The war drove
    many families to the cities", "displaced by the storm", "A knee injury ended his career", "... as costs rose",
    "With rents rising, ...", "on account of rising rents" or "Under the regulations, airlines are also forced to
    compensate passengers" (still forced, by the regulations). Also true when the rewrite turns the compulsion into
    permission, help or a choice ("The law allows companies to publish ...", "decided to step down"), and the
    forcer's own "by" phrase names no cause ("A strike by pilots forced the airline to cancel 300 flights" -> "The
    airline cancelled 300 flights during a strike by pilots")."""
    who = forcing_nouns(sent)
    if not who:
        return False
    old = {t.lemma_.lower() for t in sent}
    if any(t.lemma_.lower() in PERMITTING - old for t in cand_doc):
        return True
    forced = any(t.lemma_.lower() in FORCING_VERBS and t.pos_ == 'VERB' for t in cand_doc)
    given = stated_causes(cand_doc, sources=forced)
    own_by = {g.lemma_.lower() for t in cand_doc if t.dep_ == 'prep' and t.lower_ == 'by' and t.head.lemma_.lower() in who
              for g in t.subtree}
    other_by = {g.lemma_.lower() for t in cand_doc if (t.dep_ == 'agent' or (t.dep_ == 'prep' and t.lower_ == 'by'))
                and t.head.lemma_.lower() not in who for g in t.subtree}
    given -= own_by - other_by
    text, start = cand_doc.text, cand_doc[0].idx if len(cand_doc) else 0
    for m in CAUSE_FRAMES.finditer(text):
        for t in cand_doc:
            if t.idx - start < m.end():
                continue
            if t.text in (',', ';', '.'):
                break
            given.add(t.lemma_.lower())
    root = next(iter(cand_doc.sents)).root
    for t in cand_doc:
        if t.dep_ == 'nsubj' and (t.head.i == root.i or any(c.dep_ in ('mark', 'advmod') and c.lower_ in ('because', 'as', 'since', 'when')
                                                            for c in t.head.children)
                                  or (t.head.dep_ == 'pcomp' and t.head.head.lower_ == 'with')):     # "With rents rising"
            given |= {g.lemma_.lower() for g in t.subtree}
    return not any(kept_word(w, 'n', given, given) for w in who)

def need_restated(plain, obliged):
    """True if the sentence `plain` states as a need what the sentence `obliged` states as an obligation, or the
    other way round, keeping who needs it, what is needed and when ("A valid permit is required to park" ->
    "Parking requires a valid permit"); not "The application requires a permit" (someone new needs it), "parking
    with a valid permit is required" (parking is now what is needed) or "The town needed a new bridge" -> "The town
    must have a new bridge" (a past need made a present duty). What forced someone to act may stay as its cause
    ("The storm forced residents to leave" -> "Because of the storm, residents had to leave", "Many small shops were
    forced to close down as rents grew", "In response to falling sales, the bakery had to reduce its opening
    hours")."""
    if forcing_nouns(plain) and not forcing_dropped(plain, obliged):
        return True
    who = forcing_nouns(plain)
    if who:
        given = stated_causes(obliged)
        if any(kept_word(w, 'n', given, given) for w in who):
            return True
    if told_in_past(plain) != told_in_past(obliged):
        return False
    words = {t.lower_ for t in obliged} | {t.lemma_.lower() for t in obliged}
    def kept(word):
        return kept_word(word, 'n', words, words) or bool(set(getAllLemmas(word).get('VERB', ())) & words)
    others = set().union(*(needed for _, needed in needs_stated(obliged)))
    return any((not holders or any(map(kept, holders))) and (not needed or not others or any(
        kept_word(n, 'n', others, others) for n in needed)) for holders, needed in needs_stated(plain))

def frequency(tokens, text):
    """How often a sentence says something holds: 'often', 'sometimes', 'rarely' ("tend to", "in rare cases")."""
    out = {kind for kind, pattern in FREQUENCY_PHRASES if pattern.search(text)}
    tokens = list(tokens)
    for k, t in enumerate(tokens):
        kind = FREQUENCY.get(t.lower_)
        if not kind:
            continue
        if t.lemma_.lower() == 'tend' and not (k + 1 < len(tokens) and tokens[k + 1].lower_ == 'to'):
            continue        # "tend the garden"
        if t.lower_ == 'often' and k and tokens[k - 1].lower_ in ('how', 'as'):
            continue        # "how often", "as often as"
        out.add(kind)
    out |= {'sometimes' for t in tokens if t.lower_ in ('occasional', 'sporadic', 'periodic') and t.dep_ == 'amod'}
    return out

def evidence(tokens):
    """How strongly a sentence presents its claim: 'tentative', 'shown', 'opinion', 'correlation'."""
    out = set()
    for t in tokens:
        lemma = t.lemma_.lower()
        kind = EVIDENCE.get(lemma)
        if kind and acts_as_verb(t):
            if kind == 'correlation':
                if t.tag_ == 'VBN' or any(c.dep_ == 'prep' and c.lower_ in ('to', 'with') for c in t.children):
                    out.add(kind)
            elif (t.dep_ in ('acl', 'relcl') and t.tag_ == 'VBN') or any(
                    c.dep_ in ('ccomp', 'xcomp') or (c.dep_ == 'mark' and c.lower_ == 'as') for c in t.children):
                out.add(kind)       # "suggest that ...", "shown to", "as shown", "the results shown" (not "think about")
            elif (t.tag_ == 'VBN' and any(c.dep_ == 'auxpass' for c in t.children) and t.i + 1 < len(t.doc)
                  and t.doc[t.i + 1].lower_ == 'to'):
                out.add(kind)       # "has been demonstrated to", "is projected to" (the parser may hang "to V" anywhere)
        elif lemma in CORRELATION_NOUNS and t.pos_ == 'NOUN' and any(
                c.dep_ == 'prep' and c.lower_ in ('with', 'between') for c in t.children):
            out.add('correlation')
    return out

def quantities(tokens):
    """How much of something a sentence speaks of ('all', 'only', 'many', 'most', 'few', 'some', 'part',
    'specific'). "some" counts when it picks out some of a group ("some patients", "some of them", "certain
    cancers", "a few", "a number of"), not before a number ("some 200 people") or in "for some time"."""
    tokens = list(tokens)
    if not tokens:
        return set()
    text = tokens[0].doc[tokens[0].i: tokens[-1].i + 1].text
    base = tokens[0].idx
    idioms = [m.span() for m in IDIOM_ALL.finditer(text)]
    out = set()
    for k, t in enumerate(tokens):
        w = t.lower_
        prev = tokens[k - 1] if k else None
        nxt = tokens[k + 1] if k + 1 < len(tokens) else None
        if w in ADDITIVE_NOT and prev is not None and prev.lower_ == 'not':
            continue
        if w in ('every', 'each') and t.head.lemma_.lower() in TIME_NOUNS:
            continue        # "every year" -> "annually"
        if w == 'each' and nxt is not None and nxt.lower_ == 'other':
            continue        # "each other"
        if w == 'few' and prev is not None and prev.lower_ == 'a':
            out.add('some')     # "a few"
            continue
        if (w == 'few' and t.head.lemma_.lower() in TIME_NOUNS and prev is not None
                and prev.lower_ in ('past', 'last', 'next', 'previous', 'recent', 'first', 'coming', 'following')):
            continue        # "over the past few decades" is a span of time
        if w == 'just' and not (nxt is not None and nxt.pos_ in ('DET', 'NUM', 'NOUN', 'ADJ', 'PROPN')):
            continue        # "just finished", "just like"
        if w in ('lot', 'lots', 'plenty') and not (nxt is not None and nxt.lower_ == 'of'):
            continue        # "improved a lot"
        if w == 'most':
            if (t.dep_ in ('amod', 'det') and t.head.pos_ in ('NOUN', 'PROPN')) or (nxt is not None and nxt.lower_ == 'of'):
                out.add('most')     # not the "most" of "the most important"
            continue
        if w == 'all' and any(s <= t.idx - base < e for s, e in idioms):
            continue        # "all over the world"
        if w == 'all' and nxt is not None and nxt.pos_ in ('VERB', 'AUX') and any(
                c.lower_ in ('and', 'or') for c in tokens[max(0, k - 8): k]):
            continue        # "X, Y and Z are all contributing": the list already names every member
        if w == 'some' or (w == 'certain' and t.dep_ == 'amod' and t.head.tag_ == 'NNS'):
            if nxt is not None and (nxt.like_num or nxt.lower_ in ('of', 'time', 'point', 'extent', 'degree', 'way',
                                                                  'sort', 'kind', 'day', 'cases', 'instances')):
                if nxt.lower_ == 'of' and not (k + 2 < len(tokens) and tokens[k + 2].like_num):
                    out.add('some')     # "some of them"
                continue
            out.add('some')
            continue
        if w == 'number' and prev is not None and prev.lower_ == 'a' and nxt is not None and nxt.lower_ == 'of':
            out.add('some')     # "a number of"
            continue
        if w == 'part' and prev is not None and prev.lower_ == 'in' and t.dep_ == 'pobj':
            out.add('part')     # "caused in part by"
            continue
        if w in WHOLE and t.dep_ == 'amod':
            if not any(c.dep_ == 'nummod' for c in t.head.children):
                out.add('all')      # "its entire database" == "all of its database" (not "six whole months")
            continue
        if w in QUANTIFIERS:
            out.add(QUANTIFIERS[w])
    return out

def bounds(text):
    kinds = {kind for kind, pattern in BOUNDS if pattern.search(text)}
    for m in OVER_TIME.finditer(text):
        before = re.search(r"([A-Za-z'’-]+)\W*$", text[:m.start()])
        word = before.group(1).lower() if before else ''
        if not word or word in DURATION_WORDS or word in NLP.Defaults.stop_words:
            kinds.add('min')    # "for over 12 weeks", "took over a year" (not "lost 4 kg over 12 weeks")
    return kinds

def negators(tokens):
    tokens = list(tokens)
    return [t for k, t in enumerate(tokens) if (t.lower_ in NEGATORS or t.lower_ in ("n't", "n’t"))
            and not (t.lower_ == 'not' and k + 1 < len(tokens) and tokens[k + 1].lower_ in ADDITIVE_NOT)
            and not (t.lower_ == 'no' and k + 2 < len(tokens) and tokens[k + 1].lower_ in BOUND_AFTER_NO
                     and tokens[k + 2].lower_ == 'than')]

FUNCTION_LEMMAS = {'be', 'have', 'do', 'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would', 'not'}
WORD_PART = {'NOUN': 'n', 'PROPN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}

def negated_clause(neg):
    """Token indices of the clause a negation belongs to: its verb and that verb's own words, plus the verbs it
    hangs off up to the main clause; without the negated phrase itself ("without relying on credit cards"),
    clauses hanging off the verb, or coordinated clauses with a subject of their own."""
    verb = neg if neg.pos_ in ('VERB', 'AUX') else neg.head
    while verb.pos_ not in ('VERB', 'AUX') and verb.head.i != verb.i:
        verb = verb.head
    out, stack = set(), [verb]
    while stack:
        t = stack.pop()
        out.add(t.i)
        for c in t.children:
            if c.i in out or (t.i == verb.i and c.dep_ in ('relcl', 'advcl', 'parataxis', 'acl')):
                continue
            if c.dep_ == 'conj' and c.pos_ in ('VERB', 'AUX') and any(
                    g.dep_ in ('nsubj', 'nsubjpass', 'expl') for g in c.children):
                continue
            stack.append(c)
    node = verb
    while node.dep_ in ('xcomp', 'ccomp', 'pcomp', 'advcl', 'prep', 'pobj') and node.head.i != node.i:
        parent = node.head
        out.add(parent.i)
        for c in parent.children:
            if c.i != node.i and c.dep_ in ('nsubj', 'nsubjpass', 'dobj', 'dative', 'attr'):
                out.update(t.i for t in c.subtree)
        node = parent
    # "without" heads the negated phrase; other negators ("not", "never", "impossible") are single words, whose
    # own dependents ("impossible with the current budget") belong to the clause
    return out - ({t.i for t in neg.subtree} if neg.lower_ == 'without' else {neg.i})

def negation_moved(sent, cand_doc, source_words):
    """True if a negation no longer sits in a clause about the same things: at least half the content words of
    each negated clause must share a clause with a negation in the rewrite ("cover costs without relying on
    credit cards" can't become "pay for costs, and it doesn't need a credit card to be present")."""
    old = negators(sent)
    if not old:
        return False
    new = negators(cand_doc) + negating_verbs(cand_doc) + [t for t in cand_doc if negated_stem(t.lower_) in source_words]
    clauses = []
    for neg in new:
        ids = negated_clause(neg)
        clauses.append(({cand_doc[i].lower_ for i in ids}, {cand_doc[i].lemma_.lower() for i in ids}))
    doc = sent.doc
    for neg in old:
        anchors = {(doc[i].lemma_.lower(), WORD_PART[doc[i].pos_]) for i in negated_clause(neg)
                   if doc[i].pos_ in WORD_PART and doc[i].is_alpha and not doc[i].is_stop
                   and doc[i].lemma_.lower() not in FUNCTION_LEMMAS}
        if not anchors:
            continue
        need = (len(anchors) + 1) // 2
        if not any(sum(kept_word(lemma, part, words, lemmas) for lemma, part in anchors) >= need
                   for words, lemmas in clauses):
            return True
    return False

def role_words(token):
    """The noun lemmas a role argument stands for: a relative pronoun stands for its noun, which after "be" also
    stands for the subject ("Japan is a destination that offers ..."); coordinated nouns count too."""
    if token.tag_ in ('WDT', 'WP') and token.head.dep_ == 'relcl':
        noun = token.head.head
        words = role_words(noun) if noun is not token else set()
        if noun.dep_ == 'attr' and noun.head.lemma_ == 'be':
            subject = next((c for c in noun.head.children if c.dep_ in ('nsubj', 'nsubjpass')), None)
            if subject is not None:
                words |= role_words(subject)
        return words
    return {('everyone' if t.lemma_.lower() in ('person', 'individual', 'one') and any(
                c.lower_ in ('every', 'each') for c in t.children) else t.lemma_.lower())
            for t in [token, *token.conjuncts] if t.pos_ in ('NOUN', 'PROPN') or t.lemma_.lower() in ROLE_PRONOUNS}

def verb_subject(verb):
    return next((c for c in verb.children if c.dep_ in ('nsubj', 'csubj')), None)

def doers(verb, depth=0):
    """The noun lemmas doing a verb's action: its subject or the "by" agent of a passive. A verb without a
    subject of its own takes it from the verb it hangs off, when that is unambiguous."""
    subject = verb_subject(verb)
    if subject is not None:
        return role_words(subject)
    agent = next((c for c in verb.children if c.dep_ == 'agent'), None)
    if agent is not None:
        obj = next((c for c in agent.children if c.dep_ == 'pobj'), None)
        return role_words(obj) if obj is not None else set()
    if any(c.dep_ == 'nsubjpass' for c in verb.children) or depth > 3:
        return set()
    head = verb.head
    if verb.dep_ == 'conj' and head.pos_ in ('VERB', 'AUX'):
        return doers(head, depth + 1)
    if verb.dep_ in ('relcl', 'acl') and verb.tag_ == 'VB' and head.dep_ in ('dobj', 'attr'):
        governing = verb_subject(head.head)     # "Everyone has something to offer": who has it offers it
        return role_words(governing) if governing is not None else set()
    if verb.dep_ in ('xcomp', 'advcl') and head.pos_ in ('VERB', 'AUX') and not any(
            c.dep_ in ('dobj', 'dative', 'prep') for c in head.children):
        return doers(head, depth + 1)
    # "Chlorophyll is responsible for absorbing light", "plants are able to absorb light": the subject does it
    holder = head.head if verb.dep_ == 'pcomp' and head.dep_ == 'prep' and head.lower_ in ('for', 'of') else head
    if verb.dep_ in ('pcomp', 'xcomp') and holder.pos_ == 'ADJ' and holder.dep_ in ('acomp', 'oprd'):
        governing = verb_subject(holder.head)
        return role_words(governing) if governing is not None else set()
    return set()

def roles(tokens):
    """verb lemma -> {noun lemma: role} for a sentence's content verbs: 'A' does the action, 'P' undergoes it,
    'R' receives it or is its goal, 'S' is its source."""
    out = {}
    for verb in tokens:
        lemma = verb.lemma_.lower()
        if verb.pos_ != 'VERB' or lemma in ROLE_SKIP or verb.dep_ in ('nsubj', 'nsubjpass', 'dobj', 'pobj', 'attr', 'appos'):
            continue
        args = out.setdefault(lemma, {})

        def put(words, role):
            for w in words:
                args.setdefault(w, role)
        put(doers(verb), 'A')
        for c in verb.children:
            if c.dep_ in ('dobj', 'nsubjpass'):
                put(role_words(c), 'P')
            elif c.dep_ == 'dative':
                put(role_words(c), 'R')
        holders = [verb] + [c for c in verb.children if c.dep_ == 'dobj']   # "offers something for everyone"
        for holder in holders:
            for p in holder.children:
                if p.dep_ == 'prep' and p.lower_ in ROLE_PREPS:
                    obj = next((g for g in p.children if g.dep_ == 'pobj'), None)
                    if obj is not None:
                        put(role_words(obj), ROLE_PREPS[p.lower_])
    return out

def swapped_roles(sent, cand_doc):
    """True if, for a verb both versions use, someone does the action in one but not the other, two words trade
    places ("Japan offers something for everyone" -> "Everyone has something to offer in Japan"), where something
    comes from changes ("served from the cache" -> "serving the cache"), or someone else does it ("the database
    reduces latency" -> "the application reduces latency")."""
    old, new = roles(sent), roles(cand_doc)
    for verb in old.keys() & new.keys():
        a, b = old[verb], new[verb]
        shared = a.keys() & b.keys()
        if any(a[w] != b[w] and ('A' in (a[w], b[w]) or {a[w], b[w]} in ({'S', 'P'}, {'S', 'R'})) for w in shared):
            return True
        if any(a[w] != a[v] and (a[w], a[v]) == (b[v], b[w]) for w in shared for v in shared if w < v):
            return True
    old_doers, new_doers = explicit_doers(sent), explicit_doers(cand_doc)
    for verb in old_doers.keys() & new_doers.keys():
        if not any(kept_word(w, 'n', new_doers[verb], new_doers[verb]) for w in old_doers[verb]):
            return True
    old_verbs = {t.lemma_.lower() for t in sent if t.pos_ == 'VERB'}
    new_verbs = {t.lemma_.lower() for t in cand_doc if t.pos_ == 'VERB'}
    if 'afford' in old_verbs - new_verbs and (new_verbs - old_verbs) & AFFORD_SWAPS:
        return True     # "unable to afford treatment" -> "unable to receive treatment"
    return transfer_reversed(sent, cand_doc) or doer_undergoes(sent, cand_doc)

# Passives that tell what something is or how it stands, not what someone does to it ("is made of", "is based on",
# "is located in", "are opposed to", "are drawn to", "should be equipped with")
STATIVE_PASSIVES = {'make', 'base', 'locate', 'situate', 'headquarter', 'dedicate', 'devote', 'commit', 'oppose', 'draw',
                    'attach', 'relate', 'connect', 'link', 'equip', 'fill', 'cover', 'pack', 'line', 'please', 'satisfy',
                    'concern', 'associate', 'involve', 'know', 'compose', 'comprise', 'interest', 'engage', 'surround',
                    'populate', 'inhabit', 'focus', 'center', 'centre', 'aim', 'gear', 'tailor', 'intend',
                    'divide', 'split', 'unite', 'inspire', 'worry',
                    'scare', 'frighten', 'excite', 'thrill', 'delight', 'impress', 'amaze', 'surprise', 'shock',
                    'disappoint', 'annoy', 'tire', 'bore', 'confuse', 'devastate', 'overwhelm', 'fascinate', 'attract',
                    'accustom', 'entitle', 'qualify', 'suit', 'bind', 'tie'}
# What a thing has or holds, and how one feels about something: told in the passive, the subject undergoes nothing
# ("Every flat has its own meter" -> "Each flat is equipped with a meter", "The bridge contains 2,000 tons of steel"
# -> "The bridge is constructed with ...", "Many children love the old park" -> "are drawn to")
HAVING_VERBS = {'have', 'contain', 'include', 'comprise', 'hold', 'carry', 'own', 'possess', 'feature', 'house', 'boast',
                'cover', 'love', 'like', 'enjoy', 'oppose', 'support', 'fear', 'prefer', 'hate'}

def doer_undergoes(sent, cand_doc):
    """True if the one who does the original's main action undergoes the rewrite's, told in the passive with
    another verb ("Those with diabetes should check their blood sugar" -> "Diabetics should be monitored for their
    blood sugar"); not the original's object made the subject ("Their blood sugar should be checked"), someone
    given something ("Students learn English" -> "Students are taught English"), someone who still does it
    ("Visitors must show a ticket" -> "Visitors are required to show a ticket", "was engaged in producing"), or a
    passive that tells what something is or has ("The company has its headquarters in Lyon" -> "The company is
    headquartered in Lyon", "The film tells a true story" -> "The film is based on a true story")."""
    old_root, new_root = sent.root, next(iter(cand_doc.sents)).root
    subject = next((c for c in old_root.children if c.dep_ == 'nsubj'), None)
    passive = next((c for c in new_root.children if c.dep_ == 'nsubjpass'), None)
    if (old_root.pos_ != 'VERB' or subject is None or passive is None
            or not any(c.dep_ == 'dobj' for c in old_root.children)
            or old_root.lemma_.lower() in HAVING_VERBS or new_root.lemma_.lower() in STATIVE_PASSIVES):
        return False
    def action(token):
        return token.tag_ == 'VBG' or (token.lower_.endswith('ing') and bool(getAllLemmas(token.lower_).get('VERB')))
    if any(c.dep_ in ('agent', 'dobj', 'dative', 'xcomp', 'ccomp')
           or (c.dep_ == 'prep' and any(g.dep_ in ('pcomp', 'pobj') and action(g) for g in c.children))
           for c in new_root.children):
        return False        # a "by" agent, someone given something, or the subject still acting
    doing = {g.lemma_.lower() for g in subject.subtree if g.pos_ in ('NOUN', 'PROPN', 'PRON')}
    objects = {g.lemma_.lower() for c in old_root.children if c.dep_ == 'dobj' for g in c.subtree if g.pos_ in ('NOUN', 'PROPN')}
    word = passive.lemma_.lower()
    return kept_word(word, 'n', doing, doing) and not kept_word(word, 'n', objects, objects)

def explicit_doers(tokens):
    """verb lemma -> the words naming who does it, from its own subject or "by" agent only, with the words that
    name the same one ("Council member Rene Gonzalez", "policy makers" = "policymakers"). Numbers and amounts
    ("the 12 percent rise") are not doers."""
    out = {}
    for verb in tokens:
        lemma = verb.lemma_.lower()
        if verb.pos_ != 'VERB' or lemma in ROLE_SKIP or verb.dep_ in ('nsubj', 'nsubjpass', 'dobj', 'pobj', 'attr', 'appos'):
            continue
        subject = verb_subject(verb)
        if subject is None:
            agent = next((c for c in verb.children if c.dep_ == 'agent'), None)
            subject = next((c for c in agent.children if c.dep_ == 'pobj'), None) if agent is not None else None
        holder = verb.head.head if verb.dep_ == 'pcomp' and verb.head.dep_ == 'prep' else verb.head
        if subject is None and verb.dep_ in ('pcomp', 'xcomp') and holder.pos_ == 'ADJ' and holder.dep_ in ('acomp', 'oprd'):
            subject = verb_subject(holder.head)     # "X is responsible for absorbing", "X is able to absorb"
        if subject is None:
            continue
        named = [subject]
        if subject.tag_ in ('WDT', 'WP') and subject.head.dep_ == 'relcl':
            subject = subject.head.head
            named = [subject]
            if subject.dep_ == 'attr' and subject.head.lemma_ == 'be':     # "Japan is a destination that offers ..."
                named += [c for c in subject.head.children if c.dep_ in ('nsubj', 'nsubjpass')]
        if subject.pos_ not in ('NOUN', 'PROPN') or subject.lemma_.lower() in ('percent', 'percentage') or any(
                c.dep_ == 'nummod' for c in subject.children):
            continue
        names = set()
        for t in [t for n in named for t in (n, *n.conjuncts)]:
            relatives = [t, *(c for c in t.children if c.dep_ in ('compound', 'appos'))] + ([t.head] if t.dep_ == 'appos' else [])
            relatives += [g for c in t.children if c.dep_ == 'prep' and c.lower_ == 'of' for g in c.children if g.dep_ == 'pobj']
            names |= {r.lemma_.lower() for r in relatives}
            compounds = [c.lower_ for c in t.children if c.dep_ == 'compound']
            names.add(''.join(compounds + [t.lower_]))
            names.add(''.join(compounds + [t.lemma_.lower()]))
        out.setdefault(lemma, set()).update(names)
    return out

def transfers(tokens):
    """noun lemma -> 'giver' or 'receiver' for the giving and taking verbs of a sentence ("the app collects data
    from users": the app receives, the users give)."""
    out = {}
    for verb in tokens:
        if verb.pos_ != 'VERB':
            continue
        lemma = verb.lemma_.lower()
        kids = list(verb.children)
        giving = lemma in GIVING or (lemma == 'share' and any(c.dep_ == 'prep' and c.lower_ == 'with' for c in kids))
        if not giving and lemma not in TAKING:
            continue
        for w in doers(verb):
            out.setdefault(w, 'giver' if giving else 'receiver')
        passive = any(c.dep_ in ('nsubjpass', 'auxpass') for c in kids)
        for holder in [c for c in kids if c.dep_ in ('dobj', 'nsubjpass')] + [
                g for c in kids if c.dep_ == 'prep' and c.lower_ == 'with' for g in c.children if g.dep_ == 'pobj']:
            for c in holder.children:
                if c.dep_ == 'prep' and (c.lower_ == ('to' if giving else 'from') or (giving and passive and c.lower_ == 'by')):
                    for g in c.children:
                        if g.dep_ == 'pobj':
                            for w in role_words(g):
                                out.setdefault(w, 'giver' if (not giving or c.lower_ == 'by') else 'receiver')
        for c in kids:
            if c.dep_ == 'prep' and c.lower_ in (('to', 'for') if giving else ('from',)):
                for g in c.children:
                    if g.dep_ == 'pobj':
                        for w in role_words(g):
                            out.setdefault(w, 'receiver' if giving else 'giver')
            elif giving and c.dep_ == 'dative':
                for g in ([c] if c.pos_ in ('NOUN', 'PROPN', 'PRON') else [g for g in c.children if g.dep_ == 'pobj']):
                    for w in role_words(g):
                        out.setdefault(w, 'receiver')     # "gave them a prize", "was given to the council"
            elif giving and c.dep_ in ('dobj', 'nsubjpass') and (any(k.dep_ == 'prep' and k.lower_ == 'with' for k in kids)
                                                                  or (c.dep_ == 'nsubjpass' and any(k.dep_ == 'dobj' for k in kids))):
                for w in role_words(c):
                    out.setdefault(w, 'receiver')   # "provides users with data", "users are given data"
    return out

def transfer_reversed(sent, cand_doc):
    """True if someone who gives in one version receives in the other ("the app does not collect data from its
    users" -> "users are not given data by the app")."""
    old, new = transfers(sent), transfers(cand_doc)
    if any(old[w] != new[w] for w in old.keys() & new.keys()):
        return True
    acted = {}      # noun -> verbs it does, its own subject's or that of the verb it hangs off ("voted to approve")
    for verb in sent:
        if verb.pos_ != 'VERB':
            continue
        node, subject = verb, None
        while subject is None and node is not None:
            subject = verb_subject(node)
            node = node.head if node.dep_ in ('xcomp', 'conj') and node.head.i != node.i else None
        for name in (role_words(subject) if subject is not None else ()):
            acted.setdefault(name, set()).add(verb.lemma_.lower())
    for verb in cand_doc:
        if verb.pos_ != 'VERB' or verb.lemma_.lower() not in GIVING:
            continue
        themes = {stem(c.lemma_.lower()) for c in verb.children if c.dep_ in ('dobj', 'nsubjpass')} - {None}
        receivers = {w for c in verb.children if c.dep_ in ('prep', 'dative') and c.lower_ in ('to', 'for')
                     for g in c.children if g.dep_ == 'pobj' for w in role_words(g)}
        receivers |= {w for c in verb.children if c.dep_ == 'dative' and c.pos_ in ('NOUN', 'PROPN') for w in role_words(c)}
        if any(stem(v) in themes for w in receivers & acted.keys() for v in acted[w]):
            return True
    return False

SINGLE_DETERMINERS = {'a', 'an', 'one', 'each', 'every', 'single'}

def plural_made_single(sent, cand_doc):
    """True if a noun the original only uses in the plural comes back only as one of them ("small
    contributions add up" -> "a small contribution can")."""
    for lemma in {t.lemma_.lower() for t in sent if t.pos_ == 'NOUN'}:
        uses = [t for t in sent if t.lemma_.lower() == lemma and t.pos_ == 'NOUN' and t.dep_ != 'compound']
        if not uses or any(t.tag_ != 'NNS' for t in uses):
            continue
        if any(t.head.lower_ == 'of' and t.head.head.lower_ in ('one', 'each', 'any', 'some', 'none') for t in uses):
            continue    # "one of the most important steps"
        new = [t for t in cand_doc if t.lemma_.lower() == lemma and t.pos_ == 'NOUN' and t.dep_ != 'compound']
        if new and all(t.tag_ == 'NN' and any(c.lower_ in SINGLE_DETERMINERS for c in t.children) for t in new):
            return True
    return False

def cut_off(sent, cand_doc):
    """True if a rewrite ends on a describing word that needs its noun ("... under the new.")."""
    last = next((t for t in reversed(cand_doc) if not t.is_punct and not t.is_space), None)
    return (last is not None and last.pos_ == 'ADJ' and last.dep_ in ('pobj', 'dobj', 'nsubj', 'nsubjpass', 'appos')
            and any(c.dep_ in ('det', 'poss') for c in last.children)
            and any(t.lower_ == last.lower_ and t.dep_ == 'amod' for t in sent))

def synonyms_of(lemma, part):
    entry = SYNONYMS.get(f'{lemma}|{part}')
    return {w for w, _ in entry[1][:8]} if entry else set()

# Words a rewrite may use for each other that the synonym table leaves out (it only lists swaps that are safe
# everywhere: "a study" can't become "a research", but a rewrite that says "research" keeps the topic)
_EQUIVALENT_GROUPS = [
    # nouns
    {'learner', 'student'}, {'study', 'research'}, {'effect', 'impact'}, {'result', 'finding', 'outcome'},
    {'benefit', 'advantage'}, {'lack', 'absence'}, {'price', 'cost'}, {'cost', 'expense'},
    {'drug', 'medication', 'medicine'}, {'company', 'firm'}, {'soldier', 'troop'}, {'teenager', 'adolescent'},
    {'people', 'individual', 'person'}, {'interaction', 'contact'}, {'isolation', 'loneliness'}, {'evidence', 'data'},
    {'scientist', 'researcher'}, {'transition', 'shift'}, {'cooperation', 'collaboration', 'partnership'},
    {'issue', 'problem'}, {'decisive', 'bold', 'determined', 'resolute', 'firm', 'forceful'},
    {'frequent', 'frequently', 'often'}, {'realistic', 'feasible', 'achievable', 'attainable', 'reasonable'},
    {'seasoned', 'experienced', 'veteran'},
    {'work', 'job'}, {'duty', 'obligation', 'responsibility'}, {'complicated', 'complex'},
    {'power', 'energy', 'electricity'}, {'importance', 'significance'}, {'connection', 'relationship', 'tie', 'bond'},
    {'celebration', 'commemoration', 'festivity'}, {'provider', 'service', 'vendor'},
    {'challenge', 'problem', 'difficulty', 'obstacle'}, {'habit', 'routine'}, {'change', 'adjustment', 'modification'},
    {'goal', 'target', 'aim'}, {'illness', 'disease'}, {'trip', 'journey'}, {'job', 'position'}, {'staff', 'employee'},
    # describing words
    {'large', 'considerable', 'substantial', 'sizable', 'sizeable', 'extensive'}, {'deadly', 'fatal', 'lethal'},
    {'short', 'brief', 'concise'}, {'fast', 'rapid', 'quick', 'rapidly', 'quickly'}, {'bustling', 'lively', 'busy', 'vibrant'},
    {'serene', 'tranquil', 'peaceful', 'calm'}, {'famous', 'renowned', 'celebrated', 'well-known'},
    {'exceptional', 'outstanding', 'remarkable', 'excellent'}, {'easily', 'readily'}, {'slowly', 'gradually'},
    {'bravely', 'courageously'}, {'deep', 'thorough', 'in-depth'}, {'further', 'additional'}, {'small', 'minor', 'modest'},
    {'poor', 'substandard', 'inadequate'}, {'vulnerable', 'susceptible'}, {'unexpected', 'unforeseen', 'unanticipated'},
    {'sharply', 'steeply', 'drastically', 'dramatically', 'significantly', 'substantially', 'considerably',
     'markedly', 'greatly', 'significant', 'substantial', 'considerable', 'marked', 'dramatic', 'drastic', 'sharp'},
    {'international', 'global', 'worldwide'},
    {'connected', 'touch'}, {'top', 'leading', 'premier', 'top-tier'}, {'hot', 'thermal'},
    {'mainly', 'mostly', 'largely', 'primarily', 'chiefly', 'principally', 'predominantly', 'main', 'primary', 'chief',
     'principal'},
    {'country', 'nation'}, {'atmosphere', 'air'}, {'commitment', 'dedication'}, {'action', 'step', 'measure'},
]
# Words a rewrite may use for each other that the synonym table leaves out (it only lists swaps that are safe
# everywhere: "a study" can't become "a research", but a rewrite that says "research" keeps the topic)
KEPT_EQUIVALENTS = {'approach': {'method', 'technique'}}
for _group in _EQUIVALENT_GROUPS:
    for _word in _group:
        KEPT_EQUIVALENTS.setdefault(_word, set()).update(_group - {_word})

PREFIXES = ('trans', 'inter', 'under', 'over', 'super', 'multi', 'micro', 'macro', 'counter', 'contra', 'extra',
            'ultra', 'semi', 'auto', 'tele', 'photo', 'hyper', 'post', 'self')

def stem(word):
    """The start a word shares with its related forms ("celeb" in "celebrate"/"celebration"): five letters,
    or more after a common prefix ("transpiration" isn't related to "transform")."""
    for prefix in PREFIXES:
        if word.startswith(prefix):
            return word[:len(prefix) + 4] if len(word) >= len(prefix) + 4 else None
    return word[:5] if len(word) >= 5 else None

def kept_word(lemma, part, present, present_lemmas, by_stem=True):
    """True if a word survives in the candidate: as itself, inflected, by a WordNet synonym, or by a
    related form ("remote" in "remotely", "celebration" as "celebrate"). With `by_stem` off, a word that only
    starts the same way doesn't count ("screen" is not kept by "screening")."""
    if lemma in present or lemma in present_lemmas or KEPT_EQUIVALENTS.get(lemma, set()) & (present | present_lemmas):
        return True
    if part == 'n' and set(getInflection(lemma, tag='NNS') or ()) & present:
        return True
    if synonyms_of(lemma, part) & present_lemmas:
        return True
    if part == 'n' and NOMINALIZATIONS.get(lemma) in present_lemmas:
        return True
    start = stem(lemma) if by_stem else None
    return start is not None and any(w.startswith(start) for w in present_lemmas | present)

def name_parts(word):
    return [p for p in re.sub(r"['’]s$", '', word).split('-') if p]

def refuse(why, reason):
    """Reject a candidate, noting the reason when the caller collects them."""
    if why is not None:
        why.append(reason)
    return None

# Real words the lemma tables don't know, so they aren't taken for made-up ones
KNOWN_WORDS = {
    'despite', 'feasible', 'scenery', 'wellbeing', 'subpar', 'iconic',
    'affordability', 'repurchase', 'repurchased', 'occurrence', 'reshaped', 'sightseeing', 'sensible', 'timeline',
    'noticeable', 'negligible', 'website', 'websites', 'gameplay', 'standout', 'hardworking', 'pollinator', 'pollinators',
    'repopulation', 'bioenergy', 'responsiveness', 'proactivity', 'camaraderie', 'convergence', 'correlation',
    'correlate', 'correlated', 'enrollees', 'tramways', 'ultralight', 'online', 'offline', 'smartphone', 'smartphones',
    'healthcare', 'workplace', 'workforce', 'lifespan', 'lifestyle', 'lifestyles', 'downtime', 'uptime', 'onboarding',
    'eco-friendly', 'nonprofit', 'stakeholders', 'stakeholder', 'upskill', 'upskilling', 'reskilling', 'cybersecurity',
    'data-driven', 'policymakers', 'policymaker', 'teammates', 'coworkers', 'coworker', 'wildfires', 'wildfire',
    'heatwave', 'heatwaves', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
    'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty',
    'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million', 'billion', 'dozen', 'eighteenth',
    'nineteenth', 'twentieth', 'seventeenth', 'sixteenth', 'fifteenth'}
NUMBER_WORDS = {'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
                'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty',
                'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million', 'billion', 'dozen', 'dozens',
                'hundreds', 'thousands', 'millions', 'billions'}
# "a village in Scotland" may become "a Scottish village"
DEMONYMS = {'scotland': 'scottish', 'britain': 'british', 'japan': 'japanese', 'america': 'american', 'europe': 'european',
            'india': 'indian', 'france': 'french', 'germany': 'german', 'china': 'chinese', 'spain': 'spanish',
            'england': 'english', 'ireland': 'irish', 'italy': 'italian', 'canada': 'canadian', 'mexico': 'mexican',
            'australia': 'australian', 'africa': 'african', 'asia': 'asian', 'portugal': 'portuguese', 'wales': 'welsh'}
# Nouns of stock frames a rewrite may bring in ("over the course of", "at a minimum")
LIGHT_EXTRA = {'course', 'moment', 'occasion', 'minimum', 'maximum'}
# Who a sentence is about: "her life" can't come into a sentence about a car
PERSON_NOUNS = {'minister', 'president', 'teacher', 'author', 'mayor', 'scientist', 'doctor', 'man', 'woman', 'boy', 'girl',
                'mother', 'father', 'king', 'queen', 'member', 'officer', 'chief', 'leader', 'student', 'child', 'person',
                'people', 'patient', 'writer', 'artist', 'researcher', 'worker', 'employee', 'resident', 'founder'}
THIRD_PERSON = {'he': 'he', 'him': 'he', 'his': 'he', 'himself': 'he', 'she': 'she', 'her': 'she', 'hers': 'she',
                'herself': 'she'}
# A rewrite that ends on a possessive stopped mid-phrase ("... with the new policy's.")
POSSESSIVE_END = re.compile(r"(?:['’]s|\b(?:its|their|our|your|my|whose))\s*[.!?][\"”’)\]]?\s*$", re.I)
ARTICLES = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'its', 'their'}
# Nouns that may take a plural verb in the singular ("a number of users are", "the staff are")
COLLECTIVE = {'number', 'majority', 'lot', 'rest', 'percent', 'percentage', 'half', 'third', 'quarter', 'part',
              'proportion', 'fraction', 'total', 'variety', 'range', 'host', 'couple', 'kind', 'type', 'sort', 'series',
              'team', 'staff', 'family', 'government', 'committee', 'police', 'public', 'crowd', 'group', 'data', 'media',
              'audience', 'council', 'board', 'jury', 'class', 'faculty', 'personnel', 'species', 'means', 'news', 'set',
              'pair', 'body', 'array', 'plenty', 'mathematics', 'physics', 'economics', 'politics', 'ethics', 'statistics',
              'headquarters', 'crew', 'army', 'company', 'firm', 'band', 'population', 'community', 'generation', 'youth',
              'cattle', 'clergy', 'remainder', 'bulk', 'minority', 'handful', 'dozen', 'pile', 'collection', 'list'}
# Comparatives keep their claim even when the plain word is a light one ("stronger job growth")
COMPARATIVE_FREE = {'more', 'most', 'less', 'least', 'better', 'best', 'greater', 'greatest', 'further', 'furthest',
                    'other', 'later', 'earlier', 'latest'}
# Curated replacements count as keeping a describing word ("substantial" -> "large")
CURATED_OPTIONS = {}
for _key, _options in PLAIN_WORDS.items():
    CURATED_OPTIONS.setdefault(_key.split('|')[0], set()).update(w for o in _options for w in o.split())
for _key, _by_tone in AI_WORDS.items():
    for _options in _by_tone.values():
        CURATED_OPTIONS.setdefault(_key.split('|')[0], set()).update(w for o in _options for w in o.split())

def mentions(text, item):
    """True if `item` stands in `text` as a whole: "Ann" is not in "Annabel", nor "3" in "3.5"."""
    return re.search(r'(?<![\w.@/-])' + re.escape(item) + r'(?![\w@/]|[.,]\d)', text) is not None

def finite_clause(span):
    """True if the sentence has a main clause: a verb at its root that is finite or has a subject or an auxiliary
    (the tagger often reads "scored" or "involved" as participles), or a finite verb outside relative clauses ("The
    introduction of new machines ... led to ..."). "Caching, which involves storing ..." and "Vitamin A 800 IU and
    vitamin C 60 mg a tablet each." have none."""
    root = span.root
    if root.pos_ in ('VERB', 'AUX') and (root.tag_ not in ('VBG', 'VBN') or any(
            c.dep_ in ('nsubj', 'nsubjpass', 'aux', 'auxpass', 'expl') for c in root.children)):
        return True
    inside = set()
    for t in span:
        if t.dep_ == 'relcl' or (t.dep_ == 'acl' and (t.tag_ not in FINITE_TAGS or any(
                c.tag_ in WH_WORDS or (c.dep_ == 'mark' and c.lower_ == 'that') for c in t.children))):
            inside.update(x.i for x in t.subtree)
    if any(t.tag_ in FINITE_TAGS and t.i not in inside for t in span):
        return True
    return any(t.pos_ == 'NOUN' and t.lower_.endswith(('s', 'ed')) and getAllLemmas(t.lower_).get('VERB')
               and t.i + 1 < len(t.doc) and t.doc[t.i + 1].pos_ in ('DET', 'ADJ', 'NOUN', 'PROPN', 'NUM', 'ADV', 'ADP', 'PRON')
               and t.i > span.start for t in span)

# Nouns whose plural is the singular ("fish are sold at the market")
ZERO_PLURALS = {'fish', 'sheep', 'deer', 'species', 'series', 'aircraft', 'offspring'}
# ", which" with one of these verbs may speak of the whole clause before it ("weighs 150 grams, which makes it easy
# to carry")
WHICH_CONSEQUENCES = CAUSAL_VERBS | {'mean', 'reduce', 'lower', 'increase', 'raise', 'narrow', 'widen', 'show', 'suggest',
                                    'indicate', 'give', 'save', 'cut', 'boost', 'improve', 'add', 'put', 'set', 'bring',
                                    'cost'}

def which_numbers(tokens):
    """The numbers ('sg', 'pl') of the verbs of a sentence's "which" clauses ("which affects" -> {'sg'})."""
    out = set()
    for t in tokens:
        if t.lower_ == 'which' and t.dep_ == 'nsubj' and t.head.dep_ == 'relcl':
            verb = t.head
            finite = [v for v in [*[c for c in verb.children if c.dep_ in ('aux', 'auxpass')], verb]
                      if v.tag_ in ('VBZ', 'VBP') or v.lower_ in ('was', 'were')]
            if finite:
                out.add('sg' if finite[0].tag_ == 'VBZ' or finite[0].lower_ == 'was' else 'pl')
    return out

def number_disagreements(tokens, numbers=None):
    """(subject, verb) pairs whose number doesn't agree ("public transportation make", "plants, which uses", "the
    quality of the products were improved", "The high efficiency of public transport, make it"). Left out: a past
    tense that looks like a present ("the move that cost $2 million"), ", which makes it" after its clause's verb
    (which may speak of the whole clause; one before it is about its noun and must agree: "Electric cars, which
    produces no fumes, are"), a "which" on the object of "of" or "in" that agrees with the noun the phrase hangs on
    ("The rise in food prices, which affects every family"; not "by": "the absorption of light by plants, which
    then converts"), one noun for two things ("physical and mental health are"), and nouns that may take either.
    With `numbers` (the original's which_numbers) a "which" verb that agrees with the outer noun is left out only
    in a number the original's own "which" verbs have: "The quality of the roads, which are full of potholes" can't
    become "..., which is full of potholes"."""
    out = set()
    for t in tokens:
        if t.dep_ not in ('nsubj', 'nsubjpass'):
            continue
        outer = None        # the noun a "which" clause may be about instead of its own ("the rise in food prices")
        if t.tag_ in ('WDT', 'WP') and t.head.dep_ == 'relcl':
            verb = t.head.head      # the verb of the clause the ", which" clause hangs on
            while verb.pos_ not in ('VERB', 'AUX') and verb.dep_ != 'ROOT':
                verb = verb.head
            if t.dep_ != 'nsubj' or (t.lower_ == 'which' and t.i and t.doc[t.i - 1].text == ','
                                     and t.head.lemma_.lower() in WHICH_CONSEQUENCES and t.i > verb.i):
                continue
            subject = t.head.head           # "plants, which uses": the relative pronoun stands for the noun
            if (t.lower_ == 'which' and subject.dep_ == 'pobj' and subject.head.dep_ == 'prep'
                    and subject.head.lower_ in ('of', 'in') and subject.head.head.pos_ == 'NOUN'):
                outer = subject.head.head
        elif t.pos_ == 'NOUN':
            subject = t
        else:
            continue
        if (subject.pos_ != 'NOUN' or subject.lemma_.lower() in COLLECTIVE or subject.lower_ in COLLECTIVE
                or subject.lower_ in ZERO_PLURALS or subject.tag_ not in ('NN', 'NNS')
                or any(c.dep_ in ('conj', 'dobj', 'preconj') for c in subject.children)
                or any(c.dep_ == 'amod' and any(g.dep_ == 'conj' for g in c.children) for c in subject.children)):
            continue        # a "noun" with an object is a misread verb ("a car emits less carbon")
        verb = t.head
        aux = [c for c in verb.children if c.dep_ in ('aux', 'auxpass')]
        finite = [v for v in [*aux, verb] if v.tag_ in ('VBZ', 'VBP') or v.lower_ in ('was', 'were')]
        if not finite and verb.tag_ == 'VB' and verb.dep_ == 'ROOT' and not aux and t.dep_ == 'nsubj':
            finite = [verb]         # a bare verb with a subject of its own is a present tense
        if not finite:
            continue
        v = finite[0]
        if v.tag_ in ('VBP', 'VB') and v.lower_ in (getInflection(v.lemma_.lower(), 'VBD') or ()):
            continue
        if outer is not None and ((outer.tag_ == 'NN' and (v.tag_ == 'VBZ' or v.lower_ == 'was'))
                                  or (outer.tag_ == 'NNS' and (v.tag_ == 'VBP' or v.lower_ == 'were'))) \
                and (numbers is None or ('sg' if v.tag_ == 'VBZ' or v.lower_ == 'was' else 'pl') in numbers):
            continue
        if (subject.tag_ == 'NN' and (v.tag_ in ('VBP', 'VB') or v.lower_ == 'were')) or \
                (subject.tag_ == 'NNS' and (v.tag_ == 'VBZ' or v.lower_ == 'was')):
            out.add((subject.lower_, v.lower_))
    return out

# ...but these take a preposition instead ("The law provides for exceptions", "makes for a smooth ride")
PREPOSITIONAL_OBJECTS = {'provide': {'for'}, 'make': {'for'}, 'deliver': {'on'}}

def objectless_verbs(doc):
    """Lemmas of the verbs that need an object and have none ("The Industrial Revolution initiated in Britain",
    "help comfort over long sessions"). A verb whose phrase ends at it shares the object of a verb coordinated after
    it ("makes and sells bikes", "designs, makes and sells bikes"); "make sure that" has its complement."""
    def shares_object(t):
        return not any(c.i > t.i and c.dep_ not in ('cc', 'conj', 'punct') for c in t.children) and any(
            c.i > t.i and any(g.dep_ == 'dobj' for g in c.children) for c in t.conjuncts)
    return {t.lemma_.lower() for t in doc if t.pos_ == 'VERB' and t.lemma_.lower() in TRANSITIVE_ONLY
            and t.tag_ != 'VBN' and t.dep_ not in ('amod', 'acl', 'relcl', 'pcomp', 'advcl')
            and not any(c.dep_ in ('dobj', 'ccomp', 'xcomp', 'prt', 'dative', 'nsubjpass', 'auxpass', 'attr', 'oprd', 'acomp')
                        or (c.dep_ == 'prep' and c.lower_ in PREPOSITIONAL_OBJECTS.get(t.lemma_.lower(), ()))
                        for c in t.children)
            and not (t.dep_ == 'conj' and any(c.dep_ == 'dobj' for c in t.head.children)) and not shares_object(t)}

def participle_joined(doc):
    """True if a finite verb without a subject is joined to an opening participle that has none either ("Designed
    to be lightweight and has soft ear cushions, ...")."""
    return any(t.dep_ == 'conj' and t.tag_ in ('VBZ', 'VBP', 'VBD') and t.head.tag_ in ('VBN', 'VBG')
               and t.head.dep_ in ('advcl', 'acl') and t.head.left_edge.i == t.sent.start
               and not any(c.dep_ in ('nsubj', 'nsubjpass', 'expl') for c in t.children)
               and not any(c.dep_ in ('nsubj', 'nsubjpass', 'aux', 'auxpass', 'mark') for c in t.head.children)
               for t in doc)

# A pronoun's object form after it as the subject names a second person ("she nearly gives up" -> "she almost
# abandons her")
SUBJECT_OBJECT_PRONOUNS = {'she': 'her', 'he': 'him', 'they': 'them', 'we': 'us', 'i': 'me'}

def subject_as_object(doc):
    """True if a verb has a pronoun subject and that pronoun's object form as its object ("she abandons her")."""
    for t in doc:
        subject = next((c for c in t.children if c.dep_ == 'nsubj'), None) if t.pos_ == 'VERB' else None
        if subject is not None and subject.lower_ in SUBJECT_OBJECT_PRONOUNS and any(
                c.dep_ == 'dobj' and c.lower_ == SUBJECT_OBJECT_PRONOUNS[subject.lower_] for c in t.children):
            return True
    return False

# Particles of phrasal verbs whose noun is written with a hyphen ("follow-up", "set-up", "check-in", "roll-out")
HYPHENATED_VERB_PARTICLES = {'up', 'down', 'in', 'out', 'off', 'on', 'over', 'back', 'away', 'through'}

def hyphenated_verbs(tokens):
    """The phrasal verbs written as their noun and used as a verb ("to follow-up progress", "will set-up the
    device", "Our staff follow-up every complaint"): right after "to" or a modal, or with an auxiliary, or with a
    subject and an object (the tagger may take them for nouns). "a follow-up visit", "the set-up takes" and "look
    forward to follow-up talks" are nouns."""
    modifiers = ('amod', 'compound', 'nmod', 'poss')
    out = set()
    for t in tokens:
        doc = t.doc
        if not (t.is_alpha and t.i + 2 < len(doc) and doc[t.i + 1].text == '-' and not t.whitespace_
                and not doc[t.i + 1].whitespace_ and doc[t.i + 2].lower_ in HYPHENATED_VERB_PARTICLES):
            continue
        if ((t.i > 0 and doc[t.i - 1].tag_ in ('TO', 'MD') and not (t.dep_ in modifiers and t.head.i > t.i + 2))
                or (t.dep_ not in modifiers and (any(c.dep_ == 'aux' for c in t.children)
                                                 or {'nsubj', 'dobj'} <= {c.dep_ for c in t.children}))):
            out.add(t.lower_ + '-' + doc[t.i + 2].lower_)
    return out

def garbled(sent, cand_doc):
    """Word salad the T5 scores don't catch: a disagreeing verb, "the primary driving a 12 percent rise", "said
    that however,", "costs almost 18 percent rose.", "the small of contributions", a verb missing its object
    ("initiated in Britain"), "Designed to be lightweight and has ...", "she almost abandons her", "to follow-up
    progress"."""
    if number_disagreements(cand_doc, which_numbers(sent)) - number_disagreements(sent):
        return True
    if (objectless_verbs(cand_doc) - objectless_verbs(sent) or (participle_joined(cand_doc) and not participle_joined(sent))
            or (subject_as_object(cand_doc) and not subject_as_object(sent))
            or hyphenated_verbs(cand_doc) - hyphenated_verbs(sent)):
        return True
    toks = [t for t in cand_doc if not t.is_space]
    for k, (a, b, c) in enumerate(zip(toks, toks[1:], toks[2:])):
        if ((a.pos_ == 'DET' or a.tag_ == 'JJS' or (a.pos_ == 'ADJ' and k and toks[k - 1].pos_ == 'DET'))
                and b.tag_ == 'VBG' and c.lower_ in ('a', 'an', 'the')):
            return True
    if AFTER_THAT.search(cand_doc.text) and not AFTER_THAT.search(sent.text):
        return True
    words = [t for t in cand_doc if not t.is_punct and not t.is_space]
    if (len(words) >= 2 and words[-1].tag_ in ('VBD', 'VBZ', 'VBP')
            and (words[-2].like_num or words[-2].lemma_.lower() in ('percent', 'percentage', 'point', 'cent'))):
        return True
    if (len(words) >= 2 and words[-1].pos_ == 'VERB' and words[-1].lemma_.lower() in TRANSITIVE_ONLY and not any(
            c.dep_ in ('dobj', 'ccomp', 'xcomp', 'prep', 'prt', 'dative', 'nsubjpass', 'auxpass') for c in words[-1].children)
            and words[-1].dep_ not in ('amod', 'acl', 'relcl', 'xcomp', 'advcl', 'pcomp')):
        return True
    adjective_heads = lambda d: {t.lower_ for t in d if t.pos_ == 'ADJ' and t.tag_ == 'JJ'
                                 and any(c.dep_ == 'det' for c in t.children)
                                 and any(c.dep_ == 'prep' and c.lower_ == 'of' for c in t.children)}
    return bool(adjective_heads(cand_doc) - adjective_heads(sent))

def dropped_article(sent, cand_doc):
    """True if a noun lost its article after a preposition ("in the modern world" -> "in modern world", "the
    importance of the discovery" -> "the importance of discovery"). A dropped possessive is no dropped article
    ("focus on your music" -> "focus on music", "25 percent of its value" -> "25 percent in value")."""
    had = {t.lemma_.lower() for t in sent if t.tag_ == 'NN' and t.dep_ == 'pobj'
           and any(c.dep_ == 'det' and c.lower_ in ('the', 'a', 'an') for c in t.children)}
    bare = {t.lemma_.lower() for t in sent if t.tag_ == 'NN' and not any(c.dep_ in ('det', 'poss') for c in t.children)}
    return any(t.tag_ == 'NN' and t.dep_ == 'pobj' and t.lemma_.lower() in had - bare
               and not any(c.dep_ in ('det', 'poss', 'nummod') for c in t.children)
               for t in cand_doc)

SOURCE_NOUNS = {'officer', 'expert', 'report', 'study', 'survey', 'data', 'researcher', 'scientist', 'official',
                'government', 'department', 'ministry', 'mayor', 'minister', 'president', 'author', 'source', 'analyst',
                'police', 'spokesperson', 'spokesman', 'spokeswoman', 'agency', 'company', 'research', 'statement', 'poll',
                'estimate', 'figure', 'record', 'census', 'organization', 'organisation', 'institute', 'university',
                'committee', 'council', 'board', 'court', 'witness', 'resident', 'critic', 'economist', 'doctor', 'paper'}

def attributed(tokens, text):
    """True if a sentence reports what someone said ("residents said the detour would ...", "according to the
    mayor"); "needs vary according to climate" reports nothing."""
    tokens = list(tokens)
    if AS_REPORTED.search(text):
        return True
    for t in tokens:
        if t.lower_ == 'according' and t.i + 1 < len(t.doc) and t.doc[t.i + 1].lower_ == 'to':
            obj = next((c for c in t.doc[t.i + 1].children if c.dep_ == 'pobj'), None)
            if obj is not None and (obj.ent_type_ in ('PERSON', 'ORG', 'NORP', 'GPE') or obj.pos_ == 'PROPN'
                                    or obj.lemma_.lower() in SOURCE_NOUNS or obj.lemma_.lower() in PERSON_NOUNS):
                return True
        if t.lower_ == 'per' and t.i and t.doc[t.i - 1].lower_ == 'as':
            return True
        if acts_as_verb(t) and t.lemma_.lower() in REPORTING_VERBS and (
                t.dep_ == 'parataxis' or any(c.dep_ == 'ccomp' for c in t.children)):
            subject = verb_subject(t)
            if t.dep_ == 'parataxis' or (subject is not None and subject.lower_ != 'it'):
                return True
    return False

def strengths(tokens):
    """How strongly a sentence recommends something: 'needed', 'helpful', 'optional' ("is essential", "a key role")."""
    out = set()
    for t in tokens:
        kind = STRENGTH.get(t.lemma_.lower()) or STRENGTH.get(t.lower_)
        if kind and t.pos_ == 'ADJ' and (t.dep_ in ('acomp', 'oprd', 'attr') or (t.dep_ == 'conj' and t.head.dep_ in (
                'acomp', 'oprd')) or (t.dep_ == 'amod' and t.head.lemma_.lower() in LIGHT_NOUNS | {
                'step', 'component', 'element', 'ingredient', 'priority', 'requirement', 'skill', 'tool'})):
            out.add(kind)
    return out

HYPHENATED = re.compile(r"[\w'’]+(?:-[\w'’]+)+")

def margin_given(token):
    """True if a word has a margin hanging off it: "by three points", "by a goal", or "five points" before it (the
    parser may read "Brazil led by two goals" as a passive, but goals lead no one)."""
    for c in token.children:
        if c.lower_ == 'by' and any(g.dep_ == 'pobj' and (g.like_num or g.lemma_.lower() in POSITION_MARGINS or (
                c.dep_ == 'prep' and any(x.dep_ == 'nummod' for x in g.children))) for g in c.children):
            return True
        if c.dep_ == 'npadvmod' and any(x.dep_ == 'nummod' for x in c.children):
            return True
    return False

def standing_kept(token):
    """True if the verb "lead" or "trail" (or a "trail" the tagger took for a noun) says where something stands: it
    has a margin, a score, a table as its object or after "in"/"at", or, as a clause's verb, nothing that makes it
    "go" or "cause" ("leads to", "led into") or "guide" (an object). Not "the leading cause" or "was led by"."""
    kids = list(token.children)
    if token.dep_ in ('amod', 'compound', 'nmod') or any(c.dep_ in ('auxpass', 'nsubjpass') for c in kids):
        return False
    if margin_given(token) or any(c.pos_ == 'NUM' and c.dep_ in ('npadvmod', 'dobj', 'advmod') for c in kids):
        return True         # "lead Arsenal by three points", "led 2-1"
    if any(c.dep_ == 'agent' for c in kids):
        return False        # "the team, led by Dr. Smith"
    tables = [c for c in kids if c.dep_ == 'dobj'] + [g for c in kids if c.dep_ == 'prep' and c.lower_ in ('in', 'at', 'atop')
                                                      for g in c.children if g.dep_ == 'pobj']
    if any({t.lemma_.lower(), t.lower_} & STANDINGS_NOUNS for t in tables):
        return True         # "have led the table", "trailed in the polls"
    return token.pos_ == 'VERB' and not any(
        c.dep_ in ('dobj', 'ccomp', 'xcomp', 'prt', 'dative') or (c.dep_ == 'prep' and c.lower_ in LEAD_ROUTES) for c in kids)

def position_side(token):
    """0 for a word that puts something behind ("behind", "below", "trails"), 1 for one that puts it ahead
    ("ahead", "above", "a lead", "lead Arsenal by three points", "five points clear", "top the table"), else None.
    A "trail" with an object, or a bare one that is no part of a noun phrase, is the verb whatever the tagger says
    ("Sheffield trail the leaders", "United trail at half-time")."""
    lemma = token.lemma_.lower()
    if lemma in POSITION_BEHIND or token.lower_ in POSITION_BEHIND:
        return 0
    if lemma in POSITION_AHEAD or token.lower_ in POSITION_AHEAD:
        return 1
    if lemma == 'trail':
        if token.pos_ == 'VERB':
            return 0
        if any(c.dep_ in ('det', 'poss', 'amod', 'nummod') for c in token.children):
            return None     # "the trail", "a hiking trail"
        bare = (token.tag_ == 'NN' and token.dep_ not in ('pobj', 'dobj', 'attr', 'appos')
                and not (token.dep_ in ('compound', 'nmod') and token.head.pos_ == 'NOUN'))
        return 0 if bare or any(c.dep_ == 'dobj' for c in token.children) or standing_kept(token) else None
    if lemma == 'lead':
        return 1 if token.pos_ == 'NOUN' or standing_kept(token) else None
    if lemma == 'clear':
        return 1 if token.pos_ == 'ADJ' and margin_given(token) else None
    if lemma in TABLE_TOPPING_VERBS and (token.pos_ == 'VERB' or token.dep_ == 'ROOT'):
        return 1 if margin_given(token) or any(
            c.dep_ == 'dobj' and {c.lemma_.lower(), c.lower_} & STANDINGS_NOUNS for c in token.children) else None
    return None

def opposite_added(sent, cand_doc):
    """True if the rewrite brings in a word whose opposite the original uses (and the original doesn't use both):
    "absorb" -> "release", "began" -> "ended", "frequent" -> "rare", "two points behind" -> "a two-point lead"; or,
    where the original uses both, uses one more and the other less ("1.6 times more likely ... less than two
    hours" -> "1.6 times less likely"). A negated "allow" counts as "prevent". The parts of a hyphenated word both
    versions use ("low-lying") and the words of a bound ("at least 10 seconds" -> "10 seconds or more") don't count."""
    shared = {w.lower() for w in HYPHENATED.findall(sent.text)} & {w.lower() for w in HYPHENATED.findall(cand_doc.text)}
    def kept_tokens(doc):
        start = doc[0].idx if len(doc) else 0
        skip = [m.span() for _, pattern in BOUNDS for m in pattern.finditer(doc.text)]
        skip += [m.span() for m in HYPHENATED.finditer(doc.text) if m.group(0).lower() in shared]
        return [t for t in doc if not any(s <= t.idx - start < e for s, e in skip)]
    def sides(toks, text):
        words = ({t.lemma_.lower() for t in toks} | {t.lower_ for t in toks}
                 | {w.lower() for w in plain_words(text) if '-' in w}
                 | ({'open'} if IN_SESSION.search(text) else set()))
        words |= {v for t in toks if t.pos_ == 'NOUN' and t.lower_.endswith('ing') for v in getAllLemmas(t.lower_).get('VERB', ())}
        flipped = {t.lemma_.lower() for t in toks if t.lemma_.lower() in ENABLE_WORDS | RESTRICT_WORDS
                   and any(c.dep_ == 'neg' for c in t.children)}
        return words & OPPOSITE_WORDS, flipped, {t.lemma_.lower() for t in toks if t.pos_ == 'VERB'}
    def counts(toks, group):
        a, b = group[0], group[1]
        toks = [t for t in toks if t.pos_ == 'VERB'] if len(group) > 2 else toks
        return (sum(bool({t.lemma_.lower(), t.lower_} & a) for t in toks),
                sum(bool({t.lemma_.lower(), t.lower_} & b) for t in toks))
    old_toks, new_toks = kept_tokens(sent), kept_tokens(cand_doc)
    (old, old_flipped, old_verbs), (new, new_flipped, new_verbs) = sides(old_toks, sent.text), sides(new_toks, cand_doc.text)
    for group in OPPOSITES:
        a, b = group[0], group[1]
        if len(group) > 2:      # "began" (a verb) is no opposite of "the end of the century"
            old_a, old_b, new_a, new_b = old_verbs & a, old_verbs & b, new_verbs & a, new_verbs & b
        else:
            old_a, old_b, new_a, new_b = old & a, old & b, new & a, new & b
        if a is ENABLE_WORDS:       # "does not allow" is "prevents"; "does not prevent" is "allows"
            old_a, old_b = (old_a - old_flipped) | (old_b & old_flipped), (old_b - old_flipped) | (old_a & old_flipped)
            new_a, new_b = (new_a - new_flipped) | (new_b & new_flipped), (new_b - new_flipped) | (new_a & new_flipped)
        if (new_b and not old_b and old_a) or (new_a and not old_a and old_b):
            return True
        if a is not ENABLE_WORDS:
            (old_a, old_b), (new_a, new_b) = counts(old_toks, group), counts(new_toks, group)
            if (new_b > old_b and new_a < old_a) or (new_a > old_a and new_b < old_b):
                return True
    old_sides, new_sides = [position_side(t) for t in old_toks], [position_side(t) for t in new_toks]
    (old_a, old_b), (new_a, new_b) = (old_sides.count(0), old_sides.count(1)), (new_sides.count(0), new_sides.count(1))
    return bool((new_b and not old_b and old_a) or (new_a and not old_a and old_b)
                or (new_b > old_b and new_a < old_a) or (new_a > old_a and new_b < old_b))

# A lower risk is not prevention: "reduce the risk of X" keeps its risk, chance or likelihood
REDUCED_RISK = re.compile(r"\b(?:reduc|lower|cut|decreas|minimi[sz]|lessen|halv)\w*\b(?:\s+\w+){0,3}?\s+(?:risks?|chances?|"
                          r"likelihood|odds|probability|incidence)\s+of\b", re.I)
RISK_WORDS = re.compile(r"\b(?:risks?|chances?|likelihood|odds|probability|incidence|likely|likelier)\b", re.I)
# What nearly happened did not happen: "Brazil regained the lead" is not "Brazil almost regained the lead"
NEAR_MISSES = {'almost': 'almost', 'nearly': 'almost', 'barely': 'barely', 'hardly': 'barely', 'scarcely': 'barely'}

def claim_added(sent, cand_doc, source_vocab):
    """True if the rewrite adds a claim: a claim word ("optimal", "successfully", "a preliminary study", "the late
    Henri Becquerel", "a different value"), a degree the original doesn't give ("significantly lessened"), an
    outcome of an expectation ("analysts had expected" -> "expectations were met"), prevention for a lower risk
    ("reduce the risk of" -> "prevent"), a near miss ("regained" -> "almost regained"), or a mere effect for a
    direction ("reduce the risk" -> "have an impact on the risk")."""
    stems = {stem(w) for w in source_vocab} - {None}
    for t in cand_doc:
        w = t.lower_
        if w in ADDED_CLAIMS and w not in source_vocab and t.lemma_.lower() not in source_vocab and stem(w) not in stems \
                and not CLAIM_EQUIVALENTS.get(w, set()) & source_vocab:
            return True
        if w == 'late' and t.dep_ == 'amod' and t.head.pos_ == 'PROPN' and 'late' not in source_vocab:
            return True
        if (w == 'different' and t.dep_ == 'amod' and t.head.tag_ == 'NN' and any(c.lower_ in ('a', 'an') for c in t.head.children)
                and not {'different', 'new', 'another', 'other', 'alternative', 'separate', 'various', 'varied', 'distinct',
                         'diverse', 'differ'} & source_vocab):
            return True
    intensity = source_vocab & INTENSITY_WORDS
    if 'marked' in intensity and not any(t.lower_ == 'marked' and t.pos_ == 'ADJ' for t in sent):
        intensity.discard('marked')     # "marked a turning point" says nothing of how much
    if not intensity and any(t.lower_ in INTENSIFIERS and t.lower_ not in source_vocab for t in cand_doc):
        return True
    if REDUCED_RISK.search(sent.text) and not RISK_WORDS.search(cand_doc.text):
        return True     # "can reduce the risk of a second heart attack" -> "can prevent a second heart attack"
    near = {NEAR_MISSES[t.lower_] for t in sent if t.lower_ in NEAR_MISSES}
    if any(t.lower_ in NEAR_MISSES and NEAR_MISSES[t.lower_] not in near and t.dep_ == 'advmod' and t.head.pos_ == 'VERB'
           for t in cand_doc):
        return True     # "Brazil regained the lead" -> "Brazil almost regained the lead"
    old_lemmas = {t.lemma_.lower() for t in sent}
    if old_lemmas & EXPECTING and any(t.pos_ == 'VERB' and t.lemma_.lower() in OUTCOMES and t.lemma_.lower() not in old_lemmas
                                      for t in cand_doc):
        return True
    new_lemmas = {t.lemma_.lower() for t in cand_doc}
    if old_lemmas & {'contribute', 'contributor', 'contribution'} and not new_lemmas & {
            'contribute', 'contributor', 'contribution', 'factor', 'partly', 'part', 'role', 'partially'}:
        return True     # "pesticides ... are all contributing to the decline" -> "the decline is caused by pesticides"
    for m in re.finditer(r"\bstatistically\s+(\w+)", sent.text, re.I):
        if m.group(1).lower() not in cand_doc.text.lower():
            return True     # "not statistically significant" -> "not statistically sound"
    if 'take' in old_lemmas and old_lemmas & MEDICINES and (new_lemmas & EATING) - old_lemmas:
        return True     # "people who take insulin" -> "those who consume insulin"
    if any(WRONG_SENSE[w] & new_lemmas - old_lemmas for w in old_lemmas & WRONG_SENSE.keys()):
        return True     # "address rising rents" -> "accommodate the increase in rents"
    if any(t.pos_ == 'VERB' and t.lemma_.lower() in WRONG_VERB_SENSE[w] - old_lemmas
           for w in old_lemmas & WRONG_VERB_SENSE.keys() for t in cand_doc):
        return True     # "must still be approved" -> "required to bill the treaty"
    old_content = {t.lemma_.lower() for t in sent if t.pos_ in ('VERB', 'NOUN')}
    new_content = {t.lemma_.lower() for t in cand_doc if t.pos_ in ('VERB', 'NOUN')}
    return bool(old_content & DIRECTED) and not new_content & DIRECTED and bool(new_content & UNDIRECTED - old_content)

def causal_cue(sent):
    """True if a sentence already relates cause and effect: a causal word or verb, ", which" + verb or a participle
    clause ("..., making it easy", "Using chlorophyll, plants ...")."""
    toks = [t for t in sent if not t.is_space]
    for k, t in enumerate(toks):
        lemma = t.lemma_.lower()
        if t.lower_ in CAUSAL_WORDS or (lemma in CAUSAL_WORDS and t.pos_ == 'NOUN'):
            return True
        if t.pos_ == 'VERB' and lemma in CAUSAL_VERBS and (lemma not in ('make', 'keep', 'leave') or any(
                c.dep_ in ('ccomp', 'xcomp', 'oprd') for c in t.children)):
            return True
        if t.dep_ == 'mark' and t.lower_ in ('as', 'since', 'because', 'if', 'when', 'whenever', 'once', 'unless'):
            return True
        if t.lower_ == 'which' and k and toks[k - 1].text == ',' and t.head.tag_ in FINITE_TAGS:
            return True
        if t.tag_ == 'VBG' and t.dep_ in ('advcl', 'xcomp', 'acl', 'conj') and (k == 0 or toks[k - 1].text == ','):
            return True
    return False

def changes_stated(sent):
    """(noun lemmas, object lemmas) of the clauses on a noun that say what it changed ("new machines that
    dramatically increased productivity"). One on the subject only picks out a group ("Patients who lowered their
    salt intake had fewer heart problems"), and states no cause of what the sentence says of them."""
    out = []
    for t in sent:
        if (t.pos_ == 'VERB' and t.dep_ in ('relcl', 'acl') and t.lemma_.lower() in DIRECTED
                and t.head.dep_ not in ('nsubj', 'nsubjpass')):
            objects = {c.lemma_.lower() for c in t.children if c.dep_ == 'dobj'}
            if objects:
                out.append(({x.lemma_.lower() for x in (t.head, *t.head.conjuncts)}, objects))
    return out

# Verbs by which a sentence's subject only finds, follows or names what a clause on its object did ("The study
# tracked 500 firms that cut their energy use"): the subject caused none of it
FINDING_VERBS = {'find', 'follow', 'track', 'study', 'survey', 'identify', 'report', 'name', 'list', 'visit', 'interview',
                 'observe', 'record', 'see', 'note', 'count', 'compare', 'praise', 'rank', 'describe', 'mention',
                 'highlight', 'cite', 'examine', 'analyse', 'analyze', 'measure', 'monitor', 'include', 'show', 'reveal',
                 'feature', 'select', 'choose', 'contact', 'ask', 'question', 'poll', 'meet', 'reward', 'thank'}

def only_observes(sent):
    """True if the main verb only finds, follows or reports what the sentence tells ("The survey found many firms
    that raised wages", "Officials said ...", "Studies suggest ...")."""
    lemma = sent.root.lemma_.lower()
    return lemma in FINDING_VERBS or lemma in REPORTING_VERBS or bool(EVIDENCE.get(lemma))

def change_restated(sent, cand_doc):
    """True if every causal link of the rewrite restates a change a clause of the original states: the thing that
    changed something on the cause side and what it changed on the effect side ("machines that increased
    productivity" -> "The introduction of new machines led to an increase in productivity", "Productivity rose
    because of the new machines"), or keeps that change whole on the effect side of the sentence's own subject ("It
    resulted in the introduction of new machines that boosted productivity"); not what a group did as the cause
    of what the sentence says of them ("Fewer heart problems were seen in patients who lowered their salt intake"
    -> "Patients had fewer heart problems because they lowered their salt intake"), nor the change told of a group
    that the subject only found or followed as the subject's doing ("The study tracked 500 firms that cut their
    energy use" -> "... as a result of the study")."""
    stated = changes_stated(sent)
    if not stated or not len(cand_doc):
        return False
    observed = only_observes(sent)
    subjects = set() if observed else {c.lemma_.lower() for c in sent.root.children if c.dep_ in ('nsubj', 'nsubjpass')}
    text, start = cand_doc.text, cand_doc[0].idx
    links = [(m.start(), m.end(), bool(CAUSE_AFTER.match(text, m.start()))) for m in CAUSAL_LINK.finditer(text)]
    by = BY_DOING.match(text)
    if by and ',' in text[by.end():] and not observed:
        comma = text.index(',', by.end())
        links.append((comma, comma + 1, False))     # "By introducing new machines, it increased productivity"
    def words(lo, hi):
        toks = [t for t in cand_doc if lo <= t.idx - start < hi]
        return {t.lower_ for t in toks}, {t.lemma_.lower() for t in toks}
    def holds(lemmas, side):
        return any(kept_word(w, 'n', *side) for w in lemmas)
    for lo, hi, cause_after in links:
        before, after = words(0, lo), words(hi, len(text))
        cause, effect = (after, before) if cause_after else (before, after)
        if not any((holds(nouns, cause) or (holds(subjects, cause) and holds(nouns, effect))) and holds(objects, effect)
                   for nouns, objects in stated):
            return False
    return bool(links)

def observer_made_cause(sent, cand_doc):
    """True if a link of the rewrite gives as the cause the subject of an original whose main verb only finds,
    follows or reports ("The survey found many firms that raised wages" -> "Many firms raised wages because of the
    survey"). The cause side is the link's own phrase, up to the next comma (or, for a link with the cause before
    it, from the last comma before it): "Journey times fell because of the new road, officials said" names the road,
    not the officials."""
    if not only_observes(sent):
        return False
    subjects = {c.lemma_.lower() for c in sent.root.children if c.dep_ in ('nsubj', 'nsubjpass') and c.pos_ in ('NOUN', 'PROPN')}
    text, start = cand_doc.text, cand_doc[0].idx if len(cand_doc) else 0
    for m in CAUSAL_LINK.finditer(text):
        if CAUSE_AFTER.match(text, m.start()):
            side = []
            for t in cand_doc:
                if t.idx - start < m.end():
                    continue
                if t.text in (',', ';', '.'):
                    break
                side.append(t)
        else:
            cut = max(text.rfind(',', 0, m.start()), text.rfind(';', 0, m.start()))
            side = [t for t in cand_doc if cut < t.idx - start < m.start()]
        words, lemmas = {t.lower_ for t in side}, {t.lemma_.lower() for t in side}
        if any(kept_word(w, 'n', words, lemmas) for w in subjects):
            return True
    return False

def causal_added(sent, cand_doc):
    """True if the rewrite brings in cause and effect the original doesn't state ("..., resulting in a drop", "By
    approving a buyback, the board raised ..."), or turns them around (the original's subject, a cause, becomes
    the effect: "costs rose, which narrowed the margin" -> "the margin narrowed, resulting in higher costs"), or
    makes the one who only found or followed a group the cause of what the group did ("The survey found many firms
    that raised wages" -> "Many firms raised wages because of the survey")."""
    text = cand_doc.text
    cued = causal_cue(sent)
    cue = cued or change_restated(sent, cand_doc)
    if PARTICIPLE_EFFECT.search(text) and not PARTICIPLE_EFFECT.search(sent.text) and not cue:
        return True     # "When the bridge opened in 1883, it was the longest" -> "..., making it the longest"
    if not (CAUSAL_LINK.search(text) or BY_DOING.match(text)):
        return False
    if not cued and observer_made_cause(sent, cand_doc):
        return True
    if not cue:
        if BY_DOING.match(text):
            return True
        # the link may name the original's own actor as the cause ("the move ... cut hosting expenses" -> "hosting
        # costs fell due to the move", "social media has changed X" -> "social media has caused a change in X")
        actors = {t.lemma_.lower() for t in sent if t.dep_ in ('nsubj', 'nsubjpass') and t.pos_ in ('NOUN', 'PROPN')}
        causes = []
        for m in CAUSE_NP.finditer(text):
            causes.append(next((t for t in cand_doc if t.idx >= m.end() and t.pos_ in ('NOUN', 'PROPN')
                                and t.dep_ not in ('compound', 'amod', 'poss', 'nmod')), None))
        for t in cand_doc:
            if t.pos_ == 'VERB' and t.lemma_.lower() == 'cause':
                causes.append(next(iter(sorted(doers(t))), None) if doers(t) else None)
        links = len(CAUSAL_LINK.findall(text))
        named = [c for c in causes if c is not None]
        return not (named and len(named) >= links and all(
            kept_word(c if isinstance(c, str) else c.lemma_.lower(), 'n', actors, actors) for c in named))
    m = EFFECT_AFTER.search(text)
    if m:
        subjects = {c.lemma_.lower() for c in sent.root.children if c.dep_ in ('nsubj', 'nsubjpass') and c.pos_ in ('NOUN', 'PROPN')}
        cause_side = {t.lemma_.lower() for t in cand_doc if t.idx < m.start()}
        effect_side = {t.lemma_.lower() for t in cand_doc if t.idx >= m.end()}
        if any(w in effect_side and w not in cause_side for w in subjects):
            return True
    return False

def time_phrase_objects(sent):
    """Lemmas and words of what the sentence's time phrases order by: the object of each one's last word ("Ahead of
    the royal visit" -> visit, "on the eve of the final" -> final), else the next noun, describing word or number."""
    start = sent[0].idx if len(sent) else 0
    out = set()
    for m in TIME_ORDER_PHRASES.finditer(sent.text):
        last = next((t for t in sent if t.idx - start < m.end() <= t.idx - start + len(t.text)), None)
        objects = [c for c in last.children if c.dep_ == 'pobj'] if last is not None else []
        if not objects:
            objects = [next((t for t in sent if t.idx - start >= m.end() and t.pos_ in ('NOUN', 'PROPN', 'ADJ', 'NUM')), None)]
        out |= {w for o in objects if o is not None for w in (o.lemma_.lower(), o.lower_)}
    return out

def connective_added(sent, cand_doc):
    """True if the rewrite relates the sentence to its neighbours in a new way: a contrast ("However, the authors
    warn ..."), an example ("For example, ..."), an addition ("As part of a deal, ...", "Additionally, ...") or a
    time limit in a sentence that orders nothing in time ("When it opened" -> "Until its opening"); or drops the
    contrast the original draws with the sentence before ("However, ...")."""
    old, new = sent.text, cand_doc.text
    if CONTRAST_ADDED.search(new) and not CONTRAST_INSIDE.search(old):
        return True
    if (CONTRAST_OPENERS.match(old) or CONTRAST_ASIDE.search(old)) and not CONTRAST_INSIDE.search(new):
        return True     # a contrast with the sentence before, dropped ("However, the transition ..." -> "The transition ...")
    if EXAMPLE_OPENERS.match(new) and not EXAMPLE_INSIDE.search(old):
        return True
    # ("before" + a gerund only puts two actions in order: "hoists the flag and addresses the nation" -> "hoists the
    # flag before addressing the nation"). A time phrase of the original licenses only a "before" on the same thing
    # ("Ahead of the royal visit" -> "Before the royal visit"): "ahead of" also means in front of ("well ahead of its
    # rivals when it was sold" is not "Until its sale, ... ahead of its rivals").
    if not {t.lower_ for t in sent} & TIME_ORDER_WORDS:
        limits = [t for t in cand_doc if t.lower_ in TIME_LIMITS and t.dep_ in ('mark', 'prep')
                  and not any(c.dep_ == 'pcomp' for c in t.children)]
        objects = time_phrase_objects(sent) if limits else set()
        if any(not (t.lower_ == 'before' and any(c.dep_ == 'pobj' and kept_word(c.lemma_.lower(), 'n', objects, objects)
                                                for c in t.children)) for t in limits):
            return True
    return bool(ADDING_OPENERS.match(new) and not ADDING_INSIDE.search(old))

def counted(tokens):
    """number -> the noun lemmas it counts in an "of" phrase ("99 percent of lead and chlorine")."""
    out = {}
    for t in tokens:
        if not (t.like_num or any(c.isdigit() for c in t.text)):
            continue
        noun = t.head if t.dep_ in ('nummod', 'quantmod', 'compound') else None
        while noun is not None and noun.dep_ in ('nummod', 'quantmod') and noun.head.i != noun.i:
            noun = noun.head
        if noun is None:
            continue
        of = next((c for c in noun.children if c.dep_ == 'prep' and c.lower_ == 'of'), None)
        obj = next((c for c in of.children if c.dep_ == 'pobj'), None) if of is not None else None
        if obj is not None:
            out[t.text] = {g.lemma_.lower() for g in [obj, *obj.conjuncts]}
    return out

def restricting_words(doc):
    """Lemmas of what picks out some of a group: a "who" or "that" clause ("participants who fasted"), a participle
    ("staff working from home") or a "with" or "on" phrase ("those on insulin"); not a clause with a subject of its
    own ("a trial where they lost ...")."""
    out = set()
    for t in doc:
        if t.pos_ not in ('NOUN', 'PRON', 'PROPN'):
            continue
        for c in t.children:
            if ((c.dep_ == 'relcl' and c.pos_ in ('VERB', 'AUX') and any(
                    g.dep_ in ('nsubj', 'nsubjpass') and g.tag_ in ('WP', 'WDT') for g in c.children))
                    or (c.dep_ == 'acl' and c.tag_ in ('VBG', 'VBN'))):
                out.add(c.lemma_.lower())
            elif c.dep_ == 'prep' and c.lower_ in ('with', 'on'):
                out.add(c.lower_)
    return out

def scope_changed(sent, cand_doc):
    """True if a number now counts something else ("99 percent of lead and chlorine" -> "99 percent of tap water"),
    or a group the original narrows down ("those who fasted", "those with diabetes") is no longer narrowed by the
    same thing ("214 participants, those who fasted" -> "214 individuals ... by fasting"). The restriction survives
    in a clause or phrase on a noun ("those residing near the coast", "people who have received vaccinations") or on
    the subject itself ("smokers", "diabetics", "diabetes patients", "vaccinated people"); not in an adverbial on the
    verb ("lost weight by fasting") or a noun of its own ("Regular exercise helps people sleep")."""
    old, new = counted(sent), counted(cand_doc)
    if any(num in new and not any(kept_word(o, 'n', new[num], new[num]) for o in objs) for num, objs in old.items()):
        return True
    def nouns(phrase):
        return {g.lemma_.lower() for g in phrase.subtree if g.pos_ in ('NOUN', 'PROPN') and g.dep_ in ('dobj', 'pobj')
                and g.lemma_.lower() not in TIME_NOUNS}
    narrowed = {}       # the restriction -> the words that carry it: its verb and the nouns it names
    for t in sent:
        if t.lower_ == 'those' and t.pos_ == 'PRON':
            for c in t.children:
                if c.dep_ in ('relcl', 'acl') and c.pos_ in ('VERB', 'AUX'):
                    narrowed[c.lemma_.lower()] = {c.lemma_.lower()} | nouns(c)
                elif c.dep_ == 'prep' and c.lower_ in ('with', 'on'):
                    narrowed[c.lower_] = nouns(c)
    if not narrowed:
        return False
    kept = restricting_words(cand_doc)
    carried, subjects = set(), set()      # the words of the candidate's restrictions, and its subject nouns
    for t in cand_doc:
        if t.pos_ not in ('NOUN', 'PRON', 'PROPN'):
            continue
        for c in t.children:
            if ((c.dep_ == 'relcl' and any(g.dep_ in ('nsubj', 'nsubjpass') and g.tag_ in WH_WORDS for g in c.children))
                    or (c.dep_ == 'acl' and c.tag_ in ('VBG', 'VBN')) or (c.dep_ == 'prep' and c.lower_ in ('with', 'on'))):
                carried |= {g.lower_ for g in c.subtree} | {g.lemma_.lower() for g in c.subtree}
        if t.dep_ in ('nsubj', 'nsubjpass'):
            carried |= {w for c in t.children if c.dep_ in ('compound', 'amod') for w in (c.lower_, c.lemma_.lower())}
            subjects.add(t.lemma_.lower())
    def survives(key, words):
        return kept_word(key, 'v', kept, kept) or any(
            kept_word(w, 'n', carried, carried) or kept_word(w, 'v', carried, carried)
            or any(s != w and stem(s) is not None and stem(s) == stem(w) for s in subjects) for w in words)
    return not any(survives(key, words) for key, words in narrowed.items())

# Different things of one kind: a rewrite may not put one in place of another
SIBLINGS = [
    {'town', 'city', 'village', 'suburb', 'county', 'province', 'capital'}, {'mother', 'father'}, {'son', 'daughter'},
    {'brother', 'sister'}, {'husband', 'wife'}, {'boy', 'girl'}, {'man', 'woman'}, {'grandmother', 'grandfather'},
    {'aunt', 'uncle'}, {'king', 'queen'}, {'prince', 'princess'}, {'doctor', 'nurse', 'surgeon', 'pharmacist'},
    {'teacher', 'student', 'pupil'}, {'buyer', 'seller'}, {'employer', 'employee'}, {'landlord', 'tenant'},
    {'parent', 'child'}, {'patient', 'doctor'}, {'author', 'reader'}, {'host', 'guest'}, {'winner', 'loser'},
    {'lender', 'borrower'}, {'plaintiff', 'defendant'}, {'predator', 'prey'}, {'morning', 'afternoon', 'evening', 'night'},
    {'spring', 'summer', 'autumn', 'fall', 'winter'},
    {'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'},
    {'second', 'minute', 'hour', 'day', 'week', 'month', 'year', 'decade', 'century'}, {'breakfast', 'lunch', 'dinner'},
    {'car', 'truck', 'bus', 'train', 'plane', 'bicycle', 'bike', 'ship', 'boat', 'tram', 'subway'}, {'cat', 'dog'},
    {'bee', 'wasp', 'butterfly', 'ant'}, {'mouse', 'rat'}, {'heart', 'lung', 'liver', 'kidney', 'brain', 'stomach'},
    {'arm', 'leg'}, {'hand', 'foot'}, {'gold', 'silver', 'bronze', 'copper', 'iron', 'steel', 'aluminum', 'aluminium'},
    {'wood', 'metal', 'plastic', 'glass', 'paper', 'concrete'}, {'tea', 'coffee'}, {'wine', 'beer'},
    {'physics', 'chemistry', 'biology', 'medicine', 'mathematics', 'economics', 'literature'}, {'history', 'geography'},
    {'oxygen', 'hydrogen', 'nitrogen', 'carbon', 'helium'}, {'glucose', 'sucrose', 'fructose', 'starch'},
    {'dollar', 'euro', 'pound', 'yen', 'rupee'},
    {'cancer', 'diabetes', 'stroke', 'asthma', 'obesity', 'dementia', 'heart disease', 'heart attack', 'arthritis',
     'depression', 'anxiety', 'hypertension', 'alzheimer'}, {'save', 'invest', 'spend', 'borrow', 'lend', 'donate'},
    {'heatwave', 'drought', 'flood', 'storm', 'hurricane', 'wildfire', 'earthquake'},
    {'wind', 'solar', 'hydro', 'nuclear', 'coal', 'gas', 'oil'}, {'bacterium', 'bacteria', 'virus', 'fungus', 'mold',
                                                                  'mould', 'yeast', 'parasite'},
    {'aspirin', 'ibuprofen', 'paracetamol', 'acetaminophen', 'insulin', 'penicillin'},
    {'laptop', 'desktop', 'tablet', 'phone', 'smartphone', 'television'}, {'weekday', 'weekend'},
    {'rain', 'snow', 'hail', 'fog'}, {'ocean', 'sea', 'lake', 'river'}, {'mountain', 'hill', 'valley'},
    {'castle', 'temple', 'church', 'palace', 'cathedral', 'mosque'}, {'school', 'university', 'college', 'kindergarten'},
    {'elementary', 'primary', 'secondary', 'high', 'middle'}, {'undergraduate', 'graduate', 'postgraduate'},
    {'quarter', 'semester'}, {'mile', 'kilometer', 'kilometre', 'meter', 'metre', 'foot', 'inch'},
    {'kilogram', 'gram', 'pound', 'ounce', 'ton', 'tonne'}, {'liter', 'litre', 'milliliter', 'millilitre', 'gallon'},
]

def sibling_swapped(sent, cand_doc):
    """True if a word the original uses is gone and a sibling of it has come in ("towns" -> "cities", "heart
    disease" -> "stroke", "saving" -> "investing")."""
    def words(doc):
        toks = [t for t in doc if t.is_alpha]
        out = {t.lemma_.lower() for t in toks} | {t.lower_ for t in toks}
        return out | {a.lower_ + ' ' + b.lemma_.lower() for a, b in zip(toks, toks[1:])}
    old, new = words(sent), words(cand_doc)
    counted = bool(re.search(r"\d", cand_doc.text)) or bool({t.lower_ for t in cand_doc} & (NUMBER_WORDS | {'one'}) - old)
    return any((old & group) - new and (new & group) - old for group in SIBLINGS
               if not (counted and 'decade' in group))     # a unit converted with its number

# Names that may own what they described ("the Apple laptop" -> "Apple's laptop", "the London mayor" -> "London's
# mayor"): the rest name the thing itself (a model or product: "the Galaxy phone" is not "Galaxy's phone")
POSSESSOR_ENTITIES = {'ORG', 'GPE', 'NORP', 'PERSON'}
# A company's sites, which a place name before them only locates ("The London office" is not "London's office")
SITE_NOUNS = {'office', 'store', 'shop', 'branch', 'plant', 'factory', 'site', 'campus', 'warehouse', 'headquarters',
              'studio', 'outlet', 'showroom', 'depot', 'facility', 'location', 'hotel', 'restaurant', 'clinic', 'lab',
              'laboratory', 'bureau', 'subsidiary', 'division', 'unit', 'centre', 'center', 'franchise'}

def made_possessor(sent, cand_doc):
    """True if a model's name that described a noun now owns it ("The Vega 14 laptop" -> "Vega 14's laptop", "The
    Galaxy phone" -> "Galaxy's phone"): a name with a number in it, or one the original doesn't use for a company,
    place, people or person; or if a place that told where a site is now owns the site ("The London office" ->
    "London's office", "The Munich plant" -> "Munich's plant"). A common noun may ("the company website" -> "the
    company's website")."""
    described = {(t.lower_, t.head.lemma_.lower()): t for t in sent if t.dep_ in ('compound', 'nmod', 'nummod', 'amod')}
    def numbered(token):
        return token.like_num or any(ch.isdigit() for ch in token.text)
    for t in cand_doc:
        if t.dep_ != 'poss' or not any(c.tag_ == 'POS' for c in t.children):
            continue
        owner = [x for x in t.subtree if not x.is_punct and x.tag_ != 'POS']
        was = [described[key] for key in ((x.lower_, t.head.lemma_.lower()) for x in owner) if key in described]
        if was and (any(map(numbered, owner + was))
                    or any(x.pos_ == 'PROPN' and x.ent_type_ not in POSSESSOR_ENTITIES for x in was)):
            return True
        if t.head.lemma_.lower() in SITE_NOUNS and any(
                x.ent_type_ in ('GPE', 'LOC', 'FAC') or (x.pos_ == 'PROPN' and x.ent_type_ not in ('ORG', 'PERSON')) for x in was):
            return True
    return False

def title_added(cand_doc, source_lemmas, related):
    """True if the rewrite puts a new noun on a name, a title or description the original doesn't give ("two
    points behind Uruguay" -> "two points behind leaders Uruguay", "level with champions Croatia")."""
    def is_name(t):
        return t.pos_ == 'PROPN' and t.ent_type_ in ('PERSON', 'ORG', 'GPE', 'NORP', 'LOC', 'FAC')
    for t in cand_doc:
        lemma = t.lemma_.lower()
        if t.pos_ != 'NOUN' or not t.is_alpha or lemma in source_lemmas or lemma in related or t.lower_ in source_lemmas:
            continue
        if ((t.dep_ in ('compound', 'appos', 'nmod') and is_name(t.head))
                or any(c.dep_ in ('appos', 'compound', 'nmod') and is_name(c) for c in t.children)
                or (t.i + 1 < len(cand_doc) and is_name(cand_doc[t.i + 1]))):
            return True
    return False

EXAMPLE_MARKS = re.compile(r"\b(?:such\s+(?:\w+\s+){0,3}?as|like|including|for\s+example|for\s+instance|e\.g\.|among|"
                           r"especially|notably|in\s+particular|examples?\s+of|(?:and|or)\s+(?:\w+\s+)?(?:other|similar)|"
                           r"(?:" + _COUNT + r"|some|several|many|a\s+few)\s+of\s+(?:whom|which|them))\b", re.I)

def examples_dropped(sent, candidate):
    """True if the original gives examples of a group ("low-lying coastal cities such as Jakarta and Miami") and the
    rewrite makes them the whole claim ("Jakarta and Miami, as low-lying coastal cities, are ..."), not if it keeps
    them examples in other words ("Kyoto, Nara and other historic cities", "45 employees, 12 of whom are ..."; not
    "both of them"). "like" gives examples only after a noun that isn't the whole clause (the parser reads "Many
    listeners like these songs" as a noun at the root with "like these songs" on it)."""
    toks = [t for t in sent if not t.is_space]
    for k, t in enumerate(toks[:-1]):
        if t.tag_ not in ('NNS', 'NNPS'):
            continue
        j = k + 2 if toks[k + 1].text == ',' and k + 2 < len(toks) else k + 1
        after = toks[j]
        if ((after.lower_ == 'such' and j + 1 < len(toks) and toks[j + 1].lower_ == 'as') or after.lower_ == 'including'
                or (after.lower_ == 'like' and after.pos_ != 'VERB' and after.head.i == t.i and t.dep_ != 'ROOT')):
            return not EXAMPLE_MARKS.search(candidate)
    return False

# A verb that needs an object, left without one ("... and she almost abandons.", "The Industrial Revolution
# initiated in Britain")
TRANSITIVE_ONLY = {'abandon', 'relinquish', 'forsake', 'discard', 'renounce', 'waive', 'abolish', 'acquire', 'obtain',
                   'attain', 'utilize', 'utilise', 'require', 'provide', 'offer', 'give', 'make', 'deliver', 'enhance',
                   'facilitate', 'ensure', 'address', 'mitigate', 'foster', 'leverage', 'implement', 'establish',
                   'initiate', 'comfort'}
# Swaps the translation models make in the wrong sense, seen in review ("does nothing to address rising rents" ->
# "does nothing to accommodate the increase in rents", "hoists the flag" -> "wails the flag", "students who
# exercised regularly" -> "students who practiced regularly", "must carry their license" -> "maintain their
# license")
WRONG_SENSE = {'address': {'accommodate'}, 'hoist': {'wail'}, 'grasp': {'seize', 'grab'}, 'deliver': {'liberate'},
               'charge': {'accuse', 'indict'}, 'treat': {'regale'}, 'run': {'sprint', 'jog'}, 'exercise': {'practice', 'exert'},
               'carry': {'maintain'}}
# ...and these as a verb only ("must still be approved by both parliaments" -> "are still required to bill the
# treaty"); the noun is the usual word ("The Senate approved the measure" -> "The Senate passed the bill")
WRONG_VERB_SENSE = {'approve': {'bill'}}
# Medicines are taken, not eaten ("people who take insulin" -> "those who consume insulin")
MEDICINES = {'insulin', 'medication', 'medicine', 'drug', 'pill', 'tablet', 'aspirin', 'ibuprofen', 'antibiotic',
             'paracetamol', 'vaccine', 'dose', 'prescription', 'supplement', 'vitamin', 'painkiller', 'steroid'}
EATING = {'consume', 'eat', 'ingest', 'drink', 'swallow', 'devour'}

# Nouns of linking phrases ("In contrast", "As a result"), which a rewrite may word another way
LINKING_NOUNS = {'contrast', 'comparison', 'addition', 'summary', 'conclusion', 'example', 'instance', 'fact', 'result',
                 'short', 'general', 'particular', 'end', 'sum', 'essence', 'brief', 'whole', 'turn', 'meantime'}

def unkept_replacement(sent, cand_doc, lower_source, lower_new, change=0.0):
    """For a near-copy: True if all it does is drop an article ("the importance of the discovery" -> "of
    discovery"), or if a noun it swaps isn't kept by the new one: any noun in a very close copy, and in a close
    one the head of a compound ("battery life" -> "battery backup", "share buyback" -> "share takeover") or a
    noun made from a verb ("Caching is" -> "The cache is"). Another noun that only starts the same way keeps no
    noun here ("screen time" -> "screening time"); its possessive does ("the company website" -> "the company's
    website")."""
    ops = [op for op in difflib.SequenceMatcher(None, lower_source, lower_new, autojunk=False).get_opcodes() if op[0] != 'equal']
    if ops and all(tag == 'delete' and set(lower_source[i1:i2]) <= ARTICLES for tag, i1, i2, _, _ in ops):
        return True
    stop_words = NLP.Defaults.stop_words
    for tag, i1, i2, j1, j2 in ops:
        if tag != 'replace' or i2 - i1 > 2:
            continue
        new_words = {re.sub(r"['’]s?$", '', w) for w in lower_new[j1:j2]}
        new_lemmas = {t.lemma_.lower() for t in cand_doc if t.lower_ in new_words}
        for w in lower_source[i1:i2]:
            token = next((t for t in sent if t.lower_ == w), None)
            # verbs and describing words vary more freely than the nouns a sentence is about; an abbreviation the
            # tagger took for a name ("weighs approx. 3 kg") is a bound, which the bounds check sees to
            if (w in stop_words or not w.isalpha() or token is None or token.ent_type_
                    or token.pos_ not in ('NOUN', 'PROPN') or acts_as_verb(token)
                    or (w in ABBREVIATIONS and token.tag_ == 'NNP')
                    or token.lemma_.lower() in LIGHT_NOUNS | TIME_NOUNS | GENERIC_NOUNS | LINKING_NOUNS):
                continue
            lemma = token.lemma_.lower()
            if kept_word(lemma, 'n', new_words, new_lemmas, False) or kept_word(w, 'n', new_words, new_lemmas, False):
                continue
            start = stem(lemma)
            if start and any(t.lower_ in new_words and t.lower_.startswith(start) and t.pos_ != 'NOUN' for t in cand_doc):
                continue        # a related word of another kind ("celebration" -> "a celebratory event")
            if change < 0.15 or any(c.dep_ == 'compound' for c in token.children) or w.endswith('ing'):
                return True
    return False

def paraphrase_score(sent, candidate, cand_doc, why=None):
    """How much `candidate` changes the sentence, or None if it isn't a safe rewrite. A safe rewrite keeps
    every name, number, date and acronym and adds none; keeps the topic's nouns, its subject, the key
    describing words and adverbs, who does what to whom (and who gives to whom), the negations and where they
    apply, what the sentence claims (must/should/can/may, hedges, how often, how strong the evidence is and who
    said it, quantities and bounds, how strongly it recommends, tense) and its point of view; adds no new
    details, definitions, stronger claims, opposites, causes, contrasts, made-up words or punctuation; is a
    whole, grammatical sentence of a similar length; and doesn't repeat itself."""
    source, new = plain_words(sent.text), plain_words(candidate)
    if not source or not new or not LENGTH_RANGE[0] <= len(new) / len(source) <= LENGTH_RANGE[1]:
        return refuse(why, 'length')
    lower_source, lower_new = [w.lower() for w in source], [w.lower() for w in new]
    if lower_new == lower_source or not odd_marks(candidate) <= odd_marks(sent.text):
        return refuse(why, 'same or new punctuation')
    if (TRUNCATED.search(candidate) or (END_GAP.search(candidate) and not END_GAP.search(sent.text))
            or (POSSESSIVE_END.search(candidate) and not POSSESSIVE_END.search(sent.text)) or cut_off(sent, cand_doc)):
        return refuse(why, 'cut off')       # the model stopped mid-sentence ("..., and communities,", "the new policy's.")
    if finite_clause(sent) and not all(finite_clause(s) for s in cand_doc.sents):
        return refuse(why, 'fragment')      # no main verb: "Caching, which involves storing ...", "Vitamin A 800 IU ... each."
    stop_words = NLP.Defaults.stop_words
    # lemminflect knows content words; spaCy's stop words and a short list of its gaps cover the rest
    source_words = set(LONG_WORDS.findall(sent.text.lower()))
    stems = {w[:5] for w in source_words if len(w) >= 5}
    demonyms = {DEMONYMS[w] for w in source_words if w in DEMONYMS}
    contrastive = CONTRAST_OPENERS.match(sent.text)
    if any(not getAllLemmas(w) and w not in stop_words and w[:5] not in stems and w not in demonyms
           and (w not in KNOWN_WORDS or (w == 'despite' and not contrastive))
           for w in set(LONG_WORDS.findall(candidate.lower())) - source_words):
        return refuse(why, 'made-up word')     # a misspelt or made-up word ("marchs")
    # Names whole (without a leading "the"); of dates, amounts and the like, the numbers and month or day
    # names ("on this day" may become "on this particular day", but "15th August" must stay). "one" counts
    # as a number only when it counts something ("one reason", not "one of the most ..."), and may become "a"
    # ("one year" -> "a year"); "first" in "at first" is no number. A common noun the tagger took for a name at
    # the start ("Photosynthesis is ...") may lose its capital.
    cand_root = next(iter(cand_doc.sents)).root
    cand_subject = {t.lower_ for c in cand_root.children if c.dep_ in ('nsubj', 'nsubjpass') for t in c.subtree}
    protected = [re.sub(r'^[Tt]he\s+', '', e.text) for e in sent.ents if e.label_ in NAME_ENTITIES
                 and not (e.start == sent.start and len(e) == 1 and getAllLemmas(e.text.lower()).get('NOUN')
                          and e.text.lower() in cand_subject)]
    for t in sent:
        if not ((t.like_num and not (t.lower_ == 'one' and t.dep_ != 'nummod')) or any(c.isdigit() for c in t.text)
                or (t.ent_type_ in NUMBER_ENTITIES and t.pos_ == 'PROPN') or (t.is_upper and t.is_alpha and len(t.text) > 1)
                or t.like_email or t.like_url):
            continue
        after = t.doc[t.i + 1] if t.i + 1 < len(t.doc) else None
        if t.lower_ == 'first' and ((t.i > sent.start and t.doc[t.i - 1].lower_ == 'at') or (after is not None and after.text == '-')):
            continue        # "at first", "first-time"
        if t.lower_ == 'one' and after is not None and re.search(r'\ban?\s+' + re.escape(after.lower_) + r'\b', candidate, re.I):
            continue        # "one year" -> "a year"
        protected.append(t.text)
    if any(not mentions(candidate, p) for p in protected):
        return refuse(why, 'protected')
    # A date keeps the way it is written ("15th August" doesn't become "August 15th"); spaCy may split such
    # a date into two entities, so each number next to a month name is checked
    for t in sent[:-1]:
        right = sent.doc[t.i + 1]
        if ((any(c.isdigit() for c in t.text) and right.lower_ in MONTHS)
                or (t.lower_ in MONTHS and any(c.isdigit() for c in right.text))):
            if sent.doc[t.i: t.i + 2].text not in candidate:
                return refuse(why, 'date format')
    if Counter(NUMBER_RUN.findall(candidate)) != Counter(NUMBER_RUN.findall(sent.text)):
        return refuse(why, 'numbers')       # a number dropped, changed or repeated ("12" next to "1200")
    old_words, new_words = set(re.findall(r'[a-z]+', sent.text.lower())), set(re.findall(r'[a-z]+', candidate.lower()))
    if (new_words & NUMBER_WORDS - old_words and not NUMBER_RUN.search(sent.text)
            and not old_words & NUMBER_WORDS and not re.search(r"\ban?\s+" + TIME_UNIT + r"\b", sent.text, re.I)):
        return refuse(why, 'numbers')       # a count the original doesn't give ("studies" -> "three studies")
    known = set(lower_source) | {p.lower() for w in source for p in name_parts(w)}
    known |= {DEMONYMS[w] for w in list(known) if w in DEMONYMS}
    if any(p[0].isupper() and p.lower() not in known and p != 'I' for w in new[1:] for p in name_parts(w)):
        return refuse(why, 'invented name')     # an invented name
    names = {p for w in source[1:] for p in name_parts(w) if p[0].isupper() and p != 'I'}
    opening = next((t for t in sent if not t.is_punct and not t.is_space), None)
    if (opening is not None and opening.pos_ == 'PROPN' and opening.ent_type_ in NAME_ENTITIES
            and not getAllLemmas(opening.lower_).get('NOUN')):
        names |= set(name_parts(source[0]))     # a name that starts the sentence ("Ann and Annabel opened ...")
    if not names <= {p for w in new for p in name_parts(w)}:
        return refuse(why, 'dropped name')     # a dropped name or title ("Prime Minister" isn't an entity)
    present = set(re.findall(r'[a-z]+', candidate.lower()))      # "time-to-live" counts as "time"
    present_lemmas = {t.lemma_.lower() for t in cand_doc}
    # Topic nouns, counted by unit ("safety net" is one): a rewrite must keep most of them. Time words ("every
    # year" -> "annually") are often turned into adverbs, and units of a number ("15 percent") are kept with it
    units, unit_tokens = {}, {}
    for t in sent:
        lemma = t.lemma_.lower()
        if (t.pos_ != 'NOUN' or not t.is_alpha or lemma in LIGHT_NOUNS or lemma in TIME_NOUNS
                or any(c.dep_ == 'nummod' and c.like_num for c in t.children)):
            continue
        head = t
        while head.dep_ == 'compound' and head.head.pos_ == 'NOUN' and head.head.i != head.i:
            head = head.head
        units.setdefault(head.lemma_.lower(), set()).add(lemma)
        unit_tokens.setdefault(head.lemma_.lower(), []).append(t)
    lost_units = [u for u, unit in units.items() if not any(kept_word(n, 'n', present, present_lemmas) for n in unit)]
    if len(units) >= 2 and 1 - len(lost_units) / len(units) < NOUN_RETENTION:
        return refuse(why, 'topic nouns')     # drifted off topic
    # ...and always the main clause's subject ("Regular exercise has been shown ..." -> "Eating regularly ..."),
    # unless it is a generic noun for a way of doing things ("A common approach is ..." -> "A frequent method is ...")
    subjects = {t.lemma_.lower() for t in sent if t.pos_ == 'NOUN' and t.is_alpha and t.dep_ in ('nsubj', 'nsubjpass')
                and t.head.dep_ == 'ROOT'} - LIGHT_NOUNS - TIME_NOUNS - GENERIC_NOUNS
    if any(not kept_word(n, 'n', present, present_lemmas) for n in subjects):
        return refuse(why, 'subject')
    if plural_made_single(sent, cand_doc):
        return refuse(why, 'number')          # "small contributions add up" -> "a small contribution"
    if swapped_roles(sent, cand_doc) or made_possessor(sent, cand_doc):
        return refuse(why, 'roles')           # who does what to whom, who gives to whom, or what is whose changed
    if scope_changed(sent, cand_doc) or examples_dropped(sent, candidate):
        return refuse(why, 'scope')           # "99 percent of lead" -> "99 percent of tap water", "cities such as X" -> "X"
    # The words that pin the meaning down: "remote work", "an empty stomach", "top universities", "is efficient",
    # "disproportionately affected", "stronger job growth", "a substantial safety net" (a curated word is kept by
    # its own replacements: "substantial" -> "large"), but not a word in a bound ("no more than")
    bound_words = {w.lower() for _, pattern in BOUNDS for m in pattern.finditer(sent.text) for w in plain_words(m.group(0))}
    describers = {}
    for t in sent:
        lemma = t.lemma_.lower()
        if (not t.is_alpha or t.ent_type_ or t.lower_ in bound_words or lemma in EVALUATIVE_ADJECTIVES
                or lemma in ('own', 'other', 'such')):
            continue
        comparative = t.tag_ in ('JJR', 'JJS', 'RBR', 'RBS') and t.lower_ not in COMPARATIVE_FREE
        if (lemma in LIGHT_DESCRIBERS or t.lower_ in LIGHT_DESCRIBERS) and not comparative:
            continue
        if ((t.pos_ == 'ADJ' and (t.dep_ == 'amod' or (
                (t.dep_ in ('acomp', 'oprd') or (t.dep_ == 'conj' and t.head.pos_ == 'ADJ' and t.head.dep_ in ('acomp', 'oprd')))
                and t.lower_ not in HEDGES and lemma not in ('able', 'unable'))))
                or (t.pos_ == 'ADV' and t.lower_.endswith('ly') and len(t.text) >= 4 and t.lower_ not in stop_words
                    and lemma not in CURATED_OPTIONS)):
            describers[(lemma, 'a' if t.pos_ == 'ADJ' else 'r')] = t
    for (lemma, part), t in describers.items():
        # a describing word isn't kept by a verb of the same form ("an empty stomach" -> "emptying their stomachs")
        verb_forms = {v.lower_ for v in cand_doc if v.pos_ == 'VERB' and v.lemma_.lower() == lemma}
        if verb_forms and not any(v.pos_ != 'VERB' and v.lemma_.lower() == lemma for v in cand_doc):
            words, lemmas = present - verb_forms - {lemma}, present_lemmas - {lemma}
        else:
            words, lemmas = present, present_lemmas
        if kept_word(lemma, part, words, lemmas) or (t.lower_ in words and t.lower_ != lemma):
            continue
        if CURATED_OPTIONS.get(lemma, set()) & (words | lemmas):
            continue
        if STRENGTH.get(lemma) and any(STRENGTH.get(w) == STRENGTH[lemma] for w in words | lemmas):
            continue
        if lemma in OBLIGING_PREDICATES | NEED_STATING_WORDS and t.dep_ in ('acomp', 'oprd') and (
                'must' in modality(candidate, cand_doc)[0] or needs_stated(cand_doc)):
            continue        # "is mandatory" -> "must", "is necessary" -> "requires": the modality check sees to it
        return refuse(why, 'describer')
    source_nouns = {t.lemma_.lower() for t in sent if t.pos_ in ('NOUN', 'PROPN')}
    source_lemmas = {t.lemma_.lower() for t in sent}
    related = set().union(*(synonyms_of(n, 'n') | KEPT_EQUIVALENTS.get(n, set()) for n in source_nouns)) if source_nouns else set()
    hyphenated = {x.i for t in cand_doc if t.text == '-' and not t.whitespace_ and t.i and not cand_doc[t.i - 1].whitespace_
                  for x in (cand_doc[t.i - 1], t, cand_doc[min(t.i + 1, len(cand_doc) - 1)])}
    added = {t.lemma_.lower() for t in cand_doc if t.pos_ == 'NOUN' and t.is_alpha and t.lower_ not in stop_words
             and t.i not in hyphenated} - source_lemmas - related - LIGHT_NOUNS - LIGHT_EXTRA - TIME_NOUNS
    if source_nouns & GENERIC_NOUNS:
        added -= GENERIC_NOUNS                  # "a common approach" -> "a typical method"
    # ...but a noun made from the original's own words isn't new ("reflect" -> "reflection", "grow" -> "growth")
    source_stems = {stem(w) for w in source_lemmas} - {None}
    short_stems = {w for w in source_lemmas if len(w) == 4 and w.isalpha()}
    added = {n for n in added if NOMINALIZATIONS.get(n) not in source_lemmas and stem(n) not in source_stems
             and not any(n.startswith(w) for w in short_stems)}
    if len(added) > max(1, len(source) // 12):
        return refuse(why, 'new details')     # new details ("work done from the office", "for the cause of national unity")
    # ...nor one thing put in place of its sibling ("in many small towns" -> "in many small cities", "viruses" ->
    # "bacteria", "a week" -> "a month")
    if sibling_swapped(sent, cand_doc):
        return refuse(why, 'topic nouns')
    if ((GLOSS.search(candidate) and not GLOSS.search(sent.text)) or (OR_GLOSS.search(candidate) and not OR_GLOSS.search(sent.text))
            or any(t.dep_ == 'appos' and t.lemma_.lower() in added for t in cand_doc)
            or any(t.pos_ == 'NOUN' and t.lemma_.lower() in added and t.left_edge.lower_ in ('a', 'an', 'the')
                   and t.left_edge.i > 0 and cand_doc[t.left_edge.i - 1].text == ',' for t in cand_doc)
            or title_added(cand_doc, source_lemmas, related)):
        return refuse(why, 'new gloss')       # "Green algae, also known as ...", "..., a necessary mass", "leaders Uruguay"
    if 'another' in lower_new and 'another' not in lower_source:
        return refuse(why, 'new details')     # "this process" -> "another process"
    if {PERSONAL[w] for w in lower_new if w in PERSONAL} - {PERSONAL[w] for w in lower_source if w in PERSONAL}:
        return refuse(why, 'point of view')     # a new point of view ("we", "you")
    if not any(t.ent_type_ == 'PERSON' or t.pos_ == 'PROPN' or t.lemma_.lower() in PERSON_NOUNS for t in sent) and \
            {THIRD_PERSON[w] for w in lower_new if w in THIRD_PERSON} - {THIRD_PERSON[w] for w in lower_source if w in THIRD_PERSON}:
        return refuse(why, 'new details')     # a new person ("over the course of her life" for a car)
    trigrams = [tuple(lower_new[i:i + 3]) for i in range(len(lower_new) - 2)]
    if len(trigrams) != len(set(trigrams)):
        return refuse(why, 'repeated trigram')
    if any(lower_new.count(w) > max(1, lower_source.count(w)) for w in set(lower_new) if len(w) >= 4 and w not in stop_words):
        return refuse(why, 'doubled word')     # a doubled word ("feedback and feedback", "insights from these insights")
    if any(a == b and a not in lower_source for a, b in zip(lower_new, lower_new[1:])):
        return refuse(why, 'doubled word')
    source_pairs = set(zip(lower_source, lower_source[1:]))
    if any('-' in a + b and (a.rsplit('-', 1)[-1] == b or b.split('-', 1)[0] == a) and (a, b) not in source_pairs
           for a, b in zip(lower_new, lower_new[1:])):
        return refuse(why, 'doubled word')     # across a hyphen: "using mixed-methods methods", "a well well-known"
    # Negations must match one for one, counting the verbs that negate ("prevents ... from", "avoid", "failed to");
    # only "not X" <-> "unX/inX" on the same word may trade places, and a word that merely starts like a negation
    # ("incredibly", "unexpected" kept on both sides) loosens nothing
    source_set, new_set = set(lower_source), set(lower_new)
    gained = {s for s in map(negated_stem, new_set - source_set) if s and s in source_set}
    lost = {s for s in map(negated_stem, source_set - new_set) if s and s in new_set}
    if (negations(lower_new) + soft_negations(cand_doc)) - (negations(lower_source) + soft_negations(sent)) != len(lost) - len(gained):
        return refuse(why, 'negation')     # a negation lost, added or flipped ("efficient" -> "inefficient", "prevents" -> "allows")
    if negation_moved(sent, cand_doc, source_set):
        return refuse(why, 'negation')     # the negation now applies to something else
    source_vocab = source_set | source_lemmas
    old_modality, new_modality = modality(sent.text, sent), modality(candidate, cand_doc)
    if 'must' in old_modality[0] ^ new_modality[0] and need_restated(
            *((cand_doc, sent) if 'must' in old_modality[0] else (sent, cand_doc))):
        # an obligation stated as a need: "A permit is required to fish" -> "Fishing requires a permit"
        old_modality = (old_modality[0] | {'must'}, old_modality[1])
        new_modality = (new_modality[0] | {'must'}, new_modality[1])
    if old_modality != new_modality and not (
            new_modality[0] - old_modality[0] == {'can'} and old_modality[0] <= new_modality[0]
            and old_modality[1] == new_modality[1] and ((ENABLING.search(sent.text) and not ENABLING.search(candidate))
                                                        or soft_negations(sent) > soft_negations(cand_doc))):
        return refuse(why, 'modality')     # "must" -> "should", "may" -> "can", "shown" -> "purportedly"
    if 'must' not in new_modality[0] and forcing_dropped(sent, cand_doc):
        return refuse(why, 'modality')     # "The war forced families to move" -> "Families moved after the war"
    old_freq, new_freq = frequency(sent, sent.text), frequency(cand_doc, candidate)
    old_soft = {FREQUENCY_LICENSES[w] for w in source_vocab if w in FREQUENCY_LICENSES}
    new_soft = {FREQUENCY_LICENSES[w] for w in new_set | present_lemmas if w in FREQUENCY_LICENSES}
    if (old_freq - new_freq - new_soft or new_freq - old_freq - old_soft
            or bool(UNAIDED.search(sent.text)) != bool(UNAIDED.search(candidate))):
        return refuse(why, 'hedge')        # "tended to score higher" -> "scored higher", "go away on their own" -> "go away"
    old_evidence, new_evidence = evidence(sent), evidence(cand_doc)
    if old_evidence != new_evidence and not (
            old_evidence - new_evidence <= {'tentative', 'opinion'} and new_evidence <= old_evidence
            and ATTRIBUTION.search(candidate) and not ATTRIBUTION.search(sent.text)):
        return refuse(why, 'evidence')     # "studies suggest" -> "studies show", "the relationship between" -> "the impact of"
    if attributed(sent, sent.text) != attributed(cand_doc, candidate):
        return refuse(why, 'attribution')  # "residents said the detour would add ..." -> "the detour would add ..."
    old_quantities, new_quantities = quantities(sent), quantities(cand_doc)
    sharing = {'contribute', 'contribution', 'contributor', 'factor', 'role'}
    if 'part' in old_quantities ^ new_quantities and (
            source_lemmas & sharing if 'part' in new_quantities else present_lemmas & sharing):
        old_quantities, new_quantities = old_quantities - {'part'}, new_quantities - {'part'}   # "contributing" == "in part"
    if old_quantities != new_quantities:
        return refuse(why, 'quantity')     # "many others" -> "hundreds of others", "some patients" -> "patients"
    if bounds(sent.text) != bounds(candidate):
        return refuse(why, 'bound')        # "up to 15 percent" -> "15 percent", "longer than" -> "shorter than"
    stronger = lambda doc, words: ({STRONGER[w] for w in words if w in STRONGER and STRONGER[w] != 'fame'}
                                   | {'fame' for t in doc if STRONGER.get(t.lower_) == 'fame' and t.pos_ == 'ADJ'}
                                   | {'fame' for w in words if '-' in w and STRONGER.get(w) == 'fame'})
    old_strength, new_strength = strengths(sent), strengths(cand_doc)
    if (stronger(cand_doc, lower_new) - stronger(sent, lower_source)
            or sum(t.tag_ in ('JJS', 'RBS') for t in cand_doc) > sum(t.tag_ in ('JJS', 'RBS') for t in sent)
            or {t.lower_ for t in cand_doc if t.ent_type_ == 'ORDINAL' and t.dep_ in ('amod', 'nummod')}
            - {t.lower_ for t in sent}
            or (old_strength and new_strength - old_strength)
            or claim_added(sent, cand_doc, source_vocab)):
        return refuse(why, 'stronger claim')   # "completely", "the most exceptional", "renowned", "essential" -> "advisable"
    if opposite_added(sent, cand_doc):
        return refuse(why, 'opposite')     # "absorb" -> "release", "frequent" -> "rare", "allows" -> "is limited"
    if causal_added(sent, cand_doc):
        return refuse(why, 'cause')        # "..., resulting in a drop", "By approving ..., the board raised ..."
    if connective_added(sent, cand_doc):
        return refuse(why, 'connective')   # "However, the authors warn ...", "For example, ...", "As part of a deal, ..."
    if garbled(sent, cand_doc) or dropped_article(sent, cand_doc):
        return refuse(why, 'grammar')      # "public transportation make", "the primary driving a", "in modern world"
    first = next(iter(cand_doc.sents), None)
    old_tense, new_tense = main_tense(sent), (main_tense(first) if first is not None else None)
    if old_tense and new_tense and old_tense != new_tense:
        return refuse(why, 'tense')        # "brought" -> "has posed"
    if lower_source.count('here') != lower_new.count('here'):
        return refuse(why, 'idiom')        # "here to stay" -> "there to stay"
    change = 1 - difflib.SequenceMatcher(None, lower_source, lower_new, autojunk=False).ratio()
    if change < NEAR_COPY and unkept_replacement(sent, cand_doc, lower_source, lower_new, change):
        return refuse(why, 'near-copy')    # "address rising rents" -> "accommodate the increase", "Caching is" -> "The cache is"
    return change

NEAR_COPY = 0.25    # below this much change a rewrite is a near-copy, whose every swapped noun must keep its sense

def finish_sentence(candidate, original):
    text = candidate.strip()
    first, gap, rest = text.partition(' ')
    text = capitalized(first) + gap + rest
    if text[-1] not in '.!?':
        text += original.strip()[-1] if original.strip()[-1] in '.!?' else '.'
    return text

# The translation models learned British spelling as well as American. If the original never uses a
# British form, the rewrite uses the American one ("organisation" -> "organization").
BRITISH_IZE = re.compile(
    r"\b((?:organ|recogn|real|prior|emphas|minim|maxim|optim|util|summar|character|special|standard|critic|apolog|"
    r"categor|custom|digital|final|harmon|industrial|international|mobil|modern|normal|personal|polar|stabil|"
    r"symbol|synchron|visual|civil|global|legal|local|central|author|capital|commercial|democrat|familiar|general|"
    r"ideal|memor|monet|neutral|privat|public|rational|revolution|sanit|scrutin|social|stigmat|sympath|theor|urban|"
    r"material|energ|fertil|immun|jeopard|magnet|marginal|metabol|moistur|motor|national|natural|oxid|patron|"
    r"popular|pressur|random|regular|sensit|serial|terror|trivial|vocal|mechan|computer|dramat|equal|formal|"
    r"individual|institutional|vandal|western|human|econom|hospital|item)i)s(e|es|ed|ing|ation|ations|er|ers)\b",
    re.I)
BRITISH_WORDS = {      # not "analyses": it is also the plural of "analysis" in American English
    'analyse': 'analyze', 'analysed': 'analyzed', 'analysing': 'analyzing',
    'paralyse': 'paralyze', 'paralysed': 'paralyzed', 'catalyse': 'catalyze', 'catalysed': 'catalyzed',
    'colour': 'color', 'colours': 'colors', 'coloured': 'colored', 'colourful': 'colorful', 'favour': 'favor',
    'favours': 'favors', 'favoured': 'favored', 'favourite': 'favorite', 'favourites': 'favorites',
    'favourable': 'favorable', 'behaviour': 'behavior', 'behaviours': 'behaviors', 'behavioural': 'behavioral',
    'honour': 'honor', 'honours': 'honors', 'honoured': 'honored', 'honouring': 'honoring', 'honourable': 'honorable',
    'labour': 'labor', 'labours': 'labors', 'neighbour': 'neighbor', 'neighbours': 'neighbors',
    'neighbourhood': 'neighborhood', 'neighbourhoods': 'neighborhoods', 'humour': 'humor', 'rumour': 'rumor',
    'rumours': 'rumors', 'vigour': 'vigor', 'harbour': 'harbor', 'flavour': 'flavor', 'flavours': 'flavors',
    'endeavour': 'endeavor', 'endeavours': 'endeavors', 'odour': 'odor', 'tumour': 'tumor', 'tumours': 'tumors',
    'armour': 'armor', 'centre': 'center', 'centres': 'centers', 'centred': 'centered', 'theatre': 'theater',
    'theatres': 'theaters', 'metre': 'meter', 'metres': 'meters', 'litre': 'liter', 'litres': 'liters',
    'fibre': 'fiber', 'fibres': 'fibers', 'defence': 'defense', 'offence': 'offense', 'licence': 'license',
    'programme': 'program', 'programmes': 'programs', 'travelled': 'traveled', 'travelling': 'traveling',
    'traveller': 'traveler', 'travellers': 'travelers', 'modelling': 'modeling', 'modelled': 'modeled',
    'labelled': 'labeled', 'labelling': 'labeling', 'cancelled': 'canceled', 'cancelling': 'canceling',
    'fuelled': 'fueled', 'fuelling': 'fueling', 'signalled': 'signaled', 'counselling': 'counseling',
    'jewellery': 'jewelry', 'catalogue': 'catalog', 'ageing': 'aging', 'enrol': 'enroll', 'fulfil': 'fulfill',
    'skilful': 'skillful', 'grey': 'gray', 'whilst': 'while', 'amongst': 'among', 'co-operation': 'cooperation',
    'co-operate': 'cooperate', 'co-ordinate': 'coordinate', 'co-ordination': 'coordination',
    'enrolment': 'enrollment', 'enrolments': 'enrollments', 'enrols': 'enrolls', 'practise': 'practice',
    'practises': 'practices', 'practised': 'practiced', 'practising': 'practicing', 'sceptical': 'skeptical',
    'sceptic': 'skeptic', 'sceptics': 'skeptics', 'scepticism': 'skepticism', 'manoeuvre': 'maneuver',
    'manoeuvres': 'maneuvers', 'tyre': 'tire', 'tyres': 'tires', 'cheque': 'check', 'cheques': 'checks',
    'mould': 'mold', 'moulds': 'molds', 'mouldy': 'moldy', 'storey': 'story', 'storeys': 'stories',
    'aluminium': 'aluminum', 'anaemia': 'anemia', 'anaemic': 'anemic', 'anaesthesia': 'anesthesia',
    'anaesthetic': 'anesthetic', 'anaesthetics': 'anesthetics', 'paediatric': 'pediatric', 'paediatrician': 'pediatrician',
    'paediatricians': 'pediatricians', 'haemorrhage': 'hemorrhage', 'haemoglobin': 'hemoglobin', 'diarrhoea': 'diarrhea',
    'oedema': 'edema', 'oestrogen': 'estrogen', 'orthopaedic': 'orthopedic', 'foetus': 'fetus', 'foetal': 'fetal',
    'leukaemia': 'leukemia', 'oesophagus': 'esophagus', 'gynaecologist': 'gynecologist', 'gynaecology': 'gynecology',
}
BRITISH_WORD = re.compile(r"\b(" + "|".join(sorted(map(re.escape, BRITISH_WORDS), key=len, reverse=True)) + r")\b", re.I)
# Metric units with a prefix ("millimetres" / "millimeters"); "metre" and "litre" on their own are in the list above
BRITISH_UNIT = re.compile(r"\b((?:milli|centi|kilo|deci|micro|nano)(?:met|lit))re(s?)\b", re.I)
AMERICAN_UNIT = re.compile(r"\b((?:milli|centi|kilo|deci|micro|nano)(?:met|lit))er(s?)\b", re.I)

AMERICAN_WORDS = {american: british for british, american in BRITISH_WORDS.items()}
AMERICAN_WORD = re.compile(r"\b(" + "|".join(sorted(map(re.escape, AMERICAN_WORDS), key=len, reverse=True)) + r")\b", re.I)
AMERICAN_IZE = re.compile(BRITISH_IZE.pattern.replace(')i)s(', ')i)z('), re.I)
# American forms that are also right in British English in another sense ("a computer program", "a gas meter",
# "to license"), or part of names ("Labor Day"): changed back only when the text itself spells that very word the
# British way (and "program", "meter" and "license" only in the British form's sense, below)
AMBIGUOUS_AMERICAN = {'program', 'programs', 'meter', 'meters', 'labor', 'labors', 'center', 'centers', 'centered',
                      'license', 'gray', 'catalog', 'liter', 'liters', 'theater', 'theaters', 'fiber', 'fibers', 'mold',
                      'molds'}
# ...and these, common words in another sense ("check the figures", "the story of the town", "tires easily", "a
# medical practice", "pay at a meter", "will license new drivers", "a computer program"), only in the sense the
# British form has (a noun "check" or "tire", the verb "practice", floors counted: "a 30-story tower", a length: "200
# meters", anything but the verb "license", a "program" that is no software) and when the text spells that word the
# British way; "check" and "story" only when the original sentence itself does (a text about cheques still runs
# credit checks, and one about a 40-storey block still tells "three stories"): word -> (sense, British form, whether
# the rest of the text counts)
AMBIGUOUS_SENSES = {'check': ('NOUN', 'cheque', False), 'checks': ('NOUN', 'cheque', False),
                    'tire': ('NOUN', 'tyre', True), 'tires': ('NOUN', 'tyre', True),
                    'practice': ('VERB', 'practis', True), 'practices': ('VERB', 'practis', True),
                    'story': ('NUM', 'storey', False), 'stories': ('NUM', 'storey', False),
                    'meter': ('LENGTH', 'metre', True), 'meters': ('LENGTH', 'metre', True),
                    'license': ('NOT VERB', 'licence', True),
                    'program': ('NOT SOFTWARE', 'programme', True), 'programs': ('NOT SOFTWARE', 'programme', True)}
# Words that make a "meter" next to them a length: before it ("one meter", "a 100-meter race", "square meters", "half a
# meter", "hundreds of meters", "rose by a meter", "only meters from", "within meters of") or after it ("a meter
# wide", "meters away", "meters per second")
METRE_BEFORE = {'few', 'several', 'square', 'cubic', 'per', 'half', 'tens', 'hundreds', 'thousands', 'dozens', 'by',
                'nearly', 'almost', 'about', 'around', 'roughly', 'only', 'just', 'within'}
METRE_AFTER = {'long', 'wide', 'high', 'tall', 'deep', 'thick', 'short', 'narrow', 'low', 'away', 'apart', 'per',
               'squared'}

def british_sense(doc, start, word):
    """True if the word at character `start` of a parsed rewrite is used in the sense its British spelling has:
    floors counted ("a 30-story tower"), a verb without an article or describing word of its own ("doctors practice
    medicine"), or a noun with one, or after a preposition ("four new tires", "paid by check"); the tagger also calls
    the verb in "Long shifts tire the nurses" a noun, and the verb "practice" one ("musicians practice for hours"
    at the root, "Doctors here practice medicine" as a compound after a plural, an adverb or a pronoun). A "meter"
    is a length beside a number or a measuring word ("200 meters", "a meter wide"; not "pay at a meter"), "license"
    anything but the verb (not "the agency will license new drivers", nor "States license drivers", which the tagger
    reads as a noun), and a "program" no software ("a computer program", "software programs")."""
    sense = AMBIGUOUS_SENSES[word][0]
    token = next((t for t in doc if t.idx == start), None)
    if token is None:
        return False
    if sense == 'NUM':
        return any(c.dep_ == 'nummod' and c.like_num for c in token.children)
    before = doc[token.i - 1] if token.i > 0 else None
    if sense == 'LENGTH':
        if (token.i + 1 < len(doc) and doc[token.i + 1].lower_ == 'of'
                and any(c.dep_ == 'det' and c.lower_ in ('a', 'an', 'every', 'per') for c in token.children)):
            return True     # "a meter of snow", "every meter of cable"
        k = token.i - 1 if before is None or before.text != '-' else token.i - 2     # "a 100-meter race"
        if k >= 1 and doc[k].lower_ in ('a', 'an', 'of'):
            k -= 1          # "half a meter", "hundreds of meters"
        j = token.i + 2 if token.i + 2 < len(doc) and doc[token.i + 1].text == '-' else token.i + 1     # "meter-long"
        return ((k >= 0 and (doc[k].like_num or doc[k].lower_ in METRE_BEFORE))
                or (j < len(doc) and doc[j].lemma_.lower() in METRE_AFTER))
    if sense == 'NOT VERB':
        # a noun before a noun after "to", whatever the tagger made of "to" ("Changes to license rules")
        return not (token.pos_ in ('VERB', 'AUX') or (before is not None and (
            before.tag_ == 'MD' or (before.tag_ == 'TO' and not (token.pos_ == 'NOUN' and token.dep_ == 'compound'))
            or (token.dep_ == 'compound' and before.tag_ in ('NNS', 'PRP')))))
    if sense == 'NOT SOFTWARE':
        return before is None or before.lower_ not in ('computer', 'software')
    kids = {c.dep_ for c in token.children}
    determined = bool(kids & {'det', 'poss', 'amod', 'nummod'})
    if sense == 'VERB':
        return not determined and (token.pos_ == 'VERB' or token.dep_ in ('ROOT', 'conj') or (
            token.dep_ == 'compound' and before is not None and before.tag_ in ('NNS', 'RB', 'PRP')))
    return token.pos_ == 'NOUN' and 'dobj' not in kids and (determined or token.dep_ == 'pobj')

# Word pairs a text spells one way throughout
SPELLING_PAIRS = (('toward', 'towards'), ('afterward', 'afterwards'))
# A straight quote before a letter opens a quotation, except in an elision ("rock 'n' roll", "'em", "'til")
OPENING_QUOTE = re.compile(r"(?:^|(?<=[\s(\[{“]))'(?=[A-Za-z])(?!(?:n|em|til|cause|tis|twas|bout|cuz|kay)\b)", re.I)

def uses_british(text):
    return bool(BRITISH_WORD.search(text) or BRITISH_IZE.search(text) or BRITISH_UNIT.search(text))

def text_style(text):
    """How the whole text spells: British or American, -ise or -ize, "toward" or "towards", "internet" or
    "Internet" (decided once per text, so every rewritten sentence matches the rest)."""
    low = text.lower()
    prefer = {}
    for a, b in SPELLING_PAIRS:
        has_a, has_b = re.search(r'\b%s\b' % a, low), re.search(r'\b%s\b' % b, low)
        if has_a and not has_b:
            prefer[b] = a
        elif has_b and not has_a:
            prefer[a] = b
    inside = re.sub(r'(?:^|[.!?]\s+)\w+', ' ', text)      # words not at the start of a sentence
    internet = ('internet' if 'internet' in inside and 'Internet' not in inside
                else 'Internet' if 'Internet' in inside and 'internet' not in inside else None)
    return {'british': uses_british(text), 'british_words': {m.group(0).lower() for m in BRITISH_WORD.finditer(text)},
            'ise': bool(BRITISH_IZE.search(text)), 'prefer': prefer, 'internet': internet}

def restore_style(candidate, original, british=False, style=None):
    """A model's rewrite in the original's spelling and quote style. `british` is whether the whole text uses
    British spelling anywhere, and `style` how the whole text spells (text_style()): spelling is decided once per
    text, not per sentence. A British text gets its British forms back where the models wrote American ones
    ("organization" -> "organisation"), but names keep theirs ("Labor Day", "World Health Organization")."""
    if style is not None:
        british = british or style['british']
    if not british and not uses_british(original):
        candidate = BRITISH_WORD.sub(lambda m: match_case(m.group(0), BRITISH_WORDS[m.group(0).lower()]), candidate)
        candidate = BRITISH_UNIT.sub(lambda m: m.group(1) + ('ER' if m.group(0).isupper() else 'er') + m.group(2), candidate)
        candidate = BRITISH_IZE.sub(lambda m: m.group(1) + ('Z' if m.group(0)[len(m.group(1))].isupper() else 'z') + m.group(2), candidate)
    elif style is not None:
        def name(m):        # capitalised inside a sentence: part of a name
            return m.group(0)[0].isupper() and m.start() > 0 and not re.search(r'[.!?]\s*$', m.string[:m.start()])
        def kept(m):        # a name, or a word the original sentence itself spells this way
            return name(m) or re.search(r'\b%s\b' % re.escape(m.group(0)), original, re.I)
        original_words = set(re.findall(r'[a-z]+', original.lower()))
        parsed = []         # the rewrite, parsed when a word's sense decides its spelling
        def british_word(m):
            word = m.group(0).lower()
            if kept(m) or (word in AMBIGUOUS_AMERICAN and AMERICAN_WORDS[word] not in style['british_words']):
                return m.group(0)
            if word in AMBIGUOUS_SENSES:
                _, form, text_wide = AMBIGUOUS_SENSES[word]
                spelled = original_words | style['british_words'] if text_wide else original_words
                if NLP is None or not any(w.startswith(form) for w in spelled):
                    return m.group(0)
                if not parsed:
                    with _NLP_LOCK:
                        parsed.append(NLP(m.string))
                if not british_sense(parsed[0], m.start(), word):
                    return m.group(0)
            return match_case(m.group(0), AMERICAN_WORDS[word])
        candidate = AMERICAN_WORD.sub(british_word, candidate)
        candidate = AMERICAN_UNIT.sub(lambda m: m.group(0) if kept(m) else m.group(1) + (
            'RE' if m.group(0).isupper() else 're') + m.group(2), candidate)
        if style['ise'] and not AMERICAN_IZE.search(original):
            candidate = AMERICAN_IZE.sub(lambda m: m.group(0) if name(m) else m.group(1) + (
                'S' if m.group(0)[len(m.group(1))].isupper() else 's') + m.group(2), candidate)
    if style is not None:
        for form, wanted in style['prefer'].items():
            candidate = re.sub(r'\b%s\b' % form, lambda m: match_case(m.group(0), wanted), candidate, flags=re.I)
        if style['internet'] == 'internet':
            candidate = re.sub(r'(?<![.!?]\s)(?<!^)\bInternet\b', 'internet', candidate)
        elif style['internet'] == 'Internet':
            candidate = re.sub(r'\binternet\b', 'Internet', candidate)
    if '’' in original and "'" not in original:
        candidate = OPENING_QUOTE.sub('‘', candidate).replace("'", '’')    # opening quote, then apostrophes
    return candidate

# Phrasing typical of AI text. Among equally faithful rewrites, one that adds such phrasing is ranked lower.
STOCK_PATTERNS = [
    r"\b(?:additionally|moreover|furthermore|nonetheless|nevertheless)\b", r"\bin addition\b",
    r"\bit(?:'s|’s| is) (?:important|crucial|essential|vital|worth) to (?:note|mention|acknowledge|remember|recognize)\b",
    r"\bit should be (?:noted|emphasi[sz]ed|mentioned|stressed)\b", r"\bcannot be overstated\b",
    r"\b(?:crucial|vital|pivotal|paramount|indispensable|integral|instrumental|invaluable|imperative)\b",
    r"\b(?:facilitat|utiliz|utilis|leverag|foster|underscor|showcas|delv|navigat|empower|streamlin|elevat|harness)\w*",
    r"\b(?:a diverse range|a broad spectrum|a wide range|a plethora|a myriad|an array) of\b",
    r"\bgame-chang\w*", r"\bpotent\b", r"\bwitness(?:ed|es)? the\b", r"\bbr(?:ing|ought|ings) about\b",
    r"\bpave[sd]? the way\b", r"\bin a \w+ manner\b", r"\bserves? as a reminder\b",
    r"\bthe (?:significance|importance|necessity) of\b",
    r"\b(?:landscape|realm|tapestry|testament|journey|beacon|cornerstone)\b", r"\bin (?:today's|today’s|the current) \w+\b",
    r"\b(?:seamless|robust|comprehensive|transformative|multifaceted|holistic|dynamic|profound|significant)\w*",
    r"\bplays? an? \w+ role\b", r"\bnot only\b", r"\bin regards? to\b", r"\bmitigat\w*", r"\bnecessitat\w*",
]
STOCK_RE = [re.compile(p, re.I) for p in STOCK_PATTERNS]
STYLE_PENALTY = 0.15

def stock_count(text):
    return sum(len(r.findall(text)) for r in STOCK_RE)

# The checks with the T5, all relative to how readily it reproduces the original from the original:
# - round trip: how readily it paraphrases the rewrite back into the original (meaning shifted, detail
#   dropped or added), on average and for the single worst word ("hoists the flag" -> "holds it down")
# - forward: how plausible the rewrite is as a paraphrase of the original (garbled or off-topic output)
FIDELITY_DROP = 0.5     # the most the round trip's average log-probability per token may fall short
WORST_TOKEN_DROP = 4.5  # ...and its worst token's
FORWARD_DROP = 0.7      # ...and the forward check's average
# Rewrites this close to the most faithful one count as equally faithful, and of those the most changed is
# used. Faithfulness is the mean of the round trip and the forward check: a near-copy that swapped one word
# for the wrong sense (back-translation's "Caching" -> "The cache") round-trips well but reads as an unlikely
# paraphrase. The band is narrow: the scores are noisy at this resolution, and a wide band turns that noise into
# a preference for the least faithful rewrite.
FIDELITY_MARGIN = 0.05

# Chaos Mode asks the T5 for more rewrites and gives sentences it changed only a little a second round
CHAOS_CANDIDATES = 4
SECOND_PASS_CHANGE = 0.25
# A near-copy that swaps a word or two may take one in another sense ("screen time" -> "screening time", "log in"
# -> "connect"), which the whole sentence's scores miss. Each swap is scored on its own, as vet_many scores a
# WordNet swap: how much less likely the T5 finds the rewrite as a paraphrase of the original than the rewrite
# with just that swap undone (for a single swap, the original itself). Undoing one swap at a time leaves the
# others in place, so swaps that only make sense together ("there are no fixed times" -> "there is no fixed
# time") don't count against each other. In review, wrong-sense swaps cost 8.7 nats or more and nearly all
# faithful ones less than 7; a few rarer faithful ones cost more ("hot springs" -> "thermal springs"), and their
# sentences keep their own wording.
SWAP_DROP = 8.0

def swaps_undone(source, candidate):
    """The candidate with each of its changes undone on its own, if every change swaps one or two words for one or
    two others; else none."""
    old, new = list(re.finditer(r"[\w'’-]+", source)), list(re.finditer(r"[\w'’-]+", candidate))
    ops = [op for op in difflib.SequenceMatcher(None, [m.group(0).lower() for m in old], [m.group(0).lower() for m in new],
                                                autojunk=False).get_opcodes() if op[0] != 'equal']
    if not ops or any(tag != 'replace' or i2 - i1 > 2 or j2 - j1 > 2 for tag, i1, i2, j1, j2 in ops):
        return []
    return [candidate[:new[j1].start()] + ' '.join(m.group(0) for m in old[i1:i2]) + candidate[new[j2 - 1].end():]
            for _, i1, i2, j1, j2 in ops]

def pick_rewrites(eligible, sources, pool, margin=FIDELITY_MARGIN):
    """sentence number -> (rewrite, change, faithfulness) for the best safe rewrite among `pool` (number ->
    candidates)."""
    flat = [(n, c) for n, options in pool.items() for c in options if c and c.strip() != sources[n]]
    if not flat:
        return {}
    with _NLP_LOCK:
        cand_docs = list(NLP.pipe([c for _, c in flat]))
    safe = []       # (sentence number, rewrite, change) for every rewrite that passes the checks
    for (n, candidate), cand_doc in zip(flat, cand_docs):
        change = paraphrase_score(eligible[n], candidate, cand_doc)
        if change is not None:
            safe.append((n, candidate, change))
    if not safe:
        return {}
    numbers = sorted({n for n, _, _ in safe})
    undone = [(k, version) for k, (n, candidate, change) in enumerate(safe) if change < NEAR_COPY
              for version in swaps_undone(sources[n], candidate)]
    scores = paraphraser.score([(sources[n], sources[n]) for n in numbers]
                               + [(c, sources[n]) for n, c, _ in safe] + [(sources[n], c) for n, c, _ in safe]
                               + [(sources[safe[k][0]], version) for k, version in undone])
    baseline = dict(zip(numbers, scores[:len(numbers)]))
    back, forward = scores[len(numbers):len(numbers) + len(safe)], scores[len(numbers) + len(safe):len(numbers) + 2 * len(safe)]
    wrong_sense = {k for (k, _), (_, _, total) in zip(undone, scores[len(numbers) + 2 * len(safe):])
                   if total - forward[k][2] > SWAP_DROP}
    faithful = {}
    for k, ((n, candidate, change), (avg, worst, _), (ahead, _, _)) in enumerate(zip(safe, back, forward)):
        base_avg, base_worst, _ = baseline[n]
        if (k in wrong_sense or avg < base_avg - FIDELITY_DROP or worst < base_worst - WORST_TOKEN_DROP
                or ahead < base_avg - FORWARD_DROP):
            continue
        styled = change - STYLE_PENALTY * max(0, stock_count(candidate) - stock_count(sources[n]))
        faithful.setdefault(n, []).append(((avg + ahead) / 2, styled, candidate, change))
    best = {}
    for n, options in faithful.items():
        top = max(o[0] for o in options)
        fidelity, _, candidate, change = max((o for o in options if o[0] >= top - margin), key=lambda o: (o[1], o[0]))
        best[n] = (candidate, change, fidelity)
    return best

# Abbreviations that need a word or a number after them ("approx. 3 kg", "Dr. Smith", "vs. Brazil"); units, "et
# al.", "etc." and "Inc." end sentences as well ("The box weighs 3 lbs. The shipping cost ...")
LEAD_IN_ABBREVIATIONS = {'ca', 'approx', 'appr', 'nos', 'vol', 'fig', 'figs', 'pp', 'vs', 'cf', 'incl', 'excl', 'est', 'sq',
                         'op', 'nr', 'dr', 'mr', 'mrs', 'ms', 'prof', 'st', 'mt'}

def stands_alone(sent):
    """False if spaCy's sentence looks like a piece of one: it starts in lower case ("and I answered", but not
    "iPhone sales"), the next one does ("Yahoo! | is"), it follows a word or a colon, or it follows the full stop of
    an abbreviation that needs more after it ("weighs approx. | A standard shelf"). The parser is kept from such
    splits (whole_sentences); this keeps any it still makes away from the models, which would rewrite the piece as
    a sentence of its own."""
    doc = sent.doc
    def lower_start(token):
        return token.text[:1].islower() and not any(c.isupper() for c in token.text[1:])
    toks = [t for t in sent if not t.is_space]
    k, j = sent.start - 1, sent.end
    while k >= 0 and doc[k].is_space:
        k -= 1
    while j < len(doc) and doc[j].is_space:
        j += 1
    if not toks or lower_start(toks[0]) or (j < len(doc) and lower_start(doc[j])):
        return False
    if k >= 0 and (doc[k].text[-1].isalnum() or doc[k].text[-1] in ',;:-–—(/&'):
        return False
    return not (k >= 1 and doc[k].text == '.' and not doc[k - 1].whitespace_
                and doc[k - 1].lower_ in LEAD_IN_ABBREVIATIONS)

# Linking phrases a candidate may open with ("In fact, ...", "As a result, ..."); one-word links are adverbs
LINK_PHRASE_OPENERS = re.compile(r"^\W*(?:in\s+fact|of\s+course|as\s+a\s+(?:result|consequence)|at\s+the\s+same\s+time|"
                                 r"in\s+other\s+words|in\s+reality|after\s+all|above\s+all|in\s+general|in\s+short|"
                                 r"in\s+particular|in\s+turn|in\s+practice|in\s+any\s+case|for\s+this\s+reason|by\s+then|"
                                 r"until\s+then|at\s+first|in\s+the\s+end|that\s+is)\s*,", re.I)

def contrast_kept(pool, sources):
    """`pool` (sentence number -> {candidate: anything}) with the original's contrast with the sentence before
    ("However, ...", "Costs, however, rose ...") put back on the candidates that dropped it ("The margin narrowed as
    costs rose" -> "However, the margin narrowed as costs rose"); they would be refused for dropping it. A
    candidate that opens with a link of its own ("Additionally, ...", "Meanwhile, ...", "In fact, ...") keeps its
    refusal: two links in a row ("Instead, rather, ...") don't read as one."""
    todo = []
    for n, options in pool.items():
        source = sources[n]
        m = CONTRAST_OPENERS.match(source)
        if m:
            opener = source[:m.end()].strip() + (',' if source[m.end():m.end() + 1] == ',' else '')
        elif CONTRAST_ASIDE.search(source):
            opener = 'However,'
        else:
            continue
        todo += [(n, c, opener) for c in options if not CONTRAST_INSIDE.search(c) and not ADDING_OPENERS.match(c)
                 and not EXAMPLE_OPENERS.match(c) and not LINK_PHRASE_OPENERS.match(c)]
    if not todo:
        return pool
    with _NLP_LOCK:
        openings = [[t for t in doc if not t.is_space][:2] for doc in NLP.pipe([c for _, c, _ in todo])]
    repaired = {}
    for (n, candidate, opener), opening in zip(todo, openings):
        first = opening[0] if opening else None
        if first is not None and first.pos_ == 'ADV' and len(opening) > 1 and opening[1].text == ',':
            continue        # "Rather, ...", "Alternatively, ...", "Meanwhile, ..."
        # the first word keeps its capital only if it is a name, "I", an acronym or written like "iPhone"
        keep = first is not None and (first.pos_ == 'PROPN' or first.text == 'I' or (first.is_upper and len(first.text) > 1)
                                      or any(c.isupper() for c in first.text[1:]) or first.ent_type_ in NAME_ENTITIES)
        repaired[(n, candidate)] = opener + ' ' + (candidate if keep else candidate[:1].lower() + candidate[1:])
    return {n: {repaired.get((n, c), c): value for c, value in options.items()} for n, options in pool.items()}

def deep_paraphrase(text, deep_mode=False):
    """Replace each sentence of running text with the most faithful safe rewrite from the models.
    Returns the new text and the content words of each sentence the models wrote, so the later layers can
    recognise them."""
    started = time.monotonic()
    parts = LINE_BREAKS.split(text)
    blocks = [k for k in range(0, len(parts), 2) if parts[k].strip() and is_prose(parts[k].strip())]
    if not blocks:
        return text, []
    starts, offset = {}, 0
    for k, part in enumerate(parts):
        starts[k] = offset + len(part) - len(part.lstrip())
        offset += len(part)
    with _NLP_LOCK:
        docs = list(NLP.pipe([parts[k].strip() for k in blocks]))
    quotes = quoted_ranges(text)         # found in the whole text: a quotation may be hard-wrapped over lines
    for k, doc in zip(blocks, docs):
        doc.user_data['quoted'] = frozenset(t.i for t in doc if overlaps_quote(text, starts[k] + t.idx, starts[k] + t.idx + 1))
    # Only English paragraphs and whole sentences: the models paraphrase into English, one sentence at a time
    eligible = [sent for doc in docs if looks_english(doc) for sent in doc.sents
                if len(plain_words(sent.text)) >= DEEP_MIN_WORDS and sent.text.strip()[-1:] in '.!?'
                and not set(sent.text) & set('"“”‘()[]:') and not has_quoted(doc, range(sent.start, sent.end))
                and not any(t.like_url or t.like_email for t in sent) and stands_alone(sent)]
    if not eligible:
        return text, []
    sources = [sent.text.strip() for sent in eligible]
    style = text_style(text)
    began = time.monotonic()
    generated = paraphraser.generate(sources, CHAOS_CANDIDATES if deep_mode else paraphraser.CANDIDATES)
    generating = time.monotonic() - began
    # each sentence's candidates, with whether back-translation wrote them
    pool = {n: dict.fromkeys((restore_style(o, sources[n], style=style) for o in options if o), False)
            for n, options in enumerate(generated)}
    # With another Deep Rewrite running, both skip the optional extras so neither keeps its user waiting long;
    # "another one is running" is checked when each extra would start, so a request that arrived meanwhile counts
    if (paraphraser.backtranslate_available() and paraphraser.in_flight() <= 1
            and time.monotonic() - started < DEEP_TIME_BUDGET * 0.4):
        for n, options in enumerate(paraphraser.backtranslate(sources)):
            for candidate in options:
                pool[n][restore_style(candidate, sources[n], style=style)] = True
    pool = contrast_kept(pool, sources)
    began = time.monotonic()
    best = pick_rewrites(eligible, sources, pool)
    picking = time.monotonic() - began
    # Chaos Mode: sentences the models changed only a little get a second, hotter round from the T5, which also
    # paraphrases the first pick once more. A second-round rewrite replaces the first pick only if it is at
    # least as faithful; sentences no candidate could rewrite safely get no second round (the models paraphrase
    # them unsafely, and more samples would only fish for one that slips through), and neither do those whose pick
    # is a back-translation (the T5 rates its own hot samples above it, inverted or garbled ones too). The round
    # runs only if, going by the first round's own timings, it finishes within the time budget.
    weak = [n for n in best if best[n][1] < SECOND_PASS_CHANGE and not pool[n].get(best[n][0])]
    projected = len(weak) / len(sources) * (1.5 * generating + picking)
    if (deep_mode and weak and paraphraser.in_flight() <= 1
            and time.monotonic() - started + projected <= DEEP_TIME_BUDGET):
        hotter = paraphraser.generate([sources[n] for n in weak], CHAOS_CANDIDATES, 1.0, 60)
        again = paraphraser.generate([best[n][0] for n in weak], 2, 0.8, 40)
        second = pick_rewrites(eligible, sources, contrast_kept(
            {n: dict.fromkeys(restore_style(c, sources[n], style=style) for c in hotter[j] + again[j] if c)
             for j, n in enumerate(weak)}, sources))
        for n, (candidate, change, fidelity) in second.items():
            if change > best[n][1] + 0.05 and fidelity >= best[n][2]:
                best[n] = (candidate, change, fidelity)
    chosen, written = {}, []
    for n, (candidate, _, _) in best.items():
        sent = eligible[n]
        chosen[(id(sent.doc), sent.start)] = finish_sentence(candidate, sent.text)
        written += [content_words(s) for s in re.split(r'(?<=[.!?])\s+', candidate)]
    for k, doc in zip(blocks, docs):
        # each sentence keeps the space after it (a second space or a tab is a token of its own in spaCy)
        rebuilt = ''.join(chosen[(id(doc), sent.start)] + sent.text_with_ws[len(sent.text_with_ws.rstrip()):]
                          if (id(doc), sent.start) in chosen else sent.text_with_ws for sent in doc.sents)
        block = parts[k]
        parts[k] = block[:len(block) - len(block.lstrip())] + rebuilt.rstrip() + block[len(block.rstrip()):]
    logger.info('Deep Rewrite: %d of %d sentences rewritten in %.1f s', len(chosen), len(eligible), time.monotonic() - started)
    return ''.join(parts), written

# --- Vetting WordNet swaps with the T5 (Deep Rewrite requests, when the model is loaded) ---
# A one-word swap that the paraphraser finds implausible as a paraphrase of the sentence changed its sense
# ("strike a balance" -> "impress a balance", "legacy" -> "bequest"). On hand-labelled swaps, good ones
# cost at most about 2 nats (once 4.9) and bad ones 4.5 or more ("suggest" -> "propose" 4.5, "lack" ->
# "deficiency" 5.2, "strike" -> "impress" 21), so the cut is 4 (3 for nouns, which carry more meaning).
# Only requests that hold a Deep Rewrite slot vet (paraphraser.reserve), so the checks are bounded by the
# slots and the Deep Rewrite size limit; Chaos Mode on its own stays fast and uses the rules without the model.
VET_DROP = 4.0
VET_DROP_NOUN = 3.0
VET_MAX_PAIRS = 80      # per request: about 3 seconds of scoring at 1.5 CPUs
VET_CHUNK = 12          # pairs per call (whole sentences), so a check never holds the model for long
VET_MAX_TOKENS = 60     # a longer sentence costs too much to score; its WordNet guesses are dropped

class SwapVetter:
    """Checks one request's WordNet swaps with the T5. Once the model is busy (another Deep Rewrite holds
    it) or the request's share of checks is used up, it stops for good: guesses made on the assumption that
    they would be checked are dropped, and later blocks use the rules for working without the model."""

    def __init__(self):
        self.busy, self.pairs = False, 0

    def __call__(self, sent, guessed):
        return vet_many([(sent, guessed)], self)

    def many(self, batch):
        return vet_many(batch, self)

def vet_swaps(sent, guessed):
    return vet_many([(sent, guessed)])

def vet_many(batch, vetter=None):
    """The swaps to keep from [(sentence, {token index: new word})], scored with the T5 a few whole sentences
    per call. A guess that can't be scored is dropped: it was chosen on the assumption it would be checked."""
    groups = []
    for sent, guessed in batch:
        if guessed and len(sent) <= VET_MAX_TOKENS:
            original = sent.text.strip()
            swaps = list(guessed.items())
            groups.append((sent, swaps, [(original, original)] + [
                (original, ''.join((word if t.i == i else t.text) + t.whitespace_ for t in sent).strip())
                for i, word in swaps]))
    kept, k = {}, 0
    while k < len(groups):
        call = [groups[k]]
        size = len(groups[k][2])
        k += 1
        while k < len(groups) and size + len(groups[k][2]) <= VET_CHUNK:
            call.append(groups[k])
            size += len(groups[k][2])
            k += 1
        if vetter is not None:
            if vetter.busy or vetter.pairs + size > VET_MAX_PAIRS:
                vetter.busy = True
                break
            vetter.pairs += size
        scores = paraphraser.try_score([pair for _, _, pairs in call for pair in pairs])
        if scores is None:      # another Deep Rewrite is using the model
            if vetter is not None:
                vetter.busy = True
            break
        at = 0
        for sent, swaps, pairs in call:
            base = scores[at][2]
            for (i, word), (_, _, total) in zip(swaps, scores[at + 1: at + len(pairs)]):
                if base - total <= (VET_DROP_NOUN if sent.doc[i].pos_ == 'NOUN' else VET_DROP):
                    kept[i] = word
            at += len(pairs)
    return kept

def swap_vetter(deep_mode, deep_rewrite):
    """A vetter for this request's WordNet swaps, or None (no noun swaps, and the Brown pair check decides)."""
    if deep_rewrite and NLP is not None and paraphraser.available() and paraphraser.is_loaded():
        return SwapVetter()
    return None

def usable(vet):
    """The vetter while it can still check swaps, else None (so the rules for working without it apply)."""
    return vet if vet is not None and not getattr(vet, 'busy', False) else None

# ==========================================
#   ENTRY POINTS
# ==========================================

def perform_nuclear_chaos(text, tone, deep_mode, deep_rewrite=False):
    text = clean_text(text)
    source = text
    model_written = []
    deep = deep_rewrite and deep_rewrite_available()
    if deep:
        text, model_written = deep_paraphrase(text, deep_mode)
    text = apply_phrase_rules(text, tone, deep_mode)
    text = apply_tone_contractions(text, tone, deep_mode)
    if NLP is not None:
        vet = swap_vetter(deep_mode, deep) if len(source) <= paraphraser.MAX_CHARS else None
        text = rewrite_with_spacy(text, tone, deep_mode, model_written, vet)
    else:
        text = rewrite_basic(text, tone, deep_mode)
    return fix_number_articles(text, source)

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
