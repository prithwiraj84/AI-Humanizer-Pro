# AI Text Humanizer

A Flask-based web application that transforms AI-generated text into human-like content using advanced linguistic techniques. This tool employs multiple layers of obfuscation and transformation to evade AI detection algorithms while maintaining readability.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Ethical Considerations](#ethical-considerations)
- [License](#license)

---

## 🎯 Overview

The **AI Text Humanizer** is designed to take AI-generated content and transform it into text that appears more naturally human-written. It uses four distinct transformation techniques working in tandem to modify text structure, vocabulary, and encoding without compromising comprehensibility.

**Target Audience:** Students, content creators, and researchers exploring text transformation and AI detection evasion techniques.

---

## ✨ Features

### Core Transformation Methods

1. **Synonym Spiking (Perplexity Attack)**
   - Replaces common words with less frequently used synonyms
   - Raises unpredictability scores that AI detectors rely on
   - Uses WordNet corpus for valid synonym selection
   - Configurable intensity (0.0 - 1.0)

2. **Unicode Spoofing (Tokenization Attack)**
   - Mixes Unicode normalization forms (NFC vs NFD)
   - Characters appear identical to humans but differ at binary level
   - Confuses character-level tokenization in detectors
   - Example: `é` (U+00E9) vs `e + ́` (U+0065 + U+0301)

3. **Invisible Character Injection (Pattern Matching Attack)**
   - Injects zero-width spaces and word joiners between words
   - Characters: U+200B (Zero Width Space), U+2060 (Word Joiner), U+200C, U+200D
   - Survives most sanitization filters
   - Breaks token sequences without affecting readability

4. **Grammatical Shattering (Syntax Attack)**
   - Routes text through linguistically distant languages
   - Chain: English (SVO) → Arabic (VSO) → Korean (SOV) → English
   - Different syntactic structures prevent reconstruction
   - Fallback to original text if translation fails

### Transformation Modes

- **Simple Mode:** Basic translation-back pipeline (faster, less aggressive)
- **Deep Mode:** All four weapons combined for maximum obfuscation
- **Tone Control:** Professional or standard variations

### Analysis Dashboard

- **Real-time metrics display**
- **Diff highlighting:** Shows changed words
- **Structural change percentage:** Quantifies transformation intensity
- **Word change count:** Total words modified
- **Longest unchanged sequence:** Shows what stayed the same

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend Framework** | Flask 2.x |
| **Frontend** | HTML5, CSS3, JavaScript (Vanilla) |
| **NLP Library** | NLTK (WordNet corpus) |
| **Translation Service** | Google Translator (deep_translator) |
| **Server** | Flask development server |
| **Language** | Python 3.8+ |

### Dependencies

```
Flask>=2.0.0
deep_translator>=1.10.0
nltk>=3.6.0
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for NLTK data and Google Translate)

### Step-by-Step Setup

1. **Clone or download the project**
   ```bash
   cd your-project-directory
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install Flask deep_translator nltk
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Open your browser and navigate to `http://localhost:5000`
   - You should see the humanizer interface

---

## 🚀 Usage

### Basic Workflow

1. **Paste your text** into the input textarea
2. **Select transformation mode:**
   - Simple Mode: Quick, less aggressive transformation
   - Deep Mode: Comprehensive obfuscation using all four techniques
3. **Choose tone variation:**
   - Professional: Targets academic/formal writing
   - Standard: General-purpose transformation
4. **Click "Humanize"** to process
5. **View results:**
   - Humanized text output
   - AI detection probability
   - Structural changes analysis
   - Word-by-word diff highlighting

### Example Input/Output

**Input:** 
```
The artificial intelligence system was designed to process large datasets efficiently.
```

**Output (Deep Mode):**
```
The artificial intelligence mechanism appeared constructed to treat broad information repositories productively.
```

*(Note: Output varies due to random synonym selection and Unicode injection)*

---

## 🧠 How It Works

### The Four-Layer Attack Strategy

#### Layer 1: Synonym Spiking
```
Original:  "The system was designed to process data"
Modified:  "The mechanism was engineered to handle information"
Score Impact: Increases perplexity (unpredictability)
```

#### Layer 2: Unicode Spoofing
```
Binary Level: Characters are decomposed/recomposed using NFD/NFC forms
Visual Result: Identical on screen, different in tokenization
Impact: Breaks byte-level pattern matching
```

#### Layer 3: Invisible Character Injection
```
Visual:    "word1 word2"
Actual:    "word1\u200Bword2" (Zero-width space injected)
Impact:    Disrupts token boundaries
```

#### Layer 4: Grammatical Shattering
```
Step 1: English → Arabic      (Changes verb-subject order)
Step 2: Arabic → Korean       (Introduces particle system)
Step 3: Korean → English      (Reconstructs with new structure)
Impact: Syntax tree becomes unrecognizable to English-trained models
```

### Processing Flow

```
Input Text
    ↓
[Simple Mode] ────→ Translation (JP/FR) → Back-translate → Output
    ↓
[Deep Mode]
    ├→ Grammatical Shattering (Syntax layer)
    ├→ Synonym Spiking (Vocabulary layer)
    ├→ Unicode Spoofing (Character layer)
    ├→ Invisible Glue Injection (Token layer)
    └→ Output
    ↓
Metrics Calculation (Diff, Changes, Structural Score)
    ↓
JSON Response to Frontend
```

---

## 📁 Project Structure

```
project-root/
├── app.py                  # Main Flask application
├── index.html             # Frontend interface
├── requirements.txt       # Python dependencies (create this)
└── README.md             # This file
```

### File Descriptions

**app.py:**
- 8,623 characters
- Contains all transformation logic
- Routes: `/` (homepage), `/humanize` (API endpoint)
- Four main transformation functions
- NLTK initialization and data download

**index.html:**
- 24,191 characters
- Complete frontend UI
- Form handling and AJAX requests
- Results display and visualization
- Responsive design

---

## 🔌 API Reference

### Humanize Endpoint

**URL:** `/humanize`  
**Method:** `POST`  
**Content-Type:** `application/json`

#### Request Format

```json
{
  "text": "Your text to humanize here",
  "tone": "professional",
  "deep_mode": true
}
```

#### Request Parameters

| Parameter | Type | Required | Values | Description |
|-----------|------|----------|--------|-------------|
| `text` | string | Yes | Any | Input text to transform |
| `tone` | string | No | "professional", "standard" | Transformation style |
| `deep_mode` | boolean | No | true, false | Use all four techniques (true) or simple mode (false) |

#### Response Format

```json
{
  "original": "Input text",
  "humanized": "Transformed text with invisible characters",
  "ai_probability": "4.2%",
  "confidence": "99.8%",
  "classification": "Human-Written",
  "diff_html": "HTML showing word changes",
  "longest_unchanged": "sequence of words",
  "structural_changes": "45.3%",
  "changed_words": 12
}
```

#### Response Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `original` | string | Original input text |
| `humanized` | string | Transformed text (includes invisible chars) |
| `ai_probability` | string | Simulated AI detection probability |
| `confidence` | string | Simulated detector confidence score |
| `classification` | string | Predicted classification |
| `diff_html` | string | HTML-formatted word changes |
| `longest_unchanged` | string | Longest sequence without changes |
| `structural_changes` | string | Percentage of structural modification |
| `changed_words` | integer | Number of modified words |

#### Example Request (cURL)

```bash
curl -X POST http://localhost:5000/humanize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The experiment was successful.",
    "tone": "professional",
    "deep_mode": true
  }'
```

#### Error Responses

**Missing Text (400):**
```json
{
  "error": "No text provided"
}
```

**Processing Failed (500):**
```json
{
  "error": "Processing failed. Text may be too long."
}
```

---

## ⚙️ Configuration

### Adjustment Parameters in app.py

**Intensity Control (Line ~110)**
```python
intensity = 0.3  # Synonym spiking intensity (0.0 = none, 1.0 = all words)
```
- Lower values = fewer synonyms changed
- Higher values = more aggressive replacement
- Recommended: 0.3 - 0.5

**Unicode Spoofing Probability (Line ~140)**
```python
if random.random() < 0.5:  # 50% chance per character
```
- Controls how many characters are normalized differently
- Range: 0.0 - 1.0
- Higher = more characters affected

**Invisible Character Injection Probability (Line ~160)**
```python
if len(word) > 3 and random.random() < 0.4:  # 40% for words > 3 chars
```
- Controls injection frequency
- Only affects words longer than 3 characters
- Recommended: 0.3 - 0.5

**Translation Chain Targets (Line ~190)**
```python
v1 = GoogleTranslator(source='auto', target='ar').translate(text)
v2 = GoogleTranslator(source='ar', target='ko').translate(v1)
```
- Change language pairs for different results
- Recommended chains: AR→KO, ZH→JA, HI→KO

### NLTK Data Download

The app automatically downloads required data on first run:
```python
nltk.download('wordnet')
nltk.download('omw-1.4')
```

To pre-download manually:
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## 🔧 Troubleshooting

### Common Issues

**Issue: "LookupError: Wordnet data not found"**
- **Solution:** The app should auto-download. If not, run manually:
  ```python
  import nltk
  nltk.download('wordnet')
  nltk.download('omw-1.4')
  ```

**Issue: Translation API fails (timeout/connection)**
- **Solution:** 
  - Check internet connection
  - Google Translate may block rapid requests
  - Add delay: `time.sleep(0.5)` between requests
  - Use `try-except` fallback (already implemented)

**Issue: "No module named 'flask'"**
- **Solution:** Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

**Issue: Port 5000 already in use**
- **Solution:** Change port in app.py:
  ```python
  app.run(debug=True, port=5001)
  ```

**Issue: Text processing takes too long**
- **Solution:**
  - Use Simple Mode instead of Deep Mode
  - Reduce text length (try under 500 words)
  - Deep Mode is slower due to multiple translation steps

**Issue: Humanized text doesn't look different**
- **Solution:**
  - Invisible characters are included but not visible
  - View HTML source to see `\u200B` characters
  - Try Deep Mode for more visible changes
  - Increase intensity parameter

---

## 📊 Performance Notes

### Processing Time (Approximate)

| Mode | Text Length | Time |
|------|-------------|------|
| Simple | 100 words | 2-4 sec |
| Simple | 500 words | 4-8 sec |
| Deep | 100 words | 5-10 sec |
| Deep | 500 words | 15-30 sec |

**Bottlenecks:**
1. Google Translate API calls (3 calls in deep mode)
2. NLTK WordNet lookup (depends on word count)
3. Unicode normalization (minimal impact)

**Optimization Tips:**
- Cache synonyms for repeated words
- Implement request queuing for multiple submissions
- Use async requests for translation

---

## ⚠️ Ethical Considerations

### Important Legal and Ethical Notes

**This tool is designed for educational and research purposes.** Users should be aware of the following:

1. **Academic Integrity:** Using this tool to submit AI-generated work as original human writing may violate academic honor codes. Institutions have specific policies against such practices.

2. **Content Detection Evasion:** While the technical approach is interesting, intentionally evading detection systems may be considered academic dishonesty.

3. **Responsible Use:** Consider using this tool to:
   - Understand how AI detection systems work
   - Learn about text transformation techniques
   - Conduct academic research on AI detection
   - Improve your own writing skills

4. **Recommended Alternative Uses:**
   - Use as an educational project to learn Flask, NLP, and APIs
   - Explore text transformation algorithms
   - Research AI detection limitations
   - Create writing enhancement tools that are transparent

5. **Disclosure:** If you use transformed text, disclose that it was AI-generated and transformed, even if it passes detection tools.

---

## 🚀 Future Enhancement Ideas

1. **Custom Dictionaries:** User-defined word replacement pools
2. **Style Preservation:** Maintain original tone/formality level
3. **Language Support:** Direct transformation in multiple languages
4. **Batch Processing:** Handle multiple texts simultaneously
5. **Metrics Dashboard:** Track transformation effectiveness across samples
6. **Advanced NLP:** Use spaCy for better POS tagging
7. **Caching Layer:** Store synonym lookups for performance
8. **Web Worker Threads:** Non-blocking processing in frontend
9. **Detection Score Feedback:** Integration with actual AI detectors
10. **Multilingual Support:** Support for non-English input text

---

## 📝 Requirements.txt

Create a `requirements.txt` file in your project root:

```txt
Flask==2.3.0
deep_translator==1.11.4
nltk==3.8.1
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

To contribute improvements:

1. Test changes locally with various input texts
2. Document any new transformation techniques
3. Update performance benchmarks
4. Add new language chains for grammatical shattering
5. Improve error handling for edge cases

---

## 📄 License

This project is provided as-is for educational purposes. Users are responsible for understanding and complying with applicable laws, regulations, and institutional policies.

---

## 👨‍💻 Development Notes

### Code Organization

The `app.py` file is organized into logical sections:

1. **Setup & Imports:** Flask, NLTK initialization
2. **Secret Weapon 1:** `get_rare_synonym()`, `spike_perplexity()`
3. **Secret Weapon 2:** `unicode_spoofing()`
4. **Secret Weapon 3:** `inject_invisible_glue()`
5. **Secret Weapon 4:** `grammatical_shatter()`
6. **Main Logic:** `calculate_diff_metrics()`, `perform_nuclear_chaos()`
7. **Routes:** Flask route handlers

### Key Functions

- `get_rare_synonym(word)` → Returns less-common synonym
- `spike_perplexity(text, intensity)` → Synonym replacement
- `unicode_spoofing(text)` → NFD/NFC mixing
- `inject_invisible_glue(text)` → Zero-width character injection
- `grammatical_shatter(text)` → Multi-language transformation chain
- `calculate_diff_metrics(original, humanized)` → Analysis metrics
- `perform_nuclear_chaos(text, tone, deep_mode)` → Orchestration function
- `humanize()` → Flask POST endpoint

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the How It Works section for understanding
3. Test with shorter text lengths first
4. Check API response error messages
5. Verify all dependencies are installed correctly

---

**Last Updated:** December 2025  
**Version:** 1.0  
**Status:** Beta (Educational/Research)
