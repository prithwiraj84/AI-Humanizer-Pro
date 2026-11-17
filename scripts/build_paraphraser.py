"""
Builds the optional Deep Rewrite models in CTranslate2's 8-bit format:
  data/paraphraser/     the T5 paraphraser (~225 MB), needed for Deep Rewrite
  data/backtranslate/   with --backtranslate: OPUS-MT translation models for back-translation through
                        another language (~150 MB per language), a second, human-trained source of rewrites

Run it once on a computer with a few GB of free disk (it downloads PyTorch and the original models);
the server only needs the results plus the small ctranslate2 and sentencepiece packages:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install "transformers<5" ctranslate2 sentencepiece
    python scripts/build_paraphraser.py                       # writes data/paraphraser/
    python scripts/build_paraphraser.py --backtranslate       # also writes data/backtranslate/ (French)
    python scripts/build_paraphraser.py --backtranslate --pivots fr,de --zip

--zip also packs each folder as data/<name>.zip. Copy the folders to the server, or upload the zips to the
server's data folder (start.py unpacks them), or host the zips and set PARAPHRASER_URL / BACKTRANSLATE_URL
so start.py downloads them. --skip-t5 builds only the back-translation models.

Models and licences (MODEL_LICENSE.txt in each folder):
- humarin/chatgpt_paraphraser_on_T5_base: OpenRAIL; its use restrictions apply to everyone who uses it.
- Helsinki-NLP/opus-mt-en-<lang> and opus-mt-<lang>-en (OPUS-MT, University of Helsinki): CC-BY 4.0,
  which requires credit to the authors when the models are used or shared.
"""
import argparse
import json
import os
import shutil

import ctranslate2

T5_MODEL = 'humarin/chatgpt_paraphraser_on_T5_base'
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

T5_LICENSE = f"""Deep Rewrite model: {T5_MODEL}
Converted to CTranslate2 (8-bit) by scripts/build_paraphraser.py.

License: OpenRAIL. The model card and license terms, including the use restrictions that apply to
anyone using or redistributing the model, are at https://huggingface.co/{T5_MODEL}
"""

OPUS_LICENSE = """Back-translation models: {models}
OPUS-MT by the Language Technology Research Group, University of Helsinki (Tiedemann & Thottingal,
"OPUS-MT - Building open translation services for the World", EAMT 2020): https://github.com/Helsinki-NLP/Opus-MT
Converted to CTranslate2 (8-bit) by scripts/build_paraphraser.py; not otherwise changed.

License: Creative Commons Attribution 4.0 International (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/
"""


def folder_size(path):
    return sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(path) for f in files) / 1e6


def pack(path):
    archive = shutil.make_archive(path, 'zip', path)
    print(f'Packed as {archive} ({os.path.getsize(archive) / 1e6:.0f} MB)')


def build_t5(out):
    converter = ctranslate2.converters.TransformersConverter(T5_MODEL, copy_files=['spiece.model'])
    converter.convert(out, quantization='int8', force=True)
    with open(os.path.join(out, 'humanizer.json'), 'w', encoding='utf-8') as f:
        json.dump({'name': T5_MODEL, 'license': 'OpenRAIL', 'prefix': 'paraphrase: ',
                   'tokenizer': 'sentencepiece', 'tokenizer_file': 'spiece.model'}, f, indent=2)
    with open(os.path.join(out, 'MODEL_LICENSE.txt'), 'w', encoding='utf-8') as f:
        f.write(T5_LICENSE)
    print(f'Deep Rewrite model written to {out} ({folder_size(out):.0f} MB)')


def build_backtranslation(out, pivots):
    os.makedirs(out, exist_ok=True)
    models, entries = [], []
    for lang in pivots:
        for name in (f'en-{lang}', f'{lang}-en'):
            repo = f'Helsinki-NLP/opus-mt-{name}'
            converter = ctranslate2.converters.TransformersConverter(repo, copy_files=['source.spm', 'target.spm'])
            converter.convert(os.path.join(out, name), quantization='int8', force=True)
            models.append(repo)
        entries.append({'lang': lang, 'forward': f'en-{lang}', 'back': f'{lang}-en'})
    with open(os.path.join(out, 'humanizer.json'), 'w', encoding='utf-8') as f:
        json.dump({'license': 'CC-BY-4.0', 'models': models, 'pivots': entries}, f, indent=2)
    with open(os.path.join(out, 'MODEL_LICENSE.txt'), 'w', encoding='utf-8') as f:
        f.write(OPUS_LICENSE.format(models=', '.join(models)))
    print(f'Back-translation models written to {out} ({folder_size(out):.0f} MB)')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--zip', action='store_true', help='also pack each folder as data/<name>.zip')
    parser.add_argument('--backtranslate', action='store_true', help='also build the back-translation models')
    parser.add_argument('--pivots', default='fr', help='comma-separated pivot languages (default: fr)')
    parser.add_argument('--skip-t5', action='store_true', help="don't rebuild the T5 paraphraser")
    args = parser.parse_args()

    if not args.skip_t5:
        out = os.path.join(DATA_DIR, 'paraphraser')
        build_t5(out)
        if args.zip:
            pack(out)
    if args.backtranslate:
        out = os.path.join(DATA_DIR, 'backtranslate')
        build_backtranslation(out, [p.strip() for p in args.pivots.split(',') if p.strip()])
        if args.zip:
            pack(out)


if __name__ == '__main__':
    main()
