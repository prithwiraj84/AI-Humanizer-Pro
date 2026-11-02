"""
Builds data/paraphraser/: the optional Deep Rewrite model in CTranslate2's 8-bit format.

Run it once on a computer with a few GB of free disk (it downloads PyTorch and the original model);
the server only needs the result (~225 MB) plus the small ctranslate2 and sentencepiece packages:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install "transformers<5" ctranslate2 sentencepiece
    python scripts/build_paraphraser.py            # writes data/paraphraser/
    python scripts/build_paraphraser.py --zip      # also writes paraphraser.zip for PARAPHRASER_URL

Then either copy data/paraphraser/ to the server, or host paraphraser.zip somewhere (e.g. a GitHub
release) and set PARAPHRASER_URL so start.py downloads it on first start.

Model: humarin/chatgpt_paraphraser_on_T5_base, a T5-base model fine-tuned for paraphrasing,
licensed under OpenRAIL. Its use restrictions apply to everyone who uses it; MODEL_LICENSE.txt in
the output points to the terms.
"""
import argparse
import json
import os
import shutil

import ctranslate2

MODEL = 'humarin/chatgpt_paraphraser_on_T5_base'
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'paraphraser')

LICENSE_NOTE = f"""Deep Rewrite model: {MODEL}
Converted to CTranslate2 (8-bit) by scripts/build_paraphraser.py.

License: OpenRAIL. The model card and license terms, including the use restrictions that apply to
anyone using or redistributing the model, are at https://huggingface.co/{MODEL}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--output', default=OUT_DIR)
    parser.add_argument('--zip', action='store_true', help='also pack the result as paraphraser.zip')
    args = parser.parse_args()

    converter = ctranslate2.converters.TransformersConverter(MODEL, copy_files=['spiece.model'])
    converter.convert(args.output, quantization='int8', force=True)
    with open(os.path.join(args.output, 'humanizer.json'), 'w', encoding='utf-8') as f:
        json.dump({'name': MODEL, 'license': 'OpenRAIL', 'prefix': 'paraphrase: ',
                   'tokenizer': 'sentencepiece', 'tokenizer_file': 'spiece.model'}, f, indent=2)
    with open(os.path.join(args.output, 'MODEL_LICENSE.txt'), 'w', encoding='utf-8') as f:
        f.write(LICENSE_NOTE)
    size = sum(os.path.getsize(os.path.join(args.output, n)) for n in os.listdir(args.output)) / 1e6
    print(f'Deep Rewrite model written to {args.output} ({size:.0f} MB)')

    if args.zip:
        archive = shutil.make_archive(os.path.join(os.path.dirname(args.output), 'paraphraser'), 'zip', args.output)
        print(f'Packed as {archive} ({os.path.getsize(archive) / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
