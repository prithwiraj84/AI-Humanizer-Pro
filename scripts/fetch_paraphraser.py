"""
Downloads the Deep Rewrite model into data/paraphraser/ from the zip at PARAPHRASER_URL (made by
scripts/build_paraphraser.py --zip). start.py runs this automatically on first start when
PARAPHRASER_URL is set; it can also be run by hand:
    python scripts/fetch_paraphraser.py

It never stops the app from starting: without the model, Deep Rewrite simply isn't offered.
"""
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'paraphraser')


def main():
    url = os.environ.get('PARAPHRASER_URL', '').strip()
    if not url:
        print('PARAPHRASER_URL is not set; Deep Rewrite stays off', file=sys.stderr)
        return
    if os.path.isfile(os.path.join(OUT_DIR, 'model.bin')):
        return
    try:
        os.makedirs(os.path.dirname(OUT_DIR), exist_ok=True)
        # Work next to the destination: panels such as Pterodactyl give /tmp only ~100 MB of memory
        with tempfile.TemporaryDirectory(dir=os.path.dirname(OUT_DIR)) as work:
            archive = os.path.join(work, 'paraphraser.zip')
            request = urllib.request.Request(url, headers={'User-Agent': 'ai-humanizer-model-fetch/1.0'})
            with urllib.request.urlopen(request, timeout=300) as response, open(archive, 'wb') as out:
                shutil.copyfileobj(response, out)
            unpacked = os.path.join(work, 'model')
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():   # refuse paths that would land outside the folder
                    if name.startswith(('/', '\\')) or '..' in name.replace('\\', '/').split('/'):
                        raise ValueError(f'unsafe path in archive: {name}')
                z.extractall(unpacked)
            if not os.path.isfile(os.path.join(unpacked, 'model.bin')):
                raise ValueError('the archive has no model.bin')
            shutil.rmtree(OUT_DIR, ignore_errors=True)
            shutil.move(unpacked, OUT_DIR)
        print(f'Deep Rewrite model saved to {OUT_DIR}')
    except Exception as e:
        print(f'Could not download the Deep Rewrite model: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
