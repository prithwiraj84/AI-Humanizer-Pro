"""
Installs the Deep Rewrite model into data/paraphraser/ from the zip made by
scripts/build_paraphraser.py --zip: either one uploaded to data/paraphraser.zip (it's unpacked,
then deleted to free the disk space) or one downloaded from PARAPHRASER_URL. start.py runs this
automatically when the model is missing; it can also be run by hand:
    python scripts/fetch_paraphraser.py

It never stops the app from starting: without the model, Deep Rewrite simply isn't offered.
"""
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUT_DIR = os.path.join(DATA_DIR, 'paraphraser')
UPLOADED = os.path.join(DATA_DIR, 'paraphraser.zip')


def unpack(archive, work):
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


def main():
    if os.path.isfile(os.path.join(OUT_DIR, 'model.bin')):
        return
    url = os.environ.get('PARAPHRASER_URL', '').strip()
    if not os.path.isfile(UPLOADED) and not url:
        print('No Deep Rewrite model (upload data/paraphraser.zip or set PARAPHRASER_URL); Deep Rewrite stays off',
              file=sys.stderr)
        return
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        # Work next to the destination: some hosts give /tmp only ~100 MB of memory
        with tempfile.TemporaryDirectory(dir=DATA_DIR) as work:
            if os.path.isfile(UPLOADED):
                unpack(UPLOADED, work)
                os.remove(UPLOADED)
            else:
                archive = os.path.join(work, 'paraphraser.zip')
                request = urllib.request.Request(url, headers={'User-Agent': 'ai-humanizer-model-fetch/1.0'})
                with urllib.request.urlopen(request, timeout=300) as response, open(archive, 'wb') as out:
                    shutil.copyfileobj(response, out)
                unpack(archive, work)
        print(f'Deep Rewrite model installed in {OUT_DIR}')
    except Exception as e:
        print(f'Could not install the Deep Rewrite model: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
