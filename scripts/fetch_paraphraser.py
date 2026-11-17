"""
Installs the Deep Rewrite models from the zips made by scripts/build_paraphraser.py --zip:
  data/paraphraser.zip   -> data/paraphraser/    (the T5 paraphraser; or downloaded from PARAPHRASER_URL)
  data/backtranslate.zip -> data/backtranslate/  (optional back-translation; or from BACKTRANSLATE_URL)
PARAPHRASER_DIR and BACKTRANSLATE_DIR move the installed folders, as they do for paraphraser.py.
An uploaded zip is unpacked, checked and swapped in (replacing a model already installed), then deleted to
free the disk space. start.py runs this at every start; it can also be run by hand:
    python scripts/fetch_paraphraser.py

It never stops the app from starting: without the models, Deep Rewrite simply isn't offered.
"""
import json
import os
import shutil
import sys
import urllib.request
import zipfile

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
# (folder, environment variable that moves it, environment variable with a download link)
MODELS = [('paraphraser', 'PARAPHRASER_DIR', 'PARAPHRASER_URL'), ('backtranslate', 'BACKTRANSLATE_DIR', 'BACKTRANSLATE_URL')]


def complete(name, path):
    """True if `path` holds every file paraphraser.py loads for this model."""
    try:
        with open(os.path.join(path, 'humanizer.json'), encoding='utf-8') as f:
            info = json.load(f)
        if name == 'paraphraser':
            return all(os.path.isfile(os.path.join(path, n)) for n in ('model.bin', info['tokenizer_file']))
        return bool(info['pivots']) and all(
            os.path.isfile(os.path.join(path, pivot[direction], n))
            for pivot in info['pivots'] for direction in ('forward', 'back')
            for n in ('model.bin', 'source.spm', 'target.spm'))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def unpack(name, archive, work, out_dir):
    """Unpack `archive` in `work`, check it, and swap it in for `out_dir`."""
    unpacked = os.path.join(work, 'model')
    with zipfile.ZipFile(archive) as z:
        for entry in z.namelist():      # refuse paths that would land outside the folder
            if entry.startswith(('/', '\\')) or ':' in entry or '..' in entry.replace('\\', '/').split('/'):
                raise ValueError(f'unsafe path in archive: {entry}')
        z.extractall(unpacked)
    root = unpacked
    if not complete(name, root):
        # a zip made from the folder itself holds it as its only entry
        inside = [os.path.join(unpacked, e) for e in os.listdir(unpacked)]
        if len(inside) == 1 and os.path.isdir(inside[0]) and complete(name, inside[0]):
            root = inside[0]
        else:
            raise ValueError('the archive is missing model files (see scripts/build_paraphraser.py --zip)')
    old = os.path.join(work, 'old')
    if os.path.exists(out_dir):
        os.replace(out_dir, old)        # same volume: a rename, so the swap is quick
    try:
        shutil.move(root, out_dir)
    except Exception:
        if os.path.exists(old) and not os.path.exists(out_dir):
            os.replace(old, out_dir)    # put the model that was there back
        raise


def install(name, dir_variable, url_variable):
    out_dir = os.environ.get(dir_variable) or os.path.join(DATA_DIR, name)
    uploaded = os.path.join(DATA_DIR, name + '.zip')
    # Work next to the destination (some hosts give /tmp only ~100 MB of memory), in a folder with a fixed name
    # so an install cut short is cleared away at the next start
    work = os.path.join(os.path.dirname(os.path.abspath(out_dir)), '.installing-' + name)
    shutil.rmtree(work, ignore_errors=True)
    url = os.environ.get(url_variable, '').strip()
    if not os.path.isfile(uploaded) and (complete(name, out_dir) or not url):
        if name == 'paraphraser' and not complete(name, out_dir):
            print(f'No Deep Rewrite model (upload data/{name}.zip or set {url_variable}); Deep Rewrite stays off',
                  file=sys.stderr)
        return
    try:
        os.makedirs(work)
        if os.path.isfile(uploaded):
            unpack(name, uploaded, work, out_dir)
            os.remove(uploaded)
        else:
            archive = os.path.join(work, name + '.zip')
            request = urllib.request.Request(url, headers={'User-Agent': 'ai-humanizer-model-fetch/1.0'})
            with urllib.request.urlopen(request, timeout=300) as response, open(archive, 'wb') as out:
                shutil.copyfileobj(response, out)
            unpack(name, archive, work, out_dir)
        print(f'Deep Rewrite model installed in {out_dir}')
    except Exception as e:
        print(f'Could not install the {name} model: {e}', file=sys.stderr)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    for name, dir_variable, url_variable in MODELS:
        install(name, dir_variable, url_variable)


if __name__ == '__main__':
    main()
