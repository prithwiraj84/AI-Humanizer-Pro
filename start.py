"""
Starts the app in production with gunicorn (settings in gunicorn.conf.py).

For hosts whose panel runs a start command, such as Botkeep: use `python start.py`. Elsewhere
`gunicorn app:app` does the same. On start it also downloads the country database used for
regional pricing if it's missing, and installs the Deep Rewrite model if data/paraphraser.zip has
been uploaded or PARAPHRASER_URL is set.

Settings come from the host's environment variables or a .env file next to this one (Botkeep
writes its Environment tab to .env).
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    os.chdir(BASE_DIR)    # gunicorn reads gunicorn.conf.py from the working directory
    sys.path.insert(0, BASE_DIR)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, '.env'))   # before gunicorn reads its settings and the downloads below
    if not os.path.exists(os.path.join(BASE_DIR, 'data', 'geoip', 'dbip-country-lite.mmdb')):
        from scripts.fetch_geoip import main as fetch_geoip
        fetch_geoip()
    if not os.path.exists(os.path.join(BASE_DIR, 'data', 'paraphraser', 'model.bin')):
        from scripts.fetch_paraphraser import main as fetch_paraphraser
        fetch_paraphraser()    # does nothing (and says so) without an uploaded zip or PARAPHRASER_URL
    from gunicorn.app.wsgiapp import run
    sys.argv = ['gunicorn', 'app:app']
    run()


if __name__ == '__main__':
    main()
