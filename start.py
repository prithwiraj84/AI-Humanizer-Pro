"""
Starts the app in production with gunicorn (settings in gunicorn.conf.py).

For hosts whose panel runs a Python file rather than a command, such as Botkeep and other
Pterodactyl-style panels: set the startup file to start.py. Elsewhere `gunicorn app:app` does the
same. On the first start it also downloads the country database used for regional pricing and,
if PARAPHRASER_URL is set, the Deep Rewrite model.

Settings come from the host's environment variables or a .env file next to this one (panels often
allow only their own variables, so .env is the usual place there).
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
    if os.environ.get('PARAPHRASER_URL') and not os.path.exists(os.path.join(BASE_DIR, 'data', 'paraphraser', 'model.bin')):
        from scripts.fetch_paraphraser import main as fetch_paraphraser
        fetch_paraphraser()
    from gunicorn.app.wsgiapp import run
    sys.argv = ['gunicorn', 'app:app']
    run()


if __name__ == '__main__':
    main()
