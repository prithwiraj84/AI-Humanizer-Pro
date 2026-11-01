"""
Downloads DB-IP's free IP-to-country database, used to show regional prices.

Run during the build (see render.yaml) or locally:
    python scripts/fetch_geoip.py

The database is updated monthly; this fetches the current month's file and falls back to the
previous month early in the month. It never fails the build: without the file, every visitor
sees the default region's prices.

IP Geolocation by DB-IP (https://db-ip.com), licensed under CC BY 4.0.
"""
import gzip
import os
import shutil
import sys
import urllib.request
from datetime import date

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'geoip')
OUT_PATH = os.path.join(OUT_DIR, 'dbip-country-lite.mmdb')
URL = 'https://download.db-ip.com/free/dbip-country-lite-{month}.mmdb.gz'


def candidate_months():
    today = date.today()
    previous = date(today.year - (today.month == 1), (today.month - 2) % 12 + 1, 1)
    return [f'{today:%Y-%m}', f'{previous:%Y-%m}']


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for month in candidate_months():
        url = URL.format(month=month)
        # DB-IP rejects Python's default user agent
        request = urllib.request.Request(url, headers={'User-Agent': 'ai-humanizer-geoip-fetch/1.0'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, gzip.GzipFile(fileobj=response) as data:
                with open(OUT_PATH + '.tmp', 'wb') as out:
                    shutil.copyfileobj(data, out)
            os.replace(OUT_PATH + '.tmp', OUT_PATH)
            print(f'GeoIP database {month} saved to {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)')
            return
        except Exception as e:
            print(f'Could not download {url}: {e}', file=sys.stderr)
    print('Warning: no GeoIP database; all visitors will see default-region prices', file=sys.stderr)


if __name__ == '__main__':
    main()
