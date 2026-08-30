import argparse
import os
import zipfile

import requests

URL = "https://database.nikonoel.fr/lichess_elite_{month}.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", required=True, help="e.g. 2024-01 2024-02")
    ap.add_argument("--dest", default=r"C:\chessdata\raw")
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    for month in args.months:
        url = URL.format(month=month)
        zip_path = os.path.join(args.dest, f"elite_{month}.zip")
        if not os.path.exists(zip_path):
            print(f"downloading {url}")
            r = requests.get(url, stream=True, timeout=60)
            if r.status_code != 200:
                raise SystemExit(
                    f"{url} -> HTTP {r.status_code}. If the mirror is down, download "
                    f"the Lichess Elite Database manually and put the .pgn files in "
                    f"{args.dest}, then skip this script."
                )
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        print(f"extracting {zip_path}")
        zipfile.ZipFile(zip_path).extractall(args.dest)
    print("done")


if __name__ == "__main__":
    main()
