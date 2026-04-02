import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.get_data import fetch_injuries_data, build_latest_availability_snapshot


def main():
    injuries = fetch_injuries_data()
    build_latest_availability_snapshot(injuries_df=injuries)


if __name__ == '__main__':
    main()
