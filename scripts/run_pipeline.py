import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.get_data import fetch_games_data, fetch_players_data, fetch_player_game_logs_data, fetch_odds_data
from scripts.fetch_schedule import fetch_upcoming_schedule
from scripts.fetch_availability import fetch_availability_for_upcoming
from scripts.build_features import main as build_features_main
from scripts.build_inference_features import main as build_inference_features_main
from scripts.generate_monitoring_report import generate_monitoring_report
from scripts.train_baseline import train_baseline
from scripts.train_tree_model import train_tree_model
from scripts.train_sequential import train_sequential
from scripts.train_ensemble import train_ensemble
from scripts.train_player_model import train_player_model


def phase0_setup():
    print("PHASE 0: Foundation -> data ingestion and initial feature engineering")
    fetch_games_data()
    fetch_players_data()
    fetch_player_game_logs_data()
    fetch_odds_data()
    fetch_upcoming_schedule(days_ahead=7)
    fetch_availability_for_upcoming()


def phase1_feature_engineering():
    print("PHASE 1: Data & Feature Engineering")
    build_features_main()
    build_inference_features_main()
    generate_monitoring_report()


def phase2_model_training():
    print("PHASE 2: Model training")
    train_baseline()
    train_tree_model()
    train_player_model()
    train_sequential()
    train_ensemble()


def main():
    phase0_setup()
    phase1_feature_engineering()
    phase2_model_training()
    print("Pipeline complete. Models trained.")


if __name__ == '__main__':
    main()
