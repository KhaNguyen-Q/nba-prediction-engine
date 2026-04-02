import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PROCESSED_PATH = ROOT_DIR / 'data' / 'processed' / 'games_with_features.csv'
BACKTEST_OUTPUT_PATH = ROOT_DIR / 'data' / 'processed' / 'backtest_results.csv'
BACKTEST_SUMMARY_PATH = ROOT_DIR / 'reports' / 'backtest_summary.csv'
MODEL_PATHS = {
    'baseline': ROOT_DIR / 'models' / 'logistic_baseline.pkl',
    'tree': ROOT_DIR / 'models' / 'xgb_tree_model.pkl',
    'ensemble': ROOT_DIR / 'models' / 'ensemble_meta.pkl',
}
BASELINE_FEATURES = [
    'pts_last5', 'reb_last5', 'ast_last5',
    'pts_last10', 'reb_last10', 'ast_last10',
    'REST_DAYS', 'BACK_TO_BACK', 'TRAVEL_DISTANCE', 'TIMEZONE_SHIFT',
    'fatigue_index', 'ADULT_ENTERTAINMENT_INDEX',
]


def load_backtest_data(path=PROCESSED_PATH):
    if not path.exists():
        raise FileNotFoundError(f'Processed dataset not found at {path}')
    df = pd.read_csv(path)
    return df


def american_odds_to_probability(odds):
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None

    if odds == 0:
        return None
    if abs(odds) >= 100:
        if odds > 0:
            return 100.0 / (odds + 100.0)
        return -odds / (-odds + 100.0)

    if odds > 1.0:
        return 1.0 / odds

    return None


def american_odds_to_profit(odds, stake=1.0):
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None

    if odds == 0:
        return None
    if abs(odds) >= 100:
        if odds > 0:
            return stake * (odds / 100.0)
        return stake * (100.0 / abs(odds))
    if odds > 1.0:
        return stake * (odds - 1.0)
    return None


def get_model_path(model_name):
    return MODEL_PATHS.get(model_name)


def load_model(model_name):
    path = get_model_path(model_name)
    if path is None or not path.exists():
        return None
    return joblib.load(path)


def prepare_model_inputs(model, df, model_name):
    if hasattr(model, 'feature_names_in_'):
        required = list(model.feature_names_in_)
        missing = [name for name in required if name not in df.columns]
        if missing:
            raise ValueError(f"Model '{model_name}' requires missing columns: {missing}")
        return df[required].fillna(0)

    if model_name == 'baseline':
        missing = [name for name in BASELINE_FEATURES if name not in df.columns]
        if missing:
            raise ValueError(f"Baseline model requires missing columns: {missing}")
        return df[BASELINE_FEATURES].fillna(0)

    if model_name == 'tree':
        features = df.select_dtypes(include=[np.number]).drop(columns=['WIN'], errors='ignore')
        return features.fillna(0)

    if model_name == 'ensemble':
        raise ValueError('Ensemble model inputs should be built from baseline/tree outputs.')

    raise ValueError(f'Unknown model {model_name}')


def model_predict_proba(model_name, df):
    model = load_model(model_name)
    if model is None:
        return None

    if model_name == 'ensemble':
        baseline_probs = model_predict_proba('baseline', df)
        tree_probs = model_predict_proba('tree', df)
        if baseline_probs is None or tree_probs is None:
            return None
        ensemble_input = pd.DataFrame({'baseline_prob': baseline_probs, 'tree_prob': tree_probs})
        return model.predict_proba(ensemble_input)[:, 1]

    X = prepare_model_inputs(model, df, model_name)
    if not hasattr(model, 'predict_proba'):
        raise AttributeError(f"Model '{model_name}' does not support probability predictions.")
    return model.predict_proba(X)[:, 1]


def resolve_odds_columns(df):
    if {'TEAM_ML_PRICE', 'TEAM_IMPLIED_PROB'}.issubset(df.columns):
        df = df.copy()
        df['ODDS_PRICE'] = df['TEAM_ML_PRICE']
        df['IMPLIED_PROB'] = df['TEAM_IMPLIED_PROB']
        return df

    if {'HOME_ML_PRICE', 'HOME_ML_PROB', 'HOME'}.issubset(df.columns):
        df = df.copy()
        df = df[df['HOME'] == 1].copy()
        df['ODDS_PRICE'] = df['HOME_ML_PRICE']
        df['IMPLIED_PROB'] = df['HOME_ML_PROB']
        return df

    if 'CLOSING_ODDS' in df.columns:
        df = df.copy()
        df['IMPLIED_PROB'] = df['CLOSING_ODDS'].apply(american_odds_to_probability)
        df['ODDS_PRICE'] = df['CLOSING_ODDS']
        return df

    return None


def compute_bet_profit(row, stake=1.0):
    price = row.get('ODDS_PRICE')
    if pd.isna(price):
        return None
    win = row.get('WIN')
    profit = american_odds_to_profit(price, stake)
    if profit is None:
        return None
    return profit if win == 1 else -stake


def run_backtest(model_name='baseline', threshold=0.05, stake=1.0):
    df = load_backtest_data()
    if 'WIN' not in df.columns:
        raise ValueError('Processed dataset must contain a WIN label.')

    odds_df = resolve_odds_columns(df)
    if odds_df is None or odds_df[['ODDS_PRICE', 'IMPLIED_PROB']].isna().all(axis=1).all():
        print('No odds fields available for backtesting. Ensure the processed dataset includes odds features.')
        summary = pd.DataFrame([{
            'model': model_name,
            'period': 'overall',
            'threshold': float(threshold),
            'stake': float(stake),
            'total_bets': 0,
            'total_staked': 0.0,
            'total_profit': 0.0,
            'roi': 0.0,
            'win_rate': 0.0,
            'sharpe_like': 0.0,
            'status': 'no_odds_data',
        }])
        BACKTEST_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(BACKTEST_SUMMARY_PATH, index=False)
        print(f'Saved backtest summary to {BACKTEST_SUMMARY_PATH}')
        return

    model_probs = model_predict_proba(model_name, odds_df)
    if model_probs is None:
        print(f'Unable to score model: {model_name}')
        summary = pd.DataFrame([{
            'model': model_name,
            'period': 'overall',
            'threshold': float(threshold),
            'stake': float(stake),
            'total_bets': 0,
            'total_staked': 0.0,
            'total_profit': 0.0,
            'roi': 0.0,
            'win_rate': 0.0,
            'sharpe_like': 0.0,
            'status': 'model_unavailable',
        }])
        BACKTEST_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(BACKTEST_SUMMARY_PATH, index=False)
        print(f'Saved backtest summary to {BACKTEST_SUMMARY_PATH}')
        return

    odds_df = odds_df.copy()
    odds_df['MODEL_PROB'] = model_probs
    odds_df = odds_df.dropna(subset=['MODEL_PROB', 'IMPLIED_PROB', 'WIN', 'ODDS_PRICE'])
    if odds_df.empty:
        print('No valid rows remain after filtering scores, odds, and labels.')
        return

    odds_df['BET_SIGNAL'] = odds_df['MODEL_PROB'] - odds_df['IMPLIED_PROB'] > threshold
    odds_df['PROFIT'] = odds_df.apply(lambda row: compute_bet_profit(row, stake) if row['BET_SIGNAL'] else 0.0, axis=1)
    odds_df['BET_OUTCOME'] = np.where(odds_df['BET_SIGNAL'], odds_df['PROFIT'], 0.0)
    odds_df['CUMULATIVE_PROFIT'] = odds_df['BET_OUTCOME'].cumsum()

    total_bets = int(odds_df['BET_SIGNAL'].sum())
    total_profit = float(odds_df['BET_OUTCOME'].sum())
    total_staked = float(total_bets * stake)
    roi = total_profit / total_staked if total_staked > 0 else 0.0
    win_rate = (odds_df.loc[odds_df['BET_SIGNAL'], 'BET_OUTCOME'] > 0).mean() if total_bets > 0 else 0.0

    print(f'Model: {model_name}')
    print(f'Threshold: {threshold:.2f}')
    print(f'Total bets: {total_bets}')
    print(f'Total staked: {total_staked:.2f}')
    print(f'Total profit: {total_profit:.2f}')
    print(f'ROI: {roi:.4f}')
    print(f'Win rate: {win_rate:.2%}')

    BACKTEST_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    odds_df.to_csv(BACKTEST_OUTPUT_PATH, index=False)
    print(f'Saved backtest results to {BACKTEST_OUTPUT_PATH}')

    summary_df = build_backtest_summary(odds_df, model_name=model_name, threshold=threshold, stake=stake)
    BACKTEST_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(BACKTEST_SUMMARY_PATH, index=False)
    print(f'Saved backtest summary to {BACKTEST_SUMMARY_PATH}')


def _summarize_slice(frame, model_name, threshold, stake, period_label):
    placed = frame[frame['BET_SIGNAL']].copy()
    bets = int(len(placed))
    profit = float(placed['BET_OUTCOME'].sum()) if bets > 0 else 0.0
    staked = float(bets * stake)
    roi = profit / staked if staked > 0 else 0.0
    win_rate = float((placed['BET_OUTCOME'] > 0).mean()) if bets > 0 else 0.0
    sharpe = 0.0
    if bets > 1:
        std = float(placed['BET_OUTCOME'].std(ddof=1))
        if std > 0:
            sharpe = float(placed['BET_OUTCOME'].mean() / std)
    return {
        'model': model_name,
        'period': period_label,
        'threshold': float(threshold),
        'stake': float(stake),
        'total_bets': bets,
        'total_staked': staked,
        'total_profit': profit,
        'roi': roi,
        'win_rate': win_rate,
        'sharpe_like': sharpe,
        'status': 'ok',
    }


def build_backtest_summary(odds_df, model_name, threshold, stake):
    work = odds_df.copy()
    if 'GAME_DATE' in work.columns:
        work['GAME_DATE'] = pd.to_datetime(work['GAME_DATE'], errors='coerce')
        work['YEAR_MONTH'] = work['GAME_DATE'].dt.to_period('M').astype(str)
    else:
        work['YEAR_MONTH'] = 'unknown'

    rows = [_summarize_slice(work, model_name, threshold, stake, 'overall')]
    for month, group in work.groupby('YEAR_MONTH', dropna=True):
        if str(month).lower() == 'nat':
            continue
        rows.append(_summarize_slice(group, model_name, threshold, stake, str(month)))
    summary = pd.DataFrame(rows)
    if not summary.empty:
        month_rows = summary[summary['period'] != 'overall'].sort_values(by='period')
        overall_rows = summary[summary['period'] == 'overall']
        summary = pd.concat([overall_rows, month_rows], ignore_index=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description='Backtest NBA model predictions against market odds.')
    parser.add_argument('--model', choices=list(MODEL_PATHS.keys()), default='baseline', help='Model to use for prediction probability scoring.')
    parser.add_argument('--threshold', type=float, default=0.05, help='Minimum edge threshold between model probability and implied odds probability.')
    parser.add_argument('--stake', type=float, default=1.0, help='Stake per placed bet.')
    return parser.parse_args()


def main():
    args = parse_args()
    run_backtest(model_name=args.model, threshold=args.threshold, stake=args.stake)


if __name__ == '__main__':
    main()
