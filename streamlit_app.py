import json
from pathlib import Path

try:
    import altair as alt
except Exception:
    alt = None

import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="NBA Prediction Dashboard",
    page_icon="🏀",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #f7f4ef 0%, #f2f7fb 100%); }
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
      .metric-card { border: 1px solid #d8dee8; border-radius: 10px; padding: 12px; background: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_teams():
    team_path = Path("config/nba_teams.json")
    if not team_path.exists():
        return []
    with team_path.open("r", encoding="utf-8") as fh:
        teams = json.load(fh)
    teams = sorted(teams, key=lambda x: x.get("name", ""))
    return teams


def label_for_team(team):
    return f"{team['name']} ({team['abbreviation']}) [{team['team_id']}]"


def safe_get_json(url, timeout=20):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:
        return None, str(exc)


def safe_post_json(url, payload, timeout=30):
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:
        return None, str(exc)


def projection_table(projection_block):
    players = projection_block.get("players", [])
    if not players:
        return pd.DataFrame()
    df = pd.DataFrame(players)
    preferred = [
        "player_name",
        "projection_source",
        "availability",
        "injury_status",
        "projected_minutes",
        "projected_minutes_ci_low",
        "projected_minutes_ci_high",
        "projected_points",
        "projected_points_ci_low",
        "projected_points_ci_high",
        "projected_rebounds",
        "projected_rebounds_ci_low",
        "projected_rebounds_ci_high",
        "projected_assists",
        "projected_assists_ci_low",
        "projected_assists_ci_high",
    ]
    existing = [col for col in preferred if col in df.columns]
    remaining = [col for col in df.columns if col not in existing]
    return df[existing + remaining]


def format_pct(value, decimals=1):
    try:
        v = float(value)
    except Exception:
        return "N/A"
    return f"{100.0 * v:.{decimals}f}%"


def format_number_pct(value, decimals=1):
    try:
        v = float(value)
    except Exception:
        return "N/A"
    return f"{v:.{decimals}f}%"


def explanation_table(explain_block):
    rows = explain_block.get("top_features", []) if isinstance(explain_block, dict) else []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "impact_pct" in df.columns:
        df["impact_pct"] = df["impact_pct"].map(lambda x: format_number_pct(x, 2))
    return df


def explanation_chart_df(explain_block):
    rows = explain_block.get("top_features", []) if isinstance(explain_block, dict) else []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty or "feature" not in df.columns or "impact_pct" not in df.columns:
        return pd.DataFrame()
    df["signed_impact_pct"] = df.apply(
        lambda r: float(r.get("impact_pct", 0.0)) * (1.0 if str(r.get("direction", "up")) == "up" else -1.0),
        axis=1,
    )
    df["direction"] = df["signed_impact_pct"].apply(lambda x: "Positive" if x >= 0 else "Negative")
    df["impact_pct_abs"] = df["signed_impact_pct"].abs()
    return df.sort_values("signed_impact_pct")[["feature", "signed_impact_pct", "direction", "impact_pct_abs"]]


def explanation_color_chart(df):
    if df.empty:
        return None
    plot_df = df.copy()
    if "direction" not in plot_df.columns:
        plot_df["direction"] = plot_df["signed_impact_pct"].apply(lambda x: "Positive" if x >= 0 else "Negative")
    if "impact_pct_abs" not in plot_df.columns:
        plot_df["impact_pct_abs"] = plot_df["signed_impact_pct"].abs()
    if alt is None:
        return None
    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("signed_impact_pct:Q", title="Signed Impact (%)"),
            y=alt.Y("feature:N", sort=alt.SortField(field="signed_impact_pct", order="ascending"), title="Feature"),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Positive", "Negative"], range=["#2e7d32", "#c62828"]),
                legend=alt.Legend(title="Direction"),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("signed_impact_pct:Q", title="Signed Impact (%)", format=".2f"),
                alt.Tooltip("impact_pct_abs:Q", title="Abs Impact (%)", format=".2f"),
                alt.Tooltip("direction:N", title="Direction"),
            ],
        )
        .properties(height=260)
    )
    return chart


def load_calibration_report(model_name):
    path = Path(f"reports/calibration_{model_name}.csv")
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_backtest_summary():
    path = Path("reports/backtest_summary.csv")
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def calibration_summary(df, model_label):
    if df.empty or not {"predicted_mean", "observed_rate"}.issubset(df.columns):
        return None
    work = df.copy()
    work["predicted_mean"] = pd.to_numeric(work["predicted_mean"], errors="coerce")
    work["observed_rate"] = pd.to_numeric(work["observed_rate"], errors="coerce")
    work = work.dropna(subset=["predicted_mean", "observed_rate"])
    if work.empty:
        return None
    gap = (work["predicted_mean"] - work["observed_rate"]).abs()
    signed_gap = (work["predicted_mean"] - work["observed_rate"]).mean()
    mae = float(gap.mean())
    max_gap = float(gap.max())
    tendency = "over-confident" if signed_gap > 0.0 else ("under-confident" if signed_gap < 0.0 else "well-centered")
    return {
        "model": model_label,
        "avg_gap": mae,
        "max_gap": max_gap,
        "signed_gap": float(signed_gap),
        "tendency": tendency,
    }


st.title("NBA Prediction Dashboard")
st.caption("Team win probabilities, player projections, and API health checks.")

teams = load_teams()
if not teams:
    st.error("Could not load team metadata from config/nba_teams.json.")
    st.stop()

team_options = {label_for_team(team): team for team in teams}
labels = list(team_options.keys())


def load_upcoming_games(api_base):
    payload, err = safe_get_json(f"{api_base}/upcoming-games?limit=100")
    if err:
        return [], err
    games = payload.get("games", []) if isinstance(payload, dict) else []
    return games, None

with st.sidebar:
    st.header("Controls")
    api_base = st.text_input("API Base URL", value="http://127.0.0.1:8000")
    model = st.selectbox("Model", options=["baseline", "tree", "champion"], index=0)
    compare_models = st.toggle("Compare Baseline vs Tree", value=True)
    include_player_projection = st.toggle("Include Player Projection", value=True)
    upcoming_games, upcoming_err = load_upcoming_games(api_base)
    game_options = []
    for game in upcoming_games:
        home_id = int(game.get("home_team_id"))
        away_id = int(game.get("away_team_id"))
        home = next((t for t in teams if int(t["team_id"]) == home_id), None)
        away = next((t for t in teams if int(t["team_id"]) == away_id), None)
        home_label = home["abbreviation"] if home else str(home_id)
        away_label = away["abbreviation"] if away else str(away_id)
        game_options.append({
            "label": f"{game.get('game_date')} | {away_label} @ {home_label} | {game.get('game_id')}",
            "game_id": str(game.get("game_id")),
            "home_team_id": home_id,
            "away_team_id": away_id,
        })

    if game_options:
        selected_game = st.selectbox("Upcoming Game", options=game_options, format_func=lambda x: x["label"])
    else:
        selected_game = None
        if upcoming_err:
            st.error(f"Upcoming schedule unavailable: {upcoming_err}")
        else:
            st.error("No upcoming games available from API.")

    run_prediction = st.button("Run Team Prediction", type="primary", use_container_width=True)

col_left, col_right = st.columns([2, 1], gap="large")

with col_right:
    st.subheader("Model Status")
    features_data, features_err = safe_get_json(f"{api_base}/features")
    if features_err:
        st.error(f"Could not load /features: {features_err}")
    else:
        st.success("Connected to API")
        st.write(
            {
                "baseline_model_loaded": features_data.get("baseline_model_loaded"),
                "tree_model_loaded": features_data.get("tree_model_loaded"),
                "player_projection_model_loaded": features_data.get("player_projection_model_loaded"),
            }
        )
        st.caption(
            f"Baseline features: {len(features_data.get('baseline_features', []))} | "
            f"Tree features: {len(features_data.get('tree_features', []))}"
        )

    st.subheader("Pipeline Status")
    pipeline_data, pipeline_err = safe_get_json(f"{api_base}/pipeline-status")
    if pipeline_err:
        st.error(f"Could not load /pipeline-status: {pipeline_err}")
    else:
        if not pipeline_data.get("available", False):
            st.warning(pipeline_data.get("detail", "Pipeline status unavailable"))
        else:
            st.write({
                "run_id": pipeline_data.get("run_id"),
                "status": pipeline_data.get("status"),
                "started_at_utc": pipeline_data.get("started_at_utc"),
                "completed_at_utc": pipeline_data.get("completed_at_utc"),
            })

    st.subheader("Monitoring")
    monitoring_data, monitoring_err = safe_get_json(f"{api_base}/monitoring")
    if monitoring_err:
        st.error(f"Could not load /monitoring: {monitoring_err}")
    else:
        if not monitoring_data.get("available", False):
            st.warning(monitoring_data.get("detail", "Monitoring report unavailable"))
        else:
            freshness = monitoring_data.get("freshness", {})
            drift = monitoring_data.get("drift", {})
            st.caption(f"Report generated: {monitoring_data.get('generated_at_utc', 'N/A')}")
            st.write({
                "drift_status": drift.get("status"),
                "avg_psi_pct": format_number_pct((drift.get("avg_psi") or 0.0) * 100.0, 2),
                "max_psi_pct": format_number_pct((drift.get("max_psi") or 0.0) * 100.0, 2),
            })
            alerts = monitoring_data.get("alerts", {})
            if alerts:
                st.write({"alert_status": alerts.get("overall_status"), "alert_count": len(alerts.get("items", []))})
                if alerts.get("items"):
                    st.dataframe(pd.DataFrame(alerts.get("items", [])), hide_index=True, use_container_width=True)
            with st.expander("Freshness Details", expanded=False):
                st.json(freshness)
            feature_psi = drift.get("feature_psi", {})
            if feature_psi:
                top = sorted(feature_psi.items(), key=lambda kv: kv[1], reverse=True)[:5]
                st.caption("Top drift features")
                top_df = pd.DataFrame(top, columns=["feature", "psi"])
                top_df["psi_percent"] = top_df["psi"].map(lambda x: format_number_pct(float(x) * 100.0, 2))
                st.dataframe(top_df[["feature", "psi_percent"]], hide_index=True, use_container_width=True)

    st.subheader("Prediction Quality")
    quality_data, quality_err = safe_get_json(f"{api_base}/prediction-quality")
    if quality_err:
        st.error(f"Could not load /prediction-quality: {quality_err}")
    else:
        if not quality_data.get("available", False):
            st.warning(quality_data.get("detail", "Prediction quality report unavailable"))
        else:
            st.caption(f"Report generated: {quality_data.get('generated_at_utc', 'N/A')}")
            overall = quality_data.get("overall", {}) if isinstance(quality_data, dict) else {}
            recent_7d = quality_data.get("recent_7d", {}) if isinstance(quality_data, dict) else {}
            recent_30d = quality_data.get("recent_30d", {}) if isinstance(quality_data, dict) else {}

            q1, q2 = st.columns(2)
            q1.metric("Overall Accuracy", format_pct(overall.get("accuracy", None), 2))
            q2.metric("Overall Brier", f"{float(overall.get('brier_score')):.4f}" if overall.get("brier_score") is not None else "N/A")
            q3, q4 = st.columns(2)
            q3.metric("7D Accuracy", format_pct(recent_7d.get("accuracy", None), 2))
            q4.metric("30D Accuracy", format_pct(recent_30d.get("accuracy", None), 2))
            q5, q6 = st.columns(2)
            q5.metric("7D Log Loss", f"{float(recent_7d.get('log_loss')):.4f}" if recent_7d.get("log_loss") is not None else "N/A")
            q6.metric("30D Log Loss", f"{float(recent_30d.get('log_loss')):.4f}" if recent_30d.get("log_loss") is not None else "N/A")

            quality_rows = [
                {
                    "window": "overall",
                    "rows": overall.get("rows"),
                    "accuracy": format_pct(overall.get("accuracy", None), 2),
                    "log_loss": f"{float(overall.get('log_loss')):.4f}" if overall.get("log_loss") is not None else "N/A",
                    "brier_score": f"{float(overall.get('brier_score')):.4f}" if overall.get("brier_score") is not None else "N/A",
                },
                {
                    "window": "recent_7d",
                    "rows": recent_7d.get("rows"),
                    "accuracy": format_pct(recent_7d.get("accuracy", None), 2),
                    "log_loss": f"{float(recent_7d.get('log_loss')):.4f}" if recent_7d.get("log_loss") is not None else "N/A",
                    "brier_score": f"{float(recent_7d.get('brier_score')):.4f}" if recent_7d.get("brier_score") is not None else "N/A",
                },
                {
                    "window": "recent_30d",
                    "rows": recent_30d.get("rows"),
                    "accuracy": format_pct(recent_30d.get("accuracy", None), 2),
                    "log_loss": f"{float(recent_30d.get('log_loss')):.4f}" if recent_30d.get("log_loss") is not None else "N/A",
                    "brier_score": f"{float(recent_30d.get('brier_score')):.4f}" if recent_30d.get("brier_score") is not None else "N/A",
                },
            ]
            st.dataframe(pd.DataFrame(quality_rows), hide_index=True, use_container_width=True)

    st.subheader("Smoke Tests")
    smoke_col1, smoke_col2 = st.columns(2)
    if smoke_col1.button("Test /health", use_container_width=True):
        payload, err = safe_get_json(f"{api_base}/health")
        if err:
            st.error(err)
        else:
            st.json(payload)
    if smoke_col2.button("Test /predict/sample", use_container_width=True):
        payload, err = safe_get_json(f"{api_base}/predict/sample?model={model}")
        if err:
            st.error(err)
        else:
            st.json(payload)

with col_left:
    st.subheader("Game Prediction")
    if run_prediction:
        if not selected_game:
            st.error("Cannot run prediction without an upcoming game selection.")
        else:
            home_team_id = int(selected_game["home_team_id"])
            away_team_id = int(selected_game["away_team_id"])
            home_team = next((t for t in teams if int(t["team_id"]) == home_team_id), None)
            away_team = next((t for t in teams if int(t["team_id"]) == away_team_id), None)
            payload = {
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "game_id": selected_game["game_id"] if selected_game else None,
                "model": model,
                "include_player_projection": include_player_projection,
            }
            result, err = safe_post_json(f"{api_base}/predict/team", payload)
            if err:
                st.error(f"Prediction request failed: {err}")
            else:
                game = result.get("game", {})
                report = result.get("probability_report", {})
                st.markdown(
                    f"**{game.get('away_team_name', 'Away')} @ {game.get('home_team_name', 'Home')}**  "
                    f"(scheduled {game.get('game_date', 'N/A')}; features as of {game.get('feature_as_of_date', 'N/A')})"
                )

                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                metric_col1.metric("Home Win %", f"{100 * report.get('home_win_probability', 0):.1f}%")
                metric_col2.metric("Away Win %", f"{100 * report.get('away_win_probability', 0):.1f}%")
                metric_col3.metric("Home Fair ML", report.get("home_fair_moneyline", "N/A"))
                metric_col4.metric("Away Fair ML", report.get("away_fair_moneyline", "N/A"))
                home_prob = float(report.get("home_win_probability", 0.5))
                away_prob = float(report.get("away_win_probability", 0.5))
                winner_confidence = max(home_prob, away_prob)
                edge_strength = abs(home_prob - away_prob)
                confidence_band = "High" if winner_confidence >= 0.70 else ("Medium" if winner_confidence >= 0.60 else "Low")
                favored_team = game.get("home_team_name", "Home") if home_prob >= away_prob else game.get("away_team_name", "Away")
                st.caption(
                    f"Model confidence: {format_pct(winner_confidence, 1)} ({confidence_band}) on {favored_team}. "
                    f"Edge strength: {format_pct(edge_strength, 1)}."
                )
                if report.get("home_win_ci_low") is not None and report.get("home_win_ci_high") is not None:
                    st.caption(
                        f"Home win confidence interval: {100*float(report.get('home_win_ci_low')):.1f}% - "
                        f"{100*float(report.get('home_win_ci_high')):.1f}% "
                        f"(model spread std={float(report.get('model_spread_std', 0.0)):.3f})"
                    )

                if compare_models:
                    compare_payload_base = {
                        "home_team_id": home_team_id,
                        "away_team_id": away_team_id,
                        "game_id": selected_game["game_id"] if selected_game else None,
                        "model": "baseline",
                        "include_player_projection": False,
                    }
                    compare_payload_tree = {
                        "home_team_id": home_team_id,
                        "away_team_id": away_team_id,
                        "game_id": selected_game["game_id"] if selected_game else None,
                        "model": "tree",
                        "include_player_projection": False,
                    }
                    base_res, base_err = safe_post_json(f"{api_base}/predict/team", compare_payload_base)
                    tree_res, tree_err = safe_post_json(f"{api_base}/predict/team", compare_payload_tree)
                    st.subheader("Model Comparison")
                    if base_err or tree_err:
                        st.warning(
                            "Could not compute baseline/tree comparison for this request: "
                            f"{base_err or tree_err}"
                        )
                    else:
                        b = base_res.get("probability_report", {})
                        t = tree_res.get("probability_report", {})
                        b_home = float(b.get("home_win_probability", 0.0))
                        t_home = float(t.get("home_win_probability", 0.0))
                        delta_home = t_home - b_home
                        cmp1, cmp2, cmp3 = st.columns(3)
                        cmp1.metric("Baseline Home Win %", f"{100 * b_home:.1f}%")
                        cmp2.metric("Tree Home Win %", f"{100 * t_home:.1f}%")
                        cmp3.metric("Tree - Baseline", f"{100 * delta_home:+.1f} pts")

                with st.expander("Feature Snapshot", expanded=False):
                    st.json(result.get("feature_snapshot", {}))

                data_quality = result.get("data_quality", {})
                if data_quality:
                    alerts = (data_quality.get("alerts") or {}).get("items", [])
                    overall_alert = (data_quality.get("alerts") or {}).get("overall_status", "pass")
                    if overall_alert != "pass":
                        st.warning(f"Data quality status: {overall_alert}")
                    else:
                        st.success("Data quality status: pass")
                    if alerts:
                        st.dataframe(pd.DataFrame(alerts), hide_index=True, use_container_width=True)

                advisory = result.get("advisory", {})
                if advisory:
                    st.subheader("AI Game Brief")
                    st.info(advisory.get("narrative", ""))
                    drivers = advisory.get("confidence_drivers", [])
                    if drivers:
                        st.markdown("**Why confidence changed**")
                        for d in drivers:
                            st.caption(f"- {d.get('factor')}: {d.get('impact')} ({d.get('detail')})")
                    top_rec = advisory.get("top_recommendation")
                    if top_rec:
                        st.success(
                            f"Top recommendation: {top_rec.get('player_name')} "
                            f"{top_rec.get('stat')} {top_rec.get('projected_value')} "
                            f"(confidence {top_rec.get('confidence_pct')}%)"
                        )
                        st.caption(top_rec.get("reason", ""))
                    headlines = advisory.get("recent_headlines", [])
                    if headlines:
                        st.markdown("**Recent Headlines**")
                        for h in headlines:
                            title = h.get("title", "Headline")
                            url = h.get("url", "")
                            source = h.get("source", "News")
                            quote = h.get("quote", "")
                            pub = h.get("published_at", "")
                            if url:
                                st.markdown(f"- [{title}]({url}) ({source})")
                            else:
                                st.markdown(f"- {title} ({source})")
                            if quote:
                                st.caption(quote)
                            if pub:
                                st.caption(f"Published: {pub}")

                with st.expander("Prediction Explainability", expanded=False):
                    explain = result.get("explainability", {})
                    h_exp = explain.get("home", {})
                    a_exp = explain.get("away", {})
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Home**")
                        if not h_exp.get("available", False):
                            st.info(h_exp.get("detail", "No explanation available"))
                        else:
                            st.caption(f"Method: {h_exp.get('method')}")
                            st.caption(f"Prob (from explanation): {format_pct(h_exp.get('probability_from_explanation', 0.0), 2)}")
                            h_df = explanation_table(h_exp)
                            if not h_df.empty:
                                st.dataframe(h_df, hide_index=True, use_container_width=True)
                                h_chart = explanation_chart_df(h_exp)
                                if not h_chart.empty:
                                    chart_obj = explanation_color_chart(h_chart)
                                    if chart_obj is not None:
                                        st.altair_chart(chart_obj, use_container_width=True)
                                    else:
                                        st.bar_chart(h_chart.set_index("feature")[["signed_impact_pct"]], use_container_width=True)
                    with c2:
                        st.markdown("**Away**")
                        if not a_exp.get("available", False):
                            st.info(a_exp.get("detail", "No explanation available"))
                        else:
                            st.caption(f"Method: {a_exp.get('method')}")
                            st.caption(f"Prob (from explanation): {format_pct(a_exp.get('probability_from_explanation', 0.0), 2)}")
                            a_df = explanation_table(a_exp)
                            if not a_df.empty:
                                st.dataframe(a_df, hide_index=True, use_container_width=True)
                                a_chart = explanation_chart_df(a_exp)
                                if not a_chart.empty:
                                    chart_obj = explanation_color_chart(a_chart)
                                    if chart_obj is not None:
                                        st.altair_chart(chart_obj, use_container_width=True)
                                    else:
                                        st.bar_chart(a_chart.set_index("feature")[["signed_impact_pct"]], use_container_width=True)

                if include_player_projection:
                    st.subheader("Projected Player Performance")
                    player_block = result.get("projected_player_performance", {})
                    home_proj = player_block.get("home", {})
                    away_proj = player_block.get("away", {})

                    home_tab_label = f"Home ({home_team['abbreviation']})" if home_team else "Home"
                    away_tab_label = f"Away ({away_team['abbreviation']})" if away_team else "Away"
                    tab_home, tab_away = st.tabs([home_tab_label, away_tab_label])
                    with tab_home:
                        st.caption(home_proj.get("projection_method", ""))
                        if home_proj.get("context_adjustment"):
                            st.caption(f"Context: {home_proj.get('context_adjustment')}")
                        if home_proj.get("coverage_note"):
                            st.info(home_proj.get("coverage_note"))
                        home_df = projection_table(home_proj)
                        if home_df.empty:
                            st.info("No player projections available for home team.")
                        else:
                            st.dataframe(home_df, use_container_width=True, hide_index=True)
                            if alt is not None and {"player_name", "projected_points", "projected_points_ci_low", "projected_points_ci_high"}.issubset(home_df.columns):
                                chart_df = home_df.head(8).copy()
                                err = (
                                    alt.Chart(chart_df)
                                    .mark_errorbar()
                                    .encode(
                                        y=alt.Y("player_name:N", sort="-x"),
                                        x=alt.X("projected_points_ci_low:Q", title="Projected PTS"),
                                        x2="projected_points_ci_high:Q",
                                    )
                                )
                                pts = (
                                    alt.Chart(chart_df)
                                    .mark_point(color="#2e7d32", size=55)
                                    .encode(y=alt.Y("player_name:N", sort="-x"), x="projected_points:Q")
                                )
                                st.altair_chart((err + pts).properties(height=260), use_container_width=True)
                            st.download_button(
                                "Download Home Projections CSV",
                                data=home_df.to_csv(index=False),
                                file_name="home_player_projections.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                    with tab_away:
                        st.caption(away_proj.get("projection_method", ""))
                        if away_proj.get("context_adjustment"):
                            st.caption(f"Context: {away_proj.get('context_adjustment')}")
                        if away_proj.get("coverage_note"):
                            st.info(away_proj.get("coverage_note"))
                        away_df = projection_table(away_proj)
                        if away_df.empty:
                            st.info("No player projections available for away team.")
                        else:
                            st.dataframe(away_df, use_container_width=True, hide_index=True)
                            if alt is not None and {"player_name", "projected_points", "projected_points_ci_low", "projected_points_ci_high"}.issubset(away_df.columns):
                                chart_df = away_df.head(8).copy()
                                err = (
                                    alt.Chart(chart_df)
                                    .mark_errorbar()
                                    .encode(
                                        y=alt.Y("player_name:N", sort="-x"),
                                        x=alt.X("projected_points_ci_low:Q", title="Projected PTS"),
                                        x2="projected_points_ci_high:Q",
                                    )
                                )
                                pts = (
                                    alt.Chart(chart_df)
                                    .mark_point(color="#2e7d32", size=55)
                                    .encode(y=alt.Y("player_name:N", sort="-x"), x="projected_points:Q")
                                )
                                st.altair_chart((err + pts).properties(height=260), use_container_width=True)
                            st.download_button(
                                "Download Away Projections CSV",
                                data=away_df.to_csv(index=False),
                                file_name="away_player_projections.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )

                with st.expander("Raw API Response", expanded=False):
                    st.json(result)
    else:
        st.info("Choose teams in the sidebar and click Run Team Prediction.")

st.subheader("Calibration & Backtest")
cal_col1, cal_col2 = st.columns(2, gap="large")
with cal_col1:
    st.markdown("**Calibration Curves**")
    st.caption(
        "How to read: if the model is perfectly calibrated, the observed line should closely match "
        "the predicted line. Bigger gaps mean probability estimates are less trustworthy."
    )
    cal_base = load_calibration_report("baseline")
    cal_tree = load_calibration_report("tree")
    if cal_base.empty and cal_tree.empty:
        st.info("Calibration reports not found. Run training scripts to generate reports/calibration_*.csv.")
    else:
        summaries = []
        base_summary = calibration_summary(cal_base, "Baseline")
        tree_summary = calibration_summary(cal_tree, "Tree")
        if base_summary:
            summaries.append(base_summary)
        if tree_summary:
            summaries.append(tree_summary)
        if summaries:
            summary_df = pd.DataFrame(summaries)
            summary_df["avg_gap_pct"] = summary_df["avg_gap"].map(lambda x: format_number_pct(float(x) * 100.0, 2))
            summary_df["max_gap_pct"] = summary_df["max_gap"].map(lambda x: format_number_pct(float(x) * 100.0, 2))
            summary_df["bias_pct"] = summary_df["signed_gap"].map(lambda x: format_number_pct(float(x) * 100.0, 2))
            st.dataframe(
                summary_df[["model", "avg_gap_pct", "max_gap_pct", "bias_pct", "tendency"]],
                hide_index=True,
                use_container_width=True,
            )
        if not cal_base.empty and {"predicted_mean", "observed_rate"}.issubset(cal_base.columns):
            chart_base = cal_base[["predicted_mean", "observed_rate"]].copy()
            chart_base.columns = ["Predicted (Baseline)", "Observed (Baseline)"]
            st.line_chart(chart_base, use_container_width=True)
        if not cal_tree.empty and {"predicted_mean", "observed_rate"}.issubset(cal_tree.columns):
            chart_tree = cal_tree[["predicted_mean", "observed_rate"]].copy()
            chart_tree.columns = ["Predicted (Tree)", "Observed (Tree)"]
            st.line_chart(chart_tree, use_container_width=True)

with cal_col2:
    st.markdown("**Backtest Trend**")
    backtest = load_backtest_summary()
    if backtest.empty:
        st.info("Backtest summary not found. Run scripts/backtest.py.")
    else:
        month = backtest[backtest["period"].astype(str) != "overall"].copy() if "period" in backtest.columns else pd.DataFrame()
        if month.empty or "roi" not in month.columns:
            st.info("Monthly backtest rows unavailable yet.")
        else:
            month["roi_percent"] = pd.to_numeric(month["roi"], errors="coerce").fillna(0.0) * 100.0
            month = month.sort_values("period")
            st.line_chart(month.set_index("period")[["roi_percent"]], use_container_width=True)
