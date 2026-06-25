import pandas as pd
from pathlib import Path
from math import exp, factorial
import joblib

#Base project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

#Load team stats, future matches, and the trained models
teams = pd.read_csv(DATA_DIR / "team_worldcup_stats.csv")
future = pd.read_csv(DATA_DIR / "worldcup_future_clean.csv")

home_model = joblib.load(MODEL_DIR / "home_goal_model.pkl")
away_model = joblib.load(MODEL_DIR / "away_goal_model.pkl")

#Features expected by the trained models
features = [
    "home_attack",
    "home_defense",
    "home_momentum",
    "home_strength",

    "away_attack",
    "away_defense",
    "away_momentum",
    "away_strength",

    "strength_diff",
    "attack_diff",
    "defense_diff"
]

def poisson_probability(goals, expected_goals):
    """
    Calculates the probability of scoring a certain number of goals

    Args:
        goals(int): Number of goals
        expected_goals(float): Model's expected goals

    Returns:
        float: probability of scoring that many goals
    """
    return (expected_goals ** goals) * exp(-expected_goals) / factorial(goals)

def get_team_stats(team):
    """
    Finds a team's stats row
    Args:
        team(str): The team name

    Returns:
        Series if the team stats were found
        or None if the team doesn't exist

    """
    found = teams[teams["team"] == team]
    if len(found) == 0:
        return None

    return found.iloc[0]

def get_probability(score_info):
    """
    Gets probability from a score dictionary.

    Args:
        score_info (dict): Dictionary with score and probability

    Returns:
        float: Score probability
    """

    return score_info["probability"]

def predict_score_probs(expected_home, expected_away):
    """
    Predicts the score using a Poisson model

    Args:
        expected_home(float): expected goals for the home team
        expected_away(float): expected goals for away team

    Returns:
        score_probs(list): score sorted by probability
        home_win(float): Home win probability
        draw(float): draw probability
        away_win(float): away win probability
    """
    score_probs = []

    home_win = 0
    draw = 0
    away_win = 0

    for home_goals in range(7):
        for away_goals in range(7):
            prob = poisson_probability(home_goals, expected_home) * poisson_probability(away_goals, expected_away)
            score_probs.append({
                "score": str(home_goals) + "-" + str(away_goals),
                "probability": prob
            })

            if home_goals > away_goals:
                home_win += prob
            elif away_goals > home_goals:
                away_win += prob
            else:
                draw += prob

    score_probs = sorted(score_probs, key=get_probability, reverse=True)

    return score_probs, home_win, draw, away_win

predictions = []

for i, row in future.iterrows():
    home = row["home_team"]
    away = row["away_team"]

    home_stats = get_team_stats(home)
    away_stats = get_team_stats(away)

    if home_stats is None or away_stats is None:
        continue

    match_features = pd.DataFrame([{
        "home_attack": home_stats["attack"],
        "home_defense": home_stats["defense"],
        "home_momentum": home_stats["momentum"],
        "home_strength": home_stats["strength"],

        "away_attack": away_stats["attack"],
        "away_defense": away_stats["defense"],
        "away_momentum": away_stats["momentum"],
        "away_strength": away_stats["strength"],

        "strength_diff": home_stats["strength"] - away_stats["strength"],
        "attack_diff": home_stats["attack"] - away_stats["attack"],
        "defense_diff": home_stats["defense"] - away_stats["defense"]
    }])

    match_features = match_features[features]

    expected_home = home_model.predict(match_features)[0]
    expected_away = away_model.predict(match_features)[0]

    if expected_home < 0.2:
        expected_home = 0.2

    if expected_away < 0.2:
        expected_away = 0.2

    score_probs, home_win, draw, away_win = predict_score_probs(expected_home, expected_away)

    predictions.append({
        "date": row["date"],
        "home_team": home,
        "away_team": away,

        "expected_home_goal": round(expected_home, 2),
        "expected_away_goals": round(expected_away,2),

        "most_likely_score": score_probs[0]["score"],

        "home_win_%": round(home_win * 100, 1),
        "draw_%": round(draw * 100, 1),
        "away_win_%": round(away_win * 100, 1),

        "top_score_1": score_probs[0]["score"],
            "top_score_1_%": round(score_probs[0]["probability"] * 100, 1),

        "top_score_2": score_probs[1]["score"],
        "top_score_2_%": round(score_probs[1]["probability"] * 100, 1),

        "top_score_3": score_probs[2]["score"],
        "top_score_3_%": round(score_probs[2]["probability"] * 100, 1)
    })

def predict_single_match(home, away):
    """
    Predicts one match using the trained models and
    the Poisson score probability

    Args:
        home(str): Home team name

        away(str): Away team name

    Returns:
        dictionary if both teams exist
        None if either team is missing
    """
    home_stats = get_team_stats(home)
    away_stats = get_team_stats(away)

    if home_stats is None or away_stats is None:
        return None

    match_features = pd.DataFrame([{
        "home_attack": home_stats["attack"],
        "home_defense": home_stats["defense"],
        "home_momentum": home_stats["momentum"],
        "home_strength": home_stats["strength"],

        "away_attack": away_stats["attack"],
        "away_defense": away_stats["defense"],
        "away_momentum": away_stats["momentum"],
        "away_strength": away_stats["strength"],

        "strength_diff": home_stats["strength"] - away_stats["strength"],
        "attack_diff": home_stats["attack"] - away_stats["attack"],
        "defense_diff": home_stats["defense"] - away_stats["defense"]
    }])

    match_features = match_features[features]

    expected_home = home_model.predict(match_features)[0]
    expected_away = away_model.predict(match_features)[0]

    if expected_home < 0.2:
        expected_home = 0.2

    if expected_away < 0.2:
        expected_away = 0.2

    score_probs, home_win, draw, away_win = predict_score_probs(
        expected_home,
        expected_away
    )

    return {
        "expected_home_goals": round(expected_home, 2),
        "expected_away_goals": round(expected_away, 2),
        "most_likely_score": score_probs[0]["score"],
        "home_win_%": round(home_win * 100, 1),
        "draw_%": round(draw * 100, 1),
        "away_win_%": round(away_win * 100, 1),
        "top_score_1": score_probs[0]["score"],
        "top_score_1_%": round(score_probs[0]["probability"] * 100, 1),
        "top_score_2": score_probs[1]["score"],
        "top_score_2_%": round(score_probs[1]["probability"] * 100, 1),
        "top_score_3": score_probs[2]["score"],
        "top_score_3_%": round(score_probs[2]["probability"] * 100, 1)
    }

