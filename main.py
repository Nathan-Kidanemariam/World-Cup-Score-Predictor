import pandas as pd
from pathlib import Path

from src.predict_future_final import predict_single_match


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def print_prediction(home, away, result):
    """
    Prints the results of the prediction
    Args:
        home(str): home team name
        away(str): Away team name
        result(dict): Prediction result
    """
    print()
    print(f"{home} vs {away}")
    print(f"Expected goals: {result['expected_home_goals']} - {result['expected_away_goals']}")
    print(f"Most likely score: {result['most_likely_score']}")
    print()
    print(f"{home} win: {result['home_win_%']}%")
    print(f"Draw: {result['draw_%']}%")
    print(f"{away} win: {result['away_win_%']}%")
    print()
    print("Top scores:")
    print(f"1. {result['top_score_1']} — {result['top_score_1_%']}%")
    print(f"2. {result['top_score_2']} — {result['top_score_2_%']}%")
    print(f"3. {result['top_score_3']} — {result['top_score_3_%']}%")
    print("-" * 40)


def predict_one_match():
    """
    Given two teams by the user it predicts one match prediction
    """
    home = input("Home team: ")
    away = input("Away team: ")

    result = predict_single_match(home, away)

    if result is None:
        print("One of those teams was not found.")
    else:
        print_prediction(home, away, result)


def predict_all_matches():
    """
    Predicts and prints every future match from the future_clean csv
    """
    future = pd.read_csv(DATA_DIR / "worldcup_future_clean.csv")

    for i, row in future.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        result = predict_single_match(home, away)

        if result is not None:
            print_prediction(home, away, result)


print("World Cup Predictor")
print("1. Predict one match")
print("2. Predict all future matches")

choice = input("Choose an option: ")

if choice == "1":
    predict_one_match()

elif choice == "2":
    predict_all_matches()

else:
    print("Invalid option.")