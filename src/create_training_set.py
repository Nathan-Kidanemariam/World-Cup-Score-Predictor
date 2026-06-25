import pandas as pd
from pathlib import Path


#Base project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

#Open the historical world cup matches and stats
matches = pd.read_csv(DATA_DIR / "worldcup_history_clean.csv")
teams = pd.read_csv(DATA_DIR / "team_worldcup_stats.csv")

def get_team_stats(team):
    """
    Find a team's statistics/feature

    Args:
        team(str): Team name

    Returns:
        Series if the team stats are found
        or None if the team doesn't exist
    """
    found = teams[teams["team"] == team]

    if len(found) == 0:
        return None

    return found.iloc[0]


#Stores rows for the machine learning training data
rows = []

#Build one training row for each match
for i, row in matches.iterrows():
    home = row["home_team"]
    away = row["away_team"]

    home_stats = get_team_stats(home)
    away_stats = get_team_stats(away)

    #Skip any matches if team stats are missing
    if home_stats is None or away_stats is None:
        continue

    rows.append({
        "home_team": home,
        "away_team": away,

        #Home team features
        "home_attack": home_stats["attack"],
        "home_defense": home_stats["defense"],
        "home_momentum": home_stats["momentum"],
        "home_strength": home_stats["strength"],

        #Away team features
        "away_attack": away_stats["attack"],
        "away_defense": away_stats["defense"],
        "away_momentum": away_stats["momentum"],
        "away_strength": away_stats["strength"],

        #Difference features
        "strength_diff": home_stats["strength"] - away_stats["strength"],
        "attack_diff": home_stats["attack"] - away_stats["attack"],
        "defense_diff": home_stats["defense"] - away_stats["defense"],

        #Target values for training
        "home_score": row["home_score"],
        "away_score": row["away_score"]
    })

#Save the training dataset
training_df = pd.DataFrame(rows)

training_df.to_csv(DATA_DIR / "training_data.csv", index=False)

print("saved training df")
print(training_df.head())
print("rows: ", len(training_df))