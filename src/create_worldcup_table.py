import pandas as pd
from pathlib import Path


#Base project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


#Load the cleaned up history results
df = pd.read_csv(DATA_DIR / "results_clean.csv")

#Filter for only the fifa world cup matches
df = df[df["tournament"] == "FIFA World Cup"]


#Drop any columns that won't be used for prediction
df = df.drop(columns=["city", "country", "neutral"], errors="ignore")

#Store whether a match has been played or not
played = []

for i, row in df.iterrows():
    if pd.notna(row["home_score"]) and pd.notna(row["away_score"]):
        played.append(True)
    else:
        played.append(False)

#Save the completed matches
history = df[played].copy()

#Create a goal difference feature
history["goal_diff"] = history["home_score"] - history["away_score"]

def get_winner(row):
    """
    Determines the winner of a match
    Args:
         row(series): Match row that contains the scores and teams

    Returns:
        str: the winning team or a draw
    """
    if row["home_score"] > row["away_score"]:
        return row["home_team"]
    elif row["away_score"] > row["home_score"]:
        return row["away_team"]
    else:
        return "Draw"

#Add a winner column
history["winner"] = history.apply(get_winner, axis=1)

#Get any future matches that haven't been played yet
future = []

for i, row in df.iterrows():

    if played[i] == False:

        future.append(row)

future = pd.DataFrame(future)

#Save outputs
history.to_csv(DATA_DIR / "worldcup_history_clean.csv", index=False)
future.to_csv(DATA_DIR / "worldcup_future_clean.csv", index=False)
print("saved worldcup matches")
print("played: ", len(history))
print("future:", len(future))