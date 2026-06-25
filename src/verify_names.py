from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

goal = pd.read_csv(DATA_DIR / "goalscorers_clean.csv")
shooting = pd.read_csv(DATA_DIR / "shootouts_clean.csv")

print("Goals: ", goal["team"].nunique())

if "winner" in shooting.columns:
    print("Winners: ", shooting["winner"].nunique())

old_names = ["Swaziland", "Upper Volta", "Dahomey", "Zaïre", "Soviet Union"]

for old in old_names:
    if old in goal.to_string() or old in shooting.to_string():
        print(old, "Found")
    else:
        print(old, "Removed")