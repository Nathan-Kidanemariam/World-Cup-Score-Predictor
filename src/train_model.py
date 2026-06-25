import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#Base project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

#Create models folder if it doesn't already exist
MODEL_DIR.mkdir(exist_ok=True)

#Load in the training data
df = pd.read_csv(DATA_DIR / "training_data.csv")

#Use these features to predict the goals
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

#X will be the features we will input
X = df[features]

#These will be the target values
y_home = df["home_score"]
y_away = df["away_score"]

#Split the data for home goal model
X_train, X_test, y_home_train, y_home_test = train_test_split(X, y_home, test_size=0.2, random_state=42)

#Split the data for away goal model
Y, Y2, y_away_train, y_away_test = train_test_split(X, y_away, test_size=0.2, random_state=42)

#Create model
home_model = RandomForestRegressor(n_estimators=300, random_state=42)
away_model = RandomForestRegressor(n_estimators=300, random_state=42)

#Train the models
home_model.fit(X_train, y_home_train)
away_model.fit(X_train, y_away_train)

#Test the model
home_predictions = home_model.predict(X_test)
away_prediction = away_model.predict(X_test)

home_mae = mean_absolute_error(y_home_test, home_predictions)
away_mae = mean_absolute_error(y_away_test, away_prediction)

print("Home goals MAE:", round(home_mae, 2))
print("Away goals MAE:", round(away_mae, 2))

#Save the trained models
joblib.dump(home_model, MODEL_DIR / "home_goal_model.pkl")
joblib.dump(away_model, MODEL_DIR / "away_goal_model.pkl")

print("saved the models")
