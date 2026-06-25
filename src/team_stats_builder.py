import pandas as pd
from pathlib import Path

#Base project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

#Open the historical world cup matches
df = pd.read_csv(DATA_DIR / "worldcup_history_clean.csv")

#Store the team stats
team_stats = {}

def add_team(team):
    """
    Creates a template stats record for a team
    as long as it doesn't already exist

    Args:
         team(str): Team name
    """
    if team not in team_stats:
        team_stats[team] = {
            "games": 0,
            "goals_for": 0,
            "goals_against": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,

            #Tracks the teams recent form
            "recent_gf": [],
            "recent_ga": [],
            "recent_results": []
        }

#Build stats from each match
for i, row in df.iterrows():
    home = row["home_team"]
    away = row["away_team"]

    home_score = row["home_score"]
    away_score = row["away_score"]

    add_team(home)
    add_team(away)

    team_stats[home]["games"] += 1
    team_stats[away]["games"] += 1

    team_stats[home]["goals_for"] += home_score
    team_stats[home]["goals_against"] += away_score

    team_stats[away]["goals_for"] += away_score
    team_stats[away]["goals_against"] += home_score

    team_stats[home]["recent_gf"].append(home_score)
    team_stats[home]["recent_ga"].append(away_score)
    team_stats[away]["recent_gf"].append(away_score)
    team_stats[away]["recent_ga"].append(home_score)

    if home_score > away_score:
        team_stats[home]["wins"] += 1
        team_stats[away]["losses"] += 1

        team_stats[home]["recent_results"].append(3)
        team_stats[away]["recent_results"].append(0)
    elif away_score > home_score:
        team_stats[away]["wins"] += 1
        team_stats[home]["losses"] += 1

        team_stats[away]["recent_results"].append(3)
        team_stats[home]["recent_results"].append(0)
    else:
        team_stats[home]["draws"] += 1
        team_stats[away]["draws"] += 1

        team_stats[home]["recent_results"].append(1)
        team_stats[away]["recent_results"].append(1)

#Convert raw stats into prediction features
rows = []
for team in team_stats:
    stats = team_stats[team]
    games = stats["games"]

    #Long term average
    avg_goals_for = stats["goals_for"]/games
    avg_goals_against = stats["goals_against"]/games

    #Form over the last 5 matches
    recent_attack = sum((stats["recent_gf"][-5:]))/min(5, games)
    recent_defense = sum((stats["recent_ga"][-5:]))/min(5, games)
    momentum = sum(stats["recent_results"][-5:]) / (min(5, games) * 3)
    attack = (avg_goals_for * 0.5) + (recent_attack * 0.5)
    defense = ((3 - avg_goals_against) * 0.5) + ((3 - recent_defense) * 0.5)
    strength = (attack * 0.45 + defense * 0.30 + momentum * 0.25)

    #Save the team features
    rows.append({
        "team": team,
        "games": games,
        "goals_for": stats["goals_for"],
        "goals_against": stats["goals_against"],
        "avg_goals_for": round(avg_goals_for, 2),
        "avg_goals_against": round(avg_goals_against, 2),
        "wins": stats["wins"],
        "draws": stats["draws"],
        "losses": stats["losses"],
        "win_rate": round(stats["wins"]/games, 2),
        "recent_attack": round(recent_attack, 2),
        "recent_defense": round(recent_defense, 2),
        "momentum": round(momentum, 2),
        "attack": round(attack, 2),
        "defense": round(defense, 2),
        "strength": round(strength, 2)
    })

stats_df = pd.DataFrame(rows)
stats_df = stats_df.sort_values(by="strength", ascending=False)

#Save the final team stats
stats_df.to_csv(DATA_DIR / "team_worldcup_stats.csv", index=False)

print("saved team wc stats")
print(stats_df.head(10))