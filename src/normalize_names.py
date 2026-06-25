import unicodedata

import pandas as pd
from pathlib import Path


#Get project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DIR = BASE_DIR / "data"

#File containing old country names will soon be converted into modern names
FORMER_NAMES_FILE = DIR / "former_names.csv"

def get_names_map():
    """
    Reads the former names csv file and creates a map of
    old country names to new country names

    Returns: a dictionary where the key is the former name and the value is current name
    """
    names = pd.read_csv(FORMER_NAMES_FILE)

    name_list = {}

    for i, row in names.iterrows():
        current = row["current"].strip()
        former = row["former"].strip()

        name_list[former] = current

    return name_list

def normalize_team_names(df, columns, name_list):
    """
    Cleans the team names and replaces the old country names
    in the selected columns

    Args:
        df(DataFrame): this is the data frame which is to be cleaned
        columns(list): this is the list of columns that have to be normalized
        name_list(dict): this is the dictionary that contains the mapped of old names to new names

    Returns:
        df: the cleaned data frame
    """
    df = df.copy()
    cleaned_teams = {}

    for old in name_list:
        new = name_list[old]
        clean_old = clean_text(old)
        clean_new = clean_text(new)

        cleaned_teams[clean_old] = clean_new

    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
            df[col] = df[col].replace(cleaned_teams)

    return df

def clean_goalscorers():
    """
    Cleans the goalscorers dataset and saves it as a csv
    """
    name_list2 = get_names_map()

    df = pd.read_csv(DIR / "goalscorers.csv")

    df = normalize_team_names(df, ["home_team", "away_team", "team"], name_list2)

    df.to_csv(DIR / "goalscorers_clean.csv", index=False)
    print("saved goalscorers")

def clean_shootouts():
    """
        Cleans the shootouts dataset and saves it as a csv
        """
    name_list3 = get_names_map()
    df = pd.read_csv(DIR / "shootouts.csv")
    df = normalize_team_names(df, ["home_team", "away_team", "winner", "first_shooter"], name_list3)
    df.to_csv(DIR / "shootouts_clean.csv", index=False)
    print("saved shootouts")


def clean_text(text):
    """
    Removes any whitespace and accent marks like the ï
    in Zaïre.

    Args:
        text(str): the text to be cleaned

    Returns:
        cleaned(str): the cleaned text
    """
    if pd.isna(text):
        return text

    text = str(text).strip()
    text = unicodedata.normalize("NFKD", text)

    cleaned = ""
    for c in text:
        if not unicodedata.combining(c):
            cleaned += c

    return cleaned

def clean_results():
    """
    Cleans historical match results.
    """

    name_lists = get_names_map()

    df = pd.read_csv(DIR / "results.csv")

    df = normalize_team_names(df, ["home_team", "away_team", "country"], name_lists)

    df.to_csv(DIR / "results_clean.csv", index=False)
    print("saved results")

if __name__ == "__main__":
    clean_goalscorers()
    clean_shootouts()
    clean_results()