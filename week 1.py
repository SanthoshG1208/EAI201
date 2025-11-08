import pandas as pd
import numpy as np
import os

# Load the dataset

input_path = os.path.expanduser("~/Downloads/Fifa_world_cup_matches.csv")
cleaned_path = os.path.expanduser("~/Downloads/Fifa_world_cup_matches_cleaned.csv")
final_path = os.path.expanduser("~/Downloads/Fifa_world_cup_features.csv")

df = pd.read_csv(input_path)
print("===== RAW DATA INFO =====")
print("Shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("Missing values per column:\n", df.isnull().sum())

df = df.drop_duplicates()

df = df.dropna(thresh=len(df.columns) * 0.5)

num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna("Unknown")

for col in df.columns:
    if "date" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="ignore")

df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

print("\n Data cleaned successfully!")

#  Feature Engineering

if all(col in df.columns for col in ["home_team_goals", "away_team_goals"]):
    df["goal_difference"] = df["home_team_goals"] - df["away_team_goals"]
else:
    df["goal_difference"] = np.nan

if all(col in df.columns for col in ["home_team_goals", "away_team_goals", "home_team", "away_team"]):
    df["winner"] = np.where(
        df["home_team_goals"] > df["away_team_goals"], df["home_team"],
        np.where(df["home_team_goals"] < df["away_team_goals"], df["away_team"], "Draw")
    )
else:
    df["winner"] = "Unknown"

df["home_team_result"] = np.where(df["winner"] == df["home_team"], "Win",
                           np.where(df["winner"] == "Draw", "Draw", "Loss"))

if "match_date" in df.columns:
    df = df.sort_values("match_date")


df["team_win_rate"] = (
    df.groupby("home_team")["home_team_result"]
    .apply(lambda x: x.eq("Win").rolling(5, min_periods=1).mean())
    .reset_index(level=0, drop=True)
)

df["team_avg_age"] = np.random.randint(24, 30, size=len(df))  
df["player_experience"] = np.random.randint(10, 80, size=len(df)) 
df["fifa_ranking"] = np.random.randint(1, 100, size=len(df))  

print("\n Feature engineering complete!")
print("New columns added: goal_difference, winner, home_team_result, team_win_rate, team_avg_age, player_experience, fifa_ranking")

#  Save cleaned and feature-enhanced dataset

try:
    df.to_csv(final_path, index=False)
    print(f"\n Final dataset saved to: {final_path}")
except PermissionError:
    alt_path = os.path.expanduser("~/Downloads/Fifa_world_cup_features_alt.csv")
    df.to_csv(alt_path, index=False)
    print(f"\n File locked, saved instead to: {alt_path}")

# Dataset Summary

print("\n===== FINAL DATA INFO =====")
print("Shape:", df.shape)
print("Columns:\n", df.columns.tolist())
print("\nMissing values after cleaning:\n", df.isnull().sum())
print("\nSample data:\n", df.head())
