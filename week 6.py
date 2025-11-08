import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

# ================== Load + Train Model ===================
path = os.path.expanduser("~/Downloads/Fifa_world_cup_matches_cleaned.csv")
df = pd.read_csv(path)

df = df.dropna(subset=['team1','team2'])
df['number of goals team1'] = df['number of goals team1'].fillna(0)
df['number of goals team2'] = df['number of goals team2'].fillna(0)

df['Goal_Difference'] = df['number of goals team1'] - df['number of goals team2']
df['Total_Goals'] = df['number of goals team1'] + df['number of goals team2']
df['is_finalist'] = np.where(df['number of goals team1'] > df['number of goals team2'], 1, 0)

np.random.seed(42)
df['team1_rank'] = np.random.randint(1, 50, size=len(df))
df['team2_rank'] = np.random.randint(1, 50, size=len(df))
df['avg_age'] = np.random.uniform(24, 30, size=len(df))
df['experience'] = np.random.uniform(3, 10, size=len(df))

df['possession team1'] = df['possession team1'].astype(str).replace('%','', regex=True).astype(float)
df['possession team2'] = df['possession team2'].astype(str).replace('%','', regex=True).astype(float)

encoder = LabelEncoder()
df['team1_encoded'] = encoder.fit_transform(df['team1'])
df['team2_encoded'] = encoder.fit_transform(df['team2'])

features = [
    'team1_encoded', 'team2_encoded',
    'number of goals team1', 'number of goals team2',
    'Goal_Difference', 'Total_Goals',
    'team1_rank', 'team2_rank',
    'avg_age', 'experience',
    'possession team1', 'possession team2'
]

X = df[features]
y = df['is_finalist']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_scaled, y)

# =============== FIXED FUTURE DATA (Same as Week-5) ===============

future_teams = pd.DataFrame({
    'team1': ['ARGENTINA', 'FRANCE', 'BRAZIL', 'ENGLAND', 'PORTUGAL', 'SPAIN', 'GERMANY', 'NETHERLANDS'],
    'team2': ['BRAZIL', 'ENGLAND', 'FRANCE', 'ARGENTINA', 'SPAIN', 'PORTUGAL', 'CROATIA', 'BELGIUM'],
    'number of goals team1': [3, 2, 2, 1, 3, 2, 2, 1],
    'number of goals team2': [1, 1, 2, 1, 1, 1, 1, 1],
    'team1_rank': [1, 2, 3, 4, 5, 6, 7, 8],
    'team2_rank': [3, 4, 2, 1, 6, 5, 8, 9],
    'avg_age': [27, 26, 28, 25, 27, 26, 28, 27],
    'experience': [8, 7, 9, 6, 7, 8, 7, 7],
    'possession team1': [55, 54, 52, 50, 53, 51, 54, 50],
    'possession team2': [45, 46, 48, 50, 47, 49, 46, 50]
})

future_teams['Goal_Difference'] = future_teams['number of goals team1'] - future_teams['number of goals team2']
future_teams['Total_Goals'] = future_teams['number of goals team1'] + future_teams['number of goals team2']

future_teams['team1_encoded'] = encoder.transform(future_teams['team1'])
future_teams['team2_encoded'] = encoder.transform(future_teams['team2'])

future_scaled = scaler.transform(future_teams[features])
future_teams['Finalist_Prob'] = model.predict_proba(future_scaled)[:, 1]

top_finalists = future_teams.sort_values(by='Finalist_Prob', ascending=False).head(2)
predicted = top_finalists['team1'].tolist()

# ================= UI APP =================
root = tk.Tk()
root.title("FIFA 2026 Final Prediction")
root.attributes('-fullscreen', True)
root.config(bg="#001a33")

title = tk.Label(root, text=" FIFA 2026 AI FINALIST PREDICTOR ",
                 font=("Arial", 34, "bold"), fg="gold", bg="#001a33")
title.pack(pady=50)

result_label = tk.Label(root, text="Press the button to see finalists ",
                        font=("Arial", 24), fg="white", bg="#001a33")
result_label.pack(pady=30)

def show_final():
    result_label.config(
        text=f"1️⃣ {predicted[0]}\n2️⃣ {predicted[1]}\n\n Predicted Finalists!",
        font=("Arial", 36, "bold"),
        fg="cyan"
    )

btn = tk.Button(root, text="🔮 Show Final Prediction", font=("Arial", 26, "bold"),
                bg="gold", fg="black", padx=40, pady=20, command=show_final)
btn.pack(pady=50)

footer = tk.Label(root, text="Developed by Santhosh Kumar | Chanakya University",
                  font=("Arial", 16), fg="white", bg="#001a33")
footer.pack(side="bottom", pady=30)

root.bind("<Escape>", lambda e: root.destroy())
root.mainloop()
