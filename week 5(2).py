import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

input_path = os.path.expanduser("~/Downloads/Fifa_world_cup_matches_cleaned.csv")
df = pd.read_csv(input_path)

print("===== RAW DATA HEAD =====")
print(df.head())

df = df.dropna(subset=['team1', 'team2'])
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

if 'possession team1' in df.columns and 'possession team2' in df.columns:
    df['possession team1'] = df['possession team1'].astype(str).str.replace('%', '').astype(float)
    df['possession team2'] = df['possession team2'].astype(str).str.replace('%', '').astype(float)
else:
    df['possession team1'] = np.random.uniform(40, 60, len(df))
    df['possession team2'] = np.random.uniform(40, 60, len(df))

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
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n===== {name} Evaluation =====")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("Precision:", round(precision_score(y_test, y_pred), 3))
    print("Recall:", round(recall_score(y_test, y_pred), 3))
    print("F1:", round(f1_score(y_test, y_pred), 3))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))

    plt.figure(figsize=(4, 3))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title(f'{name} ROC Curve')
    plt.show()

evaluate_model("Logistic Regression", log_reg, X_test, y_test)
evaluate_model("Random Forest", rf, X_test, y_test)

importances = pd.DataFrame({
    'Feature': features,
    'Importance_RF': rf.feature_importances_,
    'Importance_Log': np.abs(log_reg.coef_[0])
}).sort_values(by='Importance_RF', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance_RF', y='Feature', data=importances)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()

print("\n===== FIFA WORLD CUP 2026 FINALIST PREDICTIONS =====")

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

future_teams['team1_encoded'] = encoder.fit_transform(future_teams['team1'])
future_teams['team2_encoded'] = encoder.fit_transform(future_teams['team2'])

future_scaled = scaler.transform(future_teams[features])
future_teams['Finalist_Prob'] = rf.predict_proba(future_scaled)[:, 1]
future_teams['Predicted_Finalist'] = np.where(rf.predict(future_scaled) == 1, 'YES', 'NO')

print(future_teams[['team1', 'team2', 'Finalist_Prob', 'Predicted_Finalist']])

top_finalists = future_teams.sort_values(by='Finalist_Prob', ascending=False).head(2)
print("\nPredicted 2026 FIFA World Cup Finalists:")
for team in top_finalists['team1']:
    print(f"- {team}")
