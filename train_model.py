import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

if not os.path.exists('hand_gestures.csv'):
    print("Error: hand_gestures.csv not found!")
    exit()

# 1. Load data
data = pd.read_csv('hand_gestures.csv')

# --- THE FIX IS HERE ---
# This removes any rows that have empty (NaN) values
print(f"Rows before cleaning: {len(data)}")
data = data.dropna() 
print(f"Rows after cleaning: {len(data)}")
# -----------------------

X = data.drop('label', axis=1)
y = data['label']

# 2. Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 3. Results
score = model.score(X_test, y_test)
print(f"Model successfully trained with {X.shape[1]} features!")
print(f"Accuracy: {score * 100:.2f}%")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Updated model.pkl saved!")