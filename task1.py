import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Suppress warnings for a clean output
warnings.filterwarnings('ignore')

# --- 1. Load Dataset ---
# Ensure these .txt files are in your Desktop/CODSOFT folder
try:
    train_data = pd.read_csv("train_data.txt", sep=':::', names=['id', 'title', 'genre', 'description'], engine='python')
    test_data = pd.read_csv("test_data.txt", sep=':::', names=['id', 'title', 'genre', 'description'], engine='python')
    test_sol_data = pd.read_csv("test_data_solution.txt", sep=':::', names=['id', 'title', 'genre', 'description'], engine='python')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: Dataset files not found in the current directory.")

# --- 2. Data Preprocessing ---
train_data['description'] = train_data['description'].fillna("")
test_data['description'] = test_data['description'].fillna("")

# --- 3. Feature Extraction (Corrected Capitalization) ---
t_v = TfidfVectorizer(stop_words='english', max_features=50000)
X_train = t_v.fit_transform(train_data['description'])
X_test = t_v.transform(test_data['description'])

# --- 4. Label Encoding (Corrected Capitalization) ---
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['genre'])
y_test = label_encoder.transform(test_sol_data['genre'])

# --- 5. Model Training (Corrected Capitalization) ---
# Splitting for validation
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

clf = LinearSVC(dual=False)
clf.fit(X_train_sub, y_train_sub)

# --- 6. Results ---
y_pred = clf.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
