import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.feature_selection import mutual_info_classif, RFE

# Load Dataset
df = pd.read_csv('data/german_credit_data.csv')

# Preprocessing
df['Saving accounts'].fillna('no_info', inplace=True)
df['Checking account'].fillna(0, inplace=True)

label_encoders = {}
categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Purpose']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature and Target
X = df.drop(columns=['Risk'])
y = df['Risk'].map({'good': 0, 'bad': 1})

# Correlation Analysis
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Mutual Information
mi_scores = mutual_info_classif(X, y)
mi_scores = pd.Series(mi_scores, index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=mi_scores.values, y=mi_scores.index)
plt.title('Mutual Information Scores')
plt.show()

# Recursive Feature Elimination (RFE)
model = DecisionTreeClassifier()
rfe = RFE(model, n_features_to_select=8)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]

print("\nSelected Features after RFE:", selected_features.tolist())

# Update X based on selected features
X = X[selected_features]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Models to train
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}

# Training and Evaluation
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = [acc, prec, rec, f1]

# Show results
results_df = pd.DataFrame(results, index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
print("\nModel Evaluation Results:")
print(results_df)

# Select Best Model
best_model_name = results_df.loc['F1 Score'].idxmax()
print(f"\nBest Model based on F1 Score: {best_model_name}")

best_model = models[best_model_name]

# Save the Best Model and Scaler
import os

os.makedirs('model', exist_ok=True)

with open('model/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nBest Model and Scaler saved successfully!")
