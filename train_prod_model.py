import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

ROOT = Path.cwd()
PROJECT_ROOT = ROOT
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
TABLE_DIR = OUTPUTS_DIR / 'tables'
FIG_DIR = OUTPUTS_DIR / 'figures'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel('Dataset_Mariage_Precoce.xlsx', sheet_name='data')

categorical_cols = ['Region', 'Type_Residence', 'Niveau_Education', 'Religion', 'Indice_Richesse', 'Statut_Emploi', 'Ecoute_Radio', 'Regarde_TV']
X = pd.get_dummies(df[categorical_cols], drop_first=True)
X['Age_Actuel'] = df['Age_Actuel']
y = df['Mariage_Precoce']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train['Age_Actuel'] = scaler.fit_transform(X_train[['Age_Actuel']])
X_test['Age_Actuel'] = scaler.transform(X_test[['Age_Actuel']])

# Grille (réduite pour exécution rapide)
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.1],
    'max_depth': [3],
    'min_samples_split': [2]
}

gbc = GradientBoostingClassifier(random_state=42)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("Entraînement du modèle GridSearchCV en cours...")
grid_search = GridSearchCV(estimator=gbc, param_grid=param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Modèle entraîné. Meilleurs hyperparamètres : {grid_search.best_params_}")

joblib.dump(best_model, OUTPUTS_DIR / 'best_model_mariage_precoce_prod.joblib')
joblib.dump(best_model, OUTPUTS_DIR / 'best_model_mariage_precoce_prod.pkl')
joblib.dump(scaler, OUTPUTS_DIR / 'scaler_age_prod.pkl')
print("Modèles sauvegardés dans outputs/")
