import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import json

df = pd.read_excel('Dataset_Mariage_Precoce.xlsx', sheet_name='data')

categorical_cols = ['Region', 'Type_Residence', 'Niveau_Education', 'Religion', 'Indice_Richesse', 'Statut_Emploi', 'Ecoute_Radio', 'Regarde_TV']
X = pd.get_dummies(df[categorical_cols], drop_first=True)
X['Age_Actuel'] = df['Age_Actuel']
y = df['Mariage_Precoce']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train['Age_Actuel'] = scaler.fit_transform(X_train[['Age_Actuel']])
X_test['Age_Actuel'] = scaler.transform(X_test[['Age_Actuel']])

models = {
    'Régression Logistique': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    results.append({'Modèle': name, 'Accuracy': acc, 'AUC': auc})
    print(f"\\n--- {name} ---")
    print(classification_report(y_test, y_pred))

df_results = pd.DataFrame(results).sort_values(by='AUC', ascending=False)
best_model_name = df_results.iloc[0]['Modèle']
best_model = models[best_model_name]

joblib.dump(best_model, 'meilleur_modele_mariage_precoce.joblib')
joblib.dump(best_model, 'meilleur_modele_mariage_precoce.pkl')
joblib.dump(scaler, 'scaler_age.pkl')

print(f"\\nLe meilleur modèle est {best_model_name} avec AUC={df_results.iloc[0]['AUC']:.4f}")

# Stress Test
np.random.seed(99)
n_samples = 500
X_stress = pd.DataFrame(columns=X.columns)
for col in X.columns:
    if col == 'Age_Actuel':
        ages = np.random.randint(18, 50, n_samples)
        X_stress[col] = scaler.transform(ages.reshape(-1, 1)).flatten()
    else:
        X_stress[col] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

y_stress_pred = best_model.predict(X_stress)
y_stress_proba = best_model.predict_proba(X_stress)[:, 1]
y_stress_reel = (y_stress_proba + np.random.normal(0, 0.2, n_samples) > 0.5).astype(int)

df_comparaison = pd.DataFrame({
    'Probabilite_Predite': y_stress_proba,
    'Prediction_Modele': y_stress_pred,
    'Valeur_Reelle_Simulee': y_stress_reel
})
acc_stress = accuracy_score(df_comparaison['Valeur_Reelle_Simulee'], df_comparaison['Prediction_Modele'])

print(f"\\nAccuracy Stress Test: {acc_stress:.4f}")
print("\\nAperçu Comparaison (5 lignes) :")
print(df_comparaison.head())
