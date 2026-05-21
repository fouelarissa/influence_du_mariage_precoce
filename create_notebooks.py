import nbformat as nbf
import os

def create_notebook_1():
    nb = nbf.v4.new_notebook()
    
    nb['cells'] = [
        nbf.v4.new_markdown_cell("# Modèle Statistique - Facteurs influençant le Mariage Précoce\n\nCe notebook présente l'analyse exploratoire et le modèle de régression logistique binaire basé sur les données EDS Cameroun 2018.\n\n## Pourquoi un Modèle Statistique ?\nL'objectif d'un modèle statistique classique (comme la Régression Logistique) par rapport au Machine Learning pur, est l'**explicabilité**. Nous cherchons ici à quantifier exactement *l'impact* de chaque facteur (ex: vivre en zone rurale augmente le risque de X fois) grâce aux **Odds Ratios** (Rapports de Cotes)."),
        
        nbf.v4.new_markdown_cell("## 1. Importation des bibliothèques, Architecture et Chargement"),
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from pathlib import Path
import scipy.stats as ss

# Style graphique professionnel
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('muted')

# Architecture de sortie
ROOT = Path.cwd()
PROJECT_ROOT = ROOT
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
TABLE_DIR = OUTPUTS_DIR / 'tables'
FIG_DIR = OUTPUTS_DIR / 'figures'

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Chargement simple des données
try:
    df = pd.read_excel('Dataset_Mariage_Precoce_V2.xlsx', sheet_name='data')
    print("Données chargées avec succès. Dimensions :", df.shape)
except Exception as e:
    print("Erreur lors du chargement des données :", e)"""),
        
        nbf.v4.new_markdown_cell("## 2. Preprocessing des données\nLe nettoyage des données est une étape fondamentale. Les valeurs aberrantes ont été écartées lors de la génération du jeu de données."),
        nbf.v4.new_code_cell("""# Typage des variables catégorielles
categorical_cols = ['Region', 'Type_Residence', 'Niveau_Education', 'Religion', 'Indice_Richesse', 'Statut_Emploi', 'Ecoute_Radio', 'Regarde_TV']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Définition de X et y
y_col = 'Mariage_Precoce'
y = df[y_col]
X = df.drop(columns=[y_col, 'Statut_Matrimonial', 'Age_Premier_Mariage']) 
"""),

        nbf.v4.new_markdown_cell("## 3. Analyse Exploratoire"),
        nbf.v4.new_markdown_cell("### 3.1. Analyse Univariée\nL'analyse univariée permet de comprendre la distribution individuelle de chaque variable avant d'analyser leurs relations."),
        nbf.v4.new_code_cell("""# Distribution de la variable cible
plt.figure(figsize=(6, 4))
ax = sns.countplot(x=y_col, data=df, palette='Set2')
plt.title('Distribution du Mariage Précoce (1 = Oui, 0 = Non)')
plt.ylabel('Nombre de femmes')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=10)
plt.savefig(FIG_DIR / 'dist_mariage_precoce.png', bbox_inches='tight')
plt.show()

# Distribution de l'âge actuel
plt.figure(figsize=(8, 4))
sns.histplot(df['Age_Actuel'], bins=15, kde=True, color='teal')
plt.title('Distribution de l\'Âge Actuel')
plt.xlabel('Âge')
plt.ylabel('Fréquence')
plt.savefig(FIG_DIR / 'dist_age.png', bbox_inches='tight')
plt.show()"""),

        nbf.v4.new_markdown_cell("### 3.2. Analyse Bivariée (X vs Y)\n**Le Test du Chi-2 d'indépendance** permet de vérifier si deux variables catégorielles sont significativement liées. Une *p-value* < 0.05 indique que la relation observée n'est pas due au hasard."),
        nbf.v4.new_code_cell("""vars_to_plot = ['Type_Residence', 'Niveau_Education', 'Indice_Richesse']

biv_results = []
for var in vars_to_plot:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=var, hue=y_col, data=df, palette='Set1')
    plt.title(f'Mariage Précoce selon {var}')
    plt.ylabel('Effectif')
    plt.legend(title='Mariage Précoce', loc='upper right')
    plt.savefig(FIG_DIR / f'biv_{var}.png', bbox_inches='tight')
    plt.show()
    
    contingency_table = pd.crosstab(df[var], df[y_col])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    biv_results.append({'Variable': var, 'Chi-2': chi2, 'p-value': p})
    print(f"Test Chi-2 pour {var} : p-value = {p:.4e}")

pd.DataFrame(biv_results).to_csv(TABLE_DIR / 'chi2_results.csv', index=False)
"""),

        nbf.v4.new_markdown_cell("### 3.3. Analyse Multivariée Strictement Conforme\n**Bonnes Pratiques Statistiques :** La matrice de corrélation classique (Pearson) requiert des variables continues. Puisque la majorité de nos variables sont catégorielles, nous utilisons le **V de Cramér**, qui mesure l'intensité de l'association entre variables nominales (0 = aucune association, 1 = association parfaite). L'objectif est de vérifier l'absence de colinéarité excessive (variables qui disent exactement la même chose)."),
        nbf.v4.new_code_cell("""def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

corr_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)
for col1 in categorical_cols:
    for col2 in categorical_cols:
        corr_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix.astype(float), annot=True, cmap='coolwarm', vmin=0, vmax=1, fmt=".2f")
plt.title("Matrice d'Association (V de Cramér) des Variables Catégorielles")
plt.savefig(FIG_DIR / 'cramer_v_matrix.png', bbox_inches='tight')
plt.show()

corr_matrix.to_csv(TABLE_DIR / 'cramer_v_matrix.csv')"""),

        nbf.v4.new_markdown_cell("## 4. Modèle Statistique (Régression Logistique Binaire)\nNous modélisons la probabilité qu'une femme se marie avant 18 ans. Le modèle GLM (Generalized Linear Model) de la famille Binomiale permet de prendre en compte les **Poids de sondage (V005)** de l'EDS, ce qui est crucial pour que notre échantillon reste représentatif de la population camerounaise totale."),
        nbf.v4.new_code_cell("""# Encodage Dummy pour éviter le piège de la colinéarité (drop_first=True)
X_model = pd.get_dummies(df[categorical_cols], drop_first=True)
X_model['Age_Actuel'] = df['Age_Actuel']

X_model = sm.add_constant(X_model.astype(float))
y_model = y.astype(float)

weights = df['Poids_Sondage']
logit_model = sm.GLM(y_model, X_model, family=sm.families.Binomial(), freq_weights=weights)
result = logit_model.fit()

summary_html = result.summary().as_html()
with open(TABLE_DIR / 'regression_summary.html', 'w', encoding='utf-8') as f:
    f.write(summary_html)

print(result.summary())"""),

        nbf.v4.new_markdown_cell("## 5. Interprétation : Odds Ratios (Rapports de Cotes)\n**Théorie :** Un coefficient logit brut est difficile à interpréter. En passant ce coefficient à l'exponentielle, on obtient l'**Odds Ratio**. \n- Si OR > 1 : Le facteur augmente le risque de mariage précoce.\n- Si OR < 1 : Le facteur diminue ce risque.\n- Si OR = 1 : Aucun effet."),
        nbf.v4.new_code_cell("""coeffs = result.params
print("Équation de la régression logistique :")
print(f"Logit(P) = {coeffs['const']:.4f}", end=" ")
for var, coef in coeffs.items():
    if var != 'const':
        sign = "+" if coef >= 0 else "-"
        print(f"\\n           {sign} {abs(coef):.4f} * {var}", end=" ")
print("\\n")

odds_ratios = pd.DataFrame({'Odds Ratio': np.exp(coeffs), 'p-value': result.pvalues})
odds_ratios.to_csv(TABLE_DIR / 'odds_ratios.csv')
print("Odds Ratios sauvegardés dans outputs/tables/odds_ratios.csv")
print(odds_ratios)""")
    ]
    
    with open('1_Modele_Statistique_Mariage_Precoce.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Notebook 1 V3 (Documenté) créé avec succès.")

def create_notebook_2():
    nb = nbf.v4.new_notebook()
    
    nb['cells'] = [
        nbf.v4.new_markdown_cell("# Modèle de Machine Learning - Prédiction du Mariage Précoce\n\nContrairement à la statistique pure, le but du Machine Learning est de maximiser la **capacité prédictive** sur de nouvelles données. Nous allons entraîner un algorithme robuste pour le déploiement en production."),
        
        nbf.v4.new_markdown_cell("## 1. Importation, Architecture et Séparation des données\n**Hold-out Method :** Nous divisons les données (80% entraînement, 20% test) pour évaluer le modèle sur des données qu'il n'a jamais vues. La stratification assure que la proportion de cas positifs/négatifs est la même dans les deux ensembles."),
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

ROOT = Path.cwd()
PROJECT_ROOT = ROOT
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
TABLE_DIR = OUTPUTS_DIR / 'tables'
FIG_DIR = OUTPUTS_DIR / 'figures'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel('Dataset_Mariage_Precoce_V2.xlsx', sheet_name='data')

categorical_cols = ['Region', 'Type_Residence', 'Niveau_Education', 'Religion', 'Indice_Richesse', 'Statut_Emploi', 'Ecoute_Radio', 'Regarde_TV']
X = pd.get_dummies(df[categorical_cols], drop_first=True)
X['Age_Actuel'] = df['Age_Actuel']
y = df['Mariage_Precoce']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train['Age_Actuel'] = scaler.fit_transform(X_train[['Age_Actuel']])
X_test['Age_Actuel'] = scaler.transform(X_test[['Age_Actuel']])
"""),
        
        nbf.v4.new_markdown_cell("## 2. Optimisation des Hyperparamètres (GridSearchCV)\n**GridSearchCV** teste méthodiquement de multiples combinaisons de paramètres (ex: la profondeur des arbres) avec une **validation croisée** (K-Fold). Le modèle ne valide pas sa performance sur un seul échantillon, mais sur plusieurs, garantissant une grande stabilité pour la production."),
        nbf.v4.new_code_cell("""param_grid = {
    'n_estimators': [100, 150],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4],
    'min_samples_split': [2, 5]
}

gbc = GradientBoostingClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=gbc, param_grid=param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Meilleurs Hyperparamètres : {grid_search.best_params_}")
"""),

        nbf.v4.new_markdown_cell("## 3. Courbes d'Apprentissage (Learning Curves)\nCes courbes montrent l'évolution du score du modèle en fonction de la taille de l'échantillon d'entraînement. \n- Si les courbes d'entraînement et de validation se rejoignent à un haut niveau : le modèle est sain.\n- Si elles sont éloignées : sur-apprentissage (overfitting)."),
        nbf.v4.new_code_cell("""train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Score d'Entraînement")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Score de Validation Croisée")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.title("Courbes d'Apprentissage (Learning Curves)")
plt.xlabel("Taille de l'échantillon d'entraînement")
plt.ylabel("Score AUC")
plt.legend(loc="best")
plt.savefig(FIG_DIR / 'learning_curve.png', bbox_inches='tight')
plt.show()"""),

        nbf.v4.new_markdown_cell("## 4. Évaluation Globale de Généralisation\nL'évaluation utilise des métriques clés :\n- **Accuracy :** Le pourcentage total de prédictions correctes.\n- **AUC-ROC :** La capacité du modèle à bien séparer les deux classes. 1.0 est parfait, 0.5 équivaut au hasard.\n- **F1-Score :** Un compromis entre la Précision et le Rappel (très utile si les données sont déséquilibrées)."),
        nbf.v4.new_code_cell("""y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)

print("="*60)
print(f"POURCENTAGE DE PERFORMANCE DE GÉNÉRALISATION TOTAL (ACCURACY) : {acc*100:.2f}%")
print(f"SCORE AUC (Capacité de séparation) : {auc_score*100:.2f}%")
print(f"SCORE F1 (Équilibre) : {f1*100:.2f}%")
print("="*60)

print("\\nRapport de classification détaillé :")
print(classification_report(y_test, y_pred))

# Matrice de confusion
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de Confusion (Test Data)")
plt.xlabel("Prédictions")
plt.ylabel("Valeurs Réelles")
plt.savefig(FIG_DIR / 'confusion_matrix_prod.png', bbox_inches='tight')
plt.show()

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'AUC', 'F1-Score'],
    'Score (%)': [acc*100, auc_score*100, f1*100]
})
metrics_df.to_csv(TABLE_DIR / 'generalization_performance.csv', index=False)
"""),

        nbf.v4.new_markdown_cell("### 4.1. Courbe ROC et Densité des Probabilités\nCes graphiques permettent de juger visuellement la capacité discriminante du modèle."),
        nbf.v4.new_code_cell("""# Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig(FIG_DIR / 'roc_curve.png', bbox_inches='tight')
plt.show()

# Densité des probabilités
plt.figure(figsize=(8, 5))
sns.kdeplot(y_proba[y_test == 0], fill=True, color="blue", label="Vérité : Non Précoce (0)")
sns.kdeplot(y_proba[y_test == 1], fill=True, color="red", label="Vérité : Précoce (1)")
plt.title("Densité de la Probabilité Prédite par Classe Réelle")
plt.xlabel("Probabilité de Mariage Précoce")
plt.ylabel("Densité")
plt.legend()
plt.savefig(FIG_DIR / 'prob_density.png', bbox_inches='tight')
plt.show()
"""),

        nbf.v4.new_markdown_cell("## 5. Sauvegarde Sécurisée du Modèle Final pour Production"),
        nbf.v4.new_code_cell("""joblib.dump(best_model, OUTPUTS_DIR / 'best_model_mariage_precoce_prod.joblib')
joblib.dump(best_model, OUTPUTS_DIR / 'best_model_mariage_precoce_prod.pkl')
joblib.dump(scaler, OUTPUTS_DIR / 'scaler_age_prod.pkl')

print("Modèles sauvegardés avec succès dans le dossier 'outputs/'. Prêts pour l'application Streamlit.")
""")
    ]
    
    with open('2_Modele_Machine_Learning_Mariage_Precoce.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Notebook 2 V3 (Documenté et Graphiques) créé avec succès.")

if __name__ == "__main__":
    create_notebook_1()
    create_notebook_2()
