# 🌍 Modèle Prédictif du Mariage Précoce au Cameroun (Données EDS 2018)

**Développé par : Fouedjo Feudjou Larissa**

Ce projet est une plateforme de santé publique avancée, propulsée par l'Intelligence Artificielle (Machine Learning) et la modélisation statistique. Son objectif est d'analyser, d'expliquer et de prédire les facteurs influençant l'entrée en mariage précoce (avant 18 ans) chez les jeunes femmes au Cameroun, à partir des données de l'Enquête Démographique et de Santé (EDS-V 2018).

---

## ✨ Fonctionnalités Principales

### 1. 🧠 Modélisation Hybride (Statistiques & ML)
Le projet se divise en deux volets de recherche :
- **Analyse Statistique Régression Logistique (`1_Modele_Statistique_Mariage_Precoce.ipynb`)** : Analyse multivariée stricte utilisant des matrices de corrélation (V de Cramér) et le calcul des rapports de cotes (Odds Ratios) pour garantir une parfaite explicabilité scientifique.
- **Modèle de Production Machine Learning (`2_Modele_Machine_Learning_Mariage_Precoce.ipynb`)** : Utilisation de `GradientBoostingClassifier` avec optimisation hyperparamétrique (`GridSearchCV`). Génération de courbes d'apprentissage (Learning Curves), de courbes ROC (AUC ~77%) pour le déploiement.

### 2. 💻 Application Web Interactive (Streamlit)
Une plateforme moderne, "Grand-mère friendly" et entièrement *responsive* :
- **Formulaire de Prédiction en Temps Réel** : Renseignez les variables d'un individu (Âge, Région, Éducation, etc.) et obtenez une prédiction instantanée de l'IA accompagnée d'un paragraphe narratif et d'un diagnostic détaillé.
- **Graphique d'Influence (Explainable AI)** : Visualisation du poids de chaque facteur dans la prise de décision de l'IA.

### 3. 🗺️ Cartographie Folium Dynamique
- **Clusterisation des Données Historiques** : Visualisation des 11 525 patientes historiques réparties sur la carte du Cameroun. Chaque marqueur affiche un résumé complet et exhaustif (format texte décodé) de la patiente.
- **Suivi Utilisateur Live** : À chaque prédiction, la nouvelle observation s'ajoute sur la carte avec une icône spéciale (Étoile) pour la repérer facilement parmi les autres.

### 4. 🔄 Apprentissage Continu et Persistance
- Les nouvelles prédictions sont sauvegardées instantanément dans un fichier `Nouvelles_Observations.csv`. L'historique persiste même lorsque la page web est rafraîchie (stockage local).

### 5. 📥 Exports Professionnels
- **Rapport PDF** : Téléchargement automatique du diagnostic formaté de l'individu analysé.
- **Export Excel (.xlsx)** : Téléchargement direct de l'historique complet de la session pour analyse ultérieure.

---

## 🚀 Prise en Main (Installation)

### Prérequis
Assurez-vous d'avoir installé **Python 3.8+** sur votre machine.

### Installation
1. Clonez ce dossier ou téléchargez-le sur votre bureau.
2. Ouvrez un terminal (Invite de Commandes ou PowerShell) dans le dossier du projet : `C:/Users/DELL/Desktop/Age au premier mariage`
3. Installez les dépendances avec la commande :
   ```bash
   pip install -r requirements.txt
   ```

### Lancement de l'Application
Lancez le serveur Streamlit avec la commande suivante :
```bash
streamlit run app_mariage_precoce.py
```
L'application s'ouvrira automatiquement dans votre navigateur web à l'adresse locale `http://localhost:8501`.

---
*Projet conçu avec passion pour améliorer les politiques de santé publique et la protection des jeunes femmes en Afrique.*
