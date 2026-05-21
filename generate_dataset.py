import pandas as pd
import numpy as np
import os

def create_dataset():
    input_file = "CMIR71FL.xlsx"
    output_file = "Dataset_Mariage_Precoce.xlsx"
    
    print("Chargement des données brutes...")
    # DHS variables to extract
    # V012: Current age
    # V024: Region
    # V025: Type of place of residence
    # V106: Highest educational level
    # V130: Religion
    # V190: Wealth index
    # V714: Respondent currently working
    # V158: Reading newspaper or magazine -> Actually let's use V158 (Radio), V159 (TV), V157 (Newspaper)
    # V501: Current marital status
    # V511: Age at first marriage
    # V005: Sample weight
    columns_to_load = ["V012", "V024", "V025", "V106", "V130", "V190", "V714", "V158", "V159", "V501", "V511", "V005"]
    
    # Read the dataset (only selected columns for memory efficiency)
    try:
        df = pd.read_excel(input_file, usecols=columns_to_load)
    except Exception as e:
        # If columns_to_load are not exactly present, load all and filter
        df = pd.read_excel(input_file)
        df = df[columns_to_load]

    # Renaming for clarity in the final dataset
    rename_dict = {
        "V012": "Age_Actuel",
        "V024": "Region",
        "V025": "Type_Residence",
        "V106": "Niveau_Education",
        "V130": "Religion",
        "V190": "Indice_Richesse",
        "V714": "Statut_Emploi",
        "V158": "Ecoute_Radio",
        "V159": "Regarde_TV",
        "V501": "Statut_Matrimonial",
        "V511": "Age_Premier_Mariage",
        "V005": "Poids_Sondage"
    }
    df.rename(columns=rename_dict, inplace=True)
    
    # Pre-filtering: keep only women aged 18 to 49
    df = df[(df["Age_Actuel"] >= 18) & (df["Age_Actuel"] <= 49)].copy()
    
    # Creating the target variable: Mariage_Precoce
    # In DHS, Age_Premier_Mariage is numerical. NaN often means never married.
    # We will set Mariage_Precoce = 1 if Age_Premier_Mariage < 18
    # 0 if Age_Premier_Mariage >= 18 or NaN (never married)
    
    # Replace any potential DHS missing codes (like 97, 98, 99) in Age_Premier_Mariage with NaN if applicable.
    # In V511, 97=Inconsistent, 98=Don't know, 99=Missing.
    df["Age_Premier_Mariage"] = pd.to_numeric(df["Age_Premier_Mariage"], errors='coerce')
    df.loc[df["Age_Premier_Mariage"] >= 97, "Age_Premier_Mariage"] = np.nan
    
    # Condition for Early Marriage
    condition_yes = df["Age_Premier_Mariage"] < 18
    df["Mariage_Precoce"] = np.where(condition_yes, 1, 0)
    
    # Basic cleaning
    df["Type_Residence"] = df["Type_Residence"].fillna(df["Type_Residence"].mode()[0])
    df["Niveau_Education"] = df["Niveau_Education"].fillna(df["Niveau_Education"].mode()[0])
    df["Religion"] = df["Religion"].fillna(df["Religion"].mode()[0])
    df["Indice_Richesse"] = df["Indice_Richesse"].fillna(df["Indice_Richesse"].mode()[0])
    df["Statut_Emploi"] = df["Statut_Emploi"].fillna(df["Statut_Emploi"].mode()[0])
    df["Ecoute_Radio"] = df["Ecoute_Radio"].fillna(df["Ecoute_Radio"].mode()[0])
    df["Regarde_TV"] = df["Regarde_TV"].fillna(df["Regarde_TV"].mode()[0])
    
    # For weight, DHS divides by 1000000
    df["Poids_Sondage"] = df["Poids_Sondage"] / 1000000.0

    print("Création du dictionnaire de métadonnées...")
    # Metadata dictionary to explain encodings
    metadata = {
        "Variable": [
            "Age_Actuel", "Region", "Type_Residence", "Niveau_Education",
            "Religion", "Indice_Richesse", "Statut_Emploi", "Ecoute_Radio",
            "Regarde_TV", "Statut_Matrimonial", "Age_Premier_Mariage", "Poids_Sondage", "Mariage_Precoce"
        ],
        "Description": [
        "Âge de la femme au moment de l'enquête (continue en années)",
        "Région de résidence au Cameroun (1=Adamaoua, 2=Centre (sans Yaoundé), 3=Douala, 4=Est, 5=Extrême-Nord, 6=Littoral (sans Douala), 7=Nord, 8=Nord-Ouest, 9=Ouest, 10=Sud, 11=Sud-Ouest, 12=Yaoundé)",
        "Milieu de résidence (1 = Urbain, 2 = Rural)",
        "Niveau d'éducation (0 = Aucun, 1 = Primaire, 2 = Secondaire, 3 = Supérieur)",
        "Religion pratiquée par la répondante (Catégorielle)",
        "Indice de richesse du ménage (1 = Le plus pauvre, 2 = Pauvre, 3 = Moyen, 4 = Riche, 5 = Le plus riche)",
        "Actuellement en emploi (0 = Non, 1 = Oui)",
        "Fréquence d'écoute de la radio (0=Pas du tout, 1=Moins d'1 fois/sem, 2=Au moins 1 fois/sem, 3=Presque tous les jours)",
        "Fréquence de visionnage de la TV (0=Pas du tout, 1=Moins d'1 fois/sem, 2=Au moins 1 fois/sem, 3=Presque tous les jours)",
        "Statut matrimonial actuel (0 = Jamais mariée, 1 = Mariée, etc.)",
        "Âge au premier mariage (en années, NaN si jamais mariée)",
        "Poids de sondage normalisé (V005 / 1000000)",
        "Mariage Précoce (Cible) : 1 = Oui (Mariée avant 18 ans), 0 = Non (Mariée à 18 ans ou plus, ou jamais mariée)"
    ]
    }
    df_meta = pd.DataFrame(metadata)
    
    print("Sauvegarde dans le fichier Excel...")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='data', index=False)
        df_meta.to_excel(writer, sheet_name='metadonnees', index=False)
        
    print(f"Jeu de données généré avec succès : {output_file}")
    print(f"Taille finale : {df.shape[0]} observations.")

if __name__ == "__main__":
    create_dataset()
