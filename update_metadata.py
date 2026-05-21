import pandas as pd

output_file = "Dataset_Mariage_Precoce_V2.xlsx"

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

# Load existing data
df = pd.read_excel("Dataset_Mariage_Precoce.xlsx", sheet_name='data')

# Save both back to excel
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='data', index=False)
    df_meta.to_excel(writer, sheet_name='metadonnees', index=False)

print("Métadonnées mises à jour avec succès.")
