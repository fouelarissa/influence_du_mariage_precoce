import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import time
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io
from datetime import datetime
import os

# ---------------------------------------------------------------------------
# CONFIGURATION ET SEO
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IA Sante Publique - Prediction du Mariage Precoce au Cameroun (EDS 2018)",
    page_icon="https://cdn-icons-png.flaticon.com/512/2966/2966486.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<meta name="description" content="Plateforme d Intelligence Artificielle pour analyser et predire les facteurs de risque du mariage precoce chez les jeunes femmes au Cameroun. Donnees EDS 2018.">
<meta name="keywords" content="Mariage Precoce, Cameroun, IA, Sante Publique, EDS, Machine Learning, Prediction">
<meta name="author" content="Fouedjo Feudjou Larissa">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CHEMINS
# ---------------------------------------------------------------------------
ROOT = Path.cwd()
OUTPUTS_DIR = ROOT / 'outputs'
MODEL_PATH = ROOT / 'best_model_mariage_precoce_prod.pkl'
SCALER_PATH = ROOT / 'scaler_age_prod.pkl'
DATA_PATH = ROOT / 'Dataset_Mariage_Precoce.xlsx'
NEW_OBS_PATH = ROOT / 'Nouvelles_Observations.csv'

# ---------------------------------------------------------------------------
# CSS ET FONTAWESOME
# ---------------------------------------------------------------------------
css_code = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* --- Hero --- */
    .hero-section {
        width: 100%;
        min-height: 360px;
        background-image:
            linear-gradient(135deg, rgba(20, 30, 48, 0.85), rgba(36, 59, 85, 0.75)),
            url("https://images.unsplash.com/photo-1504198458649-3128b932f49e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80");
        background-size: cover;
        background-position: center 30%;
        border-radius: 16px;
        margin-bottom: 2rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: #ffffff;
        text-align: center;
        padding: 40px 24px;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
        animation: heroFade 1.4s ease-in-out;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 14px;
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.45);
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        max-width: 850px;
        line-height: 1.7;
        opacity: 0.92;
    }

    /* --- Boutons --- */
    .stButton>button {
        background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%);
        color: #ffffff;
        font-weight: 600;
        font-size: 1rem;
        border-radius: 30px;
        border: none;
        padding: 12px 28px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(13, 71, 161, 0.35);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(13, 71, 161, 0.5);
    }

    /* --- Cartes de resultat --- */
    .result-card {
        padding: 32px 20px;
        border-radius: 16px;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
        animation: slideUp 0.6s ease-out;
    }
    .risk-high { background: linear-gradient(135deg, #d32f2f 0%, #b71c1c 100%); }
    .risk-low  { background: linear-gradient(135deg, #00897b 0%, #26a69a 100%); }
    .result-card h2 { margin: 0 0 6px; font-weight: 600; font-size: 1.3rem; }
    .result-card .score { font-size: 3.4rem; font-weight: 700; margin: 8px 0; }

    /* --- Section titres --- */
    .section-title {
        font-size: 1.35rem;
        font-weight: 600;
        margin-bottom: 8px;
        color: #1a237e;
    }

    /* --- Footer --- */
    .app-footer {
        width: 100%;
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        color: rgba(255, 255, 255, 0.85);
        text-align: center;
        padding: 18px 12px;
        font-size: 0.88rem;
        font-weight: 400;
        margin-top: 60px;
        border-radius: 12px 12px 0 0;
    }
    .app-footer a {
        color: #90caf9;
        text-decoration: none;
        font-weight: 600;
        transition: color 0.3s;
    }
    .app-footer a:hover { color: #e3f2fd; }

    /* --- Responsive --- */
    @media (max-width: 768px) {
        .hero-title { font-size: 1.8rem; }
        .hero-subtitle { font-size: 0.95rem; }
        .hero-section { min-height: 260px; padding: 28px 16px; }
        .result-card .score { font-size: 2.6rem; }
    }

    @keyframes heroFade { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes slideUp  { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
</style>
"""

import base64
hero_img_path = ROOT / 'hero_bg.png'
if hero_img_path.exists():
    with open(hero_img_path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode('utf-8')
    bg_url = f"data:image/png;base64,{b64}"
else:
    bg_url = "https://images.unsplash.com/photo-1504198458649-3128b932f49e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80"

css_code = css_code.replace(
    'url("https://images.unsplash.com/photo-1504198458649-3128b932f49e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80")',
    f'url("{bg_url}")'
)

st.markdown(css_code, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CHARGEMENT DES MODELES ET DONNEES
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    except Exception:
        return None, None

@st.cache_data
def load_data():
    try:
        return pd.read_excel(DATA_PATH, sheet_name='data')
    except Exception:
        return pd.DataFrame()

model, scaler = load_models()
historical_data = load_data()

# ---------------------------------------------------------------------------
# DICTIONNAIRES DE DECODAGE (EDS Cameroun 2018)
# ---------------------------------------------------------------------------
REGIONS_COORDS = {
    1:  [7.3195, 13.5836],  2:  [4.4600, 11.5265],  3:  [4.0511, 9.7679],
    4:  [4.5800, 13.6836],  5:  [10.5910, 14.3274], 6:  [4.5000, 10.0000],
    7:  [9.3000, 13.4000],  8:  [6.0000, 10.2000],  9:  [5.5000, 10.4000],
    10: [2.9000, 11.2000],  11: [5.0000, 9.3000],   12: [3.8480, 11.5021],
}

regions_dict   = {1: 'Adamaoua', 2: 'Centre (sans Yaounde)', 3: 'Douala', 4: 'Est', 5: 'Extreme-Nord', 6: 'Littoral (sans Douala)', 7: 'Nord', 8: 'Nord-Ouest', 9: 'Ouest', 10: 'Sud', 11: 'Sud-Ouest', 12: 'Yaounde'}
residence_dict = {1: 'Urbain', 2: 'Rural'}
edu_dict       = {0: 'Aucun', 1: 'Primaire', 2: 'Secondaire', 3: 'Superieur'}
wealth_dict    = {1: 'Le plus pauvre', 2: 'Pauvre', 3: 'Moyen', 4: 'Riche', 5: 'Le plus riche'}
media_dict     = {0: 'Pas du tout', 1: 'Rarement', 2: 'Parfois', 3: 'Regulierement'}
religion_dict  = {1: 'Catholique', 2: 'Protestant', 3: 'Musulman', 4: 'Animiste', 5: 'Autre'}
emploi_dict    = {0: 'Sans emploi', 1: 'En activite'}
matrimonial_dict = {0: 'Celibataire', 1: 'Mariee', 2: 'Vivant ensemble', 3: 'Veuve', 4: 'Divorcee', 5: 'Separee'}

def decode(val, mapping):
    """Retourne la valeur decodee ou 'Non renseigne'."""
    try:
        return mapping.get(int(val), 'Non renseigne')
    except (ValueError, TypeError):
        return 'Non renseigne'

# ---------------------------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-section">
    <div class="hero-title">
        <i class="fas fa-heartbeat" style="margin-right:10px;"></i>
        Modele Predictif du Mariage Precoce au Cameroun
    </div>
    <div class="hero-subtitle">
        Plateforme d'Intelligence Artificielle avancee pour analyser, cartographier et predire
        les multiples facteurs de vulnerabilite conduisant a l'entree en mariage precoce (avant 18 ans)
        chez les jeunes femmes au Cameroun. Basee sur une modelisation Machine Learning robuste
        entrainee sur 11 526 observations de l'Enquete Demographique et de Sante (EDS-V, 2018).
        Evaluez instantanement un profil, explorez la carte interactive et telechargez vos rapports.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PERSISTANCE DE L'HISTORIQUE
# ---------------------------------------------------------------------------
HIST_COLS = ['Date', 'Age', 'Region', 'Residence', 'Education', 'Richesse',
             'Religion', 'Emploi', 'Radio', 'TV', 'Risque (%)', 'Diagnostic']

if 'history' not in st.session_state:
    if os.path.exists(NEW_OBS_PATH):
        try:
            st.session_state.history = pd.read_csv(NEW_OBS_PATH)
        except Exception:
            st.session_state.history = pd.DataFrame(columns=HIST_COLS)
    else:
        st.session_state.history = pd.DataFrame(columns=HIST_COLS)

# ---------------------------------------------------------------------------
# ONGLETS
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "  Diagnostic Individuel  ",
    "  Cartographie Interactive  ",
    "  Historique et Exports  "
])

# ===== ONGLET 1 : DIAGNOSTIC =====
with tab1:
    st.markdown('<div class="section-title"><i class="fas fa-clipboard-list"></i> Formulaire de saisie du profil</div>', unsafe_allow_html=True)
    st.caption("Renseignez les informations ci-dessous pour obtenir un diagnostic instantane de l'Intelligence Artificielle.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age       = st.slider("Age actuel", 18, 49, 25)
        region    = st.selectbox("Region de residence", options=list(regions_dict.keys()), format_func=lambda x: regions_dict[x])
        residence = st.selectbox("Milieu de residence", options=list(residence_dict.keys()), format_func=lambda x: residence_dict[x])
    with col2:
        education = st.selectbox("Niveau d'education", options=list(edu_dict.keys()), format_func=lambda x: edu_dict[x])
        richesse  = st.selectbox("Indice de richesse", options=list(wealth_dict.keys()), format_func=lambda x: wealth_dict[x])
        emploi    = st.selectbox("Actuellement en emploi ?", options=[0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
    with col3:
        religion_val = st.selectbox("Religion", options=list(religion_dict.keys()), format_func=lambda x: religion_dict[x])
        radio     = st.selectbox("Ecoute de la radio", options=list(media_dict.keys()), format_func=lambda x: media_dict[x])
        tv        = st.selectbox("Visionnage de la TV", options=list(media_dict.keys()), format_func=lambda x: media_dict[x])

    st.markdown("---")

    if st.button("Lancer l'Analyse Predictive", use_container_width=True):
        if model is None or scaler is None:
            st.error("Le modele de prediction est introuvable. Verifiez que les fichiers .pkl existent dans le dossier outputs/.")
        else:
            with st.spinner("Analyse de l'Intelligence Artificielle en cours ..."):
                time.sleep(0.8)

                expected_cat = ['Region', 'Type_Residence', 'Niveau_Education', 'Religion',
                                'Indice_Richesse', 'Statut_Emploi', 'Ecoute_Radio', 'Regarde_TV']
                input_df = pd.DataFrame({
                    'Age_Actuel':       [age],
                    'Region':           [region],
                    'Type_Residence':   [residence],
                    'Niveau_Education': [education],
                    'Religion':         [religion_val],
                    'Indice_Richesse':  [richesse],
                    'Statut_Emploi':    [emploi],
                    'Ecoute_Radio':     [radio],
                    'Regarde_TV':       [tv],
                })
                # Le modele a ete entraine sur les colonnes de type entier sans get_dummies.
                # On utilise input_df tel quel.
                X_input = input_df.copy()
                X_input['Age_Actuel'] = scaler.transform(input_df[['Age_Actuel']])

                X_input = X_input[model.feature_names_in_]

                prob     = model.predict_proba(X_input)[0][1]
                prob_pct = round(prob * 100, 1)
                is_high  = prob > 0.5

                full_info = {
                    'Date':       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Age':        age,
                    'Region':     regions_dict[region],
                    'Residence':  residence_dict[residence],
                    'Education':  edu_dict[education],
                    'Richesse':   wealth_dict[richesse],
                    'Religion':   religion_dict[religion_val],
                    'Emploi':     'Oui' if emploi == 1 else 'Non',
                    'Radio':      media_dict[radio],
                    'TV':         media_dict[tv],
                    'Risque (%)': prob_pct,
                    'Diagnostic': 'Risque Eleve' if is_high else 'Faible Risque',
                }

                new_row = pd.DataFrame([full_info])
                st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
                st.session_state['last_pred'] = {
                    'region_code': region,
                    'lat': REGIONS_COORDS[region][0],
                    'lon': REGIONS_COORDS[region][1],
                    'prob': prob_pct,
                    'is_high': is_high,
                    'details': full_info,
                }

                # Sauvegarde persistante sur disque
                header_needed = not os.path.exists(NEW_OBS_PATH)
                new_row.to_csv(NEW_OBS_PATH, mode='a', header=header_needed, index=False)

                # ---------- AFFICHAGE DU RESULTAT ----------
                st.markdown('<div class="section-title"><i class="fas fa-stethoscope"></i> Resultat du Diagnostic</div>', unsafe_allow_html=True)

                c_left, c_right = st.columns([1, 2])
                with c_left:
                    css_class = "risk-high" if is_high else "risk-low"
                    label     = "RISQUE ELEVE" if is_high else "FAIBLE RISQUE"
                    st.markdown(
                        f'<div class="result-card {css_class}">'
                        f'<h2>{label}</h2>'
                        f'<div class="score">{prob_pct}%</div>'
                        f'<p style="font-size:0.95rem; opacity:0.9;">Probabilite estimee de mariage precoce</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                with c_right:
                    st.markdown("**<i class='fas fa-comment-medical'></i> Ce que cela signifie pour vous :**", unsafe_allow_html=True)
                    if is_high:
                        st.write(
                            f"Votre risque est **eleve** car vous presentez plusieurs facteurs de vulnerabilite "
                            f"identifies par notre modele d'Intelligence Artificielle. "
                            f"Avoir un niveau d'education *{edu_dict[education].lower()}* est un facteur "
                            f"de risque majeur car les donnees montrent que les femmes moins scolarisees "
                            f"sont significativement plus exposees au mariage precoce. "
                            f"Vivre en milieu *{residence_dict[residence].lower()}* et appartenir a la "
                            f"categorie de richesse *{wealth_dict[richesse].lower()}* renforcent egalement "
                            f"cette vulnerabilite. Etre originaire de la region *{regions_dict[region]}* "
                            f"est aussi un element contextuel pris en compte par le modele. "
                            f"Concernant l'exposition mediatique, votre niveau d'ecoute de la radio "
                            f"(*{media_dict[radio].lower()}*) et de visionnage de la television "
                            f"(*{media_dict[tv].lower()}*) indiquent un acces limite a l'information, "
                            f"ce qui est un facteur aggravant supplementaire."
                        )
                    else:
                        st.write(
                            f"Votre risque est **faible** car vous avez un niveau d'education "
                            f"*{edu_dict[education].lower()}*, ce qui est un facteur protecteur tres fort. "
                            f"Les donnees historiques montrent clairement que les femmes eduquees sont "
                            f"beaucoup moins exposees au mariage precoce. "
                            f"Votre categorie de richesse (*{wealth_dict[richesse].lower()}*) agit "
                            f"egalement comme un bouclier protecteur important. "
                            f"Vivre en milieu *{residence_dict[residence].lower()}* et etre originaire "
                            f"de la region *{regions_dict[region]}* contribuent positivement a votre profil. "
                            f"Enfin, votre exposition mediatique (radio : *{media_dict[radio].lower()}*, "
                            f"television : *{media_dict[tv].lower()}*) temoigne d'un acces a l'information "
                            f"qui constitue un facteur de protection supplementaire reconnu."
                        )

                # ---------- GRAPHIQUE D'INFLUENCE ----------
                st.markdown("---")
                st.markdown('<div class="section-title"><i class="fas fa-balance-scale"></i> Facteurs les plus influents selon le Modele</div>', unsafe_allow_html=True)
                st.caption("Ce graphique montre quels criteres pesent le plus dans la decision de l'Intelligence Artificielle, toutes predictions confondues.")

                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:6]
                top_feats = [model.feature_names_in_[i] for i in indices]
                top_imps  = importances[indices]

                name_map = {
                    'Age_Actuel': 'Age actuel de la femme',
                }
                clean = []
                for f in top_feats:
                    if f in name_map:
                        clean.append(name_map[f])
                    elif 'Education' in f:  clean.append("Niveau d'education")
                    elif 'Richesse'  in f:  clean.append("Indice de richesse du menage")
                    elif 'Residence' in f:  clean.append("Milieu de residence (Urbain/Rural)")
                    elif 'Region'    in f:  clean.append("Region de residence")
                    elif 'Religion'  in f:  clean.append("Appartenance religieuse")
                    elif 'Radio'     in f:  clean.append("Ecoute de la radio")
                    elif 'TV'        in f:  clean.append("Visionnage de la television")
                    elif 'Emploi'    in f:  clean.append("Statut d'emploi")
                    else:                   clean.append(f)

                fig, ax = plt.subplots(figsize=(9, 3.5))
                bars = ax.barh(clean[::-1], top_imps[::-1], color=['#1565c0', '#1976d2', '#1e88e5', '#42a5f5', '#64b5f6', '#90caf9'][:len(clean)])
                ax.set_xlabel("Importance dans la decision du modele", fontsize=11)
                ax.tick_params(axis='y', labelsize=11)
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)


# ===== ONGLET 2 : CARTOGRAPHIE =====
with tab2:
    st.markdown('<div class="section-title"><i class="fas fa-map-marked-alt"></i> Carte interactive des observations par region</div>', unsafe_allow_html=True)
    st.caption(
        "Cette carte presente la repartition geographique des 11 526 femmes de l'enquete EDS 2018. "
        "Les marqueurs rouges indiquent les cas de mariage precoce (avant 18 ans) et les marqueurs verts "
        "les cas sans mariage precoce. Survolez ou cliquez sur un point pour voir le profil complet de l'individu. "
        "Si vous avez lance une prediction, votre profil apparait avec une etoile bleue."
    )

    if not historical_data.empty:
        REGION_COLORS = {
            1: '#FF595E', 2: '#FFCA3A', 3: '#8AC926', 4: '#1982C4', 
            5: '#6A4C93', 6: '#FF9F1C', 7: '#2EC4B6', 8: '#E71D36', 
            9: '#3A86FF', 10: '#F15BB5', 11: '#00F5D4', 12: '#9B5DE5'
        }
        m = folium.Map(location=[7.3, 12.3], zoom_start=6, tiles='CartoDB positron')

        cluster = MarkerCluster(
            name="Observations EDS 2018",
            options={
                'maxClusterRadius': 50,
                'spiderfyOnMaxZoom': True,
                'showCoverageOnHover': False,
            }
        ).add_to(m)

        np.random.seed(42)
        for _, row in historical_data.iterrows():
            reg_code = int(row['Region'])
            if reg_code not in REGIONS_COORDS:
                continue

            lat = REGIONS_COORDS[reg_code][0] + np.random.uniform(-0.35, 0.35)
            lon = REGIONS_COORDS[reg_code][1] + np.random.uniform(-0.35, 0.35)

            is_prec   = int(row['Mariage_Precoce']) == 1
            dot_color = REGION_COLORS.get(reg_code, '#808080')
            risk_color = '#d32f2f' if is_prec else '#2e7d32'

            reg_str  = decode(reg_code, regions_dict)
            edu_str  = decode(row['Niveau_Education'], edu_dict)
            res_str  = decode(row['Type_Residence'], residence_dict)
            wea_str  = decode(row['Indice_Richesse'], wealth_dict)
            rel_str  = decode(row['Religion'], religion_dict)
            emp_str  = decode(row['Statut_Emploi'], emploi_dict)
            rad_str  = decode(row['Ecoute_Radio'], media_dict)
            tv_str   = decode(row['Regarde_TV'], media_dict)
            mat_str  = decode(row.get('Statut_Matrimonial', ''), matrimonial_dict)
            age_pm   = row.get('Age_Premier_Mariage', '')
            age_pm_s = f"{int(age_pm)} ans" if pd.notna(age_pm) else "Non renseigne"
            statut_txt = "Mariage precoce (avant 18 ans)" if is_prec else "Pas de mariage precoce"

            tooltip_html = (
                f"<div style='font-family:Outfit,sans-serif; font-size:12px; line-height:1.6; min-width:210px; border-top: 4px solid {dot_color}; padding-top: 5px;'>"
                f"<b style='color:{risk_color};'>{'RISQUE ELEVE' if is_prec else 'RISQUE FAIBLE'}</b><br>"
                f"<b>Age :</b> {int(row['Age_Actuel'])} ans<br>"
                f"<b>Origine :</b> Originaire de {reg_str}<br>"
                f"<b>Milieu :</b> {res_str}<br>"
                f"<b>Education :</b> {edu_str}<br>"
                f"<b>Richesse :</b> {wea_str}<br>"
                f"<b>Religion :</b> {rel_str}<br>"
                f"<b>Emploi :</b> {emp_str}<br>"
                f"<b>Radio :</b> {rad_str}<br>"
                f"<b>TV :</b> {tv_str}<br>"
                f"<b>Situation :</b> {mat_str}<br>"
                f"<b>Age au 1er mariage :</b> {age_pm_s}<br>"
                f"<b>Diagnostic :</b> {statut_txt}"
                f"</div>"
            )

            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color=dot_color,
                fill=True,
                fill_color=dot_color,
                fill_opacity=0.7,
                tooltip=folium.Tooltip(tooltip_html),
            ).add_to(cluster)

        # Marqueur special pour la derniere prediction
        if 'last_pred' in st.session_state:
            p = st.session_state['last_pred']
            d = p['details']
            diag_color = '#d32f2f' if p['is_high'] else '#2e7d32'

            popup_html = (
                f"<div style='font-family:Outfit,sans-serif; min-width:230px; line-height:1.7;'>"
                f"<h4 style='margin:0 0 8px; color:{diag_color};'>"
                f"<i class='fas fa-user-check'></i> NOUVEL INDIVIDU ANALYSE</h4>"
                f"<b>Age :</b> {d['Age']} ans<br>"
                f"<b>Origine :</b> Originaire de {d['Region']}<br>"
                f"<b>Milieu :</b> {d['Residence']}<br>"
                f"<b>Education :</b> {d['Education']}<br>"
                f"<b>Richesse :</b> {d['Richesse']}<br>"
                f"<b>Religion :</b> {d['Religion']}<br>"
                f"<b>Emploi :</b> {d['Emploi']}<br>"
                f"<b>Radio :</b> {d['Radio']}<br>"
                f"<b>TV :</b> {d['TV']}<br>"
                f"<hr style='margin:6px 0;'>"
                f"<b>Diagnostic IA :</b> {d['Risque (%)']}% - {d['Diagnostic']}"
                f"</div>"
            )

            folium.Marker(
                location=[p['lat'], p['lon']],
                popup=folium.Popup(popup_html, max_width=320),
                icon=folium.Icon(color='blue', icon='star', prefix='fa'),
                tooltip="Cliquez pour voir le profil complet de l'individu analyse",
            ).add_to(m)

        folium_static(m, width=1100, height=550)
    else:
        st.warning("Les donnees historiques n'ont pas pu etre chargees. Verifiez que le fichier Dataset_Mariage_Precoce.xlsx est present.")


# ===== ONGLET 3 : HISTORIQUE ET EXPORTS =====
with tab3:
    st.markdown('<div class="section-title"><i class="fas fa-clock-rotate-left"></i> Historique des predictions</div>', unsafe_allow_html=True)
    st.caption("Toutes vos predictions sont automatiquement sauvegardees. Elles persistent meme apres un rafraichissement de la page.")

    if st.session_state.history.empty:
        st.info("Aucune prediction effectuee pour le moment. Rendez-vous dans l'onglet 'Diagnostic Individuel' pour lancer votre premiere analyse.")
    else:
        st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown('<div class="section-title"><i class="fas fa-file-export"></i> Telechargements</div>', unsafe_allow_html=True)

        col_dl1, col_dl2 = st.columns(2)

        # --- EXPORT EXCEL ---
        buf_xl = io.BytesIO()
        with pd.ExcelWriter(buf_xl, engine='openpyxl') as writer:
            st.session_state.history.to_excel(writer, index=False, sheet_name='Historique')
        col_dl1.download_button(
            label="Telecharger l'historique complet (Excel)",
            data=buf_xl.getvalue(),
            file_name=f"historique_predictions_{datetime.now():%Y%m%d_%H%M}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # --- EXPORT PDF ---
        last = st.session_state.history.iloc[-1]

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 18)
        pdf.cell(0, 12, "Rapport de Prediction - Mariage Precoce", ln=True, align='C')
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 8, "Genere par la plateforme IA Sante Publique - EDS Cameroun 2018", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(0, 10, "Profil de l'individu analyse", ln=True)
        pdf.set_font("Helvetica", size=11)
        pdf.ln(2)
        for key, val in last.items():
            safe_key = str(key).encode('latin-1', 'replace').decode('latin-1')
            safe_val = str(val).encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(0, 8, f"   {safe_key} : {safe_val}", ln=True)

        pdf.ln(8)
        pdf.set_font("Helvetica", 'I', 9)
        pdf.cell(0, 8, "Ce rapport a ete genere automatiquement. Developpe par Fouedjo Feudjou Larissa.", ln=True)

        col_dl2.download_button(
            label="Telecharger le dernier rapport (PDF)",
            data=bytes(pdf.output()),
            file_name=f"rapport_prediction_{datetime.now():%Y%m%d_%H%M}.pdf",
            mime="application/pdf",
        )


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-footer">
    <i class="fas fa-code"></i>
    &copy; 2025 - Developpe par
    <a href="https://github.com/fouelarissa" target="_blank">
        <i class="fab fa-github"></i> Fouedjo Feudjou Larissa
    </a>
    &nbsp;|&nbsp; Plateforme IA de Prediction en Sante Publique &nbsp;|&nbsp; Donnees EDS Cameroun 2018
</div>
""", unsafe_allow_html=True)
