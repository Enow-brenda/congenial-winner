
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import os
import base64

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FP Autonomy Predictor",
    page_icon="assets/favicon.png" if os.path.exists("assets/favicon.png") else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── NAV CONFIG ───────────────────────────────────────────────────────────────
NAV_ITEMS = [
    # (page_key,    label_en,         label_fr,                  icon_svg)
    ("predict",     "Predict",        "Prédire",
     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" width="17" height="17"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>'),
 
    ("performance", "Model Performance", "Performance",
     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" width="17" height="17"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M8 17V13M12 17V9M16 17V12"/></svg>'),
 
    ("features",    "Feature Importance", "Importance des Variables",
     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" width="17" height="17"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>'),
 
    ("about",       "About",          "À propos",
     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" width="17" height="17"><circle cx="12" cy="12" r="9"/><line x1="12" y1="8" x2="12" y2="8"/><path d="M12 12v4"/></svg>'),
]
 
 
# ─── EXTRA CSS (appended once, add to inject_css if you prefer) ───────────────
NAV_CSS = """
<style>
/* ── sidebar nav links ───────────────────────────────── */
.sb-nav-link {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 0.55rem 0.85rem;
  border-radius: 9px;
  border: 1px solid transparent;
  margin-bottom: 5px;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--sub);
  cursor: pointer;
  transition: all 0.15s ease;
  background: none;
  text-decoration: none !important;
  width: 100%;
  box-sizing: border-box;
  font-family: var(--font);
  line-height: 1;
}
.sb-nav-link:hover {
  color: var(--text) !important;
  background: var(--bg3) !important;
  text-decoration: none !important;
}
.sb-nav-link.active {
  color: var(--orange) !important;
  background: rgba(249,115,22,0.12) !important;
  border-color: rgba(249,115,22,0.28) !important;
}
.sb-nav-link svg {
  flex-shrink: 0;
  color: inherit;
  stroke: currentColor;
}
 
/* hide the real (invisible) Streamlit buttons used as click targets */
div[data-nav-hidden] { display: none !important; }
</style>
"""

# ─── TRANSLATIONS ─────────────────────────────────────────────────────────────
TRANSLATIONS = {
    "en": {
        "app_title": "FP Decision Autonomy",
        "app_subtitle": "ML-Powered Prediction Dashboard",
        "nav_predict": "Predict",
        "nav_performance": "Model Performance",
        "nav_features": "Feature Importance",
        "nav_about": "About",
        "lang_label": "Language",
        "pred_title": "Predict Decision Autonomy",
        "pred_subtitle": "Enter respondent characteristics to generate a real-time prediction.",
        "pred_personal": "Personal Characteristics",
        "pred_socio": "Socioeconomic Factors",
        "pred_fp": "Family Planning Context",
        "pred_media": "Media & Healthcare Access",
        "pred_btn": "Run Prediction",
        "pred_result_title": "Predicted Category",
        "pred_class_1": "Woman Alone",
        "pred_class_2": "Partner Alone",
        "pred_class_3": "Joint Decision",
        "pred_proba_title": "Category Probabilities",
        "pred_confidence": "Confidence",
        "pred_hero_title": "Women's Autonomy in Family Planning",
        "pred_hero_sub": "A machine learning study using DHS Cameroon data",
        "age": "Age (years)",
        "num_children": "Number of Children",
        "edu_woman": "Woman's Education Level",
        "edu_husband": "Husband's Education Level",
        "wealth": "Wealth Index",
        "residence": "Residence Type",
        "region": "Region",
        "religion": "Religion",
        "marital_status": "Marital Status",
        "marriage_type": "Marriage Type",
        "current_method": "Currently FP method used",
        "fertility_preference": "Fertility Preference",
        "woman_working": "Woman Currently Working",
        "husband_working": "Husband Currently Working",
        "fieldworker_fp": "Visited by FP Fieldworker",
        "facility_fp": "Visited FP Health Facility",
        "media_any": "Exposed to Any Media",
        "anc_group": "ANC Visits",
        "perf_title": "Model Performance",
        "perf_subtitle": "Test set evaluation metrics for LightGBM the best performing classifier.",
        "perf_accuracy": "Accuracy",
        "perf_f1": "F1 Score",
        "perf_precision": "Precision",
        "perf_recall": "Recall",
        "perf_cv": "CV F1 Score",
        "perf_cm_title": "Confusion Matrix",
        "perf_per_class": "Per-Class Breakdown",
        "perf_sample": "Total Samples",
        "perf_train": "Training Set",
        "perf_test": "Test Set",
        "feat_title": "Feature Importance",
        "feat_subtitle": "Key predictors ranked by LightGBM split gain.",
        "feat_note": "Importance scores reflect the model's split gain across all trees. Higher values indicate stronger predictive contribution.",
        "about_title": "About This Project",
        "about_research": "Research Overview",
        "about_desc": (
            "This dashboard is the product of an academic research project studying "
            "women's decision-making autonomy in family planning (FP) in Cameroon, "
            "using data from the Demographic and Health Survey (DHS). "
            "The goal is to predict whether FP decisions are driven by the woman alone, "
            "the partner alone, or jointly by both using 18 sociodemographic features. "
            "Seven machine learning classifiers were compared under 10-fold cross-validation, "
            "SMOTE class balancing, and RFE feature selection. LightGBM emerged as the best model."
        ),
        "about_dataset": "Dataset",
        "about_dataset_txt": "DHS Cameroon · 1,454 women · 18 features",
        "about_model_lbl": "Best Model",
        "about_model_txt": "LightGBM — F1: 0.537 · Accuracy: 57.0%",
        "about_links": "Links & Resources",
        "pred_result": "Prediction Result",
        "about_github": "GitHub Repository",
        "about_docs": "Full Report / Documentation",
        "about_dhs": "DHS Program (Data Source)",
        "about_author_title": "Author & Institution",
        "about_institution": "Institution",
        "about_supervisor": "Supervisor",
        "about_year": "Academic Year",
        "about_pipeline": "ML Pipeline",
        "husband_desired_children": "Husband Desired Preference for Children",
        "about_disclaimer": (
            "Academic disclaimer : This tool is intended for research purposes only "
            "and should not be used for clinical or policy decisions without further validation."
        ),
        "no_model_warn": "No model file found. Place best_model.pkl next to app.py. Showing demo probabilities.",
        "EDU_OPTIONS" : {0: "No Education", 1: "Primary", 2: "Secondary", 3: "Higher"},
        "HUSBAND_DESIRED" : {1: "Same as Woman", 2: "Wants more", 3: "Wants fewer", 8: "Don't Know"},
        "WEALTH_OPTIONS" : {1: "Poorest", 2: "Poorer", 3: "Middle", 4: "Richer", 5: "Richest"},
        "RESIDENCE_OPT" : {1: "Urban", 2: "Rural"},
        "RELIGION_OPT" : {1: "Catholic", 2: "Protestant", 3: "Muslim", 4: "Animist", 5: "Other Christian", 6: "No Religion"},
        "MARRIAGE_TYPE" : {1: "Monogamous", 2: "Polygamous"},
        "MARITAL_OPT" : {1: "Formally married", 2: "Cohabiting"},
        "FP_YN" : {0: "No", 1: "Yes"},
        "CURRENT_METHOD_OPT" : {1: 'Pill',
                                2: 'IUD',
                                3: 'Injections',
                                4: 'Diaphragm',
                                5: 'Male condom',
                                6: 'Female sterilization',
                                7: 'Male sterilization',
                                8: 'Periodic abstinence',
                                9: 'Withdrawal',
                                10: 'Other traditional',
                                11: 'Implants/Norplant',
                                12: 'Prolonged abstinence',
                                13: 'Lactational amenorrhea (LAM)',
                                14: 'Female condom',
                                15: 'Foam or jelly',
                                16: 'Emergency contraception',
                                17: 'Other modern method',
                                18: 'Standard days method (SDM)',
                                19: 'Specific method 1',
                                20: 'Specific method 2'}    ,
        "FERTILITY_OPT"  :{1: "Wants More", 2: "No More", 3: "Undecided"},
        "WORKING_OPT"    : {0: "No", 1: "Yes"},
        "FIELDWKR_OPT"   : {0: "No", 1: "Yes"},
        "FACILITY_OPT"   : {0: "No", 1: "Yes", 8: "Didn't visit facility"},
        "MEDIA_OPT"      : {0: "No", 1: "Yes"},
        "ANC_OPT"        : {0: "None", 1: "1–3 visits", 2: "4+ visits", 9: "No pregnancy in last 5 years"},
        "REGION_OPT"     : {
            1: "Adamaoua", 2: "Center (excl. Yaounde)", 3: "East", 4: "Far-North",
            5: "Littoral (excl. Douala)", 6: "North", 7: "North-West", 8: "West",
            9: "South", 10: "South-West", 11: "Douala", 12: "Yaounde",
        }
    },
    "fr": {
        "app_title": "Autonomie Décisionnelle PF",
        "app_subtitle": "Tableau de Bord de Prédiction ML",
        "nav_predict": "Prédire",
        "nav_performance": "Performance",
        "nav_features": "Importance des Variables",
        "nav_about": "À propos",
        "lang_label": "Langue",
        "pred_title": "Prédire l'Autonomie Décisionnelle",
        "pred_subtitle": "Renseignez les caractéristiques de la répondante pour obtenir une prédiction en temps réel.",
        "pred_personal": "Caractéristiques Personnelles",
        "pred_socio": "Facteurs Socioéconomiques",
        "pred_fp": "Contexte Planification Familiale",
        "pred_media": "Médias & Accès aux Soins",
        "pred_btn": "Lancer la Prédiction",
        "pred_result_title": "Catégorie Prédite",
        "pred_result": "Résultat de la Prédiction",
        "pred_class_1": "Femme seule",
        "pred_class_2": "Partenaire seul",
        "pred_class_3": "Décision conjointe",
        "pred_proba_title": "Probabilités par Catégorie",
        "pred_confidence": "Confiance",
        "pred_hero_title": "Autonomie des Femmes en Planification Familiale",
        "pred_hero_sub": "Une étude en apprentissage automatique sur les données EDS Cameroun",
        "age": "Âge (années)",
        "num_children": "Nombre d'enfants",
        "edu_woman": "Niveau d'éducation de la femme",
        "edu_husband": "Niveau d'éducation du mari",
        "wealth": "Indice de richesse",
        "residence": "Type de résidence",
        "region": "Région",
        "religion": "Religion",
        "marital_status": "Statut matrimonial",
        "marriage_type": "Type de mariage",
        "current_method": "Méthode PF Utilisée",
        "fertility_preference": "Préférence de fécondité",
        "woman_working": "Femme exerçant une activité",
        "husband_working": "Mari exerçant une activité",
        "fieldworker_fp": "Visite d'un agent PF",
        "facility_fp": "Visite d'un établissement PF",
        "media_any": "Exposition aux médias",
        "anc_group": "Visites CPN",
        "perf_title": "Performance du Modèle",
        "perf_subtitle": "Métriques d'évaluation sur l'ensemble de test LightGBM, meilleur classificateur.",
        "perf_accuracy": "Exactitude",
        "perf_f1": "Score F1",
        "perf_precision": "Précision",
        "perf_recall": "Rappel",
        "perf_cv": "F1 Validation Croisée",
        "perf_cm_title": "Matrice de Confusion",
        "perf_per_class": "Détail par Classe",
        "perf_sample": "Total Échantillons",
        "perf_train": "Entraînement",
        "perf_test": "Test",
        "feat_title": "Importance des Variables",
        "feat_subtitle": "Prédicteurs clés classés par gain de division LightGBM.",
        "feat_note": "Les scores reflètent le gain de division dans tous les arbres. Des valeurs plus élevées indiquent une contribution prédictive plus forte.",
        "about_title": "À Propos du Projet",
        "about_research": "Présentation de la Recherche",
        "about_desc": (
            "Ce tableau de bord est le produit d'un projet de recherche académique sur "
            "l'autonomie décisionnelle des femmes en planification familiale (PF) au Cameroun, "
            "à partir des données de l'Enquête Démographique et de Santé (EDS). "
            "L'objectif est de prédire si les décisions PF sont prises par la femme seule, "
            "le partenaire seul, ou conjointement en utilisant 18 variables sociodémographiques. "
            "Sept classificateurs ont été comparés avec validation croisée à 10 plis, "
            "équilibrage SMOTE et sélection RFE. LightGBM est le meilleur modèle."
        ),
        "about_dataset": "Données",
        "about_dataset_txt": "EDS Cameroun · 1 454 femmes · 18 variables",
        "about_model_lbl": "Meilleur Modèle",
        "about_model_txt": "LightGBM — F1 : 0.537 · Exactitude : 57.0 %",
        "about_links": "Liens & Ressources",
        "about_github": "Dépôt GitHub",
        "about_docs": "Rapport / Documentation",
        "about_dhs": "Programme DHS (Source des données)",
        "about_author_title": "Auteur & Institution",
        "about_institution": "Institution",
        "about_supervisor": "Directeur de mémoire",
        "about_year": "Année Académique",
        "about_pipeline": "Pipeline ML",
        "husband_desired_children": "Préférence du mari pour les enfants",
        "about_disclaimer": (
            "Avertissement académique : Cet outil est destiné à la recherche uniquement "
            "et ne doit pas être utilisé à des fins cliniques ou politiques sans validation supplémentaire."
        ),
        "no_model_warn": "Fichier modèle introuvable. Placez best_model.pkl à côté de app.py. Probabilités de démonstration affichées.",
        "EDU_OPTIONS" : {0: "Aucune instruction", 1: "Primaire", 2: "Secondaire", 3: "Supérieur"},
        "HUSBAND_DESIRED" : {1: "Même que la femme", 2: "En veut plus", 3: "En veut moins", 8: "Ne sait pas"},
        "WEALTH_OPTIONS" : {1: "Le plus pauvre", 2: "Pauvre", 3: "Moyen", 4: "Riche", 5: "Le plus riche"},
        "RESIDENCE_OPT" : {1: "Urbain", 2: "Rural"},
        "RELIGION_OPT" : {1: "Catholique", 2: "Protestant", 3: "Musulman", 4: "Animiste", 5: "Autre chrétien", 6: "Sans religion"},
        "MARRIAGE_TYPE" : {1: "Monogame", 2: "Polygame"},
        "MARITAL_OPT" : {1: "Marié(e) officiellement", 2: "En union libre"},
        "FP_YN" : {0: "Non", 1: "Oui"},
        "CURRENT_METHOD_OPT" : {
            1:  "Pilule",
            2:  "DIU / Stérilet",
            3:  "Injections",
            4:  "Diaphragme",
            5:  "Préservatif masculin",
            6:  "Stérilisation féminine",
            7:  "Stérilisation masculine",
            8:  "Abstinence périodique",
            9:  "Retrait",
            10: "Autre méthode traditionnelle",
            11: "Implants / Norplant",
            12: "Abstinence prolongée",
            13: "Méthode de l'allaitement maternel (MAMA)",
            14: "Préservatif féminin",
            15: "Mousse ou gelée spermicide",
            16: "Contraception d'urgence",
            17: "Autre méthode moderne",
            18: "Méthode des jours fixes (MJF)",
            19: "Méthode spécifique 1",
            20: "Méthode spécifique 2",
        },
        "FERTILITY_OPT"  : {1: "Veut plus d'enfants", 2: "Ne veut plus", 3: "Indécis(e)"},
        "WORKING_OPT"    : {0: "Non", 1: "Oui"},
        "FIELDWKR_OPT"   : {0: "Non", 1: "Oui"},
        "FACILITY_OPT"   : {0: "Non", 1: "Oui", 8: "N'a pas visité de structure"},
        "MEDIA_OPT"      : {0: "Non", 1: "Oui"},
        "ANC_OPT"        : {0: "Aucune", 1: "1–3 visites", 2: "4 visites ou plus", 9: "Pas de grossesse dans les 5 dernières années"},
        "REGION_OPT"     : {
            1:  "Adamaoua",
            2:  "Centre (hors Yaoundé)",
            3:  "Est",
            4:  "Extrême-Nord",
            5:  "Littoral (hors Douala)",
            6:  "Nord",
            7:  "Nord-Ouest",
            8:  "Ouest",
            9:  "Sud",
            10: "Sud-Ouest",
            11: "Douala",
            12: "Yaoundé",
        },
    },
}

# ─── OPTION MAPS ─────────────────────────────────────────────────────────────
EDU_OPTIONS    = {0: "No Education", 1: "Primary", 2: "Secondary", 3: "Higher"}
HUSBAND_DESIRED = {1: "Same as Woman", 2: "Wants more", 3: "Wants fewer", 8: "Don't Know"}
WEALTH_OPTIONS = {1: "Poorest", 2: "Poorer", 3: "Middle", 4: "Richer", 5: "Richest"}
RESIDENCE_OPT  = {1: "Urban", 2: "Rural"}
RELIGION_OPT   = {1: "Catholic", 2: "Protestant", 3: "Muslim", 4: "Animist", 5: "Other Christian", 6: "No Religion"}
MARRIAGE_TYPE    = {1: "Monogamous", 2: "Polygamous"}
MARITAL_OPT  = {1: "Formally married", 2: "Cohabiting"}
FP_YN          = {0: "No", 1: "Yes"}
CURRENT_METHOD_OPT  ={1: 'Pill',
                                2: 'IUD',
                                3: 'Injections',
                                4: 'Diaphragm',
                                5: 'Male condom',
                                6: 'Female sterilization',
                                7: 'Male sterilization',
                                8: 'Periodic abstinence',
                                9: 'Withdrawal',
                                10: 'Other traditional',
                                11: 'Implants/Norplant',
                                12: 'Prolonged abstinence',
                                13: 'Lactational amenorrhea (LAM)',
                                14: 'Female condom',
                                15: 'Foam or jelly',
                                16: 'Emergency contraception',
                                17: 'Other modern method',
                                18: 'Standard days method (SDM)',
                                19: 'Specific method 1',
                                20: 'Specific method 2'}    
FERTILITY_OPT  = {1: "Wants More", 2: "No More", 3: "Undecided"}
WORKING_OPT    = {0: "No", 1: "Yes"}
FIELDWKR_OPT   = {0: "No", 1: "Yes"}
FACILITY_OPT   = {0: "No", 1: "Yes", 8: "Didn't visit facility"}
MEDIA_OPT      = {0: "No", 1: "Yes"}
ANC_OPT        = {0: "None", 1: "1–3 visits", 2: "4+ visits", 9: "No pregnancy in last 5 years"}
REGION_OPT     = {
    1: "Adamaoua", 2: "Center (excl. Yaoundé)", 3: "East", 4: "Far-North",
    5: "Littoral (excl. Douala)", 6: "North", 7: "North-West", 8: "West",
    9: "South", 10: "South-West", 11: "Douala", 12: "Yaounde",
}

NUMERICAL_FEATURES   = ["age", "num_children"]
CATEGORICAL_FEATURES = [
    "residence", "edu_woman", "edu_husband", "religion", "region", "marital_status",
    "wealth", "marriage_type", "woman_working", "husband_working", "fertility_preference",
    "husband_desired_children",
    "anc_group", "fieldworker_fp", "facility_fp", "media_any",
]

# ─── UNSPLASH IMAGE URLS (free-to-use, no auth needed) ───────────────────────
IMG_HERO        = "https://static.wixstatic.com/media/038af1_c222df03e4bc408cad78ba79e2432e72~mv2.jpeg/v1/fill/w_1000,h_560,al_c,q_85,usm_0.66_1.00_0.01/038af1_c222df03e4bc408cad78ba79e2432e72~mv2.jpeg"
IMG_PREDICT_BG  = "https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=900&q=75"
IMG_ABOUT_1     = "https://cameroon.unfpa.org/sites/default/files/topics/pf_.jpeg"
IMG_ABOUT_2     = "old/brenda.jpg"

# ─── GLOBAL CSS (dark-only, orange accent) ───────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:       #0d0f14;
  --bg2:      #13161e;
  --bg3:      #1a1e2a;
  --card:     #1e2230;
  --border:   #2a2f42;
  --border2:  #363c54;
  --orange:   #f97316;
  --orange2:  #fb923c;
  --orange3:  #fdba74;
  --text:     #e8eaf2;
  --sub:      #8b92ab;
  --sub2:     #5e657a;
  --green:    #34d399;
  --blue:     #60a5fa;
  --red:      #f87171;
  --font:     'Plus Jakarta Sans', sans-serif;
  --mono:     'JetBrains Mono', monospace;
}

html, body, [class*="css"], .stApp { background-color: var(--bg); font-family: var(--font); color: var(--text); }

/* ── sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
# [data-testid="stElementContainer"] { display: none; !important}
[data-testid="stSidebar"] .block-container { padding: 0 !important; }

/* ── sidebar brand ───────────────────────────────────── */
.sb-brand {
  padding-bottom: 1.2rem;
  display: flex;
                flex-direction: column;
                gap:0;
  align-items: center;
  border-bottom: 1px solid var(--border);
}
.sb-brand .sb-logo {
  width: 40px; height: 40px; border-radius: 10px;
  background: var(--orange); display: flex; align-items: center;
  justify-content: center; margin-bottom: 0.2rem;
}
.sb-brand .sb-logo svg { width: 22px; height: 22px; color: #fff; }
.sb-brand h2 { font-size: 0.95rem; font-weight: 600; margin: 0 0 2px; color: var(--text); }
.sb-brand p  { font-size: 0.72rem; color: var(--sub); margin: 0; }

.stTabs [data-baseweb="tab-list"] {
        gap: 40px;
        margin-Top: 20px;
    }
.stTabs [data-baseweb="tab"] {
    height: 60px;
    background-color: transparent !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--orange) !important;
    
}
                
.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--orange) !important;border-radius: 10px;
}
    

/* ── predict button ──────────────────────────────────── */
.stButton > button {
  background: var(--orange) !important; color: #fff !important;
  border: none !important; border-radius: 10px !important;
  padding: 0.7rem 1.5rem !important; font-weight: 600 !important;
  font-size: 0.9rem !important; font-family: var(--font) !important;
  width: 100% !important; letter-spacing: 0.02em !important;
  transition: background 0.2s !important;
}
.stButton > button:hover { background: var(--orange2) !important; }


/* ── sb footer stats ─────────────────────────────────── */
.sb-stats {
  margin: 0 0.8rem 1rem;
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 10px; padding: 1rem;
}
.sb-stats .stat-row { display: flex; justify-content: space-between; margin-bottom: 6px; }
.sb-stats .stat-lbl { font-size: 0.71rem; color: var(--sub); }
.sb-stats .stat-val { font-size: 0.71rem; font-weight: 600; color: var(--orange); font-family: var(--mono); }

/* ── main area ───────────────────────────────────────── */
.block-container { padding: 0rem 1rem !important; margin: 0rem !important; }

/* ── hero banner ─────────────────────────────────────── */
.hero-banner {
  position: relative; height: 300px; overflow: hidden;
  border-radius: 0 0 20px 20px; margin-bottom: 0rem;
}
.hero-banner img {
  width: 100%; height: 100%; object-fit: cover; object-position: center 35%;
  filter: brightness(0.50);
}
.hero-overlay {
  position: absolute; inset: 0; display: flex;
                padding: 30px;
  flex-direction: column; justify-content: center;
  background: linear-gradient(90deg, rgba(13,15,20,0.7) 0%, transparent 100%);
}
.hero-overlay .hero-badge {
  display: inline-flex;
#  align-items: center; 
                gap: 6px;
  background: rgba(249,115,22,0.2); border: 1px solid rgba(249,115,22,0.4);
  color: var(--orange2); font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase;
  padding: 10px 12px; border-radius: 99px; margin-bottom: 0.8rem;width: fit-content;
                height: 40px
}
.hero-overlay h1 { font-size: 1.75rem; font-weight: 700; color: #fff; margin: 0 0 0.3rem; line-height: 1.25; }
.hero-overlay p  { font-size: 0.88rem; color: rgba(255,255,255,0.65); margin: 0; }

/* ── page wrapper (non-hero pages) ──────────────────── */
# .page-wrap { padding-top: 2.8rem; }

/* ── section heading ─────────────────────────────────── */
.sec-head { margin-bottom: 1.4rem; }
.sec-head h2 { font-size: 1.3rem; font-weight: 700; color: var(--text); margin: 0 0 4px; }
.sec-head p  { font-size: 0.85rem; color: var(--sub); margin: 0; }

/* ── cards ───────────────────────────────────────────── */
.card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
}
.card-title {
  display: flex; align-items: center; gap: 8px;
  font-size: 0.82rem; font-weight: 600; color: var(--orange);
  letter-spacing: 0.06em; text-transform: uppercase;
  margin-bottom: 1.1rem; padding-bottom: 0.7rem;
  border-bottom: 1px solid var(--border);
}
.card-title svg { width: 16px; height: 16px; }

/* ── metric cards ────────────────────────────────────── */
.metric-grid { display: flex; gap: 12px; margin-bottom: 1.2rem; flex-wrap: wrap; }
.metric-card {
  flex: 1; min-width: 110px;
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 1.1rem 1.3rem;
  text-align: center; transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--border2); }
.metric-card .mv {
  font-size: 1.65rem; font-weight: 700; color: var(--orange);
  font-family: var(--mono); letter-spacing: -0.03em;
  display: block; margin-bottom: 4px;
}
.metric-card .ml {
  font-size: 0.7rem; font-weight: 600; color: var(--sub);
  text-transform: uppercase; letter-spacing: 0.1em;
}

/* ── result box ──────────────────────────────────────── */
.result-box {
  background: rgba(249,115,22,0.08); border: 1.5px solid rgba(249,115,22,0.3);
  border-radius: 16px; padding: 2rem; text-align: center; margin-bottom: 1.4rem;
}
.result-box .rb-label { font-size: 0.72rem; font-weight: 600; color: var(--sub); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.5rem; }
.result-box .rb-class { font-size: 2rem; font-weight: 700; color: var(--orange); margin: 0 0 0.4rem; }
.result-box .rb-conf  { font-size: 0.85rem; color: var(--sub); font-family: var(--mono); }

/* ── probability rows ────────────────────────────────── */
.proba-item { margin-bottom: 0.9rem; }
.proba-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
.proba-name { font-size: 0.82rem; font-weight: 500; color: var(--text); }
.proba-pct  { font-size: 0.82rem; font-weight: 600; font-family: var(--mono); color: var(--orange); }
.proba-track { height: 6px; background: var(--bg3); border-radius: 99px; overflow: hidden; }
.proba-fill  { height: 100%; border-radius: 99px; transition: width 0.6s cubic-bezier(.4,0,.2,1); }

/* ── form sections ───────────────────────────────────── */
.stSelectbox label, .stNumberInput label, .stSlider label {
  font-size: 0.8rem !important; color: var(--sub) !important; font-weight: 500 !important;
}
.stSelectbox > div > div, .stNumberInput > div > div > input {
  background: var(--bg3) !important; border-color: var(--border) !important; color: var(--text) !important;
}



/* ── about image ─────────────────────────────────────── */
.about-img-wrap {
  border-radius: 14px; overflow: hidden; height: 230px;
  border: 1px solid var(--border); margin-bottom: 1.2rem;
}
.about-img-wrap img { width: 100%; height: 100%; object-fit: cover; filter: brightness(0.8); }

/* ── pipeline steps ──────────────────────────────────── */
.pipeline-steps { display: flex; flex-direction: column; gap: 0; }
.pipe-step {
  display: flex; align-items: flex-start; gap: 12px;
  padding: 0.6rem 0;
}
.pipe-dot-wrap { display: flex; flex-direction: column; align-items: center; flex-shrink: 0; }
.pipe-dot { width: 28px; height: 28px; border-radius: 50%; background: rgba(249,115,22,0.15); border: 1.5px solid var(--orange); display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.pipe-dot svg { width: 13px; height: 13px; color: var(--orange); }
.pipe-line { width: 1.5px; flex: 1; min-height: 16px; background: var(--border); margin-top: 4px; }
.pipe-content { padding-bottom: 0.5rem; }
.pipe-content .pipe-title { font-size: 0.82rem; font-weight: 600; color: var(--text); }
.pipe-content .pipe-sub   { font-size: 0.74rem; color: var(--sub); margin-top: 1px; }
                

section[data-testid="stSidebar"] .stButton > button{
  display: flex !important; align-items: center !important; gap: 5px !important;background: none !important;justify-content: start !important;
  padding: 0.6rem 0.8rem !important; border-radius: 8px !important;
                
  font-size: 0.875rem !important; font-weight: 500 !important;
  color: var(--sub) !important; cursor: pointer !important; margin-bottom: 2px !important;
  transition: all 0.15s ease !important; border: 1px solid transparent !important;
}
# .sb-nav-item:hover ,section[data-testid="stSidebar"] .stButton > button:hover{ color: var(--text) !important; background: var(--bg3) !important; }
# .sb-nav-item.active ,section[data-testid="stSidebar"] .stButton > button.active{
#   color: var(--orange) !important; background: rgba(249,115,22,0.12) !important;
#   border-color: rgba(249,115,22,0.25) !important;
# }
# .sb-nav-item svg { width: 17px; height: 17px; flex-shrink: 0; }

/* ── links ───────────────────────────────────────────── */
.link-item {
  display: flex; align-items: center; gap: 8px;
  padding: 0.55rem 0; border-bottom: 1px solid var(--border);
  font-size: 0.84rem; color: var(--orange); text-decoration: none;
}
.link-item:last-child { border-bottom: none; }
.link-item svg { width: 15px; height: 15px; flex-shrink: 0; }

/* ── warning / disclaimer ────────────────────────────── */
.disclaimer {
  background: rgba(249,115,22,0.06); border: 1px solid rgba(249,115,22,0.2);
  border-radius: 10px; padding: 0.9rem 1.1rem;
  font-size: 0.78rem; color: var(--orange3); margin-top: 0.6rem;
  display: flex; gap: 8px;
}
.disclaimer svg { width: 15px; height: 15px; flex-shrink: 0; margin-top: 1px; }

/* ── scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 99px; }

/* hide default streamlit chrome ──────────────────────── */
MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
# header    { visibility: hidden; }
# [data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ─── ICONS ───────────────────────────────────────────────────────────────────
def icon(name):
    icons = {
        "predict":    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>',
        "chart":      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M8 17V13M12 17V9M16 17V12"/></svg>',
        "star":       '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>',
        "info":       '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><line x1="12" y1="8" x2="12" y2="8"/><path d="M12 12v4"/></svg>',
        "logo":       '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 12h6M9 16h6M9 8h2M5 3h14a2 2 0 012 2v14a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2z"/><circle cx="17" cy="8" r="3" fill="currentColor" stroke="none"/></svg>',
        "user":       '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/></svg>',
        "building":   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18M9 9h6M9 13h6M9 17h6"/></svg>',
        "heart":      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M20.84 4.61a5.5 5.5 0 00-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 00-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 000-7.78z"/></svg>',
        "media":      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="2" y="4" width="20" height="14" rx="2"/><path d="M8 4v4M16 4v4M2 12h20M7 16h.01M12 16h.01M17 16h.01"/></svg>',
        "link":       '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71"/></svg>',
        "alert":      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        "db":         '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg>',
        "cpu":        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/></svg>',
        "filter":     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>',
        "balance":    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M16 16l3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1z"/><path d="M2 16l3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1z"/><path d="M7 21h10M12 3v18M3 7h2.5M18.5 7H21"/></svg>',
        "trophy":     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M6 9H4.5a2.5 2.5 0 000 5H6"/><path d="M18 9h1.5a2.5 2.5 0 010 5H18"/><path d="M4 22h16M9 22V18M15 22V18M12 18c-4-1-6-5-6-9V5h12v4c0 4-2 8-6 9z"/></svg>',
        "external":   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>',
    }
    return icons.get(name, "")


# ─── LOAD RESOURCES ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    paths = [
        os.path.join(base_dir, "best_model.pkl"),
        os.path.join(base_dir, "model", "best_model.pkl"),
        "best_model.pkl",
        "model/best_model.pkl",
    ]
    
    for p in paths:
        if os.path.exists(p):
            return joblib.load(p)
    
    st.error("Model file not found. Please check your deployment.")
    return None


@st.cache_data
def load_results():
    default = {
        "best_model": "LightGBM (LGBM)",
        "cv_performance": {"mean_f1": 0.5157, "std_f1": 0.0277},
        "test_performance": {"accuracy": 0.5704, "f1_score": 0.5371, "precision": 0.5357, "recall": 0.5704},
        "confusion_matrix": [[23, 3, 53], [8, 9, 29], [19, 13, 134]],
        "sample_size": 1454, "train_size": 1163, "test_size": 291,
    }
    for p in ["model_results.json", os.path.join(os.path.dirname(__file__), "model_results.json")]:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return default


def build_input_df(vals, model=None):
    print("Building input DataFrame from values: ", vals)
    
    # Defaults in case preprocessor loading fails
    num_cols = ["age", "num_children"]
    cat_cols = [
        "residence", "edu_woman", "edu_husband", "religion", "region", "marital_status",
        "wealth", "marriage_type", "woman_working", "husband_working", "fertility_preference",
         "husband_desired_children",
        "anc_group", "fieldworker_fp", "facility_fp", "media_any"
    ]
    
    # Retrieve feature names from the trained preprocessor dynamically
    if model is not None:
        try:
            if hasattr(model, 'named_steps') and 'preprocess' in model.named_steps:
                prep = model.named_steps['preprocess']
                num_cols = prep.transformers[0][2]
                cat_cols = prep.transformers[1][2]
            elif os.path.exists("preprocessor.pkl"):
                prep = joblib.load("preprocessor.pkl")
                num_cols = prep.transformers[0][2]
                cat_cols = prep.transformers[1][2]
        except Exception as e:
            print(f"Error loading preprocessor features: {e}")
            
    row = {f: vals.get(f, 28 if f == "age" else 2) for f in num_cols}
    for f in cat_cols:
        row[f] = vals.get(f, 0)
        
    return pd.DataFrame([row])


# ─── PAGE: PREDICT ────────────────────────────────────────────────────────────
def page_predict(T, model):
    CLASSES = [T["pred_class_1"], T["pred_class_2"], T["pred_class_3"]]
    BAR_COLORS = ["#f97316", "#fb923c", "#fdba74"]

    # ── Form
    st.markdown('<div class="page-wrap" style="padding-top:1.5rem">', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-head"><h2>{T["pred_title"]}</h2><p>{T["pred_subtitle"]}</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(f'<div class="card"><div class="card-title">{icon("user")} {T["pred_personal"]}</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        age          = r1.number_input(T["age"], 15, 49, 28, key="age")
        num_children = r2.number_input(T["num_children"], 0, 15, 2, key="nc")
        r3, r4 = st.columns(2)
        edu_w  = r3.selectbox(T["edu_woman"],   list(T["EDU_OPTIONS"].keys()),   format_func=lambda x: T["EDU_OPTIONS"][x], key="edw")
        edu_h  = r4.selectbox(T["edu_husband"], list(T["EDU_OPTIONS"].keys()),   format_func=lambda x: T["EDU_OPTIONS"][x], key="edh")
        r5, r6 = st.columns(2)
        religion  = r5.selectbox(T["religion"],     list(T["RELIGION_OPT"].keys()), format_func=lambda x: T["RELIGION_OPT"][x], key="rel")
        marital_s = r6.selectbox(T["marital_status"], list(T["MARITAL_OPT"].keys()), format_func=lambda x: T["MARITAL_OPT"][x], key="mar")
        marriage_t = st.selectbox(T["marriage_type"], list(T["MARRIAGE_TYPE"].keys()), format_func=lambda x: T["MARRIAGE_TYPE"][x], key="mart")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="card"><div class="card-title">{icon("building")} {T["pred_socio"]}</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        wealth    = r1.selectbox(T["wealth"],    list(T["WEALTH_OPTIONS"].keys()), format_func=lambda x: T["WEALTH_OPTIONS"][x], key="wlth")
        residence = r2.selectbox(T["residence"], list(T["RESIDENCE_OPT"].keys()),  format_func=lambda x: T["RESIDENCE_OPT"][x], key="res")
        region    = st.selectbox(T["region"],    list(T["REGION_OPT"].keys()),     format_func=lambda x: T["REGION_OPT"][x], key="reg")
        r3, r4 = st.columns(2)
        woman_w   = r3.selectbox(T["woman_working"],   [0, 1],    format_func=lambda x: T["FP_YN"][x], key="ww")
        husband_w = r4.selectbox(T["husband_working"], [0, 1], format_func=lambda x: T["WORKING_OPT"][x], key="hw")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="card"><div class="card-title">{icon("heart")} {T["pred_fp"]}</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        
        fertility_p = r1.selectbox(T["fertility_preference"], list(T["FERTILITY_OPT"].keys()), format_func=lambda x: T["FERTILITY_OPT"][x], key="fp")
        anc         = r2.selectbox(T["anc_group"], [0, 1, 2, 9], format_func=lambda x: T["ANC_OPT"][x], key="anc")
        husband_desired = st.selectbox(T["husband_desired_children"], list(T["HUSBAND_DESIRED"].keys()), format_func=lambda x: T["HUSBAND_DESIRED"][x], key="hd")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="card"><div class="card-title">{icon("media")} {T["pred_media"]}</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        fieldworker = r1.selectbox(T["fieldworker_fp"], [0, 1], format_func=lambda x: T["FIELDWKR_OPT"][x], key="fw")
        facility    = r2.selectbox(T["facility_fp"],    [0, 1, 8], format_func=lambda x: T["FACILITY_OPT"][x], key="fac")
        media       = st.selectbox(T["media_any"],      [0, 1],    format_func=lambda x: T["MEDIA_OPT"][x], key="med")
        st.markdown('</div>', unsafe_allow_html=True)

        # Predict button
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button(T["pred_btn"], use_container_width=True, key="pred_btn")

        if predict_btn:
            vals = {
                "age": age, "num_children": num_children,
                "edu_woman": edu_w, "edu_husband": edu_h,
                "wealth": wealth, "residence": residence,
                "region": region, "religion": religion,
                "marital_status": marital_s, "marriage_type": marriage_t,
                 "fertility_preference": fertility_p,
                "woman_working": woman_w,
                "husband_working": husband_w,
                "fieldworker_fp": fieldworker, "facility_fp": facility,
                "media_any": media, "anc_group": anc,
                "husband_desired_children": husband_desired,
            }
            if model is None:
                st.sidebar.error("❌ Model NOT loaded - using demo values")
                st.warning(T["no_model_warn"])
                probabilities = np.array([0.27, 0.16, 0.57])
                predicted_class = 2
            else:
                df_in = build_input_df(vals, model)
                try:
                    predicted_class = int(model.predict(df_in)[0])
                    probabilities   = model.predict_proba(df_in)[0]
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    print(f"PREDICTION ERROR: {e}") 
                    probabilities = np.array([0.33, 0.33, 0.34])
                    predicted_class = 2
                    st.exception(e)
                    st.stop()

            st.session_state.last_proba  = list(probabilities)
            st.session_state.last_pred   = predicted_class
            st.session_state.show_result = True
            # Scroll to top where results are shown
            st.markdown("""
            <script>
            window.parent.document.querySelector('section.main').scrollTo({top: 0, behavior: 'smooth'});
            </script>
            """, unsafe_allow_html=True)
            st.rerun()
    

    st.markdown('</div>', unsafe_allow_html=True)

    
    # Show result panel if prediction was made
    if st.session_state.get("show_result") and "last_proba" in st.session_state:
        print("Displaying prediction results...")
        print("Session Data: ", st.session_state)
        probs = st.session_state.last_proba
        pred  = st.session_state.last_pred
        conf  = float(probs[pred]) * 100

        st.markdown('<div class="page-wrap" style="padding-top:0">', unsafe_allow_html=True)
        st.markdown(f'<div class="sec-head"><h2>{T["pred_result"]}</h2></div>', unsafe_allow_html=True)
        col_rb, col_pb = st.columns([1, 1.2], gap="large")

        with col_rb:
            st.markdown(f"""
            <div class="result-box">
                <div class="rb-label">{T['pred_result_title']}</div>
                <div class="rb-class">{CLASSES[pred]}</div>
                <div class="rb-conf">{T['pred_confidence']}: {conf:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Donut chart
            fig = go.Figure(go.Pie(
                labels=CLASSES,
                values=probs,
                hole=0.65,
                marker_colors=["#f97316", "#fb923c", "#34d399"],
                textinfo="none",
                hovertemplate="%{label}: %{percent:.1%}<extra></extra>",
            ))
            fig.update_layout(
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0),
                height=180,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8b92ab", family="Plus Jakarta Sans"),
                annotations=[dict(text=f"{conf:.0f}%", x=0.5, y=0.5, font_size=20,
                                  font_color="#f97316", showarrow=False, font_family="JetBrains Mono")],
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_pb:
            st.markdown(f'<div class="card"><div class="card-title">{icon("chart")} {T["pred_proba_title"]}</div>', unsafe_allow_html=True)
            for cls, prob, color in zip(CLASSES, probs, BAR_COLORS):
                pct = prob * 100
                st.markdown(f"""
                <div class="proba-item">
                    <div class="proba-header">
                        <span class="proba-name">{cls}</span>
                        <span class="proba-pct">{pct:.1f}%</span>
                    </div>
                    <div class="proba-track">
                        <div class="proba-fill" style="width:{pct:.1f}%; background:{color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<hr style='border-color:var(--border); margin:0.5rem 2.2rem 0;'>", unsafe_allow_html=True)

     


# ─── PAGE: PERFORMANCE ────────────────────────────────────────────────────────
def page_performance(T, results):
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-head"><h2>{T["perf_title"]}</h2><p>{T["perf_subtitle"]}</p></div>', unsafe_allow_html=True)

    tp = results["test_performance"]
    cv = results["cv_performance"]

    # Metrics
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card"><span class="mv">{tp['accuracy']*100:.1f}%</span><span class="ml">{T['perf_accuracy']}</span></div>
        <div class="metric-card"><span class="mv">{tp['f1_score']:.4f}</span><span class="ml">{T['perf_f1']}</span></div>
        <div class="metric-card"><span class="mv">{tp['precision']:.4f}</span><span class="ml">{T['perf_precision']}</span></div>
        <div class="metric-card"><span class="mv">{tp['recall']:.4f}</span><span class="ml">{T['perf_recall']}</span></div>
        <div class="metric-card"><span class="mv">{cv['mean_f1']:.3f}</span><span class="ml">{T['perf_cv']} ±{cv['std_f1']:.3f}</span></div>
    </div>
    """, unsafe_allow_html=True)

    col_cm, col_bar = st.columns([1, 1.1], gap="large")
    LABELS = [T["pred_class_1"], T["pred_class_2"], T["pred_class_3"]]
    cm = np.array(results["confusion_matrix"])

    with col_cm:
        st.markdown(f'<div class="card"><div class="card-title">{icon("chart")} {T["perf_cm_title"]}</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Heatmap(
            z=cm, x=LABELS, y=LABELS,
            colorscale=[[0, "rgba(249,115,22,0.04)"], [1, "#f97316"]],
            text=cm, texttemplate="<b>%{text}</b>",
            textfont={"size": 15, "color": "white", "family": "JetBrains Mono"},
            showscale=False, hoverongaps=False,
        ))
        fig.update_layout(
            xaxis=dict(title="Predicted", title_font_color="#8b92ab", tickfont_color="#8b92ab"),
            yaxis=dict(title="Actual",    title_font_color="#8b92ab", tickfont_color="#8b92ab"),
            margin=dict(t=10, b=40, l=70, r=10), height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b92ab", family="Plus Jakarta Sans"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_bar:
        st.markdown(f'<div class="card"><div class="card-title">{icon("chart")} {T["perf_per_class"]}</div>', unsafe_allow_html=True)
        prec = cm.diagonal() / cm.sum(axis=0).clip(1)
        rec  = cm.diagonal() / cm.sum(axis=1).clip(1)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)

        fig2 = go.Figure()
        for lbl, p, r, f, color in zip(LABELS, prec, rec, f1, ["#f97316","#fb923c","#34d399"]):
            fig2.add_trace(go.Bar(
                name=lbl, x=["Precision", "Recall", "F1"],
                y=[p, r, f],
                marker_color=color,
                text=[f"{v:.2f}" for v in [p, r, f]],
                textposition="outside",
                textfont=dict(size=11, color="#8b92ab"),
            ))
        fig2.update_layout(
            barmode="group", height=300,
            margin=dict(t=10, b=40, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b92ab", family="Plus Jakarta Sans"),
            legend=dict(orientation="h", yanchor="top", y=-0.15, x=0, font_size=11),
            yaxis=dict(range=[0, 1.25], gridcolor="#2a2f42", zeroline=False),
            xaxis=dict(tickfont_color="#8b92ab"),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Dataset info
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card"><span class="mv">{results['sample_size']:,}</span><span class="ml">{T['perf_sample']}</span></div>
        <div class="metric-card"><span class="mv">{results['train_size']:,}</span><span class="ml">{T['perf_train']} (80%)</span></div>
        <div class="metric-card"><span class="mv">{results['test_size']:,}</span><span class="ml">{T['perf_test']} (20%)</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def getShapImportance(isImportance=False):
    BASE_DIR = Path(__file__).parent

    shap_file = BASE_DIR / "shap_importance.csv"
    shap_data = pd.read_csv("shap_importance.csv")
    # Extract original feature name
    shap_data['original_feature'] = shap_data['feature'].apply(lambda x: x.split('__')[1].rsplit('_', 1)[0] if '__' in x else x)
    
    if not isImportance:
        return shap_data.sort_values('importance', ascending=False)[['feature', 'importance']]

    # Sum importance for each original feature
    grouped = shap_data.groupby('original_feature')['importance'].sum().sort_values(ascending=False)
    return grouped


# ─── PAGE: FEATURES ───────────────────────────────────────────────────────────
def page_features(T, model):
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-head"><h2>{T["feat_title"]}</h2><p>{T["feat_subtitle"]}</p></div>', unsafe_allow_html=True)

    # Try real importances
    importances, feature_names = None, None

    if model is not None:
        try:
            steps = model.named_steps

            # 1. Get classifier safely (LAST step is correct)
            clf = list(steps.values())[-1]

            # 2. Get preprocessor safely
            pre = steps.get("preprocess", None) or steps.get("preprocessor", None)

            # 3. Get feature names safely
            if pre is not None and hasattr(pre, "get_feature_names_out"):
                feature_names = pre.get_feature_names_out()

            # 4. Get importances (ONLY for tree models)
            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_

        except Exception as e:
            st.warning(f"Feature importance extraction failed: {e}")
            importances = None
    # Aggregate one-hot back to original names
    orig = {}
    for name, imp in zip(feature_names, importances):
        clean = name.split("__")[-1]
        matched = False
        for f in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
            if clean.startswith(f):
                orig[f] = orig.get(f, 0) + imp
                matched = True
                break
        if not matched:
            orig[clean] = orig.get(clean, 0) + imp

    names_s = sorted(orig, key=orig.get, reverse=True)
    vals_s  = [orig[n] for n in names_s]
    total   = sum(vals_s) or 1
    vals_n  = [v / total * 100 for v in vals_s]

    n = len(names_s)
    bar_colors = ["#f97316" if i < 3 else "#fb923c" if i < 7 else "#2a2f42" for i in range(n)]

    # Get SHAP importances
    shap_grouped = getShapImportance(isImportance=True)
    shap_raw = getShapImportance(isImportance=False)

    # If SHAP returned grouped features, use those
    if isinstance(shap_grouped, pd.Series) and len(shap_grouped) > 0:
        names_s = shap_grouped.index.tolist()
        vals_s = shap_grouped.values.tolist()
    else:
        # Fallback: use model's built-in feature_importances_
        if model is not None:
            try:
                steps = model.named_steps
                clf = list(steps.values())[-1]
                pre = steps.get("preprocess", None) or steps.get("preprocessor", None)
                
                if pre is not None and hasattr(pre, "get_feature_names_out"):
                    feature_names = pre.get_feature_names_out()
                
                if hasattr(clf, "feature_importances_"):
                    importances = clf.feature_importances_
                
                # Aggregate one-hot back to original names
                orig = {}
                for name, imp in zip(feature_names, importances):
                    clean = name.split("__")[-1]
                    matched = False
                    for f in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
                        if clean.startswith(f):
                            orig[f] = orig.get(f, 0) + imp
                            matched = True
                            break
                    if not matched:
                        orig[clean] = orig.get(clean, 0) + imp
                
                names_s = sorted(orig, key=orig.get, reverse=True)
                vals_s = [orig[n] for n in names_s]
            except Exception as e:
                st.warning(f"Feature importance extraction failed: {e}")
                names_s, vals_s = [], []
        else:
            names_s, vals_s = [], []

    # Continue with plotting
    total = sum(vals_s) or 1
    vals_n = [v / total * 100 for v in vals_s]
    # ... rest of your plotting code

    fig = go.Figure(go.Bar(
        y=names_s[::-1], x=vals_n[::-1],
        orientation="h",
        marker_color=bar_colors[::-1],
        text=[f"{v:.1f}%" for v in vals_n[::-1]],
        textposition="outside",
        textfont=dict(size=11, color="#8b92ab", family="JetBrains Mono"),
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        height=max(380, n * 32),
        margin=dict(t=10, b=10, l=10, r=70),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b92ab", family="Plus Jakarta Sans"),
        xaxis=dict(showgrid=True, gridcolor="#2a2f42", zeroline=False, tickfont_color="#8b92ab"),
        yaxis=dict(showgrid=False, tickfont_color="#e8eaf2", tickfont_size=12),
    )

    st.markdown(f'<div class="card"><div class="card-title">{icon("star")} {T["feat_title"]}</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(f'<div class="disclaimer">{icon("alert")} {T["feat_note"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─── PAGE: ABOUT ─────────────────────────────────────────────────────────────
PIPELINE_STEPS = [
    ("db",      "DHS Cameroon Dataset",         "1,533 women · 25 raw variables"),
    ("filter",  "Preprocessing & Cleaning",     "Missing values, outlier removal (LOF 5%)"),
    ("filter",  "RFE Feature Selection",        "18 features selected via Logistic Regression"),
    ("cpu",     "10-Fold Cross-Validation",      "Compared 7 ML algorithms"),
    ("trophy",  "Best Model: LightGBM",         "F1 0.537 · Accuracy 55.0%"),
]

def get_base64(file):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, file)
    with open(full_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def page_about(T):
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-head"><h2>{T["about_title"]}</h2></div>', unsafe_allow_html=True)

    col_main, col_side = st.columns([1.5, 1], gap="large")

    with col_main:
        # Cover image
        st.markdown(f"""
        <div class="about-img-wrap">
            <img src="{IMG_ABOUT_1}" alt="Research"/>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="card"><div class="card-title">{icon("info")} {T["about_research"]}</div><p style="font-size:0.875rem;color:var(--sub);line-height:1.75;">{T["about_desc"]}</p>', unsafe_allow_html=True)

        # Dataset + Model chips
        st.markdown(f"""
        <div style="display:flex;gap:12px;margin-top:0.8rem;flex-wrap:wrap;">
            <div style="display:flex;align-items:center;gap:8px;background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:0.5rem 1rem;font-size:0.8rem;">
                {icon("db")}<div><div style="color:var(--sub);font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;">{T['about_dataset']}</div><div style="font-weight:600;color:var(--text)">{T['about_dataset_txt']}</div></div>
            </div>
            <div style="display:flex;align-items:center;gap:8px;background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:0.5rem 1rem;font-size:0.8rem;">
                {icon("trophy")}<div><div style="color:var(--sub);font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;">{T['about_model_lbl']}</div><div style="font-weight:600;color:var(--orange)">{T['about_model_txt']}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

             # Links
        st.markdown(f'<div class="card"><div class="card-title">{icon("link")} {T["about_links"]}</div>', unsafe_allow_html=True)
        for lbl, url in [
            (T["about_github"], "https://github.com/Enow-brenda/congenial-winner"),
            (T["about_docs"],   "https://github.com/Enow-brenda/congenial-winner/raw/refs/heads/main/Womens_Decisional_Autonomy_Research_Report.docx"),
            (T["about_dhs"],    "https://dhsprogram.com/"),
        ]:
            st.markdown(f'<a class="link-item" href="{url}" target="_blank">{icon("external")} {lbl}</a>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="disclaimer">{icon("alert")} {T["about_disclaimer"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        

    with col_side:
        # Author card
        img = get_base64("old/brenda.jpg")
        st.markdown(f"""
        <div class="about-img-wrap" style="height:20rem;">
            <img src="data:image/jpg;base64,{img}" alt="Academic"/>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="card"><div class="card-title">{icon("user")} {T["about_author_title"]}</div>', unsafe_allow_html=True)
        for label, val in [
            ("Name", "ENOW EWEH MAC BRENDA"),
            (T["about_institution"], "INSTITUT SAINT JEAN"),
            # (T["about_supervisor"], "[Supervisor Name]"),
            ("Programme", "MASTER IN DATA SCIENCE"),
            (T["about_year"], "2025 – 2026"),
        ]:
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:0.35rem 0;border-bottom:1px solid var(--border);font-size:0.8rem;"><span style="color:var(--sub)">{label}</span><span style="font-weight:500;color:var(--text)">{val}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

   

        # Pipeline
        st.markdown(f'<div class="card"><div class="card-title">{icon("cpu")} {T["about_pipeline"]}</div><div class="pipeline-steps">', unsafe_allow_html=True)
        for i, (ic, title, sub) in enumerate(PIPELINE_STEPS):
            last = i == len(PIPELINE_STEPS) - 1
            line_html = '<div></div>' if last else '<div class="pipe-line"></div>'
            st.markdown(f"""
            <div class="pipe-step">
                <div class="pipe-dot-wrap">
                    <div class="pipe-dot">{icon(ic)}</div>
                    {line_html}
                </div>
                <div class="pipe-content">
                    <div class="pipe-title">{title}</div>
                    <div class="pipe-sub">{sub}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div></div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─── SIDEBAR NAV HACK (real buttons hidden, html nav shown above) ─────────────
def real_sidebar_nav(T, results):

    with st.sidebar:
        # Inject nav CSS once
        st.markdown(NAV_CSS, unsafe_allow_html=True)
 
        # ── Brand ────────────────────────────────────────────────────────────
        heart_svg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="22" height="22"><path d="M20.84 4.61a5.5 5.5 0 00-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 00-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 000-7.78z"/></svg>'
        st.markdown(f"""
        <div class="sb-brand">
            <div class="sb-logo">{heart_svg}</div>
            <h2>{T['app_title']}</h2>
            <p>{T['app_subtitle']}</p>
        </div>
        """, unsafe_allow_html=True)
 
        # ── Language selector ────────────────────────────────────────────────
        st.markdown('<div style="padding:0.4rem 0.4rem 0;">', unsafe_allow_html=True)
        lang_choice = st.selectbox(
            T["lang_label"],
            ["English", "Français"],
            index=0 if st.session_state.lang == "en" else 1,
            key="lang_select",
        )
        new_lang = "en" if lang_choice == "English" else "fr"
        if new_lang != st.session_state.lang:
            st.session_state.lang = new_lang
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
 
        # ── Nav links ────────────────────────────────────────────────────────
        # We render each nav item as a styled <a> that uses JS to push a query
        # param into the parent window URL, which triggers Streamlit to rerun.
        lang = st.session_state.lang
        st.markdown('<div style="padding-bottom:0.5rem ;">', unsafe_allow_html=True)
 
        # for key, label_en, label_fr, svg in NAV_ITEMS:
        #     label = label_en if lang == "en" else label_fr
        #     active_cls = "active" if page == key else ""
 
        #     # The <a> tag pushes ?page=KEY to the parent iframe URL via JS.
        #     # Streamlit detects the query param change and reruns automatically.
        #     html_code = f"""
        #     <a class="sb-nav-link {active_cls}"
        #     href="#"
        #     onclick="">
        #     {svg} {label}
        #     </a>
        #     """

        #     st.markdown(html_code, unsafe_allow_html=True)
            
 
        # st.markdown('</div>', unsafe_allow_html=True)
       
 
        # Hidden real Streamlit buttons — these fire Python-side page switches.
       
        # st.markdown('<div style="display:none">', unsafe_allow_html=True)
        # for key, label_en, label_fr, svg in NAV_ITEMS:
        #     label = label_en if lang == "en" else label_fr
        #     button_label = f"{label}"
        #     if st.button(button_label, key=f"nav_{key}"):
        #         st.session_state.page = key
        #         st.session_state.show_result = False
                
        #         st.query_params["page"] = key
        #         st.rerun()
        # st.markdown('</div>', unsafe_allow_html=True)
 
        # ── Footer stats ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="sb-stats">
            <div class="stat-row"><span class="stat-lbl">Best Model</span><span class="stat-val">{results['best_model']}</span></div>
            <div class="stat-row"><span class="stat-lbl">Accuracy</span><span class="stat-val">{results['test_performance']['accuracy']:.1%}</span></div>
            <div class="stat-row"><span class="stat-lbl">F1 Score</span><span class="stat-val">{results['test_performance']['f1_score']:.3f}</span></div>
            <div class="stat-row"><span class="stat-lbl">Dataset</span><span class="stat-val">DHS Cameroon</span></div>
            <div class="stat-row" style="margin-bottom:0"><span class="stat-lbl">Samples</span><span class="stat-val">{results['sample_size']:,}</span></div>
        </div>
        """, unsafe_allow_html=True)
           


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    # ── Init session state ───────────────────────────────────────────────────
    if "lang"        not in st.session_state: st.session_state.lang        = "en"
    if "show_result" not in st.session_state: st.session_state.show_result = False
 
    # ── Read page from query params FIRST (before session_state default) ─────
    qp = st.query_params
    if "page" not in st.session_state:
        st.session_state.page = qp.get("page", "predict")
    elif "page" in qp and qp["page"] != st.session_state.page:
        # URL changed (back/forward button or direct link)
        if qp["page"] in ("predict", "performance", "features", "about"):
            st.session_state.page = qp["page"]
 
    inject_css()
 
    lang = st.session_state.lang
    T    = TRANSLATIONS[lang]
    page = st.session_state.page
 
    # Keep URL in sync with session_state
    st.query_params["page"] = page
 
    model   = load_model()
    results = load_results()
 
    real_sidebar_nav(T,results)

     # Hero
    st.markdown(f"""
    <div class="hero-banner">
        <img src="{IMG_HERO}" alt="Women's health"/>
        <div class="hero-overlay">
            <div class="hero-badge">{icon("heart")} DHS Cameroon · LightGBM</div>
            <h1>{T['pred_hero_title']}</h1>
            <p>{T['pred_hero_sub']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs([T[p] for p in ["nav_predict", "nav_performance", "nav_features","nav_about"]])
 
   
    with tabs[0]:
        page_predict(T, model)
    with tabs[1]:
        page_performance(T, results)
    with tabs[2]:
        page_features(T, model)
    with tabs[3]:
        page_about(T)
 
 
if __name__ == "__main__":
    main()