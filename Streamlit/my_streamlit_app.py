import streamlit as st

# IMPORT DES MODULES

from IPython.display import display

import pandas as pd
import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import unicodedata


st.title('Hello Vidéo futur !!!')
st.write('Nous allons créer notre application de recommendation de film !')

link_df_ref_traite = "https://raw.githubusercontent.com/FabienONOLFO/WCS_Projet2/main/DB_REF_TRAITE.csv"
link_df_ref = "https://raw.githubusercontent.com/FabienONOLFO/WCS_Projet2/main/DF_REF.csv"

@st.cache()
def import_doc(link_df_ref):
	REF1 = pd.read_csv(link_df_ref)
	return REF1

@st.cache()
def import_doc2(link_df_ref_traite):
	REF1_traite = pd.read_csv(link_df_ref_traite)
	return REF1_traite

DF_REF = import_doc(link_df_ref)
DF_REF_TRAITE = import_doc2(link_df_ref_traite)

X = DF_REF.loc[:, DF_REF.columns!='title']


# CREATION D'UN RESET DE REFERENCE DU DF_REF
DF_RESET = DF_REF_TRAITE.copy()
Film_name = ""
while len(Film_name) <= 0:
	Film_name = st.text_input("Veuillez entrer un nom de film : ")
else:
	# Si il n'y a aucune ligne qui correspond au nom du film alors on affiche Film non présent
	if len(DF_RESET.loc[DF_RESET['title'].str.contains(Film_name,case=False)])==0:    # Condition vérifié à part , ça fonctionne bien !
	    st.write("Film non présent")
	# On créé une condition pour obliger d'avoir au moins 1 lettre
	else :
	    DF_REF_FILTRE = DF_RESET.loc[DF_RESET['title'].str.contains(Film_name,case=False)]

	    scaler = StandardScaler().fit(X)
	    X_scaled = scaler.transform(X)

	    DF_X_scaled = pd.DataFrame(X_scaled,columns=X.columns)
	    X_scaled = pd.DataFrame(X_scaled,columns=X.columns)
	    distanceKNN = NearestNeighbors(n_neighbors=5).fit(X_scaled)

	    DF_X_scaled['title'] = DF_REF_FILTRE['title']
	    DF_X_scaled['Annee'] = DF_REF_FILTRE['startYear']

	    if len(DF_RESET.loc[DF_RESET['title'].str.contains(Film_name,case=False)])>20:
	        # On selectionne une année pour filtrer car il y a trop de films
	        st.write("Voici les films trouvés : \n")
	        st.dataframe(DF_REF_FILTRE[['title','startYear','averageRating']].loc[DF_REF_FILTRE['title'].str.contains(Film_name,case=False)].sort_values(by=['startYear'],ascending=True))
	        filtre_annee = st.slider("Merci de choisir une année approximative(+-10Ans)",min_value=1900,max_value=2030,step=10) 
	        if filtre_annee > 1900 :
		        annee_up = filtre_annee + 10
		        annee_down = filtre_annee - 10
		        DF_REF_FILTRE = DF_REF_FILTRE.loc[DF_REF_FILTRE['startYear']<=annee_up]
		        DF_REF_FILTRE = DF_REF_FILTRE.loc[DF_REF_FILTRE['startYear']>=annee_down]    
		        st.write("Voici les films trouvés : \n")
		        DF_NEW = DF_REF_FILTRE[['title','startYear','averageRating']].loc[DF_REF_FILTRE['title'].str.contains(Film_name,case=False)].sort_values(by=['startYear'],ascending=True)
		        st.dataframe(DF_NEW)
		        Film_ref = 0
		        while Film_ref <= 0:
		        	# Permet la selection des index via une box selections
			        Film_ref = st.selectbox("Veuillez selectionner son index: ",(DF_NEW.index.to_list()))
			        neighbors = distanceKNN.kneighbors(DF_X_scaled.loc[DF_X_scaled.index==Film_ref,X_scaled.columns.tolist()])
			        st.write("\n Voici les films recommandés : \n")
			        st.dataframe(DF_REF[['title','startYear','averageRating','runtimeMinutes']].iloc[neighbors[1][0]].sort_values(by=['startYear'],ascending=True))

	    elif len(DF_RESET.loc[DF_RESET['title'].str.contains(Film_name,case=False)])==1:
	        st.write("Voici le films trouvé : \n")
	        st.dataframe(DF_REF_FILTRE[['title','startYear','averageRating']].loc[DF_REF_FILTRE['title'].str.contains(Film_name,case=False)])
	        Film_ref = DF_RESET.index[DF_RESET['title'].str.contains(Film_name,case=False)]
	        neighbors = distanceKNN.kneighbors(DF_X_scaled.loc[Film_ref,X_scaled.columns.tolist()])
	        st.write("\n Voici les films recommandés : \n")
	        st.dataframe(DF_REF[['title','startYear','averageRating','runtimeMinutes']].iloc[neighbors[1][0]].sort_values(by=['startYear'],ascending=True))  
	        
	    elif len(DF_RESET.loc[DF_RESET['title'].str.contains(Film_name,case=False)])<=20:
	        st.write("Voici les films trouvés : \n")
	        DF_NEW2 = DF_RESET[['title','startYear','averageRating']].loc[DF_RESET['title'].str.contains(Film_name,case=False)].sort_values(by=['startYear'],ascending=True)
	        st.dataframe(DF_NEW2)
	        Film_ref = 0
	        while Film_ref <= 0:
	        	Film_ref = st.selectbox("Veuillez selectionner son index: ",(DF_NEW2.index.to_list()))
	        else:
		        neighbors = distanceKNN.kneighbors(DF_X_scaled.loc[DF_X_scaled.index==Film_ref,X_scaled.columns.tolist()])
		        st.write("\n Voici les films recommandés : \n")
		        st.dataframe(DF_REF[['title','startYear','averageRating','runtimeMinutes']].iloc[neighbors[1][0]].sort_values(by=['startYear'],ascending=True))  

