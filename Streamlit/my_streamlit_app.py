import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
link = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/cars.csv"
df_car = pd.read_csv(link)
st.title("Données récupérées")
df_car

st.title("Correlation entre la puissance HP et le time-to-60")
viz_scatterplot = sns.scatterplot(data = df_car,
                                  y="time-to-60",
                                  x="hp")
st.pyplot(viz_scatterplot.figure, clear_figure=True)
st.write("Nous pouvons remarque la corrélation entre la puissance de la voiture et la rapidité de celle-ci à atteindre les 60miles/hours")

st.title("Heatmap")
viz_correlation = sns.heatmap(df_car.corr(), center=0, cmap = sns.color_palette("vlag", as_cmap=True))
st.pyplot(viz_correlation.figure, clear_figure=True)


st.title("Scatterplot by continent")
select = st.selectbox('Veuillez choisir le pays',df_car['continent'].unique())
viz_correlation2 = sns.scatterplot(data = df_car[df_car["continent"]==select], y="time-to-60", x="hp")
st.pyplot(viz_correlation2.figure)
