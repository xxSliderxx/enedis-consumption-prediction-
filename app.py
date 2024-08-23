import streamlit as st
from eda import intro, ML,ML_explain
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import seaborn as sns
import re
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.decomposition import PCA
from io import BytesIO
import base64

# Background
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{background-color: #ffee78 !important;
                                            
                                            }}
[data-testid="stHeader"] {{
                        background: rgba(0,0,0,0);
                        }}
                        
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


# Sidebar's background 
st.markdown("""
<style>
    [data-testid="stSidebar"] {
                                background-image: url("https://img.freepik.com/free-photo/copy-space-paper-ligh-bulb_23-2148519480.jpg?t=st=1720197016~exp=1720200616~hmac=1588127bcac57e9f3416101fc19c457a27f0a130a01bda1c9d3bb84cdd2c06ce&w=2000");
                                background-size: cover;
                                background-position: left;
                                }                           
</style>
""", unsafe_allow_html=True)


# Make bigger and colored
change_font = '<p style="font-family:Courier; color:Blue; font-size: 20px;">In case want to change color of title</p>'
#st.markdown(change_font, unsafe_allow_html=True)




def main():
         
    menu = ["Introduction",'Prédiction', "Consumption Prediction",'pouet']

    choice = st.sidebar.selectbox("MENU", menu)

    if choice == "Introduction":
        intro()

    
    if choice == "Consumption Prediction":
        
        ML()
        
    if choice == 'Prédiction':
        ML_explain()
    if choice == 'pouet':
        font_title = {'family': 'sans-serif',
                                'color':  '#114b98',
                                'fontweight': 'bold'}
        DF = pd.read_csv('DF.csv', compression='gzip')
        color = {'P1 : ]0-12] kVA': 'yellow', 'P1: ]0-3] kVA': sns.color_palette()[0], 'P1: ]0-6] kVA':sns.color_palette()[1], 'P1: ]0-9] kVA':sns.color_palette()[2]
                 , 'P2: ]3-6] kVA':sns.color_palette()[3], 'P3: ]6-9] kVA':sns.color_palette()[4], 'P4: ]9-12] kVA':sns.color_palette()[5],
        'P5: ]12-15] kVA' : sns.color_palette()[6], 'P6: ]15-18] kVA' : sns.color_palette()[7], 'P6: ]15-36] kVA' : sns.color_palette()[8],
        'P7: ]18-24] kVA' : sns.color_palette()[9], 'P7: ]18-30] kVA': 'red', 'P7: ]18-36] kVA' : 'black', 'P8: ]24-30] kVA' : 'orange',
        'P9: ]30-36] kVA' : 'pink'}
        
        gb_plage_cvdl = DF[(DF['Code région']==24) & (DF['Plage de puissance souscrite'] != 'P0: Total <= 36 kVA' )].groupby('Plage de puissance souscrite')['Total énergie soutirée (MWh)'].agg('sum').sort_values(ascending = False).reset_index()
        gb_plage_hdf = DF[(DF['Code région']==32) & (DF['Plage de puissance souscrite'] != 'P0: Total <= 36 kVA' )].groupby('Plage de puissance souscrite')['Total énergie soutirée (MWh)'].agg('sum').sort_values(ascending = False).reset_index()
        gb_plage_cvdl['Total énergie soutirée (MWh)'] = gb_plage_cvdl['Total énergie soutirée (MWh)']/len(DF['Date'].unique())
        gb_plage_hdf['Total énergie soutirée (MWh)'] = gb_plage_hdf['Total énergie soutirée (MWh)']/len(DF['Date'].unique())
        
        fig,axs =plt.subplots(2,1,figsize=(12,12))
        fig.suptitle("Consumptions per day by power ranges", fontdict=font_title, fontsize = 22)
        ax2 = sns.barplot(data =gb_plage_cvdl,y ='Plage de puissance souscrite',x= 'Total énergie soutirée (MWh)',palette =color,ax =axs[0])
        ax2.set_ylabel('Power ranges')
        ax2.set_title('Profile : Centre-Val de Loire', pad=8, loc='left')
        ax2.set_xlabel('Total energy (Mwh)')


        ax1 = sns.barplot(data =gb_plage_hdf,y ='Plage de puissance souscrite',x= 'Total énergie soutirée (MWh)',palette =color,ax =axs[1])
        ax1.set_title('Profile : Hauts-de-France', pad=8, loc='left')
        ax1.set_ylabel('Power ranges')
        ax1.set_xlabel('Total energy (Mwh)')

        st.pyplot(fig)
  



main()
