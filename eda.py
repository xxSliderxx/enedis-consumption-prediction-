import streamlit as st
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


def ana():
    @st.cache_data
    def Enedis():
        # Set up a dataframe for graphiques and change the mesure unit from Wh to KWh
        
        st.title("Regional Consupmtion Analysis")
        
        st.header('Presentation of the two regions on which our analysis will focus',divider='rainbow')
        
       
        change_font = '<p style="font-family:Corbel; color:#5d0076; font-size: 18px;">(Location of two regions in this project)</p>'
        
   
        st.image("intro.png")
        st.markdown("Source: enedis.fr")
        change_font = '<p style="font-family:Corbel; color:#5d0076; font-size: 25px;">DID YOU KNOW? 😎</p>'
        st.markdown(change_font, unsafe_allow_html=True)
        change_font = '<p style="font-family:Corbel; color:#5d0076; font-size: 18px;">"The average annual consumption of electricity and gas per household in France is respectively 4.5MWh and 9.8MWh in 2022."</p>'
        st.markdown(change_font, unsafe_allow_html=True)
        st.markdown("")
        
        
        # Basic info about 2 region

        st.subheader("🔹Hauts-de-France")
        st.markdown("""
        Hauts-de-France is the northernmost region of France. With 5,709,571 inhabitants as of 2018 and a population density of 189 inhabitants per km2, it is the third most populous region in France and the second-most densely populated in metropolitan France after its southern neighbour Île-de-France. 
                                 """)
        st.subheader("🔹Centre-Val de Loire")
        st.markdown("""
        Centre-Val de Loire or Centre Region, as it was known until 2015, is one of the eighteen administrative regions of France. It straddles the middle Loire Valley in the interior of the country, with a population of 2,380,990 as of 2018. Its prefecture is Orléans, and its largest city is Tours.
                                                 """) 
   
        ### Graphiques of introduction ###

        st.header("General information", divider='rainbow')
        st.markdown(" ")
        st.markdown("""
                - This graphic gives us an overview of total electricity consumption in 2 regions in 24 months. The purpose is not to compare the indicators of 2 areas but to observe and analyse the elements which could be important for our model after. 

                - The region HDF has 3,3 millions more inhabitants  than the region CVDL (2,39 times) but its total electricity consumption is 1.8 times higher than CVDL's.  In the other hand, HDF consums 7% less electricity for heating than CVDL. From this perspective, the size of house/ appartement, type of heating energy should be considered as very important factors in electricity consumption.                 
                                """)

        # Chart of total consumption
        
        Holiday = ['période 1','période 2','période 3','période 4','période 5','période 6','période 7','période 8','période 9']
        Vacances = {'période 1':'blue','période 2':'blue','période 3':'blue','période 4':'blue','période 5':'blue','période 6':'blue',
                        'période 7':'blue','période 8':'blue','période 9':'blue','période 20':'green','période 21':'green',
                        'période 22':'green','période 23':'green','période 24':'green','période 25':'green','période 26':'green',
                        'période 27':'green','période 28':'green','période 29':'green','période 30':'green','période 31':'green'}
        
        Jour_ferie = ['2022-04-18','2022-05-01','2022-05-08','2022-05-26','2022-06-06','2022-07-14','2022-08-15','2022-11-01','2022-11-11','2022-12-25',
                          '2023-01-01','2023-04-10','2023-05-01','2023-05-08','2023-05-29','2023-07-14','2023-08-15','2023-11-01','2023-11-11','2023-12-25',
                          '2024-01-01','2024-04-01']
        
        dg = pd.read_csv('dg.csv')
        
        
        
        # Set up a dictionary of font title
        font_title = {'family': 'sans-serif',
                                'color':  '#114b98',
                                'weight': 'bold'}
        
        
        fig, ax_bar = plt.subplots(figsize =(12,7))
        ax =sns.barplot(data = dg, x = 'Région', y='Total énergie soutirée (MWh)',  errorbar=None)
        for container in ax.containers:
                ax.bar_label(container, label_type="center", fmt="{:.0f} MWH",
                                color='#ffee78', fontsize=12, fontweight ='bold')

        ax_bar.set_title("Average electricity consumption per day", fontdict=font_title, fontsize = 22)
        ax_bar.set_xlabel(" " )
        ax_bar.set_ylabel("Total consumption in MWh")

        st.pyplot(fig)
        
        st.markdown("""
                    - Even if consumption is quiet different from the two regions, the shape of the two curves is the same.
                    
                    - In January consumption is at his highest level for both regions and in August it's at his lowest level.
                    
                    - Over the two years the shape of the curves is repeated.
                    
                    """)
        
        dg['Date'] = pd.to_datetime(dg['Date'])
        
        fig,axs =plt.subplots(figsize=(12,7))
        ax = sns.lineplot(data = dg,x = 'Date',y = "Total énergie soutirée (MWh)",hue = 'Région')
        ax.set_xlabel(' ')
        ax.set_ylabel("Total consumption in MWH")
        ax.set_title('Consumption by date', fontdict=font_title, fontsize = 22)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(0.90,0.89),ncol=2)
        ax.get_legend().remove()
        st.pyplot(fig)  

        
        st.header("Segment and sub-segment", divider='rainbow')
        st.markdown(" ")
        st.subheader("🔹Consumer Profile")
        st.markdown("""
                                - There are a total of 20 contract profiles in the dataset. We tried to understand if this information will be important, and we found that they are about consumer segment of Enedis. They are like a kind of client identity, so we can not encode this information but classify them into 3 groups: company, professional and residence.
                                So the segment 'Professional' occurences 50% of the data in both regions.

                                - Energy consumption depends on each purpose which is possible to be explained by the registered profile in the contract. We have to divide the rations into smaller portions depending on the profile in order to observe better the consumption trend of each profile category later.
          
                                                        """)

        DF = pd.read_csv('DF.csv')
        DF['Categorie'] = DF['Categorie'].replace({'Pro':'Professional','Res':'Residence','Ent':'Company'})
        
        
        
        fig,ax =plt.subplots(figsize=(6,4))
        ax = DF["Categorie"].value_counts().plot.pie(explode=[0, 0.1, 0.2],autopct='%1.1f%%',shadow=False)
        ax.set_title("Segment Distribution", fontweight ='bold')
        ax.set_ylabel('')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode("utf-8")
        html = f"<div style='text-align: center'><img src='data:image/png;base64,{data}'/></div>"


        st.markdown(html, unsafe_allow_html=True)
                
        
        
       
        
        
### ajout df et creation tableau pour heat map alex     
        
        d_pivot = DF[DF['Code région'] == 24].pivot_table(index = 'Profil',columns = 'Plage de puissance souscrite',values ='Nb points soutirage',
                                                                                                  aggfunc = 'mean').reset_index().set_index('Profil')
        d_pivot1 = DF[DF['Code région'] == 32].pivot_table(index = 'Profil',columns = 'Plage de puissance souscrite',values ='Nb points soutirage',
                                                                                                   aggfunc = 'mean').reset_index().set_index('Profil')

        d_pivot = d_pivot.drop('P0: Total <= 36 kVA',axis = 1).fillna(0)
        d_pivot1 = d_pivot1.drop('P0: Total <= 36 kVA',axis = 1).fillna(0)
        d_pivot  = d_pivot.drop([ 'ENT3 (+ ENT4 + ENT5)','ENT1 (+ ENT2)'])
        d_pivot1  = d_pivot1.drop([ 'ENT3 (+ ENT4 + ENT5)','ENT1 (+ ENT2)'])
        Res = []
        for j in range(len(d_pivot)):
                s =0
                for i in d_pivot.columns:
                        s+=d_pivot[i][j]
                Res.append(s)



        Res1 = []
        for j in range(len(d_pivot1)):
                s=0
                for i in d_pivot1.columns:
                        s+=d_pivot1[i][j]
                Res1.append(s)



        d_pivot['Total']=pd.Series()

        for i in range (len(d_pivot)):
                d_pivot['Total'][i] = Res[i]


        d_pivot1['Total']=pd.Series()

        for i in range (len(d_pivot1)):
                d_pivot1['Total'][i] = Res1[i]

        for j in range(len(d_pivot)):
                for i in ['P1 : ]0-12] kVA','P1: ]0-3] kVA','P1: ]0-6] kVA','P1: ]0-9] kVA','P2: ]3-6] kVA','P3: ]6-9] kVA','P4: ]9-12] kVA',
                        'P5: ]12-15] kVA','P6: ]15-18] kVA','P6: ]15-36] kVA','P7: ]18-24] kVA','P7: ]18-30] kVA','P7: ]18-36] kVA',
                        'P8: ]24-30] kVA','P9: ]30-36] kVA']:
                        d_pivot[i][j] = d_pivot[i][j]/d_pivot['Total'][j]*100

        for j in range(len(d_pivot1)):
                for i in ['P1 : ]0-12] kVA','P1: ]0-3] kVA','P1: ]0-6] kVA','P1: ]0-9] kVA','P2: ]3-6] kVA','P3: ]6-9] kVA','P4: ]9-12] kVA',
                        'P5: ]12-15] kVA','P6: ]15-18] kVA','P6: ]15-36] kVA','P7: ]18-24] kVA','P7: ]18-30] kVA','P7: ]18-36] kVA',
                        'P8: ]24-30] kVA','P9: ]30-36] kVA']:
                        d_pivot1[i][j] = d_pivot1[i][j]/d_pivot1['Total'][j]*100

        for i in d_pivot.columns:
                d_pivot[i] = d_pivot[i].apply(lambda x: '%.2f' % x)


        for i in d_pivot1.columns:
                d_pivot1[i] = d_pivot1[i].apply(lambda x: '%.2f' % x)

        for i in d_pivot.columns:
                d_pivot[i] = d_pivot[i].astype("float64")

        for i in d_pivot1.columns:
                d_pivot1[i] = d_pivot1[i].astype("float64")


        st.markdown("""
                                - The distribution of power ranges is different depending on the contract category. It can be seen that professionals generally need more power than residents.

                                - Despite slight differences between the two regions, the distribution of power ranges is similar.

                                - We can note that we do not have data for companies.

                                """)
        
        
        
 ###  heat map alex
        tab1, tab2 = st.tabs(["Centre-Val de Loire", "Hauts-de-France"])
        
        with tab1:
                fig,ax =plt.subplots(figsize =(12,7))
                ax = sns.heatmap(d_pivot1.drop('Total',axis=1), annot=True, cmap="Blues",annot_kws={"fontsize":8},vmin=0, vmax=100,linewidth=.8,cbar = False)
                ax.set_title('Distribution of profiles according to power ranges', fontdict=font_title, fontsize = 22)
                ax.set_ylabel(' ')
                ax.set_xlabel(' ')
                st.pyplot(fig)
                
                
                
                
                
        with tab2:        
                fig,ax =plt.subplots(figsize =(12,7))
                ax = sns.heatmap(d_pivot.drop('Total',axis=1), annot=True, cmap="Blues",annot_kws={"fontsize":8},vmin=0, vmax=100,linewidth=.8,cbar = False)
                ax.set_title('Distribution of profiles according to power ranges', fontdict=font_title, fontsize = 22)
                ax.set_xlabel(' ')
                ax.set_ylabel(' ')
                st.pyplot(fig)
        
        
               

        st.markdown(" ")
        st.subheader("🔹AVG consumption distribution according to contract profile and its power allows")
        
        st.markdown("The previous tables showed us that the distribution of power ranges was different depending on the different categories. The following graph shows us that although residents need less power, they are the ones who consume the most energy.")
        
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

        
        
        
        st.markdown("""
                                While the previouses charts permit us observing better the consumption behavior of each contract category, this graphic helps us explain the  portions of energy consumption which are distributed by each registered power allows of each profile. Brief, the registered power allows explains, in general, electricity needs of consumers. The larger the limit of power allows, the more energy will be extracted.""")
        

        # Répartition de la consommation quotidienne d'un point avec sa plage de puissance par région et année.
        
        
        d_tree = DF[DF['Plage'] != 2].groupby(['Date','Year','Région','Categorie','Profil'])['Total énergie soutirée (MWh)'].sum().reset_index()
        
        tree = d_tree.groupby(['Year','Région','Categorie','Profil'])['Total énergie soutirée (MWh)'].mean().reset_index()

        
        tree = tree.rename(columns={'Total énergie soutirée (MWh)': 'Energy (MWh)'})
        tree['Energy (MWh)'] = tree['Energy (MWh)'].round(2)
        
        
        tree_map = px.treemap(tree, path=[px.Constant("France"),"Year","Région", "Categorie", "Profil"], 
                                        values = "Energy (MWh)", color = "Energy (MWh)",
                                        color_continuous_scale='blues')

        tree_map.update_layout(margin = dict(t=50, l=25, r=25, b=25), 
                                        title=dict(text="Daily average regional consumption per contract with its power range", 
                                                                font=dict(size=20)))
        tree_map.update_traces(textinfo="label+text+value")

        st.plotly_chart(tree_map, theme="streamlit")


        st.header("Consumption Analysis", divider='rainbow')

        col1, col2 = st.columns(2)

        with col1:

                        st.image("hdf.png", width = 300)
                        change_font = '<p style="font-family:Corbel; color:#5d0076; font-size: 25px;">DID YOU KNOW? 😎</p>'
                        st.markdown(change_font, unsafe_allow_html=True)
                        change_font = '<p style="font-family:Corbel; color:#5d0076; font-size: 18px;">"Hauts-de-France is the most dynamic region in the development of collective self-consumption operations, with 38 operations in service in Q3/2023"</p>'
                        st.markdown(change_font, unsafe_allow_html=True)
                        #st.markdown("Did you kow that Hauts-de-France is the most dynamic region in the development of collective self-consumption operations, with 38 operations in service in Q3/2023")
  
        with col2:

                        st.image("cvdl.png", width = 300)
                        change_font = '<p style="font-family:Corbel; color:#5d0076; font-size: 25px;">DID YOU KNOW? 😎</p>'
                        st.markdown(change_font, unsafe_allow_html=True)
                        change_font = '<p style="font-family:Corbel; color:#5d0076; font-size: 18px;">"Centre-Val de Loire is the 2nd region which produces the most nuclear energy, with 62.92TWh in 2022"</p>'
                        st.markdown(change_font, unsafe_allow_html=True)
                        st.markdown("")



        #### Consumption analysis ####
        
        st.subheader("Evolution of consumption")

       

   


        ### GRAPHIQUES by region, create 2 tab to show regional chart ###

        ### Seasonal Trend ###
        
        
        
        dg['Date'] = dg['Date'].astype('str')
        season = dg.loc[(dg["Date"] >= "2023-03-21") & (dg["Date"] < "2024-03-01")]
        season['Date'] = pd.to_datetime(season['Date'])
        
        

   
        
        st.markdown("""
                           - Whatever the region chosen, consumption fluctuates enormously over time as we have seen previously.
                           - Through the differents seasons consumption is clearly different. Winter's consumption is the highest wheareas the Summer's one is the lowest.
                           - School holidays and public holidays have no influence on consumption.             
                           """)
        return(DF,dg,font_title,season)
    
    (DF,dg,font_title,season) = Enedis()
    
    Vacances = {'période 1':'blue','période 2':'blue','période 3':'blue','période 4':'blue','période 5':'blue','période 6':'blue',
                        'période 7':'blue','période 8':'blue','période 9':'blue','période 20':'green','période 21':'green',
                        'période 22':'green','période 23':'green','période 24':'green','période 25':'green','période 26':'green',
                        'période 27':'green','période 28':'green','période 29':'green','période 30':'green','période 31':'green'}
        
    Jour_ferie = ['2022-04-18','2022-05-01','2022-05-08','2022-05-26','2022-06-06','2022-07-14','2022-08-15','2022-11-01','2022-11-11','2022-12-25',
                          '2023-01-01','2023-04-10','2023-05-01','2023-05-08','2023-05-29','2023-07-14','2023-08-15','2023-11-01','2023-11-11','2023-12-25',
                          '2024-01-01','2024-04-01']
    
    
        
        
    with st.container():
        choice = st.multiselect(
        'Choose a specification :',
        options=['Seasonal','Public holiday', 'School holiday'],
        default= 'Seasonal'
        )
        if choice == ['Seasonal']:
            

            palette = {"Spring": "pink", "Summer": "green",
                        "Autumn": "#fec44f", "Winter": "#3182bd"}
            fig,axs =plt.subplots(2,1,figsize=(12,12))
            fig.suptitle("Seasonal consumption per point in MWh", fontdict=font_title, fontsize = 22)
            ax1 = sns.lineplot(data = season[season['Code région']==24],x= 'Date',y='Total énergie soutirée (MWh)',hue="Season",ax =axs[0],palette = palette)
            ax2 = sns.lineplot(data = season[season['Code région']==32],x= 'Date',y='Total énergie soutirée (MWh)',hue="Season",ax =axs[1],palette = palette)
            ax1.set_title('Profile : Centre-Val de Loire',  loc='left')
            ax2.set_title('Profile : Hauts-de-France',  loc='left')
            ax1.set_xlabel( ' ')
            ax1.set_ylabel('Total consumption in MWh')
            ax2.set_xlabel( ' ')
            ax2.set_ylabel('Total consumption in MWh')
            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(0.90,0.91),ncol=4)
            ax1.get_legend().remove()
            ax2.get_legend().remove()
            st.pyplot(fig) 
                    
            
            
        elif choice == ['Public holiday']:
            tab1, tab2 = st.tabs(["Centre-Val de Loire", "Hauts-de-France"])

            fig,axs =plt.subplots(2,1,figsize=(12,12))
            fig.suptitle("Consumption evolution by time in MWh", fontdict=font_title, fontsize = 22)
            ax1 = sns.lineplot(data = season[season['Code région']==24],x= 'Date',y='Total énergie soutirée (MWh)',ax =axs[0],color = 'green')
            sns.scatterplot(data = season[(season['Date'].isin(Jour_ferie)) & (season['Code région']==24)],x = 'Date',y = 'Total énergie soutirée (MWh)',
                ax =axs[0],s = 80, color = 'black')

            ax2 = sns.lineplot(data = season[season['Code région']==32],x= 'Date',y='Total énergie soutirée (MWh)',ax =axs[1],color = 'green')
            sns.scatterplot(data = season[(season['Date'].isin(Jour_ferie)) & (season['Code région']==32)],x = 'Date',y = 'Total énergie soutirée (MWh)',
                ax =axs[1],s = 80, color = 'black')     
            ax2.set_title('Profile : Hauts-de-France',  loc='left')
            ax1.set_xlabel( ' ')
            ax1.set_ylabel('Total consumption in MWh')
            ax2.set_xlabel( ' ')
            ax2.set_ylabel('Total consumption in MWh')
            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(0.90,0.91),ncol=4)
            st.pyplot(fig)
            
        elif choice == ['School holiday']:
                
                fig,axs =plt.subplots(2,1,figsize=(12,12))
                fig.suptitle("Consumption evolution by time in MWh", fontdict=font_title, fontsize = 22)
                ax1 = sns.lineplot(data = season[season['Code région']==24],x= 'Date',y='Total énergie soutirée (MWh)',hue = 'Période',ax =axs[0],palette = Vacances,legend = False)
                ax2 = sns.lineplot(data = season[season['Code région']==32],x= 'Date',y='Total énergie soutirée (MWh)',hue = 'Période',ax =axs[1],palette = Vacances,legend = False)
                ax1.set_title('Profile : Centre-Val de Loire',  loc='left')
                ax2.set_title('Profile : Hauts-de-France',  loc='left')
                ax1.set_xlabel( ' ')
                ax1.set_ylabel('Total consumption in MWh')
                ax2.set_xlabel( ' ')
                ax2.set_ylabel('Total consumption in MWh')
                handles, labels = ax1.get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(0.90,0.91),ncol=4)
                st.pyplot(fig)
            
            
        else:
                fig,axs =plt.subplots(2,1,figsize=(12,12))
                fig.suptitle("Consumption evolution by time in MWh", fontdict=font_title, fontsize = 22)
                ax1 = sns.lineplot(data = season[season['Code région']==24],x= 'Date',y='Total énergie soutirée (MWh)',hue = 'Période',ax =axs[0],palette = Vacances,legend = False)
                ax2 = sns.lineplot(data = season[season['Code région']==32],x= 'Date',y='Total énergie soutirée (MWh)',hue = 'Période',ax =axs[1],palette = Vacances,legend = False)
                sns.scatterplot(data = season[(season['Date'].isin(Jour_ferie)) & (season['Code région']==32)],x = 'Date',y = 'Total énergie soutirée (MWh)',
                ax =axs[1],s = 80, color = 'black')
                sns.scatterplot(data = season[(season['Date'].isin(Jour_ferie)) & (season['Code région']==24)],x = 'Date',y = 'Total énergie soutirée (MWh)',
                ax =axs[0],s = 80, color = 'black')
                ax1.set_title('Profile : Centre-Val de Loire',  loc='left')
                ax2.set_title('Profile : Hauts-de-France',  loc='left')
                ax1.set_xlabel( ' ')
                ax1.set_ylabel('Total consumption in MWh')
                ax2.set_xlabel( ' ')
                ax2.set_ylabel('Total consumption in MWh')
                handles, labels = ax1.get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(0.90,0.91),ncol=4)
                st.pyplot(fig)
                        
            



        ### Boxplot distribution of consum by holidays ###
        
    st.markdown(""" The previouses charts and the folowing one show us electric consumption is quite equal along differents kind of days except for the school holidays. 
                Usually people travel during this period.
                
                """)

    palette_box = {"National Holidays":"#2c7fb8",
                  "Week-end": "#2ca25f", "School Holidays": "#fa9fb5",
                  "Normal Day": "#fee6ce"}
        
    

    fig,axs =plt.subplots(2,1,figsize=(12,12))
    fig.suptitle("AVG consumption per day by kind of day", fontdict=font_title, fontsize = 22)
    ax1 = sns.barplot(data = season[season['Code région']==24],x= 'Day_type',y='Total énergie soutirée (MWh)',ax =axs[0],hue = 'Day_type',palette = palette_box,errorbar = None)
    ax2 = sns.barplot(data = season[season['Code région']==32],x= 'Day_type',y='Total énergie soutirée (MWh)',ax =axs[1],hue = 'Day_type',palette = palette_box,errorbar = None)
    ax1.set_title('Profile : Centre-Val de Loire',  loc='left')
    ax1.set_ylabel(' ')
    ax2.set_ylabel(' ')
    ax1.set_xlabel(' ')
    ax2.set_xlabel(' ')
    ax2.set_title('Profile : Hauts-de-France',  loc='left')
    for i in ax1.containers:
        ax1.bar_label(i,)
    for i in ax2.containers:
        ax2.bar_label(i,)
    st.pyplot(fig)
            

        
        ### Scatter min/max temerature ###

    st.header("Correlation of weather's features", divider='rainbow') 
        
    st.markdown("Electricity consumption is clearly different depending on temperature. The more it's cold the more the consumption is important. Temperature is definitely an important factor of the electrical needs.")
        
    font_title = {'family': 'sans-serif',
                                'color':  '#114b98',
                                'weight': 'bold'}
        
    #labels = df_cvdl["season"].unique()
    df_concat = pd.read_csv('df_concat.csv')
    dl = df_concat[df_concat['Code région'] == 24]
    dr = df_concat[df_concat['Code région'] == 32]
    labels = df_concat['Season'].unique()
    tab1, tab2 = st.tabs(["Centre-Val de Loire", "Hauts-de-France"])

    with tab1:

            buttonsLabels = [dict(label = "All", method = "update",visible=True, args = [{'x' : [dl['MAX_TEMPERATURE_C']]},{'y' : [dl['TEMPERATURE_EVENING_C']]},
                                                                                {'color': [dl['Total énergie soutirée (MWh)']]},
                                                                        ]
                                                                        )]
            for label in labels:
                    buttonsLabels.append(dict(label = label,method = "update",visible = True,args = [{'x' : [dl.loc[dl['Season'] == label, "MAX_TEMPERATURE_C"]]},
                                                                                {'y' : [dl.loc[dl['Season'] == label, "TEMPERATURE_EVENING_C"]]},
                                                                                {'color' : [dl.loc[dl['Season'] == label, "Total énergie soutirée (MWh)"]]},
                                                                        ]
                                                                        ))

            
            fig1 = go.Figure(px.scatter(dl, x="MAX_TEMPERATURE_C", y="TEMPERATURE_EVENING_C",color="Total énergie soutirée (MWh)",hover_data= ["Total énergie soutirée (MWh)"],
                            labels={"MAX_TEMPERATURE_C": "Max Temperature",
                                    "TEMPERATURE_EVENING_C": "Evening Temperature",
                                    "Total énergie soutirée (MWh)": "Consum"
                                     },color_continuous_scale='turbo'),
                )

            fig1.update_layout(updatemenus = [dict(buttons = buttonsLabels, showactive = True)],
                    margin = dict(t=50, l=25, r=25, b=25),
                   title=dict(text="Daily consumption distribution by Evening/Max Temperature",font=dict(size=20)),
                  title_font=dict(size=22,family= 'sans-serif',
                                color =  '#114b98')                  )

            st.plotly_chart(fig1, theme="streamlit")         

    with tab2:

            buttonslist = [dict(label = "All", method = "update",visible=True, args = [{'x' : [dr['MAX_TEMPERATURE_C']]},{'y' : [dr['TEMPERATURE_EVENING_C']]},
                                                                                {'color': [dr['Total énergie soutirée (MWh)']]},
                                                                        ]
                                                                        )]
            for label in labels:
                    buttonslist.append(dict(label = label,method = "update",visible = True,args = [{'x' : [dr.loc[dr['Season'] == label, "MAX_TEMPERATURE_C"]]},
                                                                                {'y' : [dr.loc[dr['Season'] == label, "TEMPERATURE_EVENING_C"]]},
                                                                                {'color' : [dr.loc[dr['Season'] == label, "Total énergie soutirée (MWh)"]]},
                                                                        ]
                                                                        ))
                    fig2 = go.Figure(px.scatter(dr, x="MAX_TEMPERATURE_C", y="TEMPERATURE_EVENING_C",color="Total énergie soutirée (MWh)",hover_data= ["Total énergie soutirée (MWh)"],
                            labels={"MAX_TEMPERATURE_C": "Max Temperature",
                                    "TEMPERATURE_EVENING_C": "Evening Temperature",
                                    "Total énergie soutirée (MWh)": "Consum"
                                     },color_continuous_scale='turbo'),
                )
                    fig2.update_layout(updatemenus = [dict(buttons = buttonslist, showactive = True)],
                    margin = dict(t=50, l=25, r=25, b=25),
                   title=dict(text="Daily consumption distribution by Min/Max Temperature",font=dict(size=20)),
                  title_font=dict(size=22,family= 'sans-serif',
                                color =  '#114b98'),
                   )
            st.plotly_chart(fig2)  


        
    
    
    
    
    
    
    st.markdown(" In the following charts the consumption is split along it's rainy or not, snowy or not and humidity's levels.")
    st.markdown("""
                
                - With further analyses we can notice consumption is quite different for both regions depending on two of the three factors.
                - Rain is definitely not an important factor of consumption.
                - Snow and humidity are importants factors of consumption.
                
                """)

    ### Boxplot Rainy day consumption ###
    
    fig,axs =plt.subplots(2,1,figsize=(12,12))
    fig.suptitle("AVG consumption per day by rainy/non rainy day", fontdict=font_title, fontsize = 22)
    ax1 = sns.boxplot(data = df_concat[df_concat['Code région']==24],y= 'Rain',x='Total énergie soutirée (MWh)',ax =axs[0],hue = 'Rain')
    ax2 = sns.boxplot(data = df_concat[df_concat['Code région']==32],y= 'Rain',x='Total énergie soutirée (MWh)',ax =axs[1],hue = 'Rain')
    ax1.set_title('Profile : Centre-Val de Loire',  loc='left')
    ax1.set_ylabel(' ')
    ax2.set_ylabel(' ')
    ax2.set_title('Profile : Hauts-de-France',  loc='left')
    st.pyplot(fig)
    
    df_concat['Humidity'] = ""
    for i in range(len(df_concat)):
        if df_concat['HUMIDITY_MAX_PERCENT'][i] > 70:
                df_concat['Humidity'][i] = "Humidity > 70 %"
        else:
                df_concat['Humidity'][i] = "Suitable Humidity"
    
    fig,axs =plt.subplots(2,1,figsize=(12,12))
    fig.suptitle("AVG consumption per day by Humidity Levels", fontdict=font_title, fontsize = 22)
    ax1 = sns.boxplot(data = df_concat[df_concat['Code région']==24],y= 'Humidity',x='Total énergie soutirée (MWh)',ax =axs[0],hue = 'Humidity')
    ax2 = sns.boxplot(data = df_concat[df_concat['Code région']==32],y= 'Humidity',x='Total énergie soutirée (MWh)',ax =axs[1],hue = 'Humidity')
    ax1.set_title('Profile : Centre-Val de Loire',  loc='left')
    ax1.set_ylabel(' ')
    ax2.set_ylabel(' ')
    ax2.set_title('Profile : Hauts-de-France',  loc='left')
    st.pyplot(fig)
    
    df_concat['Date'] = pd.to_datetime(df_concat['Date'])
    df_concat['Snow'] = ""
    for i in range(len(df_concat)):
        if df_concat['TOTAL_SNOW_MM'][i] ==0:
                df_concat['Snow'][i] = "No snow"
        else:
                df_concat['Snow'][i] = "Snow"
    
    fig,axs =plt.subplots(2,1,figsize=(12,12))
    fig.suptitle("AVG consumption per day by snowy/non snowy day", fontdict=font_title, fontsize = 22)
    ax1 = sns.boxplot(data = df_concat[df_concat['Code région']==24],y= 'Snow',x='Total énergie soutirée (MWh)',ax =axs[0],hue = 'Snow')
    ax2 = sns.boxplot(data = df_concat[df_concat['Code région']==32],y= 'Snow',x='Total énergie soutirée (MWh)',ax =axs[1],hue = 'Snow')
    ax1.set_title('Profile : Centre-Val de Loire',  loc='left')
    ax1.set_ylabel(' ')
    ax2.set_ylabel(' ')
    ax2.set_title('Profile : Hauts-de-France',  loc='left')
    st.pyplot(fig)
    
    st.markdown(""" The consumption for the differents winds level are quite similar so the wind level isn't an important factor for electric consumption.
            
            """)
    df_concat['Wind'] = df_concat['WINDSPEED_MAX_KMH'].apply(lambda x : 'Light' if x <10 else 'Moderate' if 10<=x <= 40 else 'Strong')
    df_concat['Région'] = df_concat['Code région'].apply(lambda x : 'Centre-Val de Loire' if x == 24 else 'Hauts-de-France')
    fig,axs =plt.subplots(figsize=(12,7))
    fig.suptitle("AVG consumption per day by wind's levels", fontdict=font_title, fontsize = 22)
    ax = sns.barplot(data = df_concat,x= 'Wind', y ='Total énergie soutirée (MWh)',hue = 'Région',order = ['Light','Moderate','Strong'],errorbar = None)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(0.90,0.93),ncol=2)
    ax.set_xlabel(' ')
    ax.set_ylabel("Total consumption in MWH")
    ax.get_legend().remove()
    for i in ax.containers:
        ax.bar_label(i,)
    st.pyplot(fig)
 


 



    
    
    
    
    
    
    
    

def intro():
        st.title("Electricity Consumption Prediction System")
        st.header("Introduction of the project, location of analysis and datasets", divider='rainbow')

        st.markdown("""
                                  As part of my training at Wild Code School, with the very enthusiastic support from my coach, Mathieu Procis,  my classmate Mai and I implented this project which  is about the electricity consumption and how weather affects it in 2 specific region of France: Centre-Val de Loire and Hauts-de-France.
                                """)
        st.markdown(" ")
        # Create a folium map
        dic = {'region': ["France", "Hauts-de-France", "Centre-Val de Loire"],
                   "long_lat": [[46.2276, 2.2137], [49.6636, 2.5281], [47.7516, 1.6751]]}
        map = pd.DataFrame(dic)

        m = folium.Map(location=map["long_lat"][0], zoom_start=5)

        for ll in map["long_lat"][1:]:
                folium.Marker(
                location=ll,
                icon=folium.Icon(color='purple', icon='fa fa-flag', prefix='fa')).add_to(m)
        
        folium_static(m, height=500, width=700)
        change_font = '<p style="font-family:Corbel; color:#5d0076; font-size: 18px;">(Location of two regions in this project)</p>'
        st.markdown(change_font, unsafe_allow_html=True)
        

        st.markdown("""

                Special thanks to opendata of enedis.fr and historique-meteo.net, we could collect and exploit these datasets for our learning and training practice.  

                Requirement for the final product is a electricity consumption prediction system which returns user's amount of electricity consumption when user inputs certain important indicators as: user's profile, contract's registration power range, maximum and minimum temperature which user can find easily from their weather appmication on mobile phone.   
                """)
        st.subheader("Implementation steps")
        st.markdown("""
                The prediction system creation is implemented in 4 principal steps:
                1. Extract data from mentioned sources
                2. Combine and Explorate (Analyze) all the datasets
                3. Clean and feature data in order to prepare for ML
                4. Train and validate models (standardize, PCS, RandomSearchCV,...)
                                """)
        st.subheader("About the datasets:")           
        st.markdown("""     
                - 2 datasets about total electricity consumption every half hour of 2 regions from 4/2022 to 3/2024, which provides us condensed and macro information of the regional electricity consumption. 
                - 33 datasets about the weather's indicators of 11 departments in 2 regions.
                                """)
   
                
def ML():
    ### Regroupement des fichiers pour avoir une matrice de travail
    df_concat = pd.read_csv('df_concat.csv')
    
    
    
    
    ### Création du model de ML
     
    X = df_concat[['TOTAL_SNOW_MM', 'Code région', 'HUMIDITY_MAX_PERCENT', 'Day_type_mod',
                                 'Season-modified','PRECIP_TOTAL_DAY_MM','MAX_TEMPERATURE_C'] ]
    y= df_concat['Total énergie soutirée (MWh)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.75)
    
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    modelDTR = DecisionTreeRegressor(max_depth= 6, min_samples_leaf= 5, min_samples_split= 15,random_state = 42)
    modelDTR.fit(X_train_scaled, y_train)
    
    ### demande à l'utilisateur de rentrer les paramètres
    
    st.header("Consumption prediction")
    
    st.write("Hi there, How are you today?")
    st.write("Thanks for using this application to predict the ***Regional Electricity Consumption*** in France.")
    st.write("We need you to answer some important questions which will provide us enough information for the prediction 🤖")
    
    T_max = st.number_input('What is the max temperature ? ')
    
    Precip = st.number_input('What is the amount of precipitation ? ')
    
    Region = st.radio('Select your region',['Hauts-de-France','Centre-Val de Loire'])
    if Region == 'Hauts-de-France':
        code = 32
    else:
        code = 24
    
    Season = st.radio('What is the Season ? ',['Autumn','Winter','Spring','Summer'])
    
    if Season == 'Autumn':
            Season = 3
    elif Season == 'Winter':
            Season = 4
    elif Season == 'Spring':
            Season = 1
    else:
            Season = 2
            
    Day = st.radio('Is the predicted day on a special day ?',['Normal Day','Week-end','National Holiday','School Holiday'])
    
    if Day == 'Normal Day':
            Day = 0
    elif Day == 'Week-end':
            Day = 1
    elif Day == 'National Holiday':
            Day = 3
    else:
            Day = 2
            
            
    Snow = st.number_input('What is the amount of snow ? ')
    
    
    
    Humidity = st.number_input('What is the maximum humidity percentage ? ')
       
        
    ### prédiction la consommation
       
    Res = [Snow,code,Humidity,Day,Season,Precip,T_max] 
    Res = modelDTR.predict([Res])
    Res= round(Res[0],2)
    
    st.write("\nWith the previouses parameters the electric consumption will be", Res, 'MWh')
    
def ML_explain(): 
        df_concat = pd.read_csv('df_concat.csv') 
        st.header("Overview of Dataframe", divider='rainbow')
        st.markdown(' ')
        st.markdown("""
                - 1462 rows equivalent to 365 days x 2 years x 2 regions
                - Target variable: 'Total énergie soutirée (MWh)'
                - Explanatory variables are all the columns except of 'DATE' and target column
                """)
        st.markdown(' ')
        
        st.markdown(" ")
        st.markdown("This is the dataset after the step Feature Engineering")
        st.write(df_concat)


        ### Check Correlation between variables ###
        st.header("Check Correlation between variables", divider='rainbow')
        # get correlation data 
        var_corr = df_concat.corr(numeric_only = True)

        # build a matrix of booleans with same size and shape of var_corr
        ones_corr = np.ones_like(var_corr, dtype=bool)

        # create a mask to take only the half values of the table by using np.triu
        mask = np.triu(ones_corr)

        # adjust the mask and correlation data  to remove the 1st row and last column
        adjusted_mask = mask[1:,:-1]
        adjusted_corr = var_corr.iloc[1:,:-1]

        fig = plt.figure(figsize=(15,10))
        sns.heatmap(adjusted_corr, mask = adjusted_mask,
                    annot = True, cmap = 'vlag', center = 0, 
                    annot_kws={'size': 7}, linecolor="white", linewidth=0.5)
        st.pyplot(fig) 


        
        st.markdown(' ')
        st.subheader('Machine Learning workflow step by step:')
        st.markdown("""
                    0. Extract and preprocess data
                    1. Initialze X (explanatory variables), y (target variable)
                    2. Split train, test then standardize X_train, X_test
                    3. Train 2 models: LR, DTR and evaluate models using merics score on test/train set, RMSE
                    4. Apply PCA to reduce not very important demensions (explanatory variables)
                    5. Get columns names of important components then train models again with new numbers of explanatory variables
                    6. Apply RandommizedSearchCV to get good hyperparameters for model DTR. Then, train DTR again tuning hyperparameters
                    7. Final evaluation: vizualize test-score and RMSE on bar chart to compare and choose final model
                        """)
        

        st.header("Metrics vizualization", divider='rainbow')
        st.markdown(" ")
        st.markdown("""After using PCA to obtain the most important features according to theirs variance contribution in the dataset, we got  8 variables:  ['TOTAL_SNOW_MM', 'Code région', 'HUMIDITY_MAX_PERCENT', 'Day_type_mod', 'WINDSPEED_MAX_KMH', 'Season-modified','PRECIP_TOTAL_DAY_MM','MAX_TEMPERATURE_C']
                    as input for our DecisionTreeRegressor model. But according to our analysis, we observe that 'wind_speed' and 'PRESSURE_MAX_MB' don't have much influence on electricity consumption of regions, so we tried to replace them by a time variable 'hol_weekend_vac' to see what could happen. The results that we got were very interesting. We visualized all the results (test-score and RMSE) into a barplot for easier observing.
        """)
        st.markdown("Thus, in the end, we selected 7 variables as input, they are:['TOTAL_SNOW_MM', 'Code région', 'HUMIDITY_MAX_PERCENT', 'Day_type_mod', 'Season-modified','PRECIP_TOTAL_DAY_MM','MAX_TEMPERATURE_C']")
        
        
        
        # Create df of models with theirs scores

        dico_models = {
                        'model': ['LinearRegression', 'DecisionTreeRegressor', 'DecisionTreeRegressor with params', 
                                'LinearRegression', 'DecisionTreeRegressor', 'DecisionTreeRegressor with params', 
                                'LinearRegression', 'DecisionTreeRegressor', 'DecisionTreeRegressor with params',
                                'LinearRegression', 'DecisionTreeRegressor', 'DecisionTreeRegressor with params'],
                        'n_components': [24, 24, 24, 13, 13, 13, 8, 8, 8, 7, 7, 7],
                        'score_test': [0.896, 0.937, 0.955, 
                                    0.873, 0.936, 0.948, 
                                    0.868, 0.931, 0.948,
                                    0.867, 0.925, 0.953],
                        'RMSE (Mwh)': [4471, 3481, 2952, 
                                    4930, 3510, 3160,
                                    5019, 3624, 3164,
                                    5055, 3778, 2994]
                    }

        df_models = pd.DataFrame(dico_models)

        fig, axs = plt.subplots(2, 1, figsize=(15, 7))
        ax1 = sns.barplot(df_models, x = 'n_components', y = 'score_test', hue = 'model', ax=axs[0], order=[24,13,8,7])
        ax1.set_title("Test-set Score by Num of Features and Model", fontweight ='bold', pad=10,loc='left')
        ax1.set_xlabel(" ", fontweight ='bold' )
        ax1.set_ylabel("Accuracy score on Test set")

        for container in axs[0].containers:
            axs[0].bar_label(container, label_type="edge",
                                color="r", fontsize=8)


        ax2 = sns.barplot(df_models, x = 'n_components', y = 'RMSE (Mwh)', hue = 'model', ax=axs[1], order=[24,13,8,7])
        ax2.legend_.remove()
        ax2.set_title("RMSE (Mwh) by Num of Features and Model", fontweight ='bold', pad=10,loc='left')
        ax2.set_xlabel("Num of Features" )
        ax2.set_ylabel("Root Mean Squared Error")

        for container in axs[1].containers:
            axs[1].bar_label(container, label_type="edge",
                                color="r", fontsize=8)

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(0.90,0.92),ncol=3,fontsize =8)

        ax1.get_legend().remove()
        st.pyplot(fig)

        st.markdown(" ")
        st.title(":violet[Detailed Explanation]")
        
        

        st.header("Train and fit X with 24 features", divider='rainbow')
        
        # Initialize X, y
        # 1st reound: X has 24 features
        df_concat['Day_type_mod'] = df_concat['Day_type'].replace({'Normal Day':0, 'Week-end':1, 'School Holidays':2, 'National Holidays':3})
        X = df_concat.select_dtypes('number').drop('Total énergie soutirée (MWh)',axis = 1)
        y= df_concat['Total énergie soutirée (MWh)']

        # Split and standardize X
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)

        # Standardize X 
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        code = '''
                    # Initialize X, y
                    # 1st reound: X has 24 features
                    X = df_concat.select_dtypes('number').drop('Total énergie soutirée (MWh)',axis = 1)
                    y= df_concat['Total énergie soutirée (MWh)']


                    # Split and standardize X
                    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, 
                                                                            train_size=0.8)

                    # Standardize X 
                    scaler = StandardScaler().fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    '''
        st.subheader("Initialize-split X,y and standardize data")
        st.code(code, language='python')
        
        
        st.subheader("Metrics using 24 features")
        
        
        modelLR = LinearRegression().fit(X_train_scaled, y_train)
        y_test_predictLR = modelLR.predict(X_test_scaled)

        st.write("METRICS for MODEL Linear Regression with 24 features")
        st.write("Score for the Train-set :", modelLR.score(X_train_scaled, y_train))
        st.write("Score for the Test-set :", modelLR.score(X_test_scaled, y_test))
        st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predictLR)), 'Mwh')

        ### DecisionTreeGressor ###
        # Without hyperparams
        modelDTR = DecisionTreeRegressor(random_state=42)
        modelDTR.fit(X_train_scaled, y_train)
        y_test_predictDTR = modelDTR.predict(X_test_scaled)

        st.write("METRICS for MODEL DecisionTreeRegressor with 24 features")
        st.write("Score for the Train-set :", modelDTR.score(X_train_scaled, y_train))
        st.write("Score for the Test-set :", modelDTR.score(X_test_scaled, y_test))
        st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predictDTR)), 'Mwh')
        
        
        ### DecisionTreeGressor ###
        # With hyperparams
        st.subheader("Decision Tree Regressor with Hyperparameters tuning ")
        modelDTR = DecisionTreeRegressor(min_samples_split = 6, min_samples_leaf = 8, max_depth = 18, random_state=42)
        modelDTR.fit(X_train_scaled, y_train)

        y_test_predict = modelDTR.predict(X_test_scaled)

        st.write("\nScore for the Train-set :", modelDTR.score(X_train_scaled, y_train))
        st.write("\nScore for the Test-set :", modelDTR.score(X_test_scaled, y_test))
        st.write('\nRMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)), 'Mwh')

        st.header("Apply PCA to reduce dimensions", divider='rainbow')
        
        # Initialze CPA to find the n_components
        pca = PCA().fit(X_train_scaled)
        
        code_pca = '''
                    # Initialize CPA to find the n_components
                    pca = PCA().fit(X_train_scaled)
                    pca.explained_variance_ratio_                
                        '''
        st.code(code_pca, language='python')
        
        st.write("The ratio of variance explained for each of the new dimensions:")                             
        st.write('''
                    array([4.41327913e-01, 1.87846718e-01, 5.43681936e-02, 4.45644348e-02,
                        4.39413172e-02, 3.81786407e-02, 3.66545686e-02, 2.57795366e-02,
                        2.48588358e-02, 2.15889885e-02, 1.96405003e-02, 1.73239651e-02,
                        1.50945639e-02, 1.06143693e-02, 5.83812741e-03, 4.67768308e-03,
                        3.59850559e-03, 1.92093056e-03, 6.34784743e-04, 5.52194254e-04,
                        4.51762073e-04, 3.02901805e-04, 1.52935238e-04, 8.76292234e-05])
                 ''')
        
        st.markdown(" ")
        st.subheader('Project into a line chart to see how n_components explaines variance of dataset')
        
        
        # Plot to see how n_components explaines variance
        plt.rcParams["figure.figsize"] = (12,6)

        fig1, ax = plt.subplots()
        xi = np.arange(1, 25, step=1)
        y = np.cumsum(pca.explained_variance_ratio_)

        plt.ylim(0.0,1.1)
        plt.plot(xi, y, marker='o', linestyle='--', color='b')

        plt.xlabel('Number of Components')
        plt.xticks(np.arange(0, 25, step=1)) #change from 0-based array index to 1-based human-readable label
        plt.ylabel('Cumulative variance (%)')
        plt.title('The number of components needed to explain variance')

        # 95% cut-off threshold
        plt.axhline(y=0.95, color='r', linestyle='-')
        plt.text(0.5, 0.9, '95% cut-off threshold', color = 'red', fontsize=12)

        # 80% cut-off threshold
        plt.axhline(y=0.80, color='r', linestyle='-')
        plt.text(5.5, 0.75, '80% cut-off threshold', color = 'red', fontsize=12)

        ax.grid(axis='x')
        st.pyplot(fig1)

        st.subheader('Create a Dataframe to identify variance indicators of each variable')
        st.markdown("🔹With n_components = 12")
        
        
        # Get columns name of important columns by applying n_components = 11
        pca_ncompo_12 =  PCA(n_components=12).fit(X_train_scaled)
        
        # Create a df to identify important columns
        pca_df = pd.DataFrame(pca_ncompo_12.components_, columns=X_train.columns, index=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 
                                                                                        'c9', 'c10', 'c11', 'c12'])
        st.write(pca_df)
        st.markdown('Get the most 12 variables having big variance in the datset')
        important_components = pca_df.abs().sum().sort_values(ascending=False)
        cod1 = '''
                     important_components = pca_df.abs().sum().sort_values(ascending=False)
                     important_components[0:12].index
                     print(important_components)
                    '''
        st.code(cod1, language='python')

        st.write(important_components)
        st.markdown("""
                    lst_13_var = ['WINDSPEED_MAX_KMH', 'VISIBILITY_AVG_KM', 'HUMIDITY_MAX_PERCENT',
                                'TOTAL_SNOW_MM', 'PRESSURE_MAX_MB', 'Code région',
                                'WEATHER_CODE_EVENING', 'Season-modified', 'Day_type_mod',
                                'WEATHER_CODE_MORNING', 'WEATHER_CODE_NOON', 'PRECIP_TOTAL_DAY_MM','MAX_TEMPERATURE_C']

                    Note: according to the requirements of this project, two required variables are: precipitation and temperature. In this case, we need to add one temperature variable, and we choose 'MAX_TEMPERATURE_C'
                    """)
        
        st.markdown(" ")
        st.markdown("🔹With n_components = 6")
        
        # Get columns name of important columns by applying n_components = 8
        pca_ncompo_6 =  PCA(n_components=6).fit(X_train_scaled)
        
        # Create a df to identify important columns
        pca_df = pd.DataFrame(pca_ncompo_6.components_, columns=X_train.columns, index=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
        st.write(pca_df)
        st.markdown('Get the most 6 variables having big variance in the datset')
        important_components = pca_df.abs().sum().sort_values(ascending=False)
        cod2 = '''
                     important_components = pca_df.abs().sum().sort_values(ascending=False)
                     important_components[0:6].index
                     print(important_components)
                    '''
        st.code(cod2, language='python')

        st.write(important_components)
        st.markdown("""
                    lst_8_var = ['TOTAL_SNOW_MM', 'Code région', 'HUMIDITY_MAX_PERCENT', 'Day_type_mod',
                                'WINDSPEED_MAX_KMH', 'Season-modified','PRECIP_TOTAL_DAY_MM','MAX_TEMPERATURE_C'] 

                    Note: add 2 features 'PRECIP_TOTAL_DAY_MM' and 'MAX_TEMPERATURE_C'
                    """)

        
        st.header("RandomizedSearchCV for best params of model DTR", divider='rainbow')
        
        code2 = '''
                dico = {'max_depth': range(1, 20),
                'min_samples_split': range(2, 20),
                'min_samples_leaf': range(1, 20)}
                dtree_reg = DecisionTreeRegressor(random_state=42)
                rando = RandomizedSearchCV(dtree_reg, param_distributions=dico , 
                                        n_iter=100, cv=10, random_state=42)
                rando.fit(X_train_scaled, y_train)
        '''
        st.code(code2, language='python')
        
        st.markdown(" ")
        st.markdown('''
                With n_features = 24
                - Best Parameters: {'min_samples_split': 15, 'min_samples_leaf': 2, 'max_depth': 7}
                - Best Score: 0.9476689830770946
                ''')
        st.markdown(" ")
        st.markdown('''
                With n_features = 13
                - Best Parameters: {'min_samples_split': 19, 'min_samples_leaf': 10, 'max_depth': 14}
                - Best Score: 0.9483405940065023
                ''')
        st.markdown(" ")
        st.markdown('''
                With n_features = 8
                - Best Parameters: {'min_samples_split': 7, 'min_samples_leaf': 11, 'max_depth': 17}
                - Best Score: 0.949321256037212
                ''')
        st.markdown(" ")
        st.markdown('''
                With n_features = 7
                - Best Parameters: {'min_samples_split': 5, 'min_samples_leaf': 10, 'max_depth': 11}
                - Best Score: 0.9492161520342682
                ''')


        st.header("Metrics using 13 features", divider='rainbow')
        
        X = df_concat[['WINDSPEED_MAX_KMH', 'VISIBILITY_AVG_KM', 'HUMIDITY_MAX_PERCENT',
                                'TOTAL_SNOW_MM', 'PRESSURE_MAX_MB', 'Code région',
                                'WEATHER_CODE_EVENING', 'Season-modified', 'Day_type_mod',
                                'WEATHER_CODE_MORNING', 'WEATHER_CODE_NOON', 'PRECIP_TOTAL_DAY_MM','MAX_TEMPERATURE_C']]

        y= df_concat['Total énergie soutirée (MWh)']

        # Split and standardize X
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)

        # Standardize X 
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)


        st.subheader("Metrics using 13 features")
        
        
        modelLR = LinearRegression().fit(X_train_scaled, y_train)
        y_test_predictLR = modelLR.predict(X_test_scaled)

        st.write("METRICS for MODEL Linear Regression with 13 features")
        st.write("Score for the Train-set :", modelLR.score(X_train_scaled, y_train))
        st.write("Score for the Test-set :", modelLR.score(X_test_scaled, y_test))
        st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predictLR)), 'Mwh')

        ### DecisionTreeGressor ###
        # Without hyperparams
        modelDTR = DecisionTreeRegressor(random_state=42)
        modelDTR.fit(X_train_scaled, y_train)
        y_test_predictDTR = modelDTR.predict(X_test_scaled)

        st.write("METRICS for MODEL DecisionTreeRegressor with 13 features")
        st.write("Score for the Train-set :", modelDTR.score(X_train_scaled, y_train))
        st.write("Score for the Test-set :", modelDTR.score(X_test_scaled, y_test))
        st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predictDTR)), 'Mwh')
        

        ### DecisionTreeGressor ###
        # With hyperparams
        st.subheader("Decision Tree Regressor with Hyperparameters tuning ")
        modelDTR = DecisionTreeRegressor(min_samples_split = 6, min_samples_leaf = 11, max_depth = 10, random_state=42)                  
        modelDTR.fit(X_train_scaled, y_train)

        y_test_predict = modelDTR.predict(X_test_scaled)

        st.write("\nScore for the Train-set :", modelDTR.score(X_train_scaled, y_train))
        st.write("\nScore for the Test-set :", modelDTR.score(X_test_scaled, y_test))
        st.write('\nRMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)), 'Mwh')

        st.header("Metrics using 8 features", divider='rainbow')

         ### Round 3: Try with 80% cut-off for n_coponents = 6 then plus 2 required variables ###
        X = df_concat[['TOTAL_SNOW_MM', 'Code région', 'HUMIDITY_MAX_PERCENT', 'Day_type_mod',
                                'WINDSPEED_MAX_KMH', 'Season-modified','PRECIP_TOTAL_DAY_MM','MAX_TEMPERATURE_C'] ]

        y= df_concat['Total énergie soutirée (MWh)']

        # Split and standardize X
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)

        # Standardize X 
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        modelLR = LinearRegression().fit(X_train_scaled, y_train)
        y_test_predictLR = modelLR.predict(X_test_scaled)

        st.write("METRICS for MODEL Linear Regression with 8 features")
        st.write("Score for the Train-set :", modelLR.score(X_train_scaled, y_train))
        st.write("Score for the Test-set :", modelLR.score(X_test_scaled, y_test))
        st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predictLR)), 'Mwh')

        ### DecisionTreeGressor ###
        # Without hyperparams
        modelDTR = DecisionTreeRegressor(random_state=42)
        modelDTR.fit(X_train_scaled, y_train)
        y_test_predictDTR = modelDTR.predict(X_test_scaled)

        st.write("METRICS for MODEL DecisionTreeRegressor with 8 features")
        st.write("Score for the Train-set :", modelDTR.score(X_train_scaled, y_train))
        st.write("Score for the Test-set :", modelDTR.score(X_test_scaled, y_test))
        st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predictDTR)), 'Mwh')                                  

        ### DecisionTreeGressor ###
        # With hyperparams
        st.subheader("Decision Tree Regressor with Hyperparameters tuning ")
        modelDTR = DecisionTreeRegressor(min_samples_split = 6, min_samples_leaf = 8, max_depth = 19, random_state=42)                  
        modelDTR.fit(X_train_scaled, y_train)

        y_test_predict = modelDTR.predict(X_test_scaled)

        st.write("\nScore for the Train-set :", modelDTR.score(X_train_scaled, y_train))
        st.write("\nScore for the Test-set :", modelDTR.score(X_test_scaled, y_test))
        st.write('\nRMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)), 'Mwh')

        st.header("Metrics using 7 features", divider='rainbow')
        st.markdown("Following our weather influence analysis, we noticed that the duo features 'wind_speed' and 'PRESSURE_MAX_MB' do not have much impact on regional electricity consumption so we chose to remove these variables.")
        st.markdown(" ")
        ### Round 4: 7 variables ###

        X = df_concat[['TOTAL_SNOW_MM', 'Code région', 'HUMIDITY_MAX_PERCENT', 'Day_type_mod', 'Season-modified','PRECIP_TOTAL_DAY_MM','MAX_TEMPERATURE_C']]

        y= df_concat['Total énergie soutirée (MWh)']

        # Split and standardize X
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)

        # Standardize X 
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        modelLR = LinearRegression().fit(X_train_scaled, y_train)
        y_test_predictLR = modelLR.predict(X_test_scaled)

        st.write("METRICS for MODEL Linear Regression with 8 features")
        st.write("Score for the Train-set :", modelLR.score(X_train_scaled, y_train))
        st.write("Score for the Test-set :", modelLR.score(X_test_scaled, y_test))
        st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predictLR)), 'Mwh')

        ### DecisionTreeGressor ###
        # Without hyperparams
        modelDTR = DecisionTreeRegressor(random_state=42)
        modelDTR.fit(X_train_scaled, y_train)
        y_test_predictDTR = modelDTR.predict(X_test_scaled)

        st.write("METRICS for MODEL DecisionTreeRegressor with 8 features")
        st.write("Score for the Train-set :", modelDTR.score(X_train_scaled, y_train))
        st.write("Score for the Test-set :", modelDTR.score(X_test_scaled, y_test))
        st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predictDTR)), 'Mwh')                                  
                                                        

        ### DecisionTreeGressor ###
        # With hyperparams
        st.subheader("Decision Tree Regressor with Hyperparameters tuning ")
        modelDTR = DecisionTreeRegressor(min_samples_split = 15, min_samples_leaf = 1, max_depth = 8, random_state=42)                  
        modelDTR.fit(X_train_scaled, y_train)

        y_test_predict = modelDTR.predict(X_test_scaled)

        st.write("\nScore for the Train-set :", modelDTR.score(X_train_scaled, y_train))
        st.write("\nScore for the Test-set :", modelDTR.score(X_test_scaled, y_test))
        st.write('\nRMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)), 'Mwh')

        st.header("Conclusion", divider='rainbow')
        st.markdown("""
                    According to the METRICS EVALUATION graphics, we decided to keep the model using 7 features input for our prediction system:
                    - 'Code région': the code to identify the specific region
                    - 'Season-modified': a specific season
                    - 'Day_type_mod': a specific type of date (working days, weekend, school holidays or national holidays)
                    - 'TOTAL_SNOW_MM': total quantity of snowfall in mm
                    - 'HUMIDITY_MAX_PERCENT': the percentage of hulidity
                    - 'MAX_TEMPERATURE_C': the maximum teperature readings
                    - 'PRECIP_TOTAL_DAY_MM': total quantity of rainfall in mm
                    """)

