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

