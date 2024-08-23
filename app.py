limport streamlit as st
from eda import intro, ana, ML,ML_explain

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
         
    menu = ["Introduction", "Consumption Analysis",'Machine Learning', "Consumption Prediction"]

    choice = st.sidebar.selectbox("MENU", menu)

    if choice == "Introduction":
        intro()

    if choice == "Consumption Analysis":
        ana()
    if choice == "Consumption Prediction":
        
        ML()
        
    if choice == 'Machine Learning':
        ML_explain()



main()
