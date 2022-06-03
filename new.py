import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title('Bangalore House Price Predictor')



def prediction(location, bhk, bath, balcony, sqft, area_type, availability):
    
    loc_index, area_index, avail_index = -1,-1,-1
    with open(r'C:\Users\paritosh\python_vs_ML\Bangalore_proj1\pkl_file', 'rb') as file:
      X = pickle.load(file)
    
    if location!='other':
        loc_index = int(np.where(X.columns==location)[0][0])
    
    if area_type!='Super built-up  Area':
        area_index = np.where(X.columns==area_type)[0][0]
        
    if availability!='Not Ready':        
        avail_index = np.where(X.columns==availability)[0][0]
            
    x = np.zeros(len(X.columns))
    x[0] = bath
    x[1] = balcony
    x[2] = bhk
    x[3] = sqft
    
    if loc_index >= 0:
        x[loc_index] = 1
    if area_index >= 0:
        x[area_index] = 1
    if avail_index >= 0:
        x[avail_index] = 1

    with open (r'\Users\paritosh\python_vs_ML\Bangalore_proj1\pk_file','rb') as f:
        model=pickle.load(f)

        
    return st.title(model.predict([x])[0])


df=pd.read_pickle(r'C:\Users\paritosh\python_vs_ML\Bangalore_proj1\pickle_file')
location_list=df['location'].unique().tolist()
location_list.sort()
selected_location=st.selectbox('Location : ',location_list)

bhk_list=df['bhk'].unique().tolist()
bhk_list.sort()
selected_bhk=st.selectbox('BHK : ',bhk_list)

df['bath']=df['bath'].astype(int)
bath_list=df['bath'].unique().tolist()
bath_list.sort()
selected_bath=st.selectbox('Bathroom : ',bath_list)

df['balcony']=df['balcony'].astype(int)
balcony_list=df['balcony'].unique().tolist()
balcony_list.sort()
selected_balcony=st.selectbox('Balcony : ',balcony_list)

sqft_list=df['new_total_sqft'].unique().tolist()
sqft_list.sort()
selected_sqft=st.selectbox('Sqft : ',sqft_list)

areatype_list=df['area_type'].unique().tolist()
areatype_list.sort()
selected_areatype=st.selectbox('Area Type : ',areatype_list)

avl_list=df['availability'].unique().tolist()
avl_list.sort()
selected_avl=st.selectbox('Avlaiabilty : ',avl_list)

if st.button('Show Price Prediction') :
    prediction(selected_location, selected_bhk, selected_bath, selected_balcony, selected_sqft, selected_areatype, selected_avl)

