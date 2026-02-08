import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder

###load the trained model

model=tf.keras.models.load_model('regression.h5')


# load the encoders and scalers

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)


with open('one_hot_geo.pkl','rb') as file:
    one_hot_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


### streamlit app title

st.title("estimated salary prediction")


### user input


geography=st.selectbox('Geography',one_hot_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
exited=st.selectbox('Exited',[0,1])
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])


# Example input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})


### geography

geo_encoded=one_hot_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=one_hot_geo.get_feature_names_out(['Geography']))

### combine the following
input_data=pd.concat([input_data.reset_index(drop=True ),geo_encoded_df],axis=1)


# scaling the data 


input_data_scaled=scaler.transform(input_data)


#### prediction

preddict=model.predict(input_data_scaled)
predicion_salary=preddict[0][0]


st.write(f'predicted estimated salary: {predicion_salary:.2f}')