import streamlit as st
import pandas as pd
import numpy as np
import joblib 
# Load the trained model
with open('random_forest_model.joblib', 'rb') as file:
    regression_model = joblib.load(file)

st.sidebar.header('User Input Parameters')

def user_input_features():
    st.title('Song Popularity Prediction')
    duration_ms = st.sidebar.slider('duration_ms', 5000, 533800, 141000)
    acousticness = st.sidebar.slider('acousticness', 0.0, 1.0, 0.23)
    instrumentalness = st.sidebar.slider('instrumentalness', 0.0, 1.0, 0.0)
    danceability = st.sidebar.slider('danceability', 0.0, 1.0, 0.7)
    liveness = st.sidebar.slider('liveness', 0.0, 1.0, 0.5)
    loudness = st.sidebar.slider('loudness', 0.05, 3.2, 40.0)
    energy = st.sidebar.slider('energy', 0.0, 1.0, 0.5)
    speechiness = st.sidebar.slider('speechiness', 0.0, 1.0, 0.03)
    tempo = st.sidebar.slider('tempo', 0.0, 220.0, 92.0)
    mode = st.sidebar.slider('mode', 0, 1, 0)
    time_signature = st.sidebar.slider('time_signature', 0, 5, 3)
    valence = st.sidebar.slider('valence', 0.0, 1.0, 0.7)
    key = st.sidebar.slider('key', 0, 11, 3)
    data = {
        'acousticness': acousticness,
        'danceability': danceability,
        'duration_ms': duration_ms,
        'energy': energy,
        'key': key,
        'instrumentalness': instrumentalness,
        'liveness': liveness,
        'loudness': loudness,
        'speechiness': speechiness,
        'tempo': tempo,
        'mode': mode,
        'time_signature': time_signature,
        'valence': valence,
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Show user inputs
st.subheader('User Input parameters')
st.write(df)

# Select only the required columns for the model
required_columns = [
    'acousticness', 'danceability', 'duration_ms', 'energy', 'key',
    'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'mode', 'time_signature', 'valence'
]

# Ensure the input dataframe has the correct columns in the correct order
df_model_input = df[required_columns]

if st.button('Predict Song Popularity'):
    # Predict song popularity
    prediction = regression_model.predict(df_model_input)
    prediction = int(np.round(prediction, 0))
    st.subheader('Predicted Song Popularity')
    st.write(prediction)
