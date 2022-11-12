import streamlit as st
from PIL import Image
from prediction_fonction import *

# Presentation de l'application

st.title('Restaurant Reviews')

monlabel = "Quel texte analyser ? "
options = pd.DataFrame(['Avis dataset', 'Texte libre'])

image = Image.open('photo.jpg')
st.image(image, caption='Restaurant Reviews')

nbr_topic = st.number_input(label= "Le nombre de topics", min_value=0, max_value=15)


with st.sidebar:
        st.radio(monlabel, options)
        text = st.text_input(label="Donnez nous votre avis")


if st.button(label="Détecter le sujet d'insatisfaction"):
    prediction(modelEntraine, vectorizer, nbr_topic, text)
