# A simple Streamlit App to test the predictions
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load the saved models using joblib
target_encoded_gene = joblib.load('models/target_encoder_gene.joblib')
target_encoded_variation = joblib.load('models/target_encoder_variation.joblib')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
model = joblib.load('models/model_joblib')

# Function to preprocess input data
def preprocess_input(gene, variation, text):
    gene_encoded = target_encoded_gene.transform([gene])
    variation_encoded = target_encoded_variation.transform([variation])
    text_tfidf = tfidf_vectorizer.transform([text])

    return gene_encoded, variation_encoded, text_tfidf

# Streamlit app
def main():
    st.title('Predict Class from Gene, Variation, and Text')

    # Input fields
    gene = st.text_input('Enter Gene')
    variation = st.text_input('Enter Variation')
    text = st.text_area('Enter Clinical Text', height=200)

    if st.button('Predict'):
        if gene and variation and text:
            # Preprocess input data
            gene_encoded, variation_encoded, text_tfidf = preprocess_input(gene, variation, text)

            # Make prediction
            prediction = model.predict([gene_encoded, variation_encoded, text_tfidf])

            st.success(f'Predicted Class: {prediction}')
        else:
            st.warning('Please fill in all the fields.')

if __name__ == "__main__":
    main()
