import pickle
import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from category_encoders import TargetEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load saved models with pickle
with open('target_encoder_gene.pkl', 'rb') as gene_file:
    gene_encoder = pickle.load(gene_file)

with open('target_encoder_variation.pkl', 'rb') as variation_file:
    variation_encoder = pickle.load(variation_file)

with open('tfidf_vectorizer.pkl', 'rb') as tfidf_file:
    tfidf_model = pickle.load(tfidf_file)

with open('model_pickle.pkl', 'rb') as rf_model_file:
    rf_model = pickle.load(rf_model_file)


# Preprocess text outside the prediction block
def preprocess_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9\n]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\([^)]*\)', '', text)

    stop_words = set(stopwords.words('english'))
    text_tokens = word_tokenize(text)
    tw = [word for word in text_tokens if word not in stop_words and not all(char.isdigit() for char in word)]
    text = " ".join(tw)
    return text


def preprocess_input(gene, variation, text):
    gene_encoded = gene_encoder.transform(pd.DataFrame([gene], columns=['Gene']))
    variation_encoded = variation_encoder.transform(pd.DataFrame([variation], columns=['Variation']))

    # Preprocess the text
    text_preprocess = preprocess_text(text)
    # Apply TF-IDF transformation to Text
    text_tfidf = tfidf_model.transform([text_preprocess])

    # Combine all encoded features
    input_features = pd.concat([pd.DataFrame(text_tfidf.toarray(), columns=tfidf_model.get_feature_names_out()),
                                gene_encoded.rename(columns={'Gene': 'gene_target_encoded'}),
                                variation_encoded.rename(columns={'Variation': 'variation_target_encoded'})],
                               axis=1).reset_index(drop=True)

    return input_features


# Streamlit app
st.title('Genetic Mutation Prediction')

# Input fields for Gene, Variation, and Text
gene_input = st.text_input('Enter Gene:')
variation_input = st.text_input('Enter Variation:')
text_input = st.text_area('Enter Text:', height=200)

if st.button('Predict'):
    # Preprocess input data
    input_data = preprocess_input(gene_input, variation_input, text_input)

    # Predict class using RandomForestClassifier
    prediction = rf_model.predict(input_data)
    st.write(f'Predicted Class: {prediction[0]}')
