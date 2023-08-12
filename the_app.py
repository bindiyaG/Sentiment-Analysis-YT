import streamlit as st
import numpy as np
import string
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from lime.lime_text import LimeTextExplainer

nltk.download("stopwords")

# Load the pre-trained LSTM model
model = load_model("lstm_model.h5")

def clean_text(text):
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def prepare_data(texts):
    tokenizer = Tokenizer(num_words=500, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = pad_sequences(sequences, maxlen=1408)
    return sequences

def predict_fn(texts):
    if not isinstance(texts, list):
        texts = [texts]

    processed_texts = [clean_text(text) for text in texts]
    X = prepare_data(processed_texts)
    predictions = model.predict(X)
    predictions_2d = np.hstack([(1.0 - predictions), predictions])
    return predictions_2d

def main():
    st.title("Sentiment Analysis Dashboard")
    user_input = st.text_area("Enter Your Review", "")

    if st.button("Analyze"):
        if user_input:
            cleaned_text = clean_text(user_input)
            X = prepare_data([cleaned_text])
            predictions = model.predict(X)[0]
            sentiment_label = "Positive" if predictions >= 0.5 else "Negative"
            
            st.markdown("## Sentiment Analysis Result")
            st.write(f"Review: {user_input}")
            st.write(f"Sentiment Label: {sentiment_label}")

            # Explain the prediction using LIME
            explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
            explanation = explainer.explain_instance(cleaned_text, predict_fn)

            # Display LIME explanation plot
            st.markdown("## LIME Explanation")
            fig = explanation.as_pyplot_figure()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
