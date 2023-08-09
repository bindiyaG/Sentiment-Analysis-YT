import streamlit as st
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

nltk.download('stopwords')

# Load the pre-trained LSTM model
model = load_model('lstm_model.h5')

# Function to clean and filter the text
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def prepare_data(texts):
    # Initialize a tokenizer
    tokenizer = Tokenizer(num_words=500, oov_token="<OOV>")
    # Fit the tokenizer and convert the texts to sequences
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    # Pad the sequences so they all have the same length
    sequences = pad_sequences(sequences, maxlen=1408)
    return sequences

# Streamlit App
def main():
    st.title("Sentiment Analysis Web App for Yeti Analytics")
    st.sidebar.title("Settings")
    
    st.markdown("## Enter Your Review")
    user_input = st.text_area("")

    if st.button('Analyze'):
        if user_input is not None:
            # Preprocess the review
            cleaned_text = clean_text(user_input)
            
            # Prepare the review for the model
            X = prepare_data([cleaned_text])

            # Perform sentiment analysis using the loaded model
            predictions = model.predict(X)
            
            sentiment_score = predictions[0]
            sentiment_label = "Positive" if sentiment_score >= 0.5 else "Negative"

            # Display review and sentiment analysis results
            st.markdown("## Sentiment Analysis Result")
            st.table(pd.DataFrame({"Review": [user_input], "Sentiment Label": [sentiment_label]}))

            st.markdown("## Predicted Sentiment Label")
            st.bar_chart(pd.DataFrame({'Labels': ['Positive', 'Negative'], 'Count': [sentiment_score, 1-sentiment_score]}).set_index('Labels'))

if __name__ == "__main__":
    main()
