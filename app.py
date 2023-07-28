import streamlit as st
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from nltk.corpus import wordnet

wordnet.ensure_loaded()
nltk.download('averaged_perceptron_tagger')
# Set NLTK data path
nltk.data.path.append("C:\\Users\\Bindiya Gandhi/nltk_data")

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('omw-1.4')
nltk.download('wordnet')

# Read Data file as data
data = pd.read_excel('Yeti Analytics.xlsx')

# Making another column as bad review for checking
data["is_bad_review"] = data["Rating"].apply(lambda x: 1 if x < 4 else 0)
reviews_df = data[["Review", "is_bad_review"]]

# Sample reviews for faster computations
reviews_df = reviews_df.sample(frac=0.1, replace=False, random_state=42)

# Remove 'No Negative' or 'No Positive' from text
reviews_df["Review"] = reviews_df["Review"].apply(lambda x: x.replace("No Negative", "").replace("No Positive", ""))

# Cleaning text
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    text = text.lower()
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    text = [word for word in text if not any(c.isdigit() for c in word)]
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    text = [t for t in text if len(t) > 0]
    pos_tags = nltk.pos_tag(text)
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    text = [t for t in text if len(t) > 1]
    text = " ".join(text)
    return text

reviews_df["review_clean"] = reviews_df["Review"].apply(clean_text)

# Adding sentiment analysis columns
sid = SentimentIntensityAnalyzer()
reviews_df["sentiments"] = reviews_df["Review"].apply(lambda x: sid.polarity_scores(x))
reviews_df = pd.concat([reviews_df.drop(['sentiments'], axis=1), reviews_df['sentiments'].apply(pd.Series)], axis=1)

# Feature engineering
reviews_df["nb_chars"] = reviews_df["Review"].apply(lambda x: len(x))
reviews_df["nb_words"] = reviews_df["Review"].apply(lambda x: len(x.split(" ")))

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(reviews_df["review_clean"].apply(lambda x: x.split(" ")))]
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

doc2vec_df = reviews_df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
reviews_df = pd.concat([reviews_df, doc2vec_df], axis=1)

tfidf = TfidfVectorizer(min_df=10)
tfidf_result = tfidf.fit_transform(reviews_df["review_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns=tfidf.get_feature_names_out())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = reviews_df.index
reviews_df = pd.concat([reviews_df, tfidf_df], axis=1)

# Function to fit the model
def fit_model(df):
    X = df.drop(['is_bad_review', 'Review', 'review_clean'], axis=1)
    y = df['is_bad_review']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    return rf, X_test, y_test

# Streamlit App
st.set_page_config(page_title="Sentiment Analysis & Review Classification", layout="wide")
st.title("Sentiment Analysis and Review Classification")

# Sidebar
st.sidebar.title("Options")
analysis_type = st.sidebar.radio("Choose Analysis", ("Word Cloud", "Positive Reviews", "Negative Reviews", "Sentiment Distribution"))

if analysis_type == "Word Cloud":
    st.subheader("Word Cloud")
    wordcloud = WordCloud(
        background_color='white',
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=42
    ).generate(str(reviews_df["Review"]))

    fig = plt.figure(1, figsize=(20, 20))
    plt.axis('off')
    plt.imshow(wordcloud)
    st.pyplot(fig)

elif analysis_type == "Positive Reviews":
    st.subheader("Top 10 Positive Reviews")
    positive_reviews = reviews_df[reviews_df["is_bad_review"] == 0].sort_values("pos", ascending=False)[["Review", "pos"]].head(10)
    st.table(positive_reviews.style.format({"pos": "{:.2%}"}))

# Negative Reviews Analysis
elif analysis_type == "Negative Reviews":
    st.subheader("Top 10 Negative Reviews")
    negative_reviews = reviews_df[reviews_df["is_bad_review"] == 1].sort_values("neg", ascending=False)[["Review", "neg"]].head(10)
    st.table(negative_reviews.style.format({"neg": "{:.2%}"}))

# Sentiment Distribution Analysis
else:
    st.subheader("Distribution of Sentiment Scores")
    fig, ax = plt.subplots()
    sns.histplot(reviews_df[reviews_df["nb_words"] >= 5]['compound'], kde=True, label='All Reviews')
    sns.histplot(reviews_df[reviews_df["is_bad_review"] == 1][reviews_df["nb_words"] >= 5]['compound'], kde=True, label='Bad Reviews')
    sns.histplot(reviews_df[reviews_df["is_bad_review"] == 0][reviews_df["nb_words"] >= 5]['compound'], kde=True, label='Good Reviews')
    plt.xlabel('Sentiment Score (Compound)')
    plt.title('Distribution of Sentiment Scores')
    plt.legend()
    st.pyplot(fig)

# Model Evaluation
st.subheader("Model Evaluation")
rf, X_test, y_test = fit_model(reviews_df)

# Receiver Operating Characteristic (ROC) Curve
st.subheader("Receiver Operating Characteristic (ROC) Curve")
y_pred = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
st.pyplot(fig)

# Precision-Recall Curve
st.subheader("Precision-Recall Curve")
average_precision = average_precision_score(y_test, y_pred)
precision, recall, _ = precision_recall_curve(y_test, y_pred)

fig, ax = plt.subplots()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve: AP={0:0.2f}'.format(average_precision))
st.pyplot(fig)
