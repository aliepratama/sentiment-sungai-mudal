import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import subprocess
import pickle
import nltk

def install_package(package):
    try:
        __import__(package)
    except ImportError:
        st.warning(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        st.success(f"{package} installed successfully!")

# Install required packages
required_packages = ['streamlit_wordcloud', 'nltk']
for package in required_packages:
    install_package(package)

# Now import after installation
import streamlit_wordcloud as wordcloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    st.info("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    st.success("NLTK data downloaded successfully!")

# Main application starts here
st.title('Hasil Sentimen Analisis pada Ulasan Objek Wisata Sungai Mudal')
st.write('Dataset yang digunakan adalah dataset hasil crawling pada objek wisata Sungai Mudal yang terdiri dari 1442 ulasan.')

# Load data with error handling
try:
    df_new = pd.read_csv('df_new.csv')
except FileNotFoundError:
    st.error("File 'df_new.csv' tidak ditemukan. Pastikan file tersebut berada di direktori yang sama dengan script ini.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.header('Data Ulasan')
plot_msg_rating = df_new[['rating', 'msg_review']].groupby('rating').count()
st.bar_chart(plot_msg_rating, y='msg_review')

st.header('Jumlah 10 Kata Terbanyak pada Ulasan')
temp_words = []
for j in [word_tokenize(str(i[0])) for i in df_new[['msg_review']].to_numpy().tolist()]:
    for k in j:
        temp_words.append(k)

listStopword =  set(stopwords.words('indonesian'))
listStopword.update({'dan', 'yang', 'di', 'untuk', 'yg', 'm', 'ke', 'it', 'itu', 'n', 'kl', 'ndk', 'k', 'dg', 'atau', 'tapi', 'nggak',
                     'pun', 'klo', 'tp', 'jg', 'nya', 'deh'})

filtered_words = []

for i in temp_words:
    if i not in listStopword:
        filtered_words.append(i)

fdist = FreqDist(i for i in filtered_words)
col1, col2 = st.columns(2)
with col1:
    wordcloud.visualize([dict({'text': i[0], 'value': i[1]}) for i in fdist.most_common(100)], per_word_coloring=False)
with col2:
    st.line_chart(pd.DataFrame(fdist.most_common(10), columns=['Kata', 'Frekuensi']).set_index('Kata'))

word_features = list(fdist.keys())

def find_features(val):
    words = word_tokenize(str(val))
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

def label_pos_neg(val):
    if val[-2] in ['4 bintang', '5 bintang']:
        label = 'pos'
    else:
        label = 'neg'
    return (find_features(val[-1]), label)

featuresets = [label_pos_neg(i) for i in df_new.to_numpy()]
pos_words = [word for features, label in featuresets if label == 'pos' for word, available in features.items() if available == True]
pos_fdist = FreqDist(pos_words)
neg_words = [word for features, label in featuresets if label == 'neg' for word, available in features.items() if available == True]
neg_fdist = FreqDist(neg_words)

st.header('Jumlah 10 Kata Terbanyak pada Ulasan Positif')
col1, col2 = st.columns(2)
with col1:
    wordcloud.visualize([dict({'text': i[0], 'value': i[1]}) for i in pos_fdist.most_common(100)], per_word_coloring=False)
with col2:
    st.line_chart(pd.DataFrame(pos_fdist.most_common(10), columns=['Kata', 'Frekuensi']).set_index('Kata'))

st.header('Jumlah 10 Kata Terbanyak pada Ulasan Negatif')
col1, col2 = st.columns(2)
with col1:
    wordcloud.visualize([dict({'text': i[0], 'value': i[1]}) for i in neg_fdist.most_common(100)], per_word_coloring=False)
with col2:
    st.line_chart(pd.DataFrame(neg_fdist.most_common(10), columns=['Kata', 'Frekuensi']).set_index('Kata'))

st.header('Coba Model Sentimen Analisis!')
st.write('Model yang digunakan adalah model Naive Bayes yang telah dilatih sebelumnya.')
confidence_threshold = st.slider(
    "Pilih threshold kepercayaan:", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.6,
    help="Semakin tinggi nilai threshold, semakin yakin model dengan prediksinya"
)
input_review = st.text_area('Masukkan ulasan yang ingin dianalisis:', key='input_review')
nbclassifier = pickle.load(open('classifier.pkl', 'rb'))

def word_feats(words):
    return dict([(word, True) for word in words])

if input_review:
    # Preprocess input
    words = word_tokenize(input_review.lower())
    featureset = word_feats(words)
    
    # Get probability distribution
    prob_dist = nbclassifier.prob_classify(featureset)
    
    # Get confidence scores
    pos_prob = prob_dist.prob('pos')
    neg_prob = prob_dist.prob('neg')
    
    # Make decision based on confidence threshold
    if max(pos_prob, neg_prob) < confidence_threshold:
        st.warning('Model tidak cukup yakin untuk mengklasifikasikan ulasan ini.')
    else:
        result = 'pos' if pos_prob > neg_prob else 'neg'
        
        # Display result with probability
        if result == 'pos':
            st.success(f'Ulasan tersebut termasuk dalam kategori positif! (Kepercayaan: {pos_prob:.2%})')
        else:
            st.error(f'Ulasan tersebut termasuk dalam kategori negatif! (Kepercayaan: {neg_prob:.2%})')
        
        # Show probability distribution
        st.write('Distribusi Probabilitas:')
        prob_df = pd.DataFrame({
            'Sentimen': ['Positif', 'Negatif'],
            'Probabilitas': [pos_prob, neg_prob]
        })
        st.bar_chart(prob_df.set_index('Sentimen'))