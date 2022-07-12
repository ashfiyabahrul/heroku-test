import streamlit as st
import nltk
import gensim


nltk.download("stopwords")
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

#Preprocess
def preprocess(fileteks):
  f = open(fileteks, 'r')
  preprocessed = []
  
  for dok in f.readlines():
    dokumen = []
    dokumen.append(dok)

    #TOKENISASI
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(dokumen)
    word_index = tokenizer.word_counts
    tokens = [kata for kata in word_index.keys()]

    #FILTERING
    factory = StopWordRemoverFactory()
    stop_word = factory.get_stop_words()
    stop_words = stopwords.words('indonesian')
    stop_word_juga = ["gak", "kalo", "yg", "utk", "dgn", "lu", "loe", "gua", "gue", "gw", "tdk"]
    tokens_filtered = []

    for token in tokens:
            if token not in gensim.parsing.preprocessing.STOPWORDS and token not in stop_word and token not in stop_word_juga and token not in stop_words and len(token) > 1 :
                tokens_filtered.append(token)
    
    #STEMMING
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    stem = ' '.join(token for token in tokens_filtered)
    stemmed = stemmer.stem(stem)
    preprocessed.append(stemmed)

  return preprocessed
  
def main():
    filename = st.text_input('Silahkan masukan path dari file txt yang ingin di preprocess:')
    #filename = file_selector()

    if filename:
        st.write('You selected `%s`' % filename)
        preprocessed = preprocess(filename)
        for hasil in preprocessed:
            st.write(hasil+"\n")

if __name__ == "__main__":
	main()
