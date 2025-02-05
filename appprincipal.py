# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:05:52 2025

@author: jperezr
"""

import streamlit as st
import nltk
from nltk.corpus import stopwords


# Estilo de fondo
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background:
radial-gradient(black 15%, transparent 16%) 0 0,
radial-gradient(black 15%, transparent 16%) 8px 8px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 0 1px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 8px 9px;
background-color:#282828;
background-size:16px 16px;
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# Descargar stopwords si no están disponibles
nltk.download('stopwords')
stop_words = stopwords.words('spanish')

def main():
    st.set_page_config(page_title="Clasificador Naive Bayes Multinomial", layout="centered")
    
    # Título en letras grandes
    st.markdown("# Clasificador Naive Bayes Multinomial")
    
    # Descripción breve
    st.write("Esta aplicación permite clasificar textos utilizando el algoritmo Naive Bayes Multinomial.")
    
    
    
if __name__ == "__main__":
    main()
