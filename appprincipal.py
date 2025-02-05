# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:05:52 2025

@author: jperezr
"""

import streamlit as st
import nltk
from nltk.corpus import stopwords

# Descargar stopwords si no están disponibles
nltk.download('stopwords')
stop_words = stopwords.words('spanish')

def main():
    st.set_page_config(page_title="Clasificador Naive Bayes Multinomial", layout="centered")
    
    # Título en letras grandes
    st.markdown("# Clasificador Naive Bayes Multinomial")
    
    # Descripción breve
    st.write("Esta aplicación permite clasificar textos de correos electrónicos como spam y ham, utilizando el algoritmo Naive Bayes Multinomial.")
    
    # Agregar secciones en la barra lateral
    with st.sidebar:
        st.markdown("## Materia:  Machine Learning Aplicado")
        st.markdown("## Alumno: Javier Horacio Pérez Ricárdez")
    
if __name__ == "__main__":
    main()
