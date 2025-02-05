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
    st.write("Esta aplicación permite clasificar textos utilizando el algoritmo Naive Bayes Multinomial.")
    


 - Desarrollado por:
    - **Javier Horacio Pérez Ricárdez**
    - © 2025. Todos los derechos reservados.
    
if __name__ == "__main__":
    main()
