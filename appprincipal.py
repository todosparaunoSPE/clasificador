# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:05:52 2025

@author: jperezr
"""

import streamlit as st

def main():
    st.set_page_config(page_title="Clasificador Naive Bayes Multinomial", layout="centered")
    
    # Título en letras grandes
    st.markdown("# Clasificador Naive Bayes Multinomial")
    
    # Descripción breve
    st.write("Esta aplicación permite clasificar textos utilizando el algoritmo Naive Bayes Multinomial.")
    
   
    
if __name__ == "__main__":
    main()
