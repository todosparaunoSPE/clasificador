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
    
    # Instrucciones
    st.markdown("### ¿Cómo usar esta aplicación?")
    st.write("1. Ingresa el texto que deseas clasificar.")
    st.write("2. Presiona el botón de clasificación.")
    st.write("3. Obtén la categoría del texto basado en el modelo Naive Bayes Multinomial.")
    
    # Input para texto
    user_input = st.text_area("Ingresa tu texto aquí:")
    
    if st.button("Clasificar"):
        st.write("Clasificación en proceso... (Aquí iría la lógica del modelo)")
    
if __name__ == "__main__":
    main()
