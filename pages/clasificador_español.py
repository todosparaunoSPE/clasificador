# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:46:47 2025

@author: jperezr
"""

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
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


# Título de la aplicación
st.title("Clasificador de Spam")
st.write("""
Este es un clasificador de mensajes de correo electrónico usando un modelo Naive Bayes.
El modelo determina si un mensaje es "Spam" o "Ham" (no spam).
""")

# Sección de Ayuda
with st.sidebar:
    st.header("Ayuda")
    st.write("""
    **¿Cómo usar esta aplicación?**
    1. Carga un archivo Excel (.xlsx) con mensajes de correo.
    2. El modelo entrenará automáticamente y mostrará métricas.
    3. Puedes ingresar un nuevo mensaje para clasificarlo.

    **Sobre el modelo**
    - Utiliza un clasificador Naive Bayes Multinomial.
    - Se basa en la técnica de bolsa de palabras (Bag of Words).
    - Elimina palabras irrelevantes (stop words) en español.
    
    **Interpretación de resultados**
    - La matriz de confusión muestra las predicciones correctas e incorrectas.
    - El reporte de clasificación detalla la precisión y el recall.

    - Desarrollado por:
    - **Javier Horacio Pérez Ricárdez**
    - © 2025. Todos los derechos reservados.
    """)

# Cargar un archivo Excel (.xlsx)
uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx"])

if uploaded_file is not None:
    # Leer el archivo Excel
    df = pd.read_excel(uploaded_file)

    # Verificar los nombres de las columnas
    st.write("Columnas del archivo Excel:")
    st.write(df.columns)

    # Asegurarse de que las columnas son las correctas
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']]  # Seleccionar solo las columnas relevantes
        df.columns = ['label', 'message']  # Renombrar las columnas
        st.write(f"Dataset cargado con {len(df)} mensajes.")
        
        # Verificar si el DataFrame tiene registros
        if df.shape[0] == 0:
            st.error("El archivo no contiene datos válidos para procesar.")
        else:
            st.write("### Todos los Mensajes:")
            st.write(df)

            # Verificar si hay valores nulos en el DataFrame
            if df.isnull().sum().any():
                st.error("El archivo contiene valores nulos. Por favor, limpia el archivo antes de cargarlo.")
            else:
                # Comprobar las etiquetas antes de realizar el mapeo
                unique_labels = df['label'].unique()
                st.write(f"Etiquetas únicas en 'label': {unique_labels}")

                # Mapeo de etiquetas
                df['label'] = df['label'].map({'ham': 0, 'spam': 1})

                # Comprobar si hay valores nulos después del mapeo
                if df['label'].isnull().sum() > 0:
                    st.error("El mapeo de las etiquetas ha producido valores nulos. Revisa las etiquetas en la columna 'label'.")
                else:
                    # Dividir los datos
                    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.3, random_state=42)

                    # Obtener stop words en español
                    stop_words = stopwords.words('spanish')

                    # Vectorizar los mensajes (bag of words)
                    vectorizer = CountVectorizer(stop_words=stop_words)
                    X_train_vec = vectorizer.fit_transform(X_train)
                    X_test_vec = vectorizer.transform(X_test)

                    # Entrenar el clasificador Bayesiano ingenuo
                    st.write("Entrenando el modelo...")
                    model = MultinomialNB()
                    model.fit(X_train_vec, y_train)

                    # Predecir y evaluar
                    st.write("Realizando predicciones...")
                    y_pred = model.predict(X_test_vec)

                    # Mostrar el reporte de clasificación con formato
                    st.write("### Reporte de Clasificación:")

                    # Obtener el reporte de clasificación
                    report = classification_report(y_test, y_pred, output_dict=True)

                    # Convertir el reporte en DataFrame para mejor visualización
                    report_df = pd.DataFrame(report).transpose()

                    # Mostrar las métricas clave en una tabla
                    st.write("#### Métricas de Clasificación")
                    st.dataframe(report_df)

                    # Mostrar una matriz de confusión
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
                    plt.xlabel('Predicción')
                    plt.ylabel('Real')
                    st.write("#### Matriz de Confusión:")
                    st.pyplot(fig)

                    # Mostrar una tabla con las predicciones vs las verdaderas etiquetas
                    predictions_df = pd.DataFrame({'Mensaje': X_test, 'Etiqueta Real': y_test, 'Predicción': y_pred})
                    st.write("### Comparación de Predicciones:")
                    st.write(predictions_df)

                    # Mostrar las métricas de desempeño del modelo
                    st.write("### Métricas del modelo:")
                    st.write(f"Exactitud del modelo: {model.score(X_test_vec, y_test):.2f}")

                    # Agregar una sección para que el usuario ingrese un nuevo mensaje
                    st.write("### Prueba con un nuevo mensaje:")
                    new_message = st.text_input("Introduce un mensaje para clasificar:")

                    if new_message:
                        # Vectorizar el nuevo mensaje y hacer la predicción
                        new_message_vec = vectorizer.transform([new_message])
                        prediction = model.predict(new_message_vec)

                        # Mostrar la predicción
                        if prediction == 0:
                            st.write("El mensaje es: **Ham** (no spam).")
                        else:
                            st.write("El mensaje es: **Spam**.")
    else:
        st.error("El archivo Excel no contiene las columnas esperadas ('v1' y 'v2').")
else:
    st.warning("Por favor, sube un archivo Excel.")
