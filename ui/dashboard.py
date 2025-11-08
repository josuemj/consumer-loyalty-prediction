import streamlit as st
import sys
from pathlib import Path

# Configuraci�n de la p�gina
st.set_page_config(
    page_title="Consumer Loyalty Prediction Dashboard",
    page_icon="=�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar m�dulos
from analisis_dash import show_analisis
from predicciones_dash import show_predicciones

def main():
    """
    Dashboard principal con tabs para analisis y predicciones
    """
    # Header
    st.title("Consumer Loyalty Prediction Dashboard")
    st.markdown("---")

    # Crear tabs
    tab1, tab2 = st.tabs(["Análisis Exploratorio", "Predicciones"])

    with tab1:
        show_analisis()

    with tab2:
        show_predicciones()
        
if __name__ == "__main__":
    main()
