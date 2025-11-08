import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import pickle
import joblib
from datetime import datetime

# Paleta de colores rojo-naranja
COLOR_PALETTE = {
    'primary': '#FF4B4B',      # Rojo
    'secondary': '#FF8C42',    # Naranja
    'tertiary': '#FFA07A',     # Salmón claro
    'quaternary': '#FF6347',   # Tomate
    'background': '#FFF5F0',   # Fondo claro
    'positive': '#FF4B4B',     # Rojo para predicción positiva
    'negative': '#FF8C42',     # Naranja para predicción negativa
}

def show_predicciones():
    """Módulo principal de predicciones"""

    st.header("Predicciones de Lealtad del Cliente")