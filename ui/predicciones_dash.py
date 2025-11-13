import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import io

# Agregar path del proyecto al sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from predict import LoyaltyPredictionPipeline

# Paleta de colores azules
COLOR_PALETTE = {
    'primary':       "#0082C8",  # Azul intenso (color principal)
    'secondary':     "#006789",  # Azul medio
    'tertiary':      "#49DBF9",  # Celeste suave
    'quaternary':    "#90E0EF",  # Azul pastel
    'dark_orange':   "#00A896",  # Verde aqua profundo (nuevo énfasis)
    
    'background':    "#E6F7F9",  # Fondo azul muy claro
    
    'recurrent':     "#0A9396",  # Verde-azulado fuerte (Recurrente)
    'non_recurrent': "#8ECAE6",  # Celeste claro (No recurrente)
    'unknown':       "#A8BFCB"   # Gris azulado (Desconocidos)
}

@st.cache_resource
def load_pipeline():
    """Cargar el pipeline de predicción con todos los modelos"""
    try:
        pipeline = LoyaltyPredictionPipeline(
            models_dir=str(project_root / 'models' / 'saved_models' / 'Random Forest'),
            xgb_models_dir=str(project_root / 'models' / 'saved_models' / 'XGBoost'),
            lgb_models_dir=str(project_root / 'models' / 'saved_models' / 'lightgbm')
        )

        pipeline.load_models()

        # Cargar datos de entrenamiento para RFM scoring
        train_path = project_root / 'data' / 'train_clean.csv'
        if train_path.exists():
            pipeline.load_training_data(str(train_path))

        return pipeline
    except Exception as e:
        st.error(f"Error cargando pipeline: {str(e)}")
        return None

def show_predicciones():
    """Módulo principal de predicciones"""

    st.header("Predicciones de Lealtad del Cliente")

    # Cargar pipeline
    with st.spinner("Cargando modelos..."):
        pipeline = load_pipeline()

    if pipeline is None:
        st.error("No se pudo cargar el pipeline de predicción")
        return

    # Mostrar modelos cargados
    models_loaded = []
    if pipeline.rf_all is not None:
        models_loaded.append("✅ Random Forest (All Features)")
    if pipeline.rf_selected is not None:
        models_loaded.append("✅ Random Forest (Selected Features)")
    if pipeline.xgb_model is not None:
        models_loaded.append("✅ XGBoost")
    if pipeline.lgb_model is not None:
        models_loaded.append("✅ LightGBM (Focal Loss)")

    with st.expander("📦 Modelos Cargados", expanded=False):
        for model in models_loaded:
            st.write(model)

    # Tabs principales
    tab_single, tab_batch, tab_comparison, tab_features = st.tabs([
        "🎯 Predicción Individual",
        "📊 Predicción en Lote",
        "⚖️ Comparación de Modelos",
        "🔝 Feature Importance"
    ])

    with tab_single:
        show_single_prediction(pipeline)

    with tab_batch:
        show_batch_prediction(pipeline)

    with tab_comparison:
        show_model_comparison(pipeline)

    with tab_features:
        show_feature_importance_analysis(pipeline)

def show_single_prediction(pipeline):
    """Interfaz para predicción individual"""

    st.subheader("Predicción Individual de Lealtad")

    # Selector de método de entrada
    input_method = st.radio(
        "Método de entrada de datos:",
        options=["Formulario manual", "Datos de ejemplo"],
        horizontal=True
    )

    if input_method == "Formulario manual":
        input_data = get_manual_input(pipeline)
    else:
        input_data = get_example_data()
        st.json(input_data)

    # Selector de modelos a usar
    st.markdown("### Modelos a Utilizar")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        use_rf = st.checkbox("Random Forest", value=True)
    with col_m2:
        use_xgb = st.checkbox("XGBoost", value=pipeline.xgb_model is not None)
    with col_m3:
        use_lgb = st.checkbox("LightGBM", value=pipeline.lgb_model is not None)
    with col_m4:
        use_ensemble = st.checkbox("Ensemble", value=True, help="Promedio de todos los modelos")

    # Botón de predicción
    if st.button("🔮 Realizar Predicción", type="primary", use_container_width=True):
        with st.spinner("Realizando predicciones..."):
            results = {}

            # Random Forest
            if use_rf:
                try:
                    rf_result = pipeline.predict_single(input_data)
                    results['Random Forest'] = rf_result
                except Exception as e:
                    st.error(f"Error en Random Forest: {str(e)}")

            # XGBoost
            if use_xgb and pipeline.xgb_model is not None:
                try:
                    xgb_result = pipeline.predict_single_xgb(input_data)
                    results['XGBoost'] = xgb_result
                except Exception as e:
                    st.error(f"Error en XGBoost: {str(e)}")

            # LightGBM
            if use_lgb and pipeline.lgb_model is not None:
                try:
                    lgb_result = pipeline.predict_single_lgb(input_data)
                    results['LightGBM'] = lgb_result
                except Exception as e:
                    st.error(f"Error en LightGBM: {str(e)}")

            # Mostrar resultados
            if results:
                display_prediction_results(results, input_data, use_ensemble)
            else:
                st.warning("No se pudieron obtener predicciones de ningún modelo")

def get_manual_input(pipeline):
    """Formulario para entrada manual de datos"""

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Información del Cliente")
        user_id = st.number_input("User ID", min_value=1, value=12345)
        merchant_id = st.number_input("Merchant ID", min_value=1, value=100)
        age_range = st.selectbox(
            "Rango de Edad",
            options=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            format_func=lambda x: {
                0: 'Desconocido', 1: '<18', 2: '18-24', 3: '25-29',
                4: '30-34', 5: '35-39', 6: '40-49', 7: '≥50', 8: '≥50'
            }[x],
            index=3
        )
        gender = st.selectbox(
            "Género",
            options=[0, 1, 2],
            format_func=lambda x: {0: 'Femenino', 1: 'Masculino', 2: 'Desconocido'}[x]
        )

    with col2:
        st.markdown("#### Actividad")
        activity_len = st.number_input("Longitud de Actividad", min_value=1, max_value=1000, value=10)
        actions_0 = st.number_input("Clics/Vistas (acción 0)", min_value=0, max_value=1000, value=5)
        actions_2 = st.number_input("Añadir al Carrito (acción 2)", min_value=0, max_value=1000, value=2)
        actions_3 = st.number_input("Compras (acción 3)", min_value=0, max_value=1000, value=1)

    with col3:
        st.markdown("#### Diversidad")
        unique_items = st.number_input("Items Únicos", min_value=1, max_value=1000, value=3)
        unique_categories = st.number_input("Categorías Únicas", min_value=1, max_value=100, value=2)
        unique_brands = st.number_input("Marcas Únicas", min_value=1, max_value=100, value=2)
        day_span = st.number_input("Días entre interacciones", min_value=0, max_value=365, value=30)
        has_1111 = st.selectbox("Participó en Double 11", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Sí')

    # Calcular date_max (asumimos una fecha reciente)
    date_max = st.date_input("Fecha de última interacción", value=datetime(2014, 11, 5))

    # Calcular merchant_freq si tenemos datos de entrenamiento
    merchant_freq = 1
    if pipeline is not None and pipeline.merchant_freq is not None and merchant_id in pipeline.merchant_freq:
        merchant_freq = int(pipeline.merchant_freq[merchant_id])

    return {
        'user_id': user_id,
        'merchant_id': merchant_id,
        'age_range': age_range,
        'gender': gender,
        'activity_len': activity_len,
        'actions_0': actions_0,
        'actions_2': actions_2,
        'actions_3': actions_3,
        'unique_items': unique_items,
        'unique_categories': unique_categories,
        'unique_brands': unique_brands,
        'day_span': day_span,
        'has_1111': has_1111,
        'date_max': str(date_max),
        'merchant_freq': merchant_freq
    }

def get_example_data():
    """Datos de ejemplo para pruebas rápidas"""
    return {
        'user_id': 163968,
        'merchant_id': 2300,
        'age_range': 3,
        'gender': 0,
        'activity_len': 11,
        'actions_0': 9,
        'actions_2': 2,
        'actions_3': 1,
        'unique_items': 4,
        'unique_categories': 1,
        'unique_brands': 1,
        'day_span': 146,
        'has_1111': 1,
        'date_max': '2014-11-11',
        'merchant_freq': 150
    }

def display_prediction_results(results, input_data, use_ensemble):
    """Mostrar resultados de predicciones de forma visual"""

    st.markdown("---")
    st.subheader("📊 Resultados de Predicciones")

    # Preparar datos para ensemble
    predictions = []
    probabilities = []
    model_names = []

    for model_name, result in results.items():
        if model_name == 'Random Forest':
            pred = result['ensemble_prediction']
            prob = result['loyalty_score']
        elif model_name == 'XGBoost':
            if 'error' not in result:
                pred = result['xgb_prediction']
                prob = result['xgb_probability']
            else:
                continue
        elif model_name == 'LightGBM':
            if 'error' not in result:
                pred = result['lgb_prediction']
                prob = result['lgb_probability']
            else:
                continue
        else:
            continue

        predictions.append(pred)
        probabilities.append(prob)
        model_names.append(model_name)

    # Calcular ensemble
    if use_ensemble and len(probabilities) > 0:
        ensemble_prob = np.mean(probabilities)
        ensemble_pred = 1 if ensemble_prob >= 0.5 else 0
    else:
        ensemble_prob = None
        ensemble_pred = None

    # Mostrar predicción principal
    if ensemble_pred is not None:
        col_main1, col_main2 = st.columns([2, 1])

        with col_main1:
            if ensemble_pred == 1:
                st.success(f"### ✅ CLIENTE RECURRENTE")
                st.write(f"**Probabilidad:** {ensemble_prob*100:.2f}%")
            else:
                st.warning(f"### ⚠️ CLIENTE NO RECURRENTE")
                st.write(f"**Probabilidad de no recurrencia:** {(1-ensemble_prob)*100:.2f}%")

        with col_main2:
            # Gauge de probabilidad
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ensemble_prob * 100,
                title={'text': "Prob. Recurrencia"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': COLOR_PALETTE['primary']},
                    'steps': [
                        {'range': [0, 33], 'color': COLOR_PALETTE['secondary']},
                        {'range': [33, 66], 'color': COLOR_PALETTE['tertiary']},
                        {'range': [66, 100], 'color': COLOR_PALETTE['primary']}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # Comparación de modelos
    st.markdown("### 🔍 Comparación por Modelo")

    comparison_data = []
    for model_name, result in results.items():
        if model_name == 'Random Forest':
            comparison_data.append({
                'Modelo': model_name,
                'Predicción': 'Recurrente' if result['ensemble_prediction'] == 1 else 'No Recurrente',
                'Probabilidad': f"{result['loyalty_score']*100:.2f}%",
                'Confianza': result['confidence']['confidence_level']
            })
        elif model_name == 'XGBoost' and 'error' not in result:
            comparison_data.append({
                'Modelo': model_name,
                'Predicción': 'Recurrente' if result['xgb_prediction'] == 1 else 'No Recurrente',
                'Probabilidad': f"{result['xgb_probability']*100:.2f}%",
                'Confianza': '-'
            })
        elif model_name == 'LightGBM' and 'error' not in result:
            comparison_data.append({
                'Modelo': model_name,
                'Predicción': 'Recurrente' if result['lgb_prediction'] == 1 else 'No Recurrente',
                'Probabilidad': f"{result['lgb_probability']*100:.2f}%",
                'Confianza': '-'
            })

    if comparison_data:
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    # Gráfico de barras de probabilidades
    if len(probabilities) > 1:
        fig_comp = go.Figure(data=[
            go.Bar(
                x=model_names,
                y=[p*100 for p in probabilities],
                marker_color=[COLOR_PALETTE['recurrent'] if pred == 1 else COLOR_PALETTE['non_recurrent']
                             for pred in predictions],
                text=[f"{p*100:.1f}%" for p in probabilities],
                textposition='auto',
            )
        ])
        fig_comp.update_layout(
            title="Probabilidad de Recurrencia por Modelo",
            yaxis_title="Probabilidad (%)",
            xaxis_title="Modelo",
            height=400
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Detalles de Random Forest (si está disponible)
    if 'Random Forest' in results:
        with st.expander("📊 Detalles de Random Forest", expanded=False):
            rf_result = results['Random Forest']

            col_rf1, col_rf2 = st.columns(2)

            with col_rf1:
                st.write("**Features de Entrada (RFM):**")
                st.json(rf_result['input_features'])

            with col_rf2:
                st.write("**Predicciones Individuales:**")
                st.write(f"- All Features: {rf_result['model_predictions']['all_features']['probability']*100:.2f}%")
                st.write(f"- Selected Features: {rf_result['model_predictions']['selected_features']['probability']*100:.2f}%")
                st.write(f"- Diferencia: {rf_result['confidence']['probability_diff']*100:.2f}%")

def show_batch_prediction(pipeline):
    """Interfaz para predicción en lote desde CSV"""

    st.subheader("Predicción en Lote desde CSV")

    st.info("""
    📋 **Formato esperado del CSV:**

    El archivo debe contener las siguientes columnas:
    - `user_id`, `merchant_id`, `age_range`, `gender`
    - `activity_len`, `actions_0`, `actions_2`, `actions_3`
    - `unique_items`, `unique_categories`, `unique_brands`
    - `day_span`, `has_1111`, `date_max`
    - `merchant_freq` (opcional)
    """)

    # Upload CSV
    uploaded_file = st.file_uploader("Cargar archivo CSV", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.write(f"**Archivo cargado:** {len(df)} registros")
            st.dataframe(df.head(10), use_container_width=True)

            # Selector de modelos
            col_model1, col_model2, col_model3 = st.columns(3)

            with col_model1:
                use_rf_batch = st.checkbox("Random Forest", value=True, key="batch_rf")
            with col_model2:
                use_xgb_batch = st.checkbox("XGBoost", value=pipeline.xgb_model is not None, key="batch_xgb")
            with col_model3:
                use_lgb_batch = st.checkbox("LightGBM", value=pipeline.lgb_model is not None, key="batch_lgb")

            if st.button("🚀 Realizar Predicciones en Lote", type="primary"):
                with st.spinner("Procesando predicciones..."):
                    results_df = df.copy()

                    # Random Forest
                    if use_rf_batch:
                        rf_batch = pipeline.predict_batch(df)
                        results_df['rf_prediction'] = rf_batch['ensemble_prediction']
                        results_df['rf_probability'] = rf_batch['loyalty_score']

                    # XGBoost
                    if use_xgb_batch and pipeline.xgb_model is not None:
                        xgb_batch = pipeline.predict_batch_xgb(df)
                        if 'error' not in xgb_batch.columns:
                            results_df['xgb_prediction'] = xgb_batch['xgb_prediction']
                            results_df['xgb_probability'] = xgb_batch['xgb_probability']

                    # LightGBM
                    if use_lgb_batch and pipeline.lgb_model is not None:
                        lgb_batch = pipeline.predict_batch_lgb(df)
                        if 'error' not in lgb_batch.columns:
                            results_df['lgb_prediction'] = lgb_batch['lgb_prediction']
                            results_df['lgb_probability'] = lgb_batch['lgb_probability']

                    # Mostrar resultados
                    st.success("✅ Predicciones completadas!")

                    # Métricas generales
                    col_m1, col_m2, col_m3 = st.columns(3)

                    if 'rf_prediction' in results_df.columns:
                        with col_m1:
                            rf_recurrent = results_df['rf_prediction'].sum()
                            st.metric("RF: Recurrentes", f"{rf_recurrent:,}",
                                     delta=f"{rf_recurrent/len(results_df)*100:.1f}%")

                    if 'xgb_prediction' in results_df.columns:
                        with col_m2:
                            xgb_recurrent = results_df['xgb_prediction'].sum()
                            st.metric("XGB: Recurrentes", f"{xgb_recurrent:,}",
                                     delta=f"{xgb_recurrent/len(results_df)*100:.1f}%")

                    if 'lgb_prediction' in results_df.columns:
                        with col_m3:
                            lgb_recurrent = results_df['lgb_prediction'].sum()
                            st.metric("LGB: Recurrentes", f"{lgb_recurrent:,}",
                                     delta=f"{lgb_recurrent/len(results_df)*100:.1f}%")

                    # Tabla de resultados
                    st.markdown("### 📊 Resultados")
                    st.dataframe(results_df.head(50), use_container_width=True)

                    # Descarga de resultados
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar Resultados (CSV)",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error procesando archivo: {str(e)}")

def show_model_comparison(pipeline):
    """Comparación visual entre modelos con métricas detalladas"""

    st.subheader("⚖️ Comparación Avanzada de Modelos")

    st.info("Esta sección permite comparar las predicciones de diferentes modelos con métricas detalladas y gráficos interactivos enlazados")

    # Generar datos de prueba o cargar
    test_option = st.radio(
        "Fuente de datos para comparación:",
        options=["Generar datos sintéticos", "Cargar desde test_clean.csv", "Cargar desde train_clean.csv (con labels)"],
        horizontal=False
    )

    # Cargar datos según opción
    has_labels = False
    if test_option == "Cargar desde test_clean.csv":
        test_path = project_root / 'data' / 'test_clean.csv'
        if test_path.exists():
            sample_size = st.slider("Tamaño de muestra", 100, 10000, 1000, step=100)
            df_test = pd.read_csv(test_path).sample(n=min(sample_size, 10000), random_state=42)
            st.write(f"📊 Cargados {len(df_test)} registros de test")
        else:
            st.error("Archivo test_clean.csv no encontrado")
            return
    elif test_option == "Cargar desde train_clean.csv (con labels)":
        train_path = project_root / 'data' / 'train_clean.csv'
        if train_path.exists():
            sample_size = st.slider("Tamaño de muestra", 100, 10000, 1000, step=100)
            df_full = pd.read_csv(train_path)
            # Filtrar solo clientes nuevos (con label 0 o 1)
            df_test = df_full[df_full['label'].isin([0, 1])].sample(n=min(sample_size, 10000), random_state=42)
            has_labels = True
            st.success(f"✅ Cargados {len(df_test)} registros con labels conocidos")
        else:
            st.error("Archivo train_clean.csv no encontrado")
            return
    else:
        sample_size = st.slider("Cantidad de registros sintéticos:", 100, 5000, 500, step=100)
        df_test = generate_synthetic_data(sample_size)
        st.write(f"🎲 Generados {len(df_test)} registros sintéticos")

    if st.button("🔄 Comparar Modelos", type="primary", use_container_width=True):
        with st.spinner("Ejecutando predicciones con todos los modelos..."):
            comparison_results = {}
            true_labels = df_test['label'].values if has_labels else None

            # Random Forest
            try:
                rf_batch = pipeline.predict_batch(df_test)
                comparison_results['Random Forest'] = {
                    'predictions': rf_batch['ensemble_prediction'].values,
                    'probabilities': rf_batch['loyalty_score'].values
                }
            except Exception as e:
                st.warning(f"Error en Random Forest: {str(e)}")

            # XGBoost
            if pipeline.xgb_model is not None:
                try:
                    xgb_batch = pipeline.predict_batch_xgb(df_test)
                    if 'error' not in xgb_batch.columns:
                        comparison_results['XGBoost'] = {
                            'predictions': xgb_batch['xgb_prediction'].values,
                            'probabilities': xgb_batch['xgb_probability'].values
                        }
                except Exception as e:
                    st.warning(f"Error en XGBoost: {str(e)}")

            # LightGBM
            if pipeline.lgb_model is not None:
                try:
                    lgb_batch = pipeline.predict_batch_lgb(df_test)
                    if 'error' not in lgb_batch.columns:
                        comparison_results['LightGBM'] = {
                            'predictions': lgb_batch['lgb_prediction'].values,
                            'probabilities': lgb_batch['lgb_probability'].values
                        }
                except Exception as e:
                    st.warning(f"Error en LightGBM: {str(e)}")

            if comparison_results:
                display_model_comparison_charts(comparison_results, df_test, true_labels)

def display_model_comparison_charts(results, df_test, true_labels=None):
    """Visualizar comparación de modelos con métricas detalladas y gráficos enlazados"""

    # Si hay labels verdaderos, calcular métricas
    if true_labels is not None:
        st.markdown("### 📊 Métricas de Performance")
        display_performance_metrics(results, true_labels)
        st.markdown("---")

    # Distribución de probabilidades con gráficos enlazados
    st.markdown("### 📈 Comparaciones Interactivas")

    tab_scatter, tab_confusion = st.tabs([
        "🔗 Gráficos Enlazados",
        "📋 Matrices de Confusión"
    ])

    with tab_scatter:
        display_linked_scatter_plots(results, df_test)

    with tab_confusion:
        if true_labels is not None:
            display_confusion_matrices(results, true_labels)
        else:
            st.warning("⚠️ Matrices de confusión requieren labels verdaderos. Usa 'train_clean.csv (con labels)' como fuente.")


def display_performance_metrics(results, true_labels):
    """Mostrar métricas de performance detalladas"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    metrics_data = []

    for model_name, data in results.items():
        y_pred = data['predictions']
        y_prob = data['probabilities']

        metrics = {
            'Modelo': model_name,
            'Accuracy': accuracy_score(true_labels, y_pred),
            'Precision': precision_score(true_labels, y_pred, zero_division=0),
            'Recall': recall_score(true_labels, y_pred, zero_division=0),
            'F1-Score': f1_score(true_labels, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(true_labels, y_prob)
        }
        metrics_data.append(metrics)

    metrics_df = pd.DataFrame(metrics_data)

    # Mostrar tabla de métricas arriba y gráfico de radar abajo en una sola columna
    st.dataframe(
        metrics_df.style.format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1-Score': '{:.4f}',
            'ROC-AUC': '{:.4f}'
        }).background_gradient(cmap='Blues', subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']),
        use_container_width=True,
        hide_index=True
    )

    # Gráfico de radar de métricas
    fig_radar = go.Figure()

    colors_radar = [COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['tertiary']]

    for idx, (_, row) in enumerate(metrics_df.iterrows()):
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score'], row['ROC-AUC']],
            theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            fill='toself',
            name=row['Modelo'],
            line=dict(color=colors_radar[idx % len(colors_radar)])
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=400,
        title="Comparación de Métricas (Radar Chart)"
    )

    st.plotly_chart(fig_radar, use_container_width=True)

def display_linked_scatter_plots(results, df_test):
    """Gráficos de dispersión enlazados interactivos"""

    st.info("💡 **Gráficos Enlazados:** Selecciona puntos en cualquier gráfico para ver detalles en los demás")

    model_list = list(results.keys())

    if len(model_list) >= 2:
        # Scatter plot comparando dos modelos
        model1 = model_list[0]
        model2 = model_list[1]

        col_scatter1, col_scatter2 = st.columns(2)

        with col_scatter1:
            # Scatter: Modelo 1 vs Modelo 2 (probabilidades)
            fig_scatter1 = go.Figure()

            # Color por acuerdo/desacuerdo
            agreement = (results[model1]['predictions'] == results[model2]['predictions']).astype(int)

            fig_scatter1.add_trace(go.Scatter(
                x=results[model1]['probabilities'],
                y=results[model2]['probabilities'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=agreement,
                    colorscale=[[0, COLOR_PALETTE['non_recurrent']], [1, COLOR_PALETTE['recurrent']]],
                    showscale=True,
                    colorbar=dict(title="Acuerdo", tickvals=[0, 1], ticktext=['No', 'Sí']),
                    opacity=0.6
                ),
                text=[f"Idx: {i}<br>{model1}: {p1:.3f}<br>{model2}: {p2:.3f}<br>Acuerdo: {'Sí' if a else 'No'}"
                      for i, (p1, p2, a) in enumerate(zip(
                          results[model1]['probabilities'],
                          results[model2]['probabilities'],
                          agreement))],
                hovertemplate='%{text}<extra></extra>',
                name='Predicciones'
            ))

            # Línea diagonal de referencia
            fig_scatter1.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Acuerdo Perfecto',
                showlegend=True
            ))

            fig_scatter1.update_layout(
                title=f"Comparación: {model1} vs {model2}",
                xaxis_title=f"Probabilidad {model1}",
                yaxis_title=f"Probabilidad {model2}",
                height=500
            )

            st.plotly_chart(fig_scatter1, use_container_width=True)

        with col_scatter2:
            # Scatter: Probabilidades vs Features
            feature_options = ['activity_len', 'actions_0', 'actions_3', 'day_span', 'unique_items']
            available_features = [f for f in feature_options if f in df_test.columns]

            if available_features:
                selected_feature = st.selectbox(
                    "Feature para eje X:",
                    options=available_features,
                    index=0
                )

                fig_scatter2 = make_subplots(
                    rows=1, cols=1,
                    subplot_titles=[f"Probabilidades vs {selected_feature}"]
                )

                for idx, (model_name, data) in enumerate(results.items()):
                    colors_list = [COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['tertiary']]
                    fig_scatter2.add_trace(go.Scatter(
                        x=df_test[selected_feature].values,
                        y=data['probabilities'],
                        mode='markers',
                        name=model_name,
                        marker=dict(
                            size=6,
                            color=colors_list[idx % len(colors_list)],
                            opacity=0.6
                        ),
                        text=[f"{model_name}<br>{selected_feature}: {x}<br>Prob: {p:.3f}"
                              for x, p in zip(df_test[selected_feature].values, data['probabilities'])],
                        hovertemplate='%{text}<extra></extra>'
                    ))

                fig_scatter2.update_layout(
                    height=500,
                    xaxis_title=selected_feature,
                    yaxis_title="Probabilidad de Recurrencia"
                )

                st.plotly_chart(fig_scatter2, use_container_width=True)

def display_confusion_matrices(results, true_labels):
    """Mostrar matrices de confusión mejoradas para cada modelo"""
    from sklearn.metrics import confusion_matrix

    st.markdown("#### 📋 Matrices de Confusión Detalladas")
    st.caption("💡 **Interpretación:** Verdaderos Positivos (VP), Falsos Negativos (FN), Falsos Positivos (FP), Verdaderos Negativos (VN)")

    cols = st.columns(len(results))

    for idx, (col, (model_name, data)) in enumerate(zip(cols, results.items())):
        with col:
            cm = confusion_matrix(true_labels, data['predictions'])
            
            # Calcular métricas adicionales
            tn, fp, fn, tp = cm.ravel()
            total = cm.sum()
            
            # Crear etiquetas más descriptivas
            labels_x = ['Predicción:<br>No Recurrente', 'Predicción:<br>Recurrente']
            labels_y = ['Real:<br>No Recurrente', 'Real:<br>Recurrente']

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=labels_x,
                y=labels_y,
                colorscale=[
                    [0, '#F8F9FA'],  # Blanco muy claro
                    [0.3, COLOR_PALETTE['quaternary']],  # Azul pastel
                    [0.6, COLOR_PALETTE['tertiary']],    # Celeste suave
                    [1, COLOR_PALETTE['primary']]        # Azul intenso
                ],
                showscale=False,
                hovertemplate='<b>%{y}</b><br><b>%{x}</b><br>Cantidad: %{z}<br>Porcentaje: %{text}<extra></extra>',
                text=[[f"{cm[i,j]/total*100:.1f}%" for j in range(2)] for i in range(2)]
            ))

            # Añadir anotaciones mejoradas sin duplicar números
            annotations = []
            metrics_labels = [['VN', 'FP'], ['FN', 'VP']]  # Verdadero Negativo, Falso Positivo, etc.
            
            for i in range(2):
                for j in range(2):
                    # Determinar color del texto basado en intensidad del fondo
                    text_color = 'white' if cm[i,j] > cm.max()/2 else 'black'
                    
                    annotations.append(
                        dict(
                            x=j,
                            y=i,
                            text=f"<b>{metrics_labels[i][j]}</b><br>{cm[i,j]}<br>({cm[i,j]/total*100:.1f}%)",
                            showarrow=False,
                            font=dict(color=text_color, size=12),
                            align='center'
                        )
                    )

            fig_cm.update_layout(
                title=f"<b>{model_name}</b><br><sub>Total: {total:,} casos</sub>",
                height=400,
                annotations=annotations,
                xaxis={'side': 'bottom', 'title': 'Predicción del Modelo'},
                yaxis={'title': 'Valor Real'}
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Mostrar métricas interpretativas
            accuracy = (tp + tn) / total
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            st.markdown(f"""
            **Métricas del Modelo:**
            - **Precisión:** {precision:.3f} ({tp}/{tp + fp})
            - **Recall:** {recall:.3f} ({tp}/{tp + fn})
            - **Especificidad:** {specificity:.3f} ({tn}/{tn + fp})
            - **Exactitud:** {accuracy:.3f}
            """, unsafe_allow_html=True)

def show_feature_importance_analysis(pipeline):
    """Mostrar análisis de importancia de features para todos los modelos"""

    st.subheader("🔝 Análisis de Feature Importance")

    st.info("""
    **Feature Importance** muestra qué características del cliente son más importantes para cada modelo.
    Esto ayuda a entender **qué factores influyen más** en las predicciones de lealtad.
    """)

    # Tabs para cada modelo
    tab_rf, tab_xgb, tab_lgb, tab_comparison = st.tabs([
        "🌲 Random Forest", 
        "⚡ XGBoost", 
        "💡 LightGBM",
        "📊 Comparación"
    ])

    with tab_rf:
        show_rf_feature_importance(pipeline)

    with tab_xgb:
        show_xgb_feature_importance_from_csv()

    with tab_lgb:
        show_lgb_feature_importance_from_csv()
        
    with tab_comparison:
        show_feature_importance_comparison()

def show_rf_feature_importance(pipeline):
    """Mostrar feature importance de Random Forest mejorado"""

    if pipeline.metadata and 'all_features' in pipeline.metadata:
        features = pipeline.metadata['all_features']

        if hasattr(pipeline.rf_all, 'feature_importances_'):
            importances = pipeline.rf_all.feature_importances_

            importance_df = pd.DataFrame({
                'feature': features,
                'importance': importances
            }).sort_values('importance', ascending=False).head(15)
            
            # Normalizar importancias
            importance_df['importance_norm'] = importance_df['importance'] / importance_df['importance'].max()

            fig = go.Figure(data=[
                go.Bar(
                    y=importance_df['feature'],
                    x=importance_df['importance'],
                    orientation='h',
                    marker_color=COLOR_PALETTE['primary'],
                    text=importance_df['importance'].round(4),
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Importancia: %{x:.4f}<br>Normalizada: %{customdata:.2%}<extra></extra>',
                    customdata=importance_df['importance_norm']
                )
            ])

            fig.update_layout(
                title="🔝 Top 15 Features - Random Forest (All Features)",
                xaxis_title="Importancia",
                yaxis_title="Feature",
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 💡 Interpretación")
            st.markdown("""
            **Random Forest** promedia múltiples árboles:
            - **Más estable** que modelos individuales
            - **Basado en splits** de decisión
            - **Ensemble** de predictores diversos
            """)
        else:
            st.warning("Feature importances no disponibles para Random Forest")
    else:
        st.warning("Metadata no disponible para Random Forest")

def show_xgb_feature_importance_from_csv():
    """Mostrar feature importance de XGBoost cargado desde CSV"""
    
    xgb_csv_path = project_root / 'models' / 'saved_models' / 'analysis' / 'fi_best_xgb_model.csv'
    
    try:
        if xgb_csv_path.exists():
            importance_df = pd.read_csv(xgb_csv_path)
            importance_df = importance_df.sort_values('importance', ascending=False).head(15)
            
            # Normalizar importancias para mejor visualización
            importance_df['importance_norm'] = importance_df['importance'] / importance_df['importance'].max()
            
            fig = go.Figure(data=[
                go.Bar(
                    y=importance_df['feature'],
                    x=importance_df['importance'],
                    orientation='h',
                    marker_color=COLOR_PALETTE['secondary'],
                    text=importance_df['importance'].round(4),
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Importancia: %{x:.4f}<br>Normalizada: %{customdata:.2%}<extra></extra>',
                    customdata=importance_df['importance_norm']
                )
            ])
            
            fig.update_layout(
                title="🔝 Top 15 Features - XGBoost",
                xaxis_title="Importancia",
                yaxis_title="Feature",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 💡 Interpretación")
            st.markdown("""
            **XGBoost** usa importancia basada en ganancia:
            - **Mayor valor** = más importante para decisiones
            - **actions_0** (vistas) suele ser muy relevante
            - **merchant_id** puede indicar patrones específicos
            """)
        else:
            st.error(f"❌ Archivo no encontrado: {xgb_csv_path}")
            
    except Exception as e:
        st.error(f"Error cargando feature importance de XGBoost: {str(e)}")

def show_lgb_feature_importance_from_csv():
    """Mostrar feature importance de LightGBM cargado desde CSV"""
    
    lgb_csv_path = project_root / 'models' / 'saved_models' / 'lightgbm' / 'feature_importance.csv'
    
    try:
        if lgb_csv_path.exists():
            importance_df = pd.read_csv(lgb_csv_path)
            importance_df = importance_df.sort_values('importance', ascending=False).head(15)
            
            # Normalizar importancias
            importance_df['importance_norm'] = importance_df['importance'] / importance_df['importance'].max()
            
            fig = go.Figure(data=[
                go.Bar(
                    y=importance_df['feature'],
                    x=importance_df['importance_norm'],
                    orientation='h',
                    marker_color=COLOR_PALETTE['tertiary'],
                    text=importance_df['importance_norm'].round(3),
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Importancia Original: %{customdata:.0f}<br>Normalizada: %{x:.3f}<extra></extra>',
                    customdata=importance_df['importance']
                )
            ])
            
            fig.update_layout(
                title="🔝 Top 15 Features - LightGBM (Focal Loss) - Escala Normalizada",
                xaxis_title="Importancia (Normalizada 0-1)",
                yaxis_title="Feature",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 💡 Interpretación")
            st.markdown("""
            **LightGBM** (Focal Loss) enfocado en casos difíciles:
            - **merchant_freq** es clave (frecuencia del merchant)
            - **unique_items** indica diversidad de productos
            - **Focal Loss** mejora detección de recurrentes
            """)
        else:
            st.error(f"❌ Archivo no encontrado: {lgb_csv_path}")
            
    except Exception as e:
        st.error(f"Error cargando feature importance de LightGBM: {str(e)}")

def show_feature_importance_comparison():
    """Comparación de feature importance entre todos los modelos"""
    
    st.markdown("### 🔍 Comparación de Features Importantes")
    st.info("Esta comparación muestra las features más importantes de cada modelo lado a lado para identificar patrones comunes.")
    
    # Cargar datos de todos los modelos
    all_features = {}
    
    # XGBoost
    xgb_path = project_root / 'models' / 'saved_models' / 'analysis' / 'fi_best_xgb_model.csv'
    if xgb_path.exists():
        xgb_df = pd.read_csv(xgb_path).head(10)
        xgb_df['model'] = 'XGBoost'
        all_features['XGBoost'] = xgb_df
    
    # LightGBM
    lgb_path = project_root / 'models' / 'saved_models' / 'lightgbm' / 'feature_importance.csv'
    if lgb_path.exists():
        lgb_df = pd.read_csv(lgb_path).head(10)
        lgb_df['model'] = 'LightGBM'
        # Normalizar LightGBM para comparación
        lgb_df['importance_norm'] = lgb_df['importance'] / lgb_df['importance'].max()
        all_features['LightGBM'] = lgb_df
    
    # Random Forest (si está disponible)
    pipeline = load_pipeline()
    if pipeline and pipeline.metadata and 'all_features' in pipeline.metadata:
        if hasattr(pipeline.rf_all, 'feature_importances_'):
            rf_df = pd.DataFrame({
                'feature': pipeline.metadata['all_features'],
                'importance': pipeline.rf_all.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            rf_df['model'] = 'Random Forest'
            all_features['Random Forest'] = rf_df
    
    if len(all_features) >= 2:
        # Gráfico de comparación lado a lado
        fig = make_subplots(
            rows=1, cols=len(all_features),
            subplot_titles=list(all_features.keys()),
            shared_yaxes=False
        )
        
        colors = [COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['tertiary']]
        
        for idx, (model_name, df) in enumerate(all_features.items()):
            # Normalizar importancias para comparación visual
            df_norm = df.copy()
            df_norm['importance_viz'] = df_norm['importance'] / df_norm['importance'].max()
            
            fig.add_trace(
                go.Bar(
                    y=df_norm['feature'],
                    x=df_norm['importance_viz'],
                    orientation='h',
                    marker_color=colors[idx % len(colors)],
                    name=model_name,
                    showlegend=False,
                    text=df_norm['importance_viz'].round(3),
                    textposition='auto'
                ),
                row=1, col=idx+1
            )
        
        fig.update_layout(
            title="📊 Comparación de Top 10 Features por Modelo (Normalizadas)",
            height=600
        )
        
        for i in range(len(all_features)):
            fig.update_xaxes(title_text="Importancia (Normalizada)", row=1, col=i+1)
        
        st.plotly_chart(fig, use_container_width=True)
        
    
    else:
        st.warning("⚠️ Se necesitan al menos 2 modelos para comparar")

def generate_synthetic_data(n_samples):
    """Generar datos sintéticos para pruebas"""
    np.random.seed(42)

    data = {
        'user_id': np.random.randint(1000, 100000, n_samples),
        'merchant_id': np.random.randint(1, 3000, n_samples),
        'age_range': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8], n_samples),
        'gender': np.random.choice([0, 1, 2], n_samples),
        'activity_len': np.random.randint(1, 50, n_samples),
        'actions_0': np.random.randint(0, 30, n_samples),
        'actions_2': np.random.randint(0, 10, n_samples),
        'actions_3': np.random.randint(0, 5, n_samples),
        'unique_items': np.random.randint(1, 20, n_samples),
        'unique_categories': np.random.randint(1, 10, n_samples),
        'unique_brands': np.random.randint(1, 10, n_samples),
        'day_span': np.random.randint(0, 180, n_samples),
        'has_1111': np.random.choice([0, 1], n_samples),
        'date_max': '2014-11-05',
        'merchant_freq': np.random.randint(1, 500, n_samples)
    }

    return pd.DataFrame(data)

# Hacer disponible la variable pipeline globalmente
pipeline = None
