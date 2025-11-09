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

# Paleta de colores rojo-naranja
COLOR_PALETTE = {
    'primary': '#FF4B4B',      # Rojo
    'secondary': '#FF8C42',    # Naranja
    'tertiary': '#FFA07A',     # Salmón claro
    'quaternary': '#FF6347',   # Tomate
    'background': '#FFF5F0',   # Fondo claro
    'positive': '#FF4B4B',     # Rojo para recurrente
    'negative': '#FF8C42',     # Naranja para no recurrente
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
    tab_single, tab_batch, tab_comparison, tab_performance = st.tabs([
        "🎯 Predicción Individual",
        "📊 Predicción en Lote",
        "⚖️ Comparación de Modelos",
        "📈 Performance de Modelos"
    ])

    with tab_single:
        show_single_prediction(pipeline)

    with tab_batch:
        show_batch_prediction(pipeline)

    with tab_comparison:
        show_model_comparison(pipeline)

    with tab_performance:
        show_model_performance(pipeline)

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
                marker_color=[COLOR_PALETTE['primary'] if pred == 1 else COLOR_PALETTE['secondary']
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
    st.markdown("### 📈 Distribuciones y Comparaciones Interactivas")

    tab_dist, tab_scatter, tab_confusion, tab_calibration = st.tabs([
        "📊 Distribuciones",
        "🔗 Gráficos Enlazados",
        "📋 Matrices de Confusión",
        "📉 Curvas de Calibración"
    ])

    with tab_dist:
        display_probability_distributions(results)

    with tab_scatter:
        display_linked_scatter_plots(results, df_test)

    with tab_confusion:
        if true_labels is not None:
            display_confusion_matrices(results, true_labels)
        else:
            st.warning("⚠️ Matrices de confusión requieren labels verdaderos. Usa 'train_clean.csv (con labels)' como fuente.")

    with tab_calibration:
        if true_labels is not None:
            display_calibration_curves(results, true_labels)
        else:
            st.warning("⚠️ Curvas de calibración requieren labels verdaderos. Usa 'train_clean.csv (con labels)' como fuente.")

    # Acuerdo entre modelos
    st.markdown("---")
    st.markdown("### 🤝 Análisis de Acuerdo entre Modelos")
    display_model_agreement(results)

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

    # Mostrar tabla de métricas
    col_table, col_viz = st.columns([1, 1])

    with col_table:
        st.dataframe(
            metrics_df.style.format({
                'Accuracy': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1-Score': '{:.4f}',
                'ROC-AUC': '{:.4f}'
            }).background_gradient(cmap='Reds', subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']),
            use_container_width=True,
            hide_index=True
        )

    with col_viz:
        # Gráfico de radar de métricas
        fig_radar = go.Figure()

        for _, row in metrics_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score'], row['ROC-AUC']],
                theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                fill='toself',
                name=row['Modelo']
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=400,
            title="Comparación de Métricas (Radar Chart)"
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    # Gráfico de barras comparativo
    st.markdown("#### Comparación Visual de Métricas")

    metrics_long = metrics_df.melt(id_vars=['Modelo'], var_name='Métrica', value_name='Valor')

    fig_metrics = px.bar(
        metrics_long,
        x='Métrica',
        y='Valor',
        color='Modelo',
        barmode='group',
        color_discrete_map={
            'Random Forest': COLOR_PALETTE['primary'],
            'XGBoost': COLOR_PALETTE['secondary'],
            'LightGBM': COLOR_PALETTE['tertiary']
        },
        text='Valor'
    )

    fig_metrics.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_metrics.update_layout(height=400, yaxis_range=[0, 1.1])

    st.plotly_chart(fig_metrics, use_container_width=True)

def display_probability_distributions(results):
    """Mostrar distribuciones de probabilidades"""

    col1, col2 = st.columns(2)

    with col1:
        # Histograma overlayed
        fig_hist = go.Figure()

        colors = [COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['tertiary']]

        for idx, (model_name, data) in enumerate(results.items()):
            fig_hist.add_trace(go.Histogram(
                x=data['probabilities'],
                name=model_name,
                opacity=0.6,
                nbinsx=50,
                marker_color=colors[idx % len(colors)]
            ))

        fig_hist.update_layout(
            barmode='overlay',
            title="Distribución de Probabilidades (Histogram)",
            xaxis_title="Probabilidad de Recurrencia",
            yaxis_title="Frecuencia",
            height=400
        )

        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Box plots
        fig_box = go.Figure()

        for idx, (model_name, data) in enumerate(results.items()):
            fig_box.add_trace(go.Box(
                y=data['probabilities'],
                name=model_name,
                marker_color=colors[idx % len(colors)]
            ))

        fig_box.update_layout(
            title="Distribución de Probabilidades (Box Plot)",
            yaxis_title="Probabilidad",
            height=400
        )

        st.plotly_chart(fig_box, use_container_width=True)

    # Violin plots
    fig_violin = go.Figure()

    for idx, (model_name, data) in enumerate(results.items()):
        fig_violin.add_trace(go.Violin(
            y=data['probabilities'],
            name=model_name,
            box_visible=True,
            meanline_visible=True,
            marker_color=colors[idx % len(colors)]
        ))

    fig_violin.update_layout(
        title="Distribución de Probabilidades (Violin Plot)",
        yaxis_title="Probabilidad",
        height=400
    )

    st.plotly_chart(fig_violin, use_container_width=True)

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
                    colorscale=[[0, COLOR_PALETTE['secondary']], [1, COLOR_PALETTE['primary']]],
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

        # Heatmap de correlación entre probabilidades de modelos
        if len(model_list) >= 2:
            st.markdown("#### Correlación entre Predicciones de Modelos")

            # Crear matriz de correlación
            prob_matrix = np.column_stack([results[m]['probabilities'] for m in model_list])
            corr_matrix = np.corrcoef(prob_matrix.T)

            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=model_list,
                y=model_list,
                colorscale=[
                    [0, '#FFFFFF'],
                    [0.5, COLOR_PALETTE['tertiary']],
                    [1, COLOR_PALETTE['primary']]
                ],
                text=corr_matrix.round(3),
                texttemplate='%{text}',
                textfont={"size": 14},
                colorbar=dict(title="Correlación")
            ))

            fig_corr.update_layout(
                title="Matriz de Correlación de Probabilidades entre Modelos",
                height=400
            )

            st.plotly_chart(fig_corr, use_container_width=True)

def display_confusion_matrices(results, true_labels):
    """Mostrar matrices de confusión para cada modelo"""
    from sklearn.metrics import confusion_matrix

    cols = st.columns(len(results))

    for idx, (col, (model_name, data)) in enumerate(zip(cols, results.items())):
        with col:
            cm = confusion_matrix(true_labels, data['predictions'])

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Pred: No Rec', 'Pred: Rec'],
                y=['Real: No Rec', 'Real: Rec'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                colorscale=[
                    [0, '#FFFFFF'],
                    [1, COLOR_PALETTE['primary']]
                ],
                showscale=False
            ))

            # Añadir anotaciones
            annotations = []
            for i in range(2):
                for j in range(2):
                    annotations.append(
                        dict(
                            x=j,
                            y=i,
                            text=f"{cm[i,j]}<br>({cm[i,j]/cm.sum()*100:.1f}%)",
                            showarrow=False,
                            font=dict(color='white' if cm[i,j] > cm.max()/2 else 'black', size=14)
                        )
                    )

            fig_cm.update_layout(
                title=f"{model_name}",
                height=350,
                annotations=annotations
            )

            st.plotly_chart(fig_cm, use_container_width=True)

def display_calibration_curves(results, true_labels):
    """Mostrar curvas de calibración de probabilidades"""

    st.info("📉 **Curvas de Calibración:** Muestran qué tan bien calibradas están las probabilidades predichas")

    # Calcular bins para calibración
    n_bins = 10

    fig_cal = go.Figure()

    colors_list = [COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['tertiary']]

    for idx, (model_name, data) in enumerate(results.items()):
        prob_true, prob_pred = calibration_curve_custom(true_labels, data['probabilities'], n_bins=n_bins)

        fig_cal.add_trace(go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode='lines+markers',
            name=model_name,
            line=dict(color=colors_list[idx % len(colors_list)], width=3),
            marker=dict(size=10)
        ))

    # Línea de calibración perfecta
    fig_cal.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfectamente Calibrado',
        line=dict(dash='dash', color='gray', width=2)
    ))

    fig_cal.update_layout(
        title="Curvas de Calibración de Probabilidades",
        xaxis_title="Probabilidad Predicha Media",
        yaxis_title="Fracción de Positivos",
        height=500,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )

    st.plotly_chart(fig_cal, use_container_width=True)

    st.caption("💡 Una curva cercana a la diagonal indica buena calibración")

def calibration_curve_custom(y_true, y_prob, n_bins=10):
    """Calcular curva de calibración personalizada"""
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=n_bins)
    bin_true = np.bincount(binids, weights=y_true, minlength=n_bins)
    bin_total = np.bincount(binids, minlength=n_bins)

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    return prob_true, prob_pred

def display_model_agreement(results):
    """Análisis de acuerdo entre modelos"""

    model_list = list(results.keys())

    if len(model_list) < 2:
        st.warning("Se necesitan al menos 2 modelos para analizar acuerdo")
        return

    # Matriz de acuerdo
    agreement_matrix = np.zeros((len(model_list), len(model_list)))

    for i, model1 in enumerate(model_list):
        for j, model2 in enumerate(model_list):
            if i == j:
                agreement_matrix[i, j] = 100
            else:
                pred1 = results[model1]['predictions']
                pred2 = results[model2]['predictions']
                agreement = (pred1 == pred2).sum() / len(pred1) * 100
                agreement_matrix[i, j] = agreement

    # Visualizar matriz de acuerdo
    fig_agreement = go.Figure(data=go.Heatmap(
        z=agreement_matrix,
        x=model_list,
        y=model_list,
        colorscale=[
            [0, '#FFFFFF'],
            [0.5, COLOR_PALETTE['secondary']],
            [1, COLOR_PALETTE['primary']]
        ],
        text=agreement_matrix.round(1),
        texttemplate='%{text}%',
        textfont={"size": 14},
        colorbar=dict(title="% Acuerdo")
    ))

    fig_agreement.update_layout(
        title="Matriz de Acuerdo entre Modelos (%)",
        height=400
    )

    st.plotly_chart(fig_agreement, use_container_width=True)

    # Métricas de acuerdo
    col1, col2, col3 = st.columns(3)

    if len(model_list) >= 2:
        with col1:
            agreement_01 = agreement_matrix[0, 1]
            st.metric(
                f"Acuerdo: {model_list[0]} - {model_list[1]}",
                f"{agreement_01:.1f}%"
            )

        if len(model_list) >= 3:
            with col2:
                agreement_02 = agreement_matrix[0, 2]
                st.metric(
                    f"Acuerdo: {model_list[0]} - {model_list[2]}",
                    f"{agreement_02:.1f}%"
                )

            with col3:
                agreement_12 = agreement_matrix[1, 2]
                st.metric(
                    f"Acuerdo: {model_list[1]} - {model_list[2]}",
                    f"{agreement_12:.1f}%"
                )

    # Análisis de desacuerdos
    if len(model_list) == 3:
        st.markdown("#### Análisis de Consenso/Desacuerdo")

        pred1 = results[model_list[0]]['predictions']
        pred2 = results[model_list[1]]['predictions']
        pred3 = results[model_list[2]]['predictions']

        # Calcular consenso
        consensus_all = ((pred1 == pred2) & (pred2 == pred3)).sum()
        consensus_2_of_3 = (((pred1 == pred2) | (pred1 == pred3) | (pred2 == pred3)) & ~((pred1 == pred2) & (pred2 == pred3))).sum()
        no_consensus = ((pred1 != pred2) & (pred1 != pred3) & (pred2 != pred3)).sum()

        consensus_data = pd.DataFrame({
            'Tipo': ['Consenso Total (3/3)', 'Mayoría (2/3)', 'Sin Consenso (0/3)'],
            'Cantidad': [consensus_all, consensus_2_of_3, no_consensus],
            'Porcentaje': [
                consensus_all / len(pred1) * 100,
                consensus_2_of_3 / len(pred1) * 100,
                no_consensus / len(pred1) * 100
            ]
        })

        fig_consensus = go.Figure(data=[
            go.Bar(
                x=consensus_data['Tipo'],
                y=consensus_data['Cantidad'],
                marker_color=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['tertiary']],
                text=[f"{v} ({p:.1f}%)" for v, p in zip(consensus_data['Cantidad'], consensus_data['Porcentaje'])],
                textposition='auto',
            )
        ])

        fig_consensus.update_layout(
            title="Distribución de Consenso entre los 3 Modelos",
            yaxis_title="Cantidad de Predicciones",
            height=400
        )

        st.plotly_chart(fig_consensus, use_container_width=True)

def show_model_performance(pipeline):
    """Mostrar métricas de performance de modelos"""

    st.subheader("📈 Performance y Métricas de Modelos")

    st.info("""
    Esta sección muestra métricas de performance pre-calculadas de los modelos.
    Para métricas en tiempo real, necesitamos un conjunto de validación con labels conocidos.
    """)

    # Mostrar feature importance
    st.markdown("### 🔝 Features Más Importantes")

    tab_rf, tab_xgb, tab_lgb = st.tabs(["Random Forest", "XGBoost", "LightGBM"])

    with tab_rf:
        show_rf_feature_importance(pipeline)

    with tab_xgb:
        if pipeline.xgb_model is not None:
            show_xgb_feature_importance(pipeline)
        else:
            st.warning("Modelo XGBoost no cargado")

    with tab_lgb:
        if pipeline.lgb_model is not None:
            show_lgb_feature_importance(pipeline)
        else:
            st.warning("Modelo LightGBM no cargado")

def show_rf_feature_importance(pipeline):
    """Mostrar feature importance de Random Forest"""

    if pipeline.metadata and 'all_features' in pipeline.metadata:
        features = pipeline.metadata['all_features']

        if hasattr(pipeline.rf_all, 'feature_importances_'):
            importances = pipeline.rf_all.feature_importances_

            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(15)

            fig = go.Figure(data=[
                go.Bar(
                    y=importance_df['Feature'],
                    x=importance_df['Importance'],
                    orientation='h',
                    marker_color=COLOR_PALETTE['primary'],
                    text=importance_df['Importance'].round(4),
                    textposition='auto',
                )
            ])

            fig.update_layout(
                title="Top 15 Features - Random Forest (All Features)",
                xaxis_title="Importancia",
                yaxis_title="Feature",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importances no disponibles")
    else:
        st.warning("Metadata no disponible")

def show_xgb_feature_importance(pipeline):
    """Mostrar feature importance de XGBoost"""

    if hasattr(pipeline.xgb_model, 'feature_importances_'):
        importances = pipeline.xgb_model.feature_importances_
        features = pipeline.xgb_features

        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        fig = go.Figure(data=[
            go.Bar(
                y=importance_df['Feature'],
                x=importance_df['Importance'],
                orientation='h',
                marker_color=COLOR_PALETTE['secondary'],
                text=importance_df['Importance'].round(4),
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="Feature Importance - XGBoost",
            xaxis_title="Importancia",
            yaxis_title="Feature",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Feature importances no disponibles para XGBoost")

def show_lgb_feature_importance(pipeline):
    """Mostrar feature importance de LightGBM"""

    if hasattr(pipeline.lgb_model, 'feature_importances_'):
        importances = pipeline.lgb_model.feature_importances_
        features = pipeline.lgb_features

        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        fig = go.Figure(data=[
            go.Bar(
                y=importance_df['Feature'],
                x=importance_df['Importance'],
                orientation='h',
                marker_color=COLOR_PALETTE['tertiary'],
                text=importance_df['Importance'].round(4),
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="Feature Importance - LightGBM (Focal Loss)",
            xaxis_title="Importancia",
            yaxis_title="Feature",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Feature importances no disponibles para LightGBM")

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
