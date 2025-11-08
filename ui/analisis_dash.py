import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# Paleta de colores rojo-naranja
COLOR_PALETTE = {
    'primary': '#FF4B4B',      # Rojo
    'secondary': '#FF8C42',    # Naranja
    'tertiary': '#FFA07A',     # Salmón claro
    'quaternary': '#FF6347',   # Tomate
    'background': '#FFF5F0',   # Fondo claro
    'recurrent': '#FF4B4B',    # Rojo para recurrentes
    'non_recurrent': '#FF8C42', # Naranja para no recurrentes
    'unknown': '#CCCCCC'       # Gris para desconocidos
}

@st.cache_data
def load_data():
    """Cargar datos del archivo train_clean.csv"""
    data_path = Path(__file__).parent.parent / 'data' / 'train_clean.csv'
    df = pd.read_csv(data_path)

    # Convertir fechas
    df['date_min'] = pd.to_datetime(df['date_min'])
    df['date_max'] = pd.to_datetime(df['date_max'])

    # Filtrar solo clientes nuevos (label != -1) para análisis
    df_new = df[df['label'] != -1].copy()

    return df, df_new

def create_age_range_labels():
    """Mapeo de rangos de edad"""
    return {
        0: 'Desconocido',
        1: '<18',
        2: '18-24',
        3: '25-29',
        4: '30-34',
        5: '35-39',
        6: '40-49',
        7: '≥50',
        8: '≥50'
    }

def create_gender_labels():
    """Mapeo de género"""
    return {
        0: 'Femenino',
        1: 'Masculino',
        2: 'Desconocido'
    }

def show_analisis():
    """Módulo principal de análisis exploratorio"""

    st.header("📊 Análisis Exploratorio de Datos")

    # Cargar datos
    df_all, df_new = load_data()

    # Sidebar con filtros
    st.sidebar.header("🎛️ Filtros Dinámicos")

    # Filtro de label
    label_options = {
        'Todos': [-1, 0, 1],
        'Solo nuevos clientes': [0, 1],
        'Recurrentes': [1],
        'No recurrentes': [0],
        'No nuevos': [-1]
    }
    label_filter = st.sidebar.selectbox(
        "Tipo de cliente:",
        options=list(label_options.keys()),
        index=1  # Por defecto "Solo nuevos clientes"
    )
    df_filtered = df_all[df_all['label'].isin(label_options[label_filter])]

    # Filtro de género
    gender_labels = create_gender_labels()
    selected_genders = st.sidebar.multiselect(
        "Género:",
        options=list(gender_labels.keys()),
        default=list(gender_labels.keys()),
        format_func=lambda x: gender_labels[x]
    )
    df_filtered = df_filtered[df_filtered['gender'].isin(selected_genders)]

    # Filtro de rango de edad
    age_labels = create_age_range_labels()
    selected_ages = st.sidebar.multiselect(
        "Rango de edad:",
        options=list(age_labels.keys()),
        default=list(age_labels.keys()),
        format_func=lambda x: age_labels[x]
    )
    df_filtered = df_filtered[df_filtered['age_range'].isin(selected_ages)]

    # Filtro de actividad (slider)
    max_activity = int(df_all['activity_len'].max())
    activity_range = st.sidebar.slider(
        "Longitud de actividad:",
        min_value=1,
        max_value=max_activity,
        value=(1, max_activity)
    )
    df_filtered = df_filtered[
        (df_filtered['activity_len'] >= activity_range[0]) &
        (df_filtered['activity_len'] <= activity_range[1])
    ]

    # Filtro de has_1111
    has_1111_filter = st.sidebar.checkbox("Solo con actividad en Double 11", value=False)
    if has_1111_filter:
        df_filtered = df_filtered[df_filtered['has_1111'] == 1]

    # Mostrar estadísticas generales
    st.subheader("📈 Estadísticas Descriptivas")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Registros", f"{len(df_filtered):,}")

    with col2:
        if label_filter == 'Solo nuevos clientes':
            recurrent_pct = (df_filtered['label'].sum() / len(df_filtered) * 100)
            st.metric("% Recurrentes", f"{recurrent_pct:.1f}%")
        else:
            st.metric("Usuarios Únicos", f"{df_filtered['user_id'].nunique():,}")

    with col3:
        st.metric("Comerciantes Únicos", f"{df_filtered['merchant_id'].nunique():,}")

    with col4:
        st.metric("Actividad Media", f"{df_filtered['activity_len'].mean():.1f}")

    with col5:
        st.metric("% Double 11", f"{(df_filtered['has_1111'].sum() / len(df_filtered) * 100):.1f}%")

    st.markdown("---")

    # Visualizaciones principales
    tab_dist, tab_corr, tab_time, tab_rfm = st.tabs([
        "📊 Distribuciones",
        "🔗 Correlaciones",
        "📅 Análisis Temporal",
        "💎 Segmentación RFM"
    ])

    with tab_dist:
        show_distributions(df_filtered, df_new)

    with tab_corr:
        show_correlations(df_filtered)

    with tab_time:
        show_temporal_analysis(df_filtered)

    with tab_rfm:
        show_rfm_analysis(df_filtered)

def show_distributions(df_filtered, df_new):
    """Visualizaciones de distribuciones"""

    st.subheader("Distribuciones de Variables Clave")

    # Fila 1: Género y Edad
    col1, col2 = st.columns(2)

    with col1:
        # Distribución por género
        gender_labels = create_gender_labels()
        gender_dist = df_filtered['gender'].value_counts().sort_index()

        fig_gender = go.Figure(data=[
            go.Bar(
                x=[gender_labels[i] for i in gender_dist.index],
                y=gender_dist.values,
                marker_color=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['tertiary']],
                text=gender_dist.values,
                textposition='auto',
            )
        ])
        fig_gender.update_layout(
            title="Distribución por Género",
            xaxis_title="Género",
            yaxis_title="Cantidad",
            height=400
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    with col2:
        # Distribución por rango de edad
        age_labels = create_age_range_labels()
        age_dist = df_filtered['age_range'].value_counts().sort_index()

        fig_age = go.Figure(data=[
            go.Bar(
                x=[age_labels[i] for i in age_dist.index],
                y=age_dist.values,
                marker_color=COLOR_PALETTE['secondary'],
                text=age_dist.values,
                textposition='auto',
            )
        ])
        fig_age.update_layout(
            title="Distribución por Rango de Edad",
            xaxis_title="Rango de Edad",
            yaxis_title="Cantidad",
            height=400
        )
        st.plotly_chart(fig_age, use_container_width=True)

    # Fila 2: Label y Actividad
    col3, col4 = st.columns(2)

    with col3:
        # Distribución de label (solo para nuevos clientes)
        if (df_filtered['label'] != -1).any():
            df_new_filtered = df_filtered[df_filtered['label'] != -1]
            label_dist = df_new_filtered['label'].value_counts().sort_index()

            fig_label = go.Figure(data=[
                go.Pie(
                    labels=['No Recurrente', 'Recurrente'],
                    values=label_dist.values,
                    marker_colors=[COLOR_PALETTE['secondary'], COLOR_PALETTE['primary']],
                    hole=0.4,
                    textinfo='label+percent+value'
                )
            ])
            fig_label.update_layout(
                title="Clientes Recurrentes vs No Recurrentes",
                height=400
            )
            st.plotly_chart(fig_label, use_container_width=True)

    with col4:
        # Distribución de longitud de actividad
        fig_activity = go.Figure(data=[
            go.Histogram(
                x=df_filtered['activity_len'],
                nbinsx=50,
                marker_color=COLOR_PALETTE['primary'],
                opacity=0.7
            )
        ])
        fig_activity.update_layout(
            title="Distribución de Longitud de Actividad",
            xaxis_title="Cantidad de Interacciones",
            yaxis_title="Frecuencia",
            height=400
        )
        st.plotly_chart(fig_activity, use_container_width=True)

    # Fila 3: Tipos de acciones
    st.subheader("Distribución de Tipos de Acciones")

    actions_data = pd.DataFrame({
        'Tipo de Acción': ['Clics/Vistas (0)', 'Añadir al Carrito (2)', 'Compras (3)'],
        'Total': [
            df_filtered['actions_0'].sum(),
            df_filtered['actions_2'].sum(),
            df_filtered['actions_3'].sum()
        ]
    })

    fig_actions = go.Figure(data=[
        go.Bar(
            x=actions_data['Tipo de Acción'],
            y=actions_data['Total'],
            marker_color=[COLOR_PALETTE['tertiary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['primary']],
            text=actions_data['Total'],
            textposition='auto',
        )
    ])
    fig_actions.update_layout(
        title="Total de Acciones por Tipo",
        xaxis_title="Tipo de Acción",
        yaxis_title="Cantidad Total",
        height=400
    )
    st.plotly_chart(fig_actions, use_container_width=True)

    # Fila 4: Diversidad (items, categorías, marcas)
    col5, col6, col7 = st.columns(3)

    with col5:
        fig_items = go.Figure(data=[
            go.Box(
                y=df_filtered['unique_items'],
                marker_color=COLOR_PALETTE['primary'],
                name='Items Únicos'
            )
        ])
        fig_items.update_layout(
            title="Distribución de Items Únicos",
            yaxis_title="Cantidad",
            height=350
        )
        st.plotly_chart(fig_items, use_container_width=True)

    with col6:
        fig_categories = go.Figure(data=[
            go.Box(
                y=df_filtered['unique_categories'],
                marker_color=COLOR_PALETTE['secondary'],
                name='Categorías Únicas'
            )
        ])
        fig_categories.update_layout(
            title="Distribución de Categorías Únicas",
            yaxis_title="Cantidad",
            height=350
        )
        st.plotly_chart(fig_categories, use_container_width=True)

    with col7:
        fig_brands = go.Figure(data=[
            go.Box(
                y=df_filtered['unique_brands'],
                marker_color=COLOR_PALETTE['tertiary'],
                name='Marcas Únicas'
            )
        ])
        fig_brands.update_layout(
            title="Distribución de Marcas Únicas",
            yaxis_title="Cantidad",
            height=350
        )
        st.plotly_chart(fig_brands, use_container_width=True)

def show_correlations(df_filtered):
    """Visualizaciones de correlaciones"""

    st.subheader("Matriz de Correlaciones")

    # Seleccionar variables numéricas
    numeric_cols = [
        'age_range', 'gender', 'label', 'activity_len',
        'actions_0', 'actions_2', 'actions_3',
        'unique_items', 'unique_categories', 'unique_brands',
        'day_span', 'has_1111'
    ]

    # Filtrar solo nuevos clientes para correlación con label
    df_corr = df_filtered[df_filtered['label'] != -1][numeric_cols].copy()

    # Calcular correlación
    corr_matrix = df_corr.corr()

    # Heatmap de correlación
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[
            [0, '#FFFFFF'],
            [0.5, COLOR_PALETTE['secondary']],
            [1, COLOR_PALETTE['primary']]
        ],
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlación")
    ))

    fig_corr.update_layout(
        title="Matriz de Correlación de Variables",
        height=600,
        xaxis={'side': 'bottom'},
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    # Correlación con label
    st.subheader("Correlación con Variable Objetivo (Label)")

    label_corr = corr_matrix['label'].drop('label').sort_values(ascending=True)

    fig_label_corr = go.Figure(data=[
        go.Bar(
            x=label_corr.values,
            y=label_corr.index,
            orientation='h',
            marker_color=[
                COLOR_PALETTE['primary'] if x > 0 else COLOR_PALETTE['secondary']
                for x in label_corr.values
            ],
            text=label_corr.values.round(3),
            textposition='auto',
        )
    ])

    fig_label_corr.update_layout(
        title="Correlación de Variables con Recurrencia del Cliente",
        xaxis_title="Coeficiente de Correlación",
        yaxis_title="Variable",
        height=500
    )

    st.plotly_chart(fig_label_corr, use_container_width=True)

    # Scatter plots interactivos
    st.subheader("Relaciones entre Variables")

    col1, col2 = st.columns(2)

    with col1:
        x_var = st.selectbox(
            "Variable X:",
            options=numeric_cols,
            index=numeric_cols.index('activity_len')
        )

    with col2:
        y_var = st.selectbox(
            "Variable Y:",
            options=numeric_cols,
            index=numeric_cols.index('actions_3')
        )

    # Crear scatter plot
    df_scatter = df_filtered[df_filtered['label'] != -1].copy()
    df_scatter['label_text'] = df_scatter['label'].map({0: 'No Recurrente', 1: 'Recurrente'})

    fig_scatter = px.scatter(
        df_scatter,
        x=x_var,
        y=y_var,
        color='label_text',
        color_discrete_map={
            'No Recurrente': COLOR_PALETTE['secondary'],
            'Recurrente': COLOR_PALETTE['primary']
        },
        opacity=0.6,
        title=f"Relación entre {x_var} y {y_var}",
        labels={'label_text': 'Tipo de Cliente'}
    )

    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

def show_temporal_analysis(df_filtered):
    """Análisis temporal"""

    st.subheader("Análisis Temporal de Interacciones")

    # Distribución de day_span
    col1, col2 = st.columns(2)

    with col1:
        fig_span = go.Figure(data=[
            go.Histogram(
                x=df_filtered['day_span'],
                nbinsx=50,
                marker_color=COLOR_PALETTE['primary'],
                opacity=0.7
            )
        ])
        fig_span.update_layout(
            title="Distribución de Duración de Actividad (días)",
            xaxis_title="Días entre primera y última interacción",
            yaxis_title="Frecuencia",
            height=400
        )
        st.plotly_chart(fig_span, use_container_width=True)

    with col2:
        # Day span por label
        df_temp = df_filtered[df_filtered['label'] != -1].copy()
        df_temp['label_text'] = df_temp['label'].map({0: 'No Recurrente', 1: 'Recurrente'})

        fig_span_label = go.Figure()

        for label, color in [('No Recurrente', COLOR_PALETTE['secondary']),
                             ('Recurrente', COLOR_PALETTE['primary'])]:
            data = df_temp[df_temp['label_text'] == label]['day_span']
            fig_span_label.add_trace(go.Box(
                y=data,
                name=label,
                marker_color=color
            ))

        fig_span_label.update_layout(
            title="Duración de Actividad por Tipo de Cliente",
            yaxis_title="Días",
            height=400
        )
        st.plotly_chart(fig_span_label, use_container_width=True)

    # Actividad por mes
    st.subheader("Actividad por Periodo")

    df_filtered['month_min'] = df_filtered['date_min'].dt.to_period('M').astype(str)
    monthly_activity = df_filtered.groupby('month_min').size().reset_index(name='count')

    fig_monthly = go.Figure(data=[
        go.Bar(
            x=monthly_activity['month_min'],
            y=monthly_activity['count'],
            marker_color=COLOR_PALETTE['secondary'],
            text=monthly_activity['count'],
            textposition='auto',
        )
    ])

    fig_monthly.update_layout(
        title="Número de Interacciones por Mes (fecha inicial)",
        xaxis_title="Mes",
        yaxis_title="Cantidad de Registros",
        height=400
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Impacto del Double 11
    st.subheader("Impacto del Double 11 (11/11)")

    col3, col4 = st.columns(2)

    with col3:
        double11_dist = df_filtered['has_1111'].value_counts().sort_index()

        fig_1111 = go.Figure(data=[
            go.Pie(
                labels=['Sin actividad en 11/11', 'Con actividad en 11/11'],
                values=double11_dist.values,
                marker_colors=[COLOR_PALETTE['secondary'], COLOR_PALETTE['primary']],
                hole=0.4,
                textinfo='label+percent+value'
            )
        ])
        fig_1111.update_layout(
            title="Distribución de Actividad en Double 11",
            height=400
        )
        st.plotly_chart(fig_1111, use_container_width=True)

    with col4:
        # Recurrencia por participación en Double 11
        df_1111 = df_filtered[df_filtered['label'] != -1].copy()

        double11_recurrence = df_1111.groupby('has_1111')['label'].agg(['sum', 'count'])
        double11_recurrence['percentage'] = (double11_recurrence['sum'] / double11_recurrence['count'] * 100)

        fig_1111_impact = go.Figure(data=[
            go.Bar(
                x=['Sin 11/11', 'Con 11/11'],
                y=double11_recurrence['percentage'].values,
                marker_color=[COLOR_PALETTE['secondary'], COLOR_PALETTE['primary']],
                text=double11_recurrence['percentage'].values.round(1),
                texttemplate='%{text}%',
                textposition='auto',
            )
        ])
        fig_1111_impact.update_layout(
            title="% Recurrencia según participación en Double 11",
            yaxis_title="% Clientes Recurrentes",
            height=400
        )
        st.plotly_chart(fig_1111_impact, use_container_width=True)

def show_rfm_analysis(df_filtered):
    """Segmentación RFM (Recency, Frequency, Monetary)"""

    st.subheader("Segmentación RFM de Clientes")

    st.info("""
    **Análisis RFM adaptado:**
    - **Recency (R)**: Inverso de `day_span` (clientes más recientes tienen menor day_span)
    - **Frequency (F)**: `activity_len` (cantidad de interacciones)
    - **Monetary (M)**: `actions_3` (cantidad de compras realizadas)
    """)

    # Calcular scores RFM
    df_rfm = df_filtered.copy()

    # Recency: invertir day_span (menor day_span = más reciente = mejor score)
    try:
        df_rfm['R_score'] = pd.qcut(df_rfm['day_span'], q=4, labels=False, duplicates='drop')
        # Invertir: menor day_span = mejor score
        df_rfm['R_score'] = 4 - df_rfm['R_score']
    except:
        # Si falla qcut, usar percentiles manualmente
        df_rfm['R_score'] = pd.cut(df_rfm['day_span'], bins=4, labels=False, duplicates='drop')
        df_rfm['R_score'] = 4 - df_rfm['R_score']

    # Frequency: activity_len
    try:
        df_rfm['F_score'] = pd.qcut(df_rfm['activity_len'], q=4, labels=False, duplicates='drop')
    except:
        df_rfm['F_score'] = pd.cut(df_rfm['activity_len'], bins=4, labels=False, duplicates='drop')

    # Monetary: actions_3 (compras)
    try:
        df_rfm['M_score'] = pd.qcut(df_rfm['actions_3'], q=4, labels=False, duplicates='drop')
    except:
        df_rfm['M_score'] = pd.cut(df_rfm['actions_3'], bins=4, labels=False, duplicates='drop')

    # Rellenar NaN con 0 y convertir a numérico
    df_rfm['R_score'] = df_rfm['R_score'].fillna(0).astype(int) + 1
    df_rfm['F_score'] = df_rfm['F_score'].fillna(0).astype(int) + 1
    df_rfm['M_score'] = df_rfm['M_score'].fillna(0).astype(int) + 1

    # Score total
    df_rfm['RFM_score'] = df_rfm['R_score'] + df_rfm['F_score'] + df_rfm['M_score']

    # Segmentos
    def rfm_segment(score):
        if score >= 10:
            return 'Champions'
        elif score >= 8:
            return 'Loyal Customers'
        elif score >= 6:
            return 'Potential Loyalists'
        elif score >= 4:
            return 'At Risk'
        else:
            return 'Lost'

    df_rfm['Segment'] = df_rfm['RFM_score'].apply(rfm_segment)

    # Visualizar distribución de segmentos
    col1, col2 = st.columns(2)

    with col1:
        segment_dist = df_rfm['Segment'].value_counts()

        colors_map = {
            'Champions': COLOR_PALETTE['primary'],
            'Loyal Customers': COLOR_PALETTE['secondary'],
            'Potential Loyalists': COLOR_PALETTE['tertiary'],
            'At Risk': COLOR_PALETTE['quaternary'],
            'Lost': COLOR_PALETTE['unknown']
        }

        fig_segments = go.Figure(data=[
            go.Pie(
                labels=segment_dist.index,
                values=segment_dist.values,
                marker_colors=[colors_map.get(seg, '#CCCCCC') for seg in segment_dist.index],
                textinfo='label+percent+value'
            )
        ])
        fig_segments.update_layout(
            title="Distribución de Segmentos RFM",
            height=400
        )
        st.plotly_chart(fig_segments, use_container_width=True)

    with col2:
        # Score RFM promedio por segmento
        rfm_by_segment = df_rfm.groupby('Segment')[['R_score', 'F_score', 'M_score']].mean()

        fig_rfm_scores = go.Figure()

        for component, color in [('R_score', COLOR_PALETTE['primary']),
                                 ('F_score', COLOR_PALETTE['secondary']),
                                 ('M_score', COLOR_PALETTE['tertiary'])]:
            fig_rfm_scores.add_trace(go.Bar(
                name=component.replace('_score', ''),
                x=rfm_by_segment.index,
                y=rfm_by_segment[component],
                marker_color=color
            ))

        fig_rfm_scores.update_layout(
            title="Scores RFM Promedio por Segmento",
            xaxis_title="Segmento",
            yaxis_title="Score Promedio",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_rfm_scores, use_container_width=True)

    # Tabla de estadísticas por segmento
    st.subheader("Estadísticas por Segmento")

    segment_stats = df_rfm.groupby('Segment').agg({
        'user_id': 'count',
        'activity_len': 'mean',
        'actions_3': 'mean',
        'day_span': 'mean',
        'RFM_score': 'mean'
    }).round(2)

    segment_stats.columns = ['Cantidad', 'Actividad Media', 'Compras Media', 'Day Span Medio', 'RFM Score']
    segment_stats = segment_stats.sort_values('RFM Score', ascending=False)

    st.dataframe(segment_stats, use_container_width=True)

    # Scatter 3D de RFM
    st.subheader("Visualización 3D de Segmentación RFM")

    # Muestreo para mejor rendimiento
    df_sample = df_rfm.sample(n=min(5000, len(df_rfm)), random_state=42)

    fig_3d = px.scatter_3d(
        df_sample,
        x='R_score',
        y='F_score',
        z='M_score',
        color='Segment',
        color_discrete_map=colors_map,
        opacity=0.6,
        labels={
            'R_score': 'Recency Score',
            'F_score': 'Frequency Score',
            'M_score': 'Monetary Score'
        },
        title="Distribución 3D de Segmentos RFM"
    )

    fig_3d.update_layout(height=600)
    st.plotly_chart(fig_3d, use_container_width=True)

    # Relación entre segmento y recurrencia
    if (df_rfm['label'] != -1).any():
        st.subheader("Segmentos RFM vs Recurrencia Real")

        df_rfm_label = df_rfm[df_rfm['label'] != -1].copy()
        segment_recurrence = df_rfm_label.groupby('Segment')['label'].agg(['sum', 'count'])
        segment_recurrence['percentage'] = (segment_recurrence['sum'] / segment_recurrence['count'] * 100)
        segment_recurrence = segment_recurrence.sort_values('percentage', ascending=False)

        fig_seg_rec = go.Figure(data=[
            go.Bar(
                x=segment_recurrence.index,
                y=segment_recurrence['percentage'],
                marker_color=[colors_map.get(seg, '#CCCCCC') for seg in segment_recurrence.index],
                text=segment_recurrence['percentage'].round(1),
                texttemplate='%{text}%',
                textposition='auto',
            )
        ])

        fig_seg_rec.update_layout(
            title="% de Recurrencia Real por Segmento RFM",
            xaxis_title="Segmento",
            yaxis_title="% Clientes Recurrentes",
            height=400
        )
        st.plotly_chart(fig_seg_rec, use_container_width=True)
