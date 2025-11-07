from predict import load_pipeline

pipeline = load_pipeline()

clientes = {
    'Cliente ELITE (P90 en todo)': {
        'date_max': '2014-11-11',
        'activity_len': 23,
        'actions_3': 1,
        'unique_items': 18,
        'unique_categories': 6,
        'unique_brands': 1,
        'day_span': 365,
        'has_1111': 1
    },
    'Cliente SUPER ELITE (P95)': {
        'date_max': '2014-11-11',
        'activity_len': 35,
        'actions_3': 2,
        'unique_items': 25,
        'unique_categories': 10,
        'unique_brands': 5,
        'day_span': 365,
        'has_1111': 1
    },
    'Cliente PROMEDIO': {
        'date_max': '2014-11-11',
        'activity_len': 6,
        'actions_3': 0,
        'unique_items': 3,
        'unique_categories': 2,
        'unique_brands': 1,
        'day_span': 30,
        'has_1111': 0
    },
    'Cliente POBRE': {
        'date_max': '2014-01-01',
        'activity_len': 1,
        'actions_3': 0,
        'unique_items': 1,
        'unique_categories': 1,
        'unique_brands': 1,
        'day_span': 0,
        'has_1111': 0
    },
    'Tu Cliente Original (recency=0)': {
        'date_max': '2014-11-11',
        'activity_len': 20,
        'actions_3': 20,
        'unique_items': 100,
        'unique_categories': 30,
        'unique_brands': 20,
        'day_span': 365,
        'has_1111': 1
    }
}

print("\n" + "="*80)
print("PREDICCIÓN DE LEALTAD - CLIENTES BASADOS EN DATOS REALES")
print("="*80)

for nombre, datos in clientes.items():
    resultado_rf = pipeline.predict_single(datos)
    resultado_xgb = pipeline.predict_single_xgb(datos)

    print(f"\n{nombre}")
    print("-"*80)
    print(f"Datos de entrada:")
    print(f"  - Última compra: {datos['date_max']}")
    print(f"  - Actividades: {datos['activity_len']}")
    print(f"  - Compras: {datos['actions_3']}")
    print(f"  - Items únicos: {datos['unique_items']}")
    print(f"  - Black Friday: {'Sí' if datos['has_1111'] == 1 else 'No'}")

    print(f"\nFeatures RFM:")
    print(f"  - Recency (días): {resultado_rf['input_features']['recency_days']}")
    print(f"  - RFM Score: {resultado_rf['input_features']['RFM_score']:.2f}")
    print(f"  - Frequency: {resultado_rf['input_features']['frequency_activity']}")
    print(f"  - Monetary: {resultado_rf['input_features']['monetary_proxy']}")

    print(f"\n{'='*25} RANDOM FOREST {'='*25}")
    print(f"Predicciones RF:")
    print(f"  - All Features: {'LEAL' if resultado_rf['model_predictions']['all_features']['prediction'] == 1 else 'NO LEAL'} ({resultado_rf['model_predictions']['all_features']['probability']*100:.2f}%)")
    print(f"  - Selected: {'LEAL' if resultado_rf['model_predictions']['selected_features']['prediction'] == 1 else 'NO LEAL'} ({resultado_rf['model_predictions']['selected_features']['probability']*100:.2f}%)")
    print(f"\nResultado Ensemble RF: {'✓ LEAL' if resultado_rf['ensemble_prediction'] == 1 else '✗ NO LEAL'}")
    print(f"Probabilidad: {resultado_rf['loyalty_score']*100:.2f}%")
    print(f"Confianza: {resultado_rf['confidence']['confidence_level']}")

    print(f"\n{'='*25} XGBOOST {'='*25}")
    if 'error' in resultado_xgb:
        print(f"⚠ {resultado_xgb['error']}")
    else:
        print(f"Predicción XGBoost: {'✓ LEAL' if resultado_xgb['xgb_prediction'] == 1 else '✗ NO LEAL'}")
        print(f"Probabilidad: {resultado_xgb['xgb_probability']*100:.2f}%")
        print(f"Umbral: {resultado_xgb['xgb_threshold']:.4f}")

print("\n" + "="*80)
print("CONCLUSIONES")
print("="*80)
print("\nComparación de Modelos:")
print("- RANDOM FOREST: Usa features derivadas (RFM, scores)")
print("- XGBOOST: Usa features crudas del dataset (age_range, merchant_id, etc.)")
print("\nCada modelo tiene su propia estrategia:")
print("  RF: Analiza patrones de compra históricos (Recency, Frequency, Monetary)")
print("  XGB: Considera características del cliente y su comportamiento general")
print("\nUmbral Óptimo XGBoost: 0.08 (encontrado en validación con F1-Score)")
print("\nSegún los datos de entrenamiento:")
print("- Recency=0 (última compra el 2014-11-11) es lo más común")
print("- P80 de frequency_activity = 14")
print("- P80 de monetary_proxy = 9")
print("- Clientes con recency > 0 tienen PEOR score en RF")
print("\nPor eso tu cliente original con recency=1 tiene score bajo en RF.")
print("Si cambias date_max a '2014-11-11', el score de RF debería mejorar.")
print("="*80 + "\n")