import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

class LoyaltyPredictionPipeline:
    def __init__(self, models_dir='./models/saved_models/Random Forest'):
        self.models_dir = Path(models_dir)
        self.rf_all = None
        self.rf_selected = None
        self.metadata = None
        self.reference_date = pd.to_datetime('2014-11-11')
        self.training_data = None
        
    def load_models(self):
        try:
            with open(self.models_dir / 'rf_optimized_all_features.pkl', 'rb') as f:
                self.rf_all = pickle.load(f)
            print("✓ Modelo RF (All Features) cargado")
            
            with open(self.models_dir / 'rf_optimized_selected_features.pkl', 'rb') as f:
                self.rf_selected = pickle.load(f)
            print("✓ Modelo RF (Selected Features) cargado")
            
            with open(self.models_dir / 'rf_metadata.pkl', 'rb') as f:
                self.metadata = pickle.load(f)
            print("✓ Metadata cargada")
            
        except Exception as e:
            raise Exception(f"Error cargando modelos: {str(e)}")
    
    def load_training_data(self, train_path):
        try:
            self.training_data = pd.read_csv(train_path)
            self.training_data = self.training_data[self.training_data['label'].isin([0, 1])].copy()
            
            self.training_data['date_max'] = pd.to_datetime(self.training_data['date_max'], errors='coerce')
            self.training_data['recency_days'] = (self.reference_date - self.training_data['date_max']).dt.days
            self.training_data['recency_days'] = self.training_data['recency_days'].fillna(9999).astype(int)
            
            self.training_data['frequency_activity'] = self.training_data['activity_len'].fillna(0).astype(int)
            self.training_data['frequency_purchases'] = self.training_data['actions_3'].fillna(0).astype(int)
            self.training_data['monetary_proxy'] = (
                self.training_data[['unique_items', 'unique_categories', 'unique_brands']]
                .fillna(0).sum(axis=1).astype(int)
            )
            
            print("✓ Datos de entrenamiento cargados")
        except Exception as e:
            raise Exception(f"Error cargando datos de entrenamiento: {str(e)}")
    
    def _quantile_score(self, value, feature_name, bins=5, invert=False):
        if self.training_data is None:
            raise Exception("Datos de entrenamiento no cargados. Llama a load_training_data() primero.")
        
        series = self.training_data[feature_name].dropna()
        
        if len(series) == 0:
            return 1
        
        if invert:
            worse_count = (series > value).sum()
            equal_count = (series == value).sum()
            percentile_rank = (worse_count + equal_count / 2) / len(series)
        else:
            better_count = (series < value).sum()
            equal_count = (series == value).sum()
            percentile_rank = (better_count + equal_count / 2) / len(series)
        
        if percentile_rank <= 0.20:
            score = 1
        elif percentile_rank <= 0.40:
            score = 2
        elif percentile_rank <= 0.60:
            score = 3
        elif percentile_rank <= 0.80:
            score = 4
        else:
            score = 5
        
        return score
    
    def preprocess_input(self, input_data):
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        df['date_max'] = pd.to_datetime(df['date_max'], errors='coerce')
        df['recency_days'] = (self.reference_date - df['date_max']).dt.days
        df['recency_days'] = df['recency_days'].fillna(9999).astype(int)
        
        df['frequency_activity'] = df.get('activity_len', 0)
        if 'activity_len' not in df.columns:
            df['frequency_activity'] = 0
        df['frequency_activity'] = df['frequency_activity'].fillna(0).astype(int)
        
        df['frequency_purchases'] = df.get('actions_3', 0)
        if 'actions_3' not in df.columns:
            df['frequency_purchases'] = 0
        df['frequency_purchases'] = df['frequency_purchases'].fillna(0).astype(int)
        
        df['monetary_proxy'] = 0
        for col in ['unique_items', 'unique_categories', 'unique_brands']:
            if col in df.columns:
                df['monetary_proxy'] += df[col].fillna(0)
        df['monetary_proxy'] = df['monetary_proxy'].astype(int)
        
        df['recency_score'] = df['recency_days'].apply(
            lambda x: self._quantile_score(x, 'recency_days', bins=5, invert=True)
        )
        df['frequency_score'] = df['frequency_activity'].apply(
            lambda x: self._quantile_score(x, 'frequency_activity', bins=5, invert=False)
        )
        df['purchases_score'] = df['frequency_purchases'].apply(
            lambda x: self._quantile_score(x, 'frequency_purchases', bins=5, invert=False)
        )
        df['monetary_score'] = df['monetary_proxy'].apply(
            lambda x: self._quantile_score(x, 'monetary_proxy', bins=5, invert=False)
        )
        
        df['RFM_score'] = (
            df['recency_score'] * 0.50 +
            df['frequency_score'] * 0.25 +
            df['monetary_score'] * 0.25
        )
        
        df['multiple_dates'] = df.get('day_span', 0).fillna(0).apply(lambda x: 1 if x > 0 else 0)
        df['interacted_1111'] = df.get('has_1111', 0).fillna(0).astype(int)
        
        all_features = self.metadata['all_features']
        selected_features = self.metadata['selected_features']
        
        X_all = df[all_features]
        X_selected = df[selected_features]
        
        return X_all, X_selected
    
    def predict_single(self, input_data):
        X_all, X_selected = self.preprocess_input(input_data)
        
        pred_all = self.rf_all.predict(X_all)[0]
        proba_all = self.rf_all.predict_proba(X_all)[0]
        
        pred_selected = self.rf_selected.predict(X_selected)[0]
        proba_selected = self.rf_selected.predict_proba(X_selected)[0]
        
        pred_all = 1 - pred_all
        proba_all_inverted = proba_all[0]
        
        pred_selected = 1 - pred_selected
        proba_selected_inverted = proba_selected[0]
        
        ensemble_pred = 1 if (pred_all + pred_selected) >= 1 else 0
        
        avg_proba = (proba_all_inverted + proba_selected_inverted) / 2
        
        max_proba_diff = abs(proba_all_inverted - proba_selected_inverted)
        if max_proba_diff < 0.1:
            confidence_level = "ALTA"
        elif max_proba_diff < 0.3:
            confidence_level = "MEDIA"
        else:
            confidence_level = "BAJA"
        
        return {
            'ensemble_prediction': ensemble_pred,
            'loyalty_score': avg_proba,
            'model_predictions': {
                'all_features': {
                    'prediction': int(pred_all),
                    'probability': float(proba_all_inverted)
                },
                'selected_features': {
                    'prediction': int(pred_selected),
                    'probability': float(proba_selected_inverted)
                }
            },
            'confidence': {
                'confidence_level': confidence_level,
                'model_agreement': max_proba_diff < 0.1,
                'probability_diff': float(max_proba_diff)
            },
            'input_features': {
                'recency_days': int(X_all['recency_days'].values[0]),
                'RFM_score': float(X_all['RFM_score'].values[0]),
                'frequency_activity': int(X_all['frequency_activity'].values[0]),
                'monetary_proxy': int(X_all['monetary_proxy'].values[0])
            }
        }
    
    def predict_batch(self, input_data):
        X_all, X_selected = self.preprocess_input(input_data)
        
        pred_all = self.rf_all.predict(X_all)
        proba_all = self.rf_all.predict_proba(X_all)[:, 1]
        
        pred_selected = self.rf_selected.predict(X_selected)
        proba_selected = self.rf_selected.predict_proba(X_selected)[:, 1]
        
        ensemble_pred = ((pred_all + pred_selected) >= 1).astype(int)
        avg_proba = (proba_all + proba_selected) / 2
        
        results = pd.DataFrame({
            'ensemble_prediction': ensemble_pred,
            'loyalty_score': avg_proba,
            'pred_all_features': pred_all,
            'proba_all_features': proba_all,
            'pred_selected_features': pred_selected,
            'proba_selected_features': proba_selected,
            'RFM_score': X_all['RFM_score'].values,
            'recency_days': X_all['recency_days'].values
        })
        
        return results
    
    def generate_report(self, predictions_df):
        total = len(predictions_df)
        predicted_loyal = (predictions_df['ensemble_prediction'] == 1).sum()
        
        avg_loyalty_score = predictions_df['loyalty_score'].mean()
        
        high_confidence = (predictions_df['loyalty_score'] > 0.7).sum()
        medium_confidence = ((predictions_df['loyalty_score'] >= 0.3) & 
                           (predictions_df['loyalty_score'] <= 0.7)).sum()
        low_confidence = (predictions_df['loyalty_score'] < 0.3).sum()
        
        report = {
            'total_predictions': total,
            'predicted_loyal': int(predicted_loyal),
            'predicted_loyal_pct': float(predicted_loyal / total * 100),
            'avg_loyalty_score': float(avg_loyalty_score),
            'confidence_distribution': {
                'high': int(high_confidence),
                'medium': int(medium_confidence),
                'low': int(low_confidence)
            },
            'top_loyal_candidates': predictions_df.nlargest(10, 'loyalty_score')[
                ['ensemble_prediction', 'loyalty_score', 'RFM_score']
            ].to_dict('records')
        }
        
        return report

def load_pipeline(models_dir='./models/saved_models/Random Forest', train_path='./data/train_clean.csv'):
    pipeline = LoyaltyPredictionPipeline(models_dir)
    pipeline.load_models()
    pipeline.load_training_data(train_path)
    return pipeline