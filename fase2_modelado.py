import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import os
import joblib

def load_data(filepath):
    print(f"Cargando datos preparados desde {filepath}...")
    return pd.read_csv(filepath)

def train_and_evaluate(df, target_col, task_name):
    print(f"\n{'='*50}\nIniciando modelado para: {task_name}\n{'='*50}")
    
    # Las targets no deben estar en X
    X = df.drop(['churn_delivery', 'conversion_salon'], axis=1)
    y = df[target_col]
    
    # Train-test split (80/20) - Stratify para mantener distribución de clases
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Inicializar XGBoost Classifier (parametrización robusta para evitar overfitting)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Entrenamiento
    print("Entrenando modelo XGBoost...")
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Evaluacion orientada a negocio
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\n--- Métricas de Evaluación ({task_name}) ---")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Guardar el modelo para Fases posteriores
    os.makedirs('models', exist_ok=True)
    model_path = f'models/xgboost_{target_col}.joblib'
    joblib.dump(model, model_path)
    print(f"Modelo guardado exitosamente en {model_path}")
    
    # Opcional: Generar un archivo temporal con métricas para parsearlo luego o mostrarlo en output
    with open(f'models/metrics_{target_col}.txt', 'w', encoding='utf-8') as f:
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(classification_report(y_test, y_pred))
        
    return model

def main():
    filepath = 'sushicorp_datos_preparados.csv'
    if not os.path.exists(filepath):
        print(f"Error: No se encontró {filepath}. Ejecuta la Fase 1 primero.")
        return
        
    df = load_data(filepath)
    
    # 1. Modelo de Churn Delivery
    train_and_evaluate(df, 'churn_delivery', 'Predicción de Fuga (Churn Delivery)')
    
    # 2. Modelo de Conversión a Salón
    train_and_evaluate(df, 'conversion_salon', 'Propensión de Visita al Salón (Cross-Selling)')
    
    print("\nFase 2 completada exitosamente.")

if __name__ == "__main__":
    main()
