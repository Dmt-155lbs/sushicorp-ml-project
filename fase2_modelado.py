import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import os
import joblib
import time

def load_data(filepath):
    print(f"Cargando datos preparados desde {filepath}...")
    return pd.read_csv(filepath)

def train_and_evaluate_ensemble(df, target_col, task_name):
    print(f"\n{'='*50}\nIniciando modelado (Ensemble) para: {task_name}\n{'='*50}")
    
    X = df.drop(['churn_delivery', 'conversion_salon'], axis=1)
    y = df[target_col]
    
    # Train-test split (80/20) - Stratify para mantener distribución de clases
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Calcular class weights
    class_0_count = y_train.value_counts()[0]
    class_1_count = y_train.value_counts()[1]
    spw = class_0_count / class_1_count
    
    print(f"Desbalance detectado: Clase 0 = {class_0_count}, Clase 1 = {class_1_count}")
    print(f"Aplicando scale_pos_weight = {spw:.2f}")
    
    # 1. XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=spw,
        eval_metric='logloss',
        random_state=42
    )
    
    # 2. LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=spw,
        random_state=42,
        verbose=-1
    )
    
    # 3. Random Forest (usa class_weight='balanced')
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=7,
        class_weight='balanced',
        random_state=42
    )
    
    # Ensemble: Voting Classifier
    # Usamos voting='soft' para tener predict_proba y calcular AUC
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('rf', rf_model)
        ],
        voting='soft'
    )
    
    print("Entrenando VotingClassifier (XGBoost + LightGBM + Random Forest)...")
    start_time = time.time()
    ensemble.fit(X_train, y_train)
    print(f"Entrenamiento completado en {time.time() - start_time:.2f} segundos.")
    
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\n--- Nuevas Métricas de Evaluación ({task_name}) ---")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nReporte de Clasificación:")
    report = classification_report(y_test, y_pred)
    print(report)
    
    os.makedirs('models', exist_ok=True)
    model_path = f'models/ensemble_{target_col}_v3.joblib'
    joblib.dump(ensemble, model_path)
    print(f"Modelo ensemble guardado exitosamente en {model_path}")
    
    with open(f'models/metrics_{target_col}_ensemble_v3.txt', 'w', encoding='utf-8') as f:
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Ensemble: VotingClassifier (XGB, LGBM, RF)\n")
        f.write(f"Scale_pos_weight / class_weight='balanced' aplicado\n")
        f.write("Reporte de Clasificacion:\n")
        f.write(report)
        
    return ensemble

def main():
    filepath = 'sushicorp_datos_preparados.csv'
    if not os.path.exists(filepath):
        print(f"Error: No se encontró {filepath}. Ejecuta la Fase 1 primero.")
        return
        
    df = load_data(filepath)
    
    # 1. Modelo de Churn Delivery
    train_and_evaluate_ensemble(df, 'churn_delivery', 'Predicción de Fuga (Churn Delivery)')
    
    # 2. Modelo de Conversión a Salón
    train_and_evaluate_ensemble(df, 'conversion_salon', 'Propensión de Visita al Salón (Cross-Selling)')
    
    print("\nFase 2 completada exitosamente.")

if __name__ == "__main__":
    main()