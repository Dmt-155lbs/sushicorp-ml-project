import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, make_scorer, recall_score
import os
import joblib
import time

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
    
    # Calcular scale_pos_weight para el desbalance
    class_0_count = y_train.value_counts()[0]
    class_1_count = y_train.value_counts()[1]
    spw = class_0_count / class_1_count
    print(f"Desbalance detectado: Clase 0 = {class_0_count}, Clase 1 = {class_1_count}")
    print(f"Aplicando scale_pos_weight = {spw:.2f}")
    
    # Definir espacio de hiperparámetros
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    # Inicializar modelo base
    base_model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric='logloss'
        # Nota: en nuevas versiones de xgboost, use_label_encoder no es necesario
    )
    # Pasar scale_pos_weight al fit a traves de params del estimador base
    base_model.set_params(scale_pos_weight=spw)
    
    # Usar Recall como métrica principal para la búsqueda
    scorer = make_scorer(recall_score, pos_label=1)
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=15,
        scoring=scorer,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Entrenamiento con Random Search
    print("Entrenando modelo con RandomizedSearchCV optimizando Recall...")
    start_time = time.time()
    random_search.fit(X_train, y_train)
    print(f"Tuning completado en {time.time() - start_time:.2f} segundos.")
    
    best_model = random_search.best_estimator_
    print(f"Mejores hiperparámetros encontrados:\n{random_search.best_params_}")
    
    # Predicciones
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Evaluacion orientada a negocio
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\n--- Nuevas Métricas de Evaluación ({task_name}) ---")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nReporte de Clasificación:")
    report = classification_report(y_test, y_pred)
    print(report)
    
    # Guardar el modelo optimizado para Fases posteriores
    os.makedirs('models', exist_ok=True)
    model_path = f'models/xgboost_{target_col}_optimizado.joblib'
    joblib.dump(best_model, model_path)
    print(f"Modelo optimizado guardado exitosamente en {model_path}")
    
    # Generar un archivo temporal con métricas
    with open(f'models/metrics_{target_col}_optimizado.txt', 'w', encoding='utf-8') as f:
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Mejores hiperparametros: {random_search.best_params_}\n")
        f.write(f"Scale_pos_weight: {spw:.2f}\n")
        f.write("Reporte de Clasificacion:\n")
        f.write(report)
        
    return best_model

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