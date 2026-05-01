import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import mlflow
import mlflow.xgboost
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os

def load_data(filepath):
    print(f"Cargando datos preparados desde {filepath}...")
    return pd.read_csv(filepath)

def mlflow_shap_evaluation(df, target_col, model_path, task_name):
    print(f"\n{'='*50}\nFase 3: MLflow & SHAP para {task_name}\n{'='*50}")
    
    # Preparar datos
    X = df.drop(['churn_delivery', 'conversion_salon'], axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Cargar modelo entrenado en Fase 2
    model = joblib.load(model_path)
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # 1. MLflow Tracking
    mlflow.set_experiment("SushiCorp_ML_Project")
    with mlflow.start_run(run_name=task_name):
        # Loggear hiperparámetros
        mlflow.log_params(model.get_params())
        # Loggear métrica principal
        mlflow.log_metric("roc_auc", roc_auc)
        # Loggear el modelo
        mlflow.xgboost.log_model(model, artifact_path=f"model_{target_col}")
        print(f"Experimento guardado en MLflow con ROC AUC: {roc_auc:.4f}")
    
    # 2. Explicabilidad con SHAP
    print("Calculando SHAP values (puede tardar un momento)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    os.makedirs('plots', exist_ok=True)
    
    # Generar Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    shap_plot_path = f'plots/shap_summary_{target_col}.png'
    plt.savefig(shap_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Gráfico de SHAP guardado en {shap_plot_path}")
    
    return shap_values

def roi_analysis():
    print(f"\n{'='*50}\nAnálisis de Valor y ROI (Ejecutivo)\n{'='*50}")
    print("Suposición de Negocio para Churn:")
    print("- Costo de Retención (Promoción/Descuento preventivo): $5 USD")
    print("- Valor de Vida Promedio del Cliente (LTV): $150 USD")
    print("- Tasa de Aceptación Esperada: 30%")
    print("\nJustificación Estratégica:")
    print("Identificar al 20% más riesgoso y ofrecerles una promoción de $5 (antes de que se vayan) nos permite salvar a un porcentaje de esos clientes.")
    print("Si retenemos a 100 clientes que iban a fugar, invertimos $500 en promociones, pero recuperamos $15,000 en LTV potencial.")
    print("El modelo genera un ROI estimado masivo comparado con las campañas ciegas a toda la base de datos o el alto costo de intentar recuperarlos cuando ya se fueron a la competencia.")
    
def main():
    filepath = 'sushicorp_datos_preparados.csv'
    if not os.path.exists(filepath):
        print(f"Error: No se encontró {filepath}. Ejecuta las fases anteriores.")
        return
        
    df = load_data(filepath)
    
    # Evaluar Churn
    model_path_churn = 'models/xgboost_churn_delivery.joblib'
    if os.path.exists(model_path_churn):
        mlflow_shap_evaluation(df, 'churn_delivery', model_path_churn, 'Prediccion_Churn')
    
    # Evaluar Conversión
    model_path_conv = 'models/xgboost_conversion_salon.joblib'
    if os.path.exists(model_path_conv):
        mlflow_shap_evaluation(df, 'conversion_salon', model_path_conv, 'Propension_Salon')
        
    roi_analysis()
    print("\nFase 3 completada exitosamente.")

if __name__ == "__main__":
    main()
