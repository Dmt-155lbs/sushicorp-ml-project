import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración de estilo profesional para gráficas
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

def load_data(filepath):
    print(f"Cargando datos desde {filepath}...")
    df = pd.read_csv(filepath)
    return df

def perform_eda(df):
    print("Iniciando Análisis Exploratorio de Datos (EDA)...")
    os.makedirs('plots', exist_ok=True)
    
    # 1. Análisis de Churn Delivery
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Churn vs Quejas Recientes
    sns.boxplot(x='churn_delivery', y='quejas_recientes', data=df, ax=axes[0], palette=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Impacto de las Quejas en la Fuga de Clientes (Churn)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Churn Delivery (0 = Retenido, 1 = Fugado)', fontsize=12)
    axes[0].set_ylabel('Cantidad de Quejas Recientes', fontsize=12)
    
    # Churn vs Tiempo de Entrega Promedio
    sns.violinplot(x='churn_delivery', y='tiempo_entrega_promedio_min', data=df, ax=axes[1], palette=['#2ecc71', '#e74c3c'])
    axes[1].set_title('Tiempo de Entrega vs Riesgo de Fuga', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Churn Delivery (0 = Retenido, 1 = Fugado)', fontsize=12)
    axes[1].set_ylabel('Tiempo de Entrega Promedio (min)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('plots/eda_churn_insights.png', dpi=300)
    plt.close()
    
    # 2. Análisis de Conversión al Salón (O2O)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Conversión vs Distancia
    sns.boxplot(x='conversion_salon', y='distancia_local_km', data=df, ax=axes[0], palette=['#95a5a6', '#3498db'])
    axes[0].set_title('Distancia al Local como Barrera para Visita Física', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Conversión a Salón (0 = Solo Delivery, 1 = Visitó Salón)', fontsize=12)
    axes[0].set_ylabel('Distancia al Local (km)', fontsize=12)
    
    # Conversión vs Sensibilidad a Promociones
    sns.kdeplot(data=df, x='sensibilidad_promos', hue='conversion_salon', fill=True, ax=axes[1], palette=['#95a5a6', '#3498db'])
    axes[1].set_title('Distribución de Sensibilidad a Promociones', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Score de Sensibilidad a Promociones', fontsize=12)
    axes[1].set_ylabel('Densidad', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('plots/eda_conversion_insights.png', dpi=300)
    plt.close()
    
    # 3. Matriz de Correlación
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
    plt.title('Matriz de Correlación de Variables', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/eda_correlation_matrix.png', dpi=300)
    plt.close()
    print("Gráficas de EDA generadas exitosamente en la carpeta 'plots/'.")

def feature_engineering(df):
    print("Iniciando Ingeniería de Características (Feature Engineering)...")
    df_engineered = df.copy()
    
    # Creación de variables derivadas (Insights de Negocio)
    # Valor del cliente en los últimos 6 meses
    df_engineered['valor_6m_aprox'] = df_engineered['frecuencia_6m'] * df_engineered['ticket_promedio_usd']
    
    # Ratio de quejas sobre frecuencia (intensidad de la insatisfacción)
    df_engineered['tasa_quejas_por_pedido'] = df_engineered['quejas_recientes'] / (df_engineered['frecuencia_6m'] + 1)
    
    # Atractividad de promoción vs esfuerzo físico (distancia)
    df_engineered['promocion_vs_distancia'] = df_engineered['sensibilidad_promos'] / (df_engineered['distancia_local_km'] + 0.1)
    
    # Codificación de variables categóricas
    df_engineered = pd.get_dummies(df_engineered, columns=['marca_principal'], drop_first=True)
    
    # Eliminación del identificador (no predictivo)
    df_engineered.drop('customer_id', axis=1, inplace=True)
    
    print(f"Ingeniería de Características completada. Forma final del dataset: {df_engineered.shape}")
    return df_engineered

def main():
    filepath = 'sushicorp_datos_sinteticos.csv'
    output_path = 'sushicorp_datos_preparados.csv'
    
    df = load_data(filepath)
    perform_eda(df)
    df_engineered = feature_engineering(df)
    
    # Guardar datos preparados
    df_engineered.to_csv(output_path, index=False)
    print(f"Datos preparados guardados en {output_path}")

if __name__ == "__main__":
    main()
