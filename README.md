# SushiCorp AI: Churn & Cross-Selling Intelligence Platform 🍣

## Visión Ejecutiva
SushiCorp Ecuador se enfrenta al desafío de fidelizar a sus clientes de *delivery* y migrar el tráfico hacia la experiencia *in-situ* (conversión al salón/cross-selling). 

Este proyecto implementa modelos de Inteligencia Artificial (Machine Learning) orientados a resultados financieros. Abandonamos el enfoque clásico y priorizamos una estrategia de "Data-Centric AI", logrando maximizar la tasa de detección (Recall) en ambos frentes sin sobre-complicar la arquitectura.

### 💰 Impacto en el Negocio (ROI Comprobado)
Frente a una base de prueba simulada de 3,000 clientes, nuestros modelos Champion de la Fase 2 (XGBoost con balanceo riguroso de clases) lograron:
1. **Prevención de Fuga (Churn Delivery):** 
   - Aumento masivo de la capacidad de detección temprana del 18.0% al **63%**.
   - **Impacto Económico:** Más de **$18,800 USD** en *Lifetime Value (LTV)* rescatado, optimizando las campañas preventivas asumiendo un 30% de aceptación.
2. **Conversión O2O (Online to Offline):**
   - El modelo identifica al **64%** de los clientes de delivery con alta propensión a visitar el salón (vs 11% inicial).
   - **Impacto Económico:** Incremento estimado de **~$5,000 USD** en Ticket Promedio derivado de esta conversión directa.

*Nota: Estos resultados justifican plenamente la puesta en producción del proyecto por su alto retorno de inversión frente a los reducidos costos de incentivos promocionales.*

## Arquitectura Tecnológica
- **Data Engineering:** Limpieza y feature engineering orientada a variables de comportamiento (RFM, quejas, sensibilidad a promociones).
- **Modelado:** `XGBoost` con pesos de clase (`scale_pos_weight`) para mitigar el desbalance, optimizado vía `RandomizedSearchCV`.
- **MLOps & Tracking:** Experimentos, parámetros y métricas documentadas y trackeadas nativamente con `MLflow`.
- **Explicabilidad:** Uso de `SHAP` values para proveer insights interpretables de qué afecta el comportamiento de cada grupo de usuarios.
- **Visualización:** Aplicación web interactiva desarrollada con `Streamlit` para simulaciones de clientes "al vuelo" por parte del equipo gerencial.

## 🚀 Despliegue (Producción)

El proyecto está containerizado para garantizar portabilidad en cualquier entorno (AWS, Azure, GCP o servidores locales).

### Pre-requisitos
Tener instalado [Docker](https://www.docker.com/).

### Instrucciones de Ejecución
1. Clona este repositorio o asegúrate de estar en el directorio raíz.
2. Construye la imagen de Docker:
   ```bash
   docker build -t sushicorp-ml-dashboard .
   ```
3. Ejecuta el contenedor:
   ```bash
   docker run -p 8501:8501 sushicorp-ml-dashboard
   ```
4. Abre tu navegador web en: `http://localhost:8501` para interactuar con el Dashboard Ejecutivo.
