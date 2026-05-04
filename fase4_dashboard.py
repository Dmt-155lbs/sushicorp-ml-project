import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de página - McKinsey Style (sobrio y ancho)
st.set_page_config(
    page_title="SushiCorp ML Dashboard",
    page_icon="🍣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #002D62; /* McKinsey Blue */
    }
    .risk-high {
        color: #d9534f;
        font-weight: bold;
    }
    .risk-low {
        color: #5cb85c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Cargar modelos
@st.cache_resource
def load_models():
    model_churn = joblib.load('models/xgboost_churn_delivery_optimizado.joblib')
    model_conversion = joblib.load('models/xgboost_conversion_salon_optimizado.joblib')
    return model_churn, model_conversion

model_churn, model_conversion = load_models()

# Referencia de columnas exactas del modelo
# Eliminando churn_delivery y conversion_salon
expected_columns = [
    'recency_dias', 'frecuencia_6m', 'ticket_promedio_usd', 'ltv_historico_usd',
    'tiempo_entrega_promedio_min', 'sensibilidad_promos', 'quejas_recientes', 'distancia_local_km',
    'valor_6m_aprox', 'tasa_quejas_por_pedido', 'promocion_vs_distancia',
    'marca_principal_Kobe Sushi', 'marca_principal_Noe Sushi Bar', 'marca_principal_Nubori Experience'
]

st.title("🍣 SushiCorp AI: Customer Intelligence Platform")
st.markdown("### Simulador de Probabilidad (Churn y O2O Conversion)")

# Sidebar para inputs
st.sidebar.header("Perfil del Cliente")

recency_dias = st.sidebar.slider("Recency (Días desde último pedido)", 0, 180, 30)
frecuencia_6m = st.sidebar.slider("Frecuencia (Pedidos en 6 meses)", 0, 100, 5)
ticket_promedio_usd = st.sidebar.slider("Ticket Promedio (USD)", 10.0, 150.0, 35.0)
ltv_historico_usd = st.sidebar.slider("LTV Histórico (USD)", 50.0, 3000.0, 500.0)

st.sidebar.markdown("---")
st.sidebar.header("Logística y Experiencia")
tiempo_entrega_promedio_min = st.sidebar.slider("Tiempo de Entrega Promedio (min)", 20, 120, 45)
sensibilidad_promos = st.sidebar.slider("Sensibilidad a Promos (0 a 1)", 0.0, 1.0, 0.5)
quejas_recientes = st.sidebar.slider("Quejas Recientes", 0, 10, 0)
distancia_local_km = st.sidebar.slider("Distancia al Local (km)", 0.0, 30.0, 5.0)

st.sidebar.markdown("---")
marca_principal = st.sidebar.selectbox(
    "Marca Principal Frecuentada",
    ["Kobe Sushi", "Noe Sushi Bar", "Nubori Experience", "Otra"]
)

# Ingeniería de Características en vivo
valor_6m_aprox = frecuencia_6m * ticket_promedio_usd
tasa_quejas_por_pedido = quejas_recientes / (frecuencia_6m + 1)
promocion_vs_distancia = sensibilidad_promos / (distancia_local_km + 0.1)

is_kobe = 1 if marca_principal == "Kobe Sushi" else 0
is_noe = 1 if marca_principal == "Noe Sushi Bar" else 0
is_nubori = 1 if marca_principal == "Nubori Experience" else 0

# Construir DataFrame para predicción
input_data = pd.DataFrame([[
    recency_dias, frecuencia_6m, ticket_promedio_usd, ltv_historico_usd,
    tiempo_entrega_promedio_min, sensibilidad_promos, quejas_recientes, distancia_local_km,
    valor_6m_aprox, tasa_quejas_por_pedido, promocion_vs_distancia,
    is_kobe, is_noe, is_nubori
]], columns=expected_columns)

# Realizar Predicciones
prob_churn = model_churn.predict_proba(input_data)[0][1]
prob_conversion = model_conversion.predict_proba(input_data)[0][1]

# Área Principal - Resultados
col1, col2 = st.columns(2)

with col1:
    st.subheader("🚨 Riesgo de Fuga (Churn Delivery)")
    
    # Progreso visual
    st.progress(float(prob_churn))
    
    if prob_churn >= 0.5:
        st.markdown(f"### Probabilidad: <span class='risk-high'>{prob_churn*100:.1f}%</span>", unsafe_allow_html=True)
        st.error("ALTO RIESGO: Cliente con alta propensión a fugar. Se recomienda enviar promoción de retención inmediata.")
    else:
        st.markdown(f"### Probabilidad: <span class='risk-low'>{prob_churn*100:.1f}%</span>", unsafe_allow_html=True)
        st.success("RIESGO BAJO: Cliente leal o estable por el momento.")
        
    st.markdown("---")
    st.metric(label="Valor en Riesgo (LTV)", value=f"${ltv_historico_usd:,.2f} USD")

with col2:
    st.subheader("🍣 Oportunidad O2O (Conversión a Salón)")
    
    st.progress(float(prob_conversion))
    
    if prob_conversion >= 0.5:
        st.markdown(f"### Probabilidad: <span class='risk-low'>{prob_conversion*100:.1f}%</span>", unsafe_allow_html=True)
        st.success("ALTA PROPENSIÓN: Excelente candidato para cross-selling. Ofrecer invitación o descuento para visita al salón.")
    else:
        st.markdown(f"### Probabilidad: <span style='color: gray; font-weight: bold;'>{prob_conversion*100:.1f}%</span>", unsafe_allow_html=True)
        st.info("BAJA PROPENSIÓN: Es probable que el cliente prefiera mantener su formato de Delivery.")

st.markdown("---")
st.markdown("### Resumen del Perfil Simulado")
st.dataframe(input_data.T.rename(columns={0: "Valor"}), height=300)
