import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Anthropic
from langchain.callbacks import get_openai_callback
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
from langchain_anthropic import ChatAnthropic

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Datos con IA",
    page_icon="🤖",
    layout="wide"
)

# Aplicar estilo CSS personalizado
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2E4053;
        font-size: 2.5rem !important;
    }
    .stSubheader {
        color: #566573;
    }
    </style>
    """, unsafe_allow_html=True)

# Título principal con diseño mejorado
st.title('🤖 Analítica de datos con Agentes IA')

# Crear dos columnas para mejor organización
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 📊 Carga tus datos y descubre insights
    Esta herramienta utiliza IA para analizar tus datos de forma inteligente y responder a tus preguntas en lenguaje natural.
    """)

with col2:
    image = Image.open('data_analisis.png')
    st.image(image, width=300)

# Sidebar mejorada
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    st.markdown("---")
    st.markdown("### 🔑 Configuración de API")
    ke = st.text_input('Clave de Anthropic:', type="password", help="Ingresa tu API key de Anthropic")
    
    if ke:
        os.environ['ANTHROPIC_API_KEY'] = ke
        st.success('API key configurada correctamente! ✅')
    
    st.markdown("---")
    st.markdown("### 📖 Información")
    st.info("""
    Este Agente de Pandas con Claude te ayudará a:
    - Analizar datos estadísticos
    - Identificar patrones
    - Generar visualizaciones
    - Responder preguntas sobre tus datos
    """)

# Sección principal
st.markdown("### 📤 Carga tu archivo")
uploaded_file = st.file_uploader('Selecciona un archivo CSV:', type=['csv'])

if uploaded_file is not None:
    with st.expander("👀 Vista previa de los datos", expanded=True):
        df = pd.read_csv(uploaded_file, on_bad_lines='skip')
        st.dataframe(df.head(), use_container_width=True)
        st.info(f"Dimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")

    st.markdown("### ❓ Realiza tu consulta")
    with st.form(key='query_form'):
        user_question = st.text_input("¿Qué deseas saber sobre los datos?",
                                    placeholder="Ejemplo: ¿Cuál es el promedio de ventas?")
        submit_button = st.form_submit_button(label='Analizar datos 🔍')

    def format_response_for_streamlit(response):
        """Formatea la respuesta para mostrarla en Streamlit"""
        st.markdown("### 📊 Resultados del análisis")
        
        # Crear un contenedor estilizado para la respuesta
        with st.container():
            st.markdown(f"""
            <div style="padding: 1rem; border-radius: 0.5rem; background-color: #f8f9fa;">
                {response}
            </div>
            """, unsafe_allow_html=True)

    def custom_prompt(question):
        return f"""
        Responde SIEMPRE en español.
        Analiza los siguientes datos según esta pregunta: {question}
        
        Por favor:
        1. Da una respuesta clara y concisa
        2. Si son resultados numéricos, menciónalos claramente
        3. Si es una tendencia o patrón, descríbelo específicamente
        4. Usa formato de lista o puntos cuando sea apropiado
        5. No muestres el código, solo los resultados
        """

    if submit_button and user_question and ke and uploaded_file is not None:
        try:
            with st.spinner('🔄 Analizando los datos...'):
                # Crear el agente con Claude
                agent = create_pandas_dataframe_agent(
                    ChatAnthropic(model='claude-3-5-sonnet-20241022'),
                    df,
                    verbose=True,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    handle_parsing_errors=True,
                    allow_dangerous_code=True,
                )
                
                # Ejecutar la consulta
                response = agent.run(custom_prompt(user_question))
                
                # Mostrar la respuesta formateada
                format_response_for_streamlit(response)
                
        except Exception as e:
            st.error(f"❌ Error al analizar los datos: {str(e)}")
    
    elif submit_button and (not ke or not user_question):
        st.warning("⚠️ Por favor, asegúrate de proporcionar tanto la API key como una pregunta.")
