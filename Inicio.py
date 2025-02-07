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
from langchain_anthropic import ChatAnthropic

# Configuración de la página
st.set_page_config(
    page_title="Análisis Inteligente de Datos",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Estilos personalizados usando st.markdown
st.markdown("""
    <style>
        div.stButton > button {
            width: 100%;
            background-color: #FF4B4B;
            color: white;
            font-weight: bold;
        }
        div.stButton > button:hover {
            background-color: #FF6B6B;
            color: white;
        }
        .reportview-container .main .block-container {
            padding-top: 2rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Contenedor principal con diseño mejorado
with st.container():
    st.title('📊 Análisis Inteligente de Datos')
    st.markdown("---")

# Sidebar mejorada y más profesional
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # API Key con mejor manejo de estado
    st.subheader("Configuración de API")
    ke = st.text_input(
        "API Key de Anthropic",
        type="password",
        help="Ingresa tu clave API de Anthropic para continuar"
    )
    
    if ke:
        os.environ['ANTHROPIC_API_KEY'] = ke
        st.success("API configurada correctamente")
    
    # Información del sistema
    st.markdown("---")
    st.subheader("Sobre el Sistema")
    with st.expander("ℹ️ Información", expanded=False):
        st.markdown("""
        Este sistema utiliza:
        - Claude para análisis avanzado
        - Pandas para procesamiento de datos
        - IA para interpretación de resultados
        """)

# Área principal de la aplicación
main_container = st.container()
with main_container:
    # Carga de imagen con mejor presentación
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        image = Image.open('data_analisis.png')
        st.image(image, use_column_width=True)
    
    # Carga de archivo con mejor feedback
    st.subheader("📁 Carga de Datos")
    uploaded_file = st.file_uploader(
        "Selecciona tu archivo CSV",
        type=['csv'],
        help="Por favor, asegúrate de que tu archivo esté en formato CSV"
    )

    if uploaded_file is not None:
        # Mostrar datos con métricas
        df = pd.read_csv(uploaded_file, on_bad_lines='skip')
        
        # Métricas importantes
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filas", df.shape[0])
        with col2:
            st.metric("Columnas", df.shape[1])
        with col3:
            st.metric("Campos Total", df.size)
        
        # Vista previa de datos
        with st.expander("📊 Vista Previa de Datos", expanded=True):
            st.dataframe(
                df.head(),
                use_container_width=True,
                height=200
            )
            
            # Información básica del dataset
            if st.checkbox("Mostrar información del dataset"):
                st.write("### Información del Dataset")
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())

        # Formulario de consulta
        st.subheader("🔍 Consulta")
        with st.form(key='query_form'):
            user_question = st.text_area(
                "¿Qué deseas analizar en los datos?",
                placeholder="Ejemplo: ¿Cuál es el promedio de ventas por mes?",
                help="Escribe tu pregunta en lenguaje natural"
            )
            
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                submit_button = st.form_submit_button(
                    "🔍 Analizar Datos",
                    use_container_width=True
                )

        def format_response(response):
            """Mejora el formato de la respuesta"""
            st.markdown("### 📋 Resultados del Análisis")
            st.info(response)
            
            # Agregar opciones de descarga si hay resultados numéricos
            if any(char.isdigit() for char in response):
                st.download_button(
                    label="📥 Descargar Resultados",
                    data=response,
                    file_name="analisis_resultados.txt",
                    mime="text/plain"
                )

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

        # Proceso de análisis
        if submit_button:
            if not ke:
                st.error("⚠️ Por favor, configura tu API key primero")
            elif not user_question:
                st.warning("⚠️ Por favor, ingresa una pregunta")
            else:
                try:
                    with st.spinner('⏳ Analizando datos...'):
                        agent = create_pandas_dataframe_agent(
                            ChatAnthropic(model='claude-3-5-sonnet-20241022'),
                            df,
                            verbose=True,
                            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                            handle_parsing_errors=True,
                            allow_dangerous_code=True
                        )
                        
                        response = agent.run(custom_prompt(user_question))
                        format_response(response)
                        
                except Exception as e:
                    st.error(f"❌ Error en el análisis: {str(e)}")
                    st.error("Por favor, intenta reformular tu pregunta o verifica tus datos")
