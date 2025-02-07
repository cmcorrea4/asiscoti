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
import io
from fpdf import FPDF
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Análisis Inteligente de Datos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para generar PDF
def create_pdf(response, question):
    pdf = FPDF()
    pdf.add_page()
    
    # Configurar fuente
    pdf.set_font('Arial', 'B', 16)
    
    # Título
    pdf.cell(190, 10, 'Reporte de Análisis de Datos', 0, 1, 'C')
    
    # Fecha
    pdf.set_font('Arial', '', 12)
    pdf.cell(190, 10, f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'R')
    
    # Pregunta
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(190, 10, 'Pregunta:', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(190, 10, question)
    
    # Respuesta
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(190, 10, 'Resultados:', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(190, 10, response)
    
    return pdf.output(dest='S').encode('latin-1')

# Estilos personalizados
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

# Título principal
st.title('📊 Análisis Inteligente de Datos')
st.markdown("---")

# Crear el layout de dos columnas
col_main, col_voice = st.columns([2, 1])

# Columna del asistente de voz
with col_voice:
    chat_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Chat Widget</title>
        </head>
        <body>
            <div style="width: 100%; height: 500px;">
                <elevenlabs-convai agent-id="gMh8bGtmxS5OxxPwDuKT"></elevenlabs-convai>
            </div>
            <script src="https://elevenlabs.io/convai-widget/index.js" async></script>
        </body>
        </html>
    """
    with st.expander("💬 Asistente de Voz", expanded=True):
        st.components.v1.html(chat_html, height=550, scrolling=True)

# Columna principal
with col_main:
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # API Key
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

    # Imagen principal
    image = Image.open('data_analisis.png')
    st.image(image, use_column_width=True)
    
    # Carga de archivo
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
                placeholder="Ejemplo: ¿Cuál es el p
