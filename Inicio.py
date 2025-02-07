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


# Configuración de la página de Streamlit
st.title('Analítica de datos con Agentes 🤖🔍')
image = Image.open('data_analisis.png')
st.image(image, width=350)

with st.sidebar:
    st.subheader("Este Agente de Pandas con Claude te ayudará a realizar análisis sobre tus datos")

# Input para la API key de Anthropic
ke = st.text_input('Ingresa tu Clave de Anthropic')
os.environ['ANTHROPIC_API_KEY'] = ke

# Carga de archivo
uploaded_file = st.file_uploader('Elige el archivo csv')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, on_bad_lines='skip')
    st.write(df)

st.subheader('Te ayudaré a analizar los datos que cargues.')
user_question = st.text_input("¿Qué deseas saber de los datos?:")

def format_response_for_streamlit(response):
    """Formatea la respuesta para mostrarla en Streamlit"""
    # Eliminar los bloques de código Python si existen
    clean_response = response.replace("```python", "").replace("```", "")
    
    # Mostrar la respuesta formateada
    st.write("### Análisis:")
    st.write(clean_response)
    
    # Si hay resultados numéricos, intentar mostrarlos como métricas
    try:
        if any(char.isdigit() for char in clean_response):
            numbers = [float(s) for s in clean_response.split() if s.replace('.','',1).isdigit()]
            if numbers:
                st.metric("Valor encontrado", numbers[0])
    except:
        pass

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

if user_question and ke and uploaded_file is not None:
    try:
        with st.spinner('Analizando los datos...'):
            # Crear el agente con Claude y parámetros correctos
            agent = create_pandas_dataframe_agent( ChatAnthropic(model='claude-3-5-sonnet-20241022'), #claude-3-haiku-20240307
                #Anthropic(
                #    model="claude-2.1",
                #    temperature=0,
                #    max_tokens=1500,
                #    anthropic_api_key=ke
                #),
                df,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                allow_dangerous_code=True,  # Agregado el parámetro requerido
            )
            
            # Ejecutar la consulta
            response = agent.run(custom_prompt(user_question))
            
            # Mostrar la respuesta formateada
            #format_response_for_streamlit(response)
            st.write(response)
    except Exception as e:
        st.error(f"Ocurrió un error al analizar los datos: {str(e)}")
