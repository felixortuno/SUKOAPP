import streamlit as st
import os
import googlemaps
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- CONFIGURACIÓN E INTERFAZ ---
st.set_page_config(page_title="Magdalena 2026 AI", page_icon="🌿", layout="wide")
st.title("🌿 Magdalena 2026: Asistente Inteligente")
st.markdown("Consulta eventos, puntos calientes, collas y ubicaciones en tiempo real.")

# --- GESTIÓN DE LLAVES Y CONFIGURACIÓN ---
with st.sidebar:
    st.header("🔑 Configuración de API")
    groq_key = st.text_input("Groq API Key", type="password")
    gmaps_key = st.text_input("Google Maps API Key", type="password")
    
    st.divider()
    st.info("Sube el Programa Oficial de la Magdalena 2026 (PDF) para alimentar la IA.")
    uploaded_file = st.file_uploader("Programa de Fiestas", type="pdf")

if not groq_key or not gmaps_key:
    st.warning("⚠️ Introduce las API Keys para activar el asistente.")
    st.stop()

# --- INICIALIZACIÓN DE MODELOS ---
@st.cache_resource
def init_models(api_key):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0.2)
    return embeddings, llm

embeddings, llm = init_models(groq_key)
gmaps = googlemaps.Client(key=gmaps_key)

# --- PROCESAMIENTO RAG ---
def ingest_data(file):
    temp_path = Path("magdalena_temp.pdf")
    with open(temp_path, "wb") as f:
        f.write(file.getbuffer())
    
    loader = PyMuPDFLoader(str(temp_path))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# --- LÓGICA DE LA APP ---
if uploaded_file:
    retriever = ingest_data(uploaded_file)
    
    # Prompt especializado para la Magdalena
    system_prompt = """Eres el asistente oficial de las Fiestas de la Magdalena 2026 en Castellón.
    Usa el contexto para responder sobre:
    - Horarios de Mascletaes y Pirotecnia.
    - Ubicación de Gaiatas y Collas.
    - Conciertos en el recinto de ferias.
    Si no sabes la respuesta exacta, indícalo. Responde siempre en Castellano con tono festivo."""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Pregunta: {input}\n\nContexto: {context}")
    ])

    chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # --- CHAT ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("¿A qué hora es el Pregón?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response = chain.invoke(user_input)
            st.markdown(response)
            
            # Ejemplo de integración simple con Google Maps (Buscador de sitios)
            if "dónde" in user_input.lower() or "ubicación" in user_input.lower():
                st.info("📍 Buscando ubicación en el mapa...")
                # Aquí podrías extraer la entidad (ej: 'Gaiata 15') y buscarla
                # results = gmaps.places(query=user_input + " Castellón Magdalena")
                # st.write(results)
            
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Sube el PDF del programa para comenzar a realizar consultas avanzadas.")