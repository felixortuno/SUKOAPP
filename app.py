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

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Magdalena 2026 Pro", page_icon="🌿", layout="wide")

# --- INTERFAZ DE ENTRADA DE KEYS ---
with st.sidebar:
    st.header("🔑 Acceso a la API")
    st.markdown("Introduce tus credenciales para activar la IA y los mapas.")
    
    # Intenta leer de secrets, si no, campo vacío
    groq_api_key = st.text_input(
        "Groq API Key", 
        value=st.secrets.get("GROQ_API_KEY", ""), 
        type="password",
        help="Consíguela en console.groq.com"
    )
    
    gmaps_api_key = st.text_input(
        "Google Maps API Key", 
        value=st.secrets.get("GOOGLE_MAPS_API_KEY", ""), 
        type="password",
        help="Consíguela en Google Cloud Console"
    )

    st.divider()
    uploaded_file = st.file_uploader("📂 Sube el Programa de Fiestas (PDF)", type="pdf")

# Bloqueo de seguridad: si no hay keys, no arranca la lógica pesada
if not groq_api_key or not gmaps_api_key:
    st.info("👋 ¡Bienvenido! Por favor, introduce tus **API Keys** en la barra lateral para comenzar.")
    st.stop()

# --- INICIALIZACIÓN DE COMPONENTES ---
@st.cache_resource
def get_resources(g_key):
    # Embeddings (locales)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    # LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        groq_api_key=groq_api_key, 
        temperature=0.4
    )
    # Google Maps
    gmaps_client = googlemaps.Client(key=g_key)
    return embeddings, llm, gmaps_client

try:
    embeddings, llm, gmaps = get_resources(gmaps_api_key)
except Exception as e:
    st.error(f"Error al conectar con las APIs: {e}")
    st.stop()

# --- PROCESAMIENTO RAG ---
def process_magdalena_docs(file):
    temp_path = Path("current_magdalena.pdf")
    with open(temp_path, "wb") as f:
        f.write(file.getbuffer())
    
    loader = PyMuPDFLoader(str(temp_path))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# --- LÓGICA PRINCIPAL ---
st.title("🌿 Magdalena 2026 AI Explorer")

if uploaded_file:
    with st.spinner("Analizando el programa de fiestas..."):
        retriever = process_magdalena_docs(uploaded_file)
    
    # Template para la IA
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un experto en las fiestas de la Magdalena de Castellón. "
                   "Responde usando el contexto del programa oficial. "
                   "Si te preguntan por sitios, intenta dar detalles para que el usuario pueda buscarlos."),
        ("human", "Pregunta: {input}\n\nContexto: {context}")
    ])

    rag_chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # --- INTERFAZ DE CHAT ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if query := st.chat_input("¿Qué quieres saber sobre la Magdalena?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Consultando el programa..."):
                response = rag_chain.invoke(query)
                st.markdown(response)
                
                # Integración con Google Maps para "puntos calientes"
                if any(word in query.lower() for word in ["donde", "ubicacion", "sitio", "lugar", "ir"]):
                    st.divider()
                    st.caption("📍 Información geográfica disponible vía Google Maps")
                    # Aquí el usuario puede ver que el sistema está listo para geolocalizar
        
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.warning("⚠️ Necesito el programa de fiestas en PDF para poder responder tus preguntas.")