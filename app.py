
import streamlit as st
import os
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
st.set_page_config(page_title="LegalDoc Junior AI", page_icon="⚖️", layout="wide")
st.title("⚖️ LegalDoc Junior: Asistente RAG Jurídico")
st.markdown("Analiza leyes, contratos y BOEs con precisión quirúrgica.")

# --- BARRA LATERAL: ENTRADA DE KEYS ---
with st.sidebar:
    st.header("🔑 Configuración de Acceso")
    groq_key = st.text_input("Groq API Key", type="password", help="Necesaria para el razonamiento legal.")
    
    st.divider()
    st.subheader("📚 Biblioteca Legal")
    uploaded_files = st.file_uploader("Sube archivos PDF (Leyes, Contratos...)", type="pdf", accept_multiple_files=True)

if not groq_key:
    st.info("👋 Introduce tu Groq API Key para activar el cerebro jurídico de la app.")
    st.stop()

# --- INICIALIZACIÓN DE MODELOS ---
@st.cache_resource
def load_engine(api_key):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0.1)
    return embeddings, llm

embeddings, llm = load_engine(groq_key)

# --- PROCESAMIENTO DE DOCUMENTOS ---
def process_legal_docs(files):
    all_docs = []
    for file in files:
        temp_path = Path(f"temp_{file.name}")
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        
        loader = PyMuPDFLoader(str(temp_path))
        all_docs.extend(loader.load())
        os.remove(temp_path) # Limpieza inmediata
    
    # Chunks más pequeños para precisión legal
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# --- LÓGICA DE CONSULTA ---
if uploaded_files:
    with st.spinner("Indexando jurisprudencia y normativa..."):
        retriever = process_legal_docs(uploaded_files)
    
    # Prompt diseñado para evitar inventos
    system_prompt = """Eres un consultor jurídico experto. Tu objetivo es responder preguntas basadas estrictamente en el contexto legal proporcionado.
    REGLAS DE ORO:
    1. Si la respuesta no está en el texto, di 'No he encontrado base legal en los documentos proporcionados'.
    2. Cita siempre el nombre del archivo o el artículo si aparece.
    3. Mantén un lenguaje formal y preciso."""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Contexto Legal: {context}\n\nConsulta: {input}")
    ])

    chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # --- INTERFAZ DE CHAT ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if query := st.chat_input("Ej: ¿Cuáles son las causas de rescisión de este contrato?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Analizando artículos..."):
                response = chain.invoke(query)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.warning("📥 Sube uno o varios documentos legales para empezar el análisis.")