import nest_asyncio
nest_asyncio.apply()

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
# Adicione a importação do dotenv novamente
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import fitz
import os

# 1. Re-adicione esta linha para carregar o .env localmente
_ = load_dotenv(find_dotenv())

st.set_page_config(page_title="CEFET - Chat sobre o Cefet", page_icon="🎓")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Função para extrair texto do PDF (sem alterações)
def extrai_texto_para_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

@st.cache_resource
def load_pdf_data():
    pdf_path = "perguntas2.pdf"
    if not os.path.exists(pdf_path):
        st.error(f"Arquivo '{pdf_path}' não encontrado! Verifique se ele está na mesma pasta do seu script.")
        return None
    
    texto_extraido = extrai_texto_para_pdf(pdf_path)
    
    # Dividir o texto em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(texto_extraido)
    
    # Criar documentos a partir dos chunks
    documents = [Document(page_content=chunk) for chunk in chunks]

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("A chave da API do Google não foi encontrada. Verifique os Secrets no Streamlit Cloud.")
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=google_api_key
    )

    # Criar o vectorstore a partir dos documentos (chunks)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# Carrega o retriever
retriever = load_pdf_data()

# Função para formatar o conteúdo dos documentos recuperados
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---- Interface do Streamlit ----
st.title("Infobot - Assistente Virtual 🤖")
st.write("Pergunte sobre o curso de Sistemas de Informação!")

rag_template = """
Você é um atendente virtual amigável e prestativo de uma faculdade chamada CEFET-MG (Centro Federal de Educação Tecnológica de Minas Gerais) 
no campus de Varginha. 
Seu trabalho é fornecer informações sobre o curso de Sistemas de Informação de maneira educada, empática e clara
consultando as informações extraida do texto, sempre seja organizado e detalhado.
Sempre seja gentil ao responder.

Contexto: {context}

Pergunta do cliente: {question}
"""
prompt = ChatPromptTemplate.from_template(rag_template)

# Definir a cadeia corretamente
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Caixa de entrada para o usuário
if user_input := st.chat_input("Você:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Invocar a cadeia passando diretamente a string 
    response_stream = chain.stream(user_input)  
    full_response = ""
    
    response_container = st.chat_message("assistant")
    response_text = response_container.empty()
    
    for partial_response in response_stream:
        full_response += str(partial_response.content)
        response_text.markdown(full_response + "")

    st.session_state.messages.append({"role": "assistant", "content": full_response})