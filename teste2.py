import streamlit as st
from langchain_core.prompts import ChatPromptTemplate #importarz
from dotenv import load_dotenv, find_dotenv #importar o dotenv 
from langchain_community.vectorstores import FAISS # import dos vetores 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings # import do google gemini
from langchain_core.documents import Document  # Importar a classe Document
import fitz  # PyMuPDF para leitura de PDF
import os

# Carregar variáveis de ambiente
_ = load_dotenv(find_dotenv())

# Carregar modelo do Gemini
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Função para extrair texto do PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)  # Abre o PDF
    for page in doc:
        text += page.get_text("text") + "\n"  # Extrai o texto de cada página
    return text

@st.cache_resource
def load_pdf_data():    
    # Caminho do arquivo PDF
    pdf_path = "perguntas.pdf"
    
    if not os.path.exists(pdf_path):
        st.error("Arquivo PDF não encontrado!")
        return None
    
    # Extrair texto do PDF
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Criar um documento LangChain a partir do texto extraído
    document = Document(page_content=extracted_text, metadata={"source": pdf_path})
    
    # Criar embeddings com modelo do Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Criar base de conhecimento com FAISS
    vectorstore = FAISS.from_documents([document], embeddings)
    
    return vectorstore.as_retriever()

# Carregar os dados do PDF
retriever = load_pdf_data()

st.title("CEFETMG - Assistente Virtual")

# Template do prompt
rag_template = """
Você é um atendente virtual amigável e prestativo de uma faculdade chamada CEFET-MG (Centro Federal de Educação Tecnológica de Minas Gerais) 
no campus de Varginha. 
Seu trabalho é fornecer informações sobre o curso de Sistemas de Informação de maneira educada, empática e clara, 
consultando as informações extraida do texto, sempre seja organizado e detalhado.
Sempre seja gentil ao responder.

Contexto: {context}

Pergunta do cliente: {question}
"""

prompt = ChatPromptTemplate.from_template(rag_template)


# Função para gerar resposta
def generate_response(user_question):
    context = retriever.invoke(user_question)
    response = model.invoke(prompt.format(context=context, question=user_question))
    return response.content

# Inicializar histórico do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensagens antigas
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada do usuário
if user_input := st.chat_input("Você:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Gerar resposta
    response = generate_response(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
