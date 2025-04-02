import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
import fitz
import os
from langchain_core.runnables import RunnablePassthrough

# Carrega as variaveis de ambiente
_ = load_dotenv(find_dotenv())

# Carrega o modelo do Gemini
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Função para extrair texto do PDF
def extrai_texto_para_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

@st.cache_resource
def load_pdf_data():    
    pdf_path = "perguntas.pdf"
    
    if not os.path.exists(pdf_path):
        st.error("Arquivo PDF não encontrado!")
        return None
    
    texto_extraido = extrai_texto_para_pdf(pdf_path)
    document = Document(page_content=texto_extraido, metadata={"source": pdf_path})
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_documents([document], embeddings)
    return vectorstore.as_retriever()

# Carregar os dados do PDF
retriever = load_pdf_data()

st.title("CEFET-MG - Assistente Virtual")

# Template do prompt
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