import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import fitz  # PyMuPDF
import os

# Carrega as variaveis de ambiente
_ = load_dotenv(find_dotenv())

# Carrega o modelo do Gemini
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

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
    
    # 1. Dividir o texto em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(texto_extraido)
    
    # 2. Criar documentos a partir dos chunks
    documents = [Document(page_content=chunk) for chunk in chunks]

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("A chave da API do Google não foi encontrada. Configure a variável de ambiente GOOGLE_API_KEY.")
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=google_api_key
    )

    # 3. Criar o vectorstore a partir dos documentos (chunks)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3}) # Retorna os 3 chunks mais relevantes

# Carrega o retriever
retriever = load_pdf_data()

# Função para formatar o conteúdo dos documentos recuperados
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---- Interface do Streamlit ----
st.title("CEFET-MG Varginha - Assistente Virtual 🤖")
st.write("Pergunte sobre o curso de Sistemas de Informação!")

rag_template = """
Você é um atendente virtual amigável e prestativo da faculdade CEFET-MG (Centro Federal de Educação Tecnológica de Minas Gerais) no campus de Varginha.
Seu trabalho é fornecer informações sobre o curso de Sistemas de Informação de maneira educada, empática e clara, consultando as informações extraídas do texto abaixo.
Seja sempre organizado, detalhado e gentil ao responder. Se a resposta não estiver no contexto, diga educadamente que não possui essa informação.

Contexto:
{context}

Pergunta: {question}
"""
prompt = ChatPromptTemplate.from_template(rag_template)

# Definir a cadeia RAG corretamente
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Inicializa o histórico de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Caixa de entrada para o usuário
if user_input := st.chat_input("Qual sua dúvida sobre o curso?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Resposta do assistente com streaming
    with st.chat_message("assistant"):
        # Invocar a cadeia com a entrada do usuário
        response_stream = chain.stream(user_input)
        
        # O st.write_stream é a forma mais moderna e recomendada de exibir streams
        full_response = st.write_stream(response_stream)

    st.session_state.messages.append({"role": "assistant", "content": full_response})