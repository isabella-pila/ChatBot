import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables
_ = load_dotenv(find_dotenv())

# Load Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

@st.cache_resource
def load_csv_data():    
    # Load your knowledge base
    loader = CSVLoader(file_path="perguntas.csv")
    
    # Using Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    documents = loader.load()
    
    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever()

retriever = load_csv_data()
st.title("CEFETMG - Assistente Virtual")

# Prompt configuration
rag_template = """
Você é um atendente virtual amigável e prestativo de uma empresa. 
Seu trabalho é conversar com os clientes de maneira educada, empática e clara, 
consultando a base de conhecimentos da empresa para fornecer respostas úteis.  

Sempre seja gentil e acolhedor ao responder.  

Contexto: {context}

Pergunta do cliente: {question}
"""

prompt = ChatPromptTemplate.from_template(rag_template)

def generate_response(user_question):
    context = retriever.invoke(user_question)
    response = model.invoke(prompt.format(context=context, question=user_question))
    return response.content

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Você:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    response = generate_response(user_input)
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})