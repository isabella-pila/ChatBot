import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI  # Usando a API do Google Gemini

# Carregar o modelo do Gemini da API
from langchain_google_genai import ChatGoogleGenerativeAI

API_KEY = "AIzaSyCymj5TD0XZGqzkUHRp_9FXRX95VN9pmJI"
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=API_KEY)

#model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")  # Defina o modelo Gemini que você quer usar

def scrape_course_info():
    url = "https://www.sistemasdeinformacao.varginha.cefetmg.br/informacoes-do-curso/"
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Exemplo: Extraindo parágrafos de informação sobre o curso
        course_info = "\n".join([p.text for p in soup.find_all("p")])
        return course_info
    else:
        return "Não foi possível acessar as informações do site no momento."

st.title("Cefet-MG - Sistemas de Informação")

# Obtém as informações do curso via Web Scraping
context = scrape_course_info()

# Configuração do prompt e do modelo
rag_template = """
Você é um atendente virtual amigável e prestativo de uma faculdade chamada CEFET-MG (Centro Federal de Educação Tecnológica de Minas Gerais) 
no campus de Varginha. 
Seu trabalho é fornecer informações sobre o curso de Sistemas de Informação de maneira educada, empática e clara, 
consultando as informações extraídas do site oficial.
Sempre seja gentil ao responder.

Contexto: {context}

Pergunta do cliente: {question}
"""

prompt = ChatPromptTemplate.from_template(rag_template)

def generate_response(user_question):
    response = model.invoke(prompt.format(context=context, question=user_question))
    return response.content

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
    
    response = generate_response(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
