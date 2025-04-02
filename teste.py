import os
file_path = r"C:\Users\isabe\Desktop\chatbot\cefetmg.csv"

if os.path.exists(file_path):
    print("Arquivo encontrado!")
else:
    print("Erro: O arquivo não foi encontrado!")
