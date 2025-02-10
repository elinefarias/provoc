from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.requests import Request
from jinja2 import Environment, FileSystemLoader
from server.gunicorn_server import GunicornServer
from fastapi import File, UploadFile
from datetime import datetime
import numpy as np
import pandas as pd
import openai
import os
import json
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pdfkit
from fastapi.encoders import jsonable_encoder
from weasyprint import HTML



app = FastAPI(
    title="Provoc",
    description="Agente que funciona como Teste Vocacional",
    version="0.1.0"
)

app.mount("/assets", StaticFiles(directory="assets"), name="assets")
openai_key = os.getenv('OPENAI_API_KEY')
env = Environment(loader=FileSystemLoader('templates'))


class QuestionnaireResponse(BaseModel):
    nome: str
    escolaridade: str
    respostas: list

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text


def get_embeddings(texts):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    embeddings = [embedding['embedding'] for embedding in response['data']]
    return np.array(embeddings)

def calcular_perfil_dominante(dados):
    mapeamento_perguntas = {
        1: "R", 2: "R",
        3: "I", 4: "I",
        5: "A", 6: "A",
        7: "S", 8: "S",
        9: "E", 10: "E",
        11: "C", 12: "C"
    }

    pontuacoes = {"R": 0, "I": 0, "A": 0, "S": 0, "E": 0, "C": 0}
    
    mapeamento_respostas = {
        "Concordo totalmente": 5,
        "Concordo parcialmente": 4,
        "Neutro": 3,
        "Discordo parcialmente": 2,
        "Discordo totalmente": 1
    }

    # Processa as respostas para calcular a pontuação por categoria
    for resposta in dados["respostas"]:
        pergunta = resposta["pergunta"]
        categoria = mapeamento_perguntas[pergunta]
        pontuacoes[categoria] += mapeamento_respostas[resposta["resposta"]]

    # Identifica a categoria com a maior pontuação (perfil dominante)
    perfil_dominante_letra = max(pontuacoes, key=pontuacoes.get)

    # Mapeamento das letras para descrições completas
    mapeamento_perfis = {
        "R": "Realista",
        "I": "Investigativo",
        "A": "Artístico",
        "S": "Social",
        "E": "Empreendedor",
        "C": "Convencional"
    }

    descricao_perfis = {
        "R": "Preferência por trabalhos práticos, manuais e técnicos.",
        "I": "Interesse por pesquisa, análise e ciência.",
        "A": "Inclinação para criatividade e expressão pessoal.",
        "S": "Preferência por interação e ajuda ao próximo.",
        "E": "Habilidade para liderar, persuadir e inovar.",
        "C": "Interesse por organização, estrutura e regras."
    }

    perfil_dominante = {
        "nome": mapeamento_perfis[perfil_dominante_letra],
        "descricao": descricao_perfis[perfil_dominante_letra]
    }

    # Retorna o resultado com o perfil dominante
    resultado = {
        "nome": dados["nome"],
        "escolaridade": dados["escolaridade"],
        "perfil_dominante": perfil_dominante,
        "pontuacao": pontuacoes[perfil_dominante_letra]
    }
    
    print("Perfil dominante calculado:", resultado)
    return resultado



def agente_provoc(prompt, nome, perfil_dominante):
    print("Prompt criado para a IA.")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Você é um especialista em orientação vocacional."},
            {"role": "user", "content": f"O perfil Primário é {perfil_dominante['nome']}"},
            {"role": "user", "content": "Por favor, forneça informações detalhadas sobre carreiras adequadas para este perfil, incluindo possíveis áreas de interesse, habilidades necessárias e dicas de desenvolvimento pessoal."},
            {"role": "user", "content": prompt}
        ]
    )
    print("Resposta da IA recebida.")
    
    report_text = response['choices'][0]['message']['content']
    report_path = f"database/{nome}_relatorio.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Relatório salvo em: {report_path}")
    
    # Salvar a resposta da IA em um arquivo JSON
    report_json_path = f"database/{nome}_relatorio.json"
    with open(report_json_path, "w") as f:
        json.dump({"report_text": report_text}, f)
    
    return {"message": "Relatório gerado com sucesso"}


@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    template = env.get_template('index.html')
    return template.render(request=request)


@app.post("/salvar_questionario")
async def salvar_questionario(response: QuestionnaireResponse):
    data = response.dict()
    os.makedirs('database', exist_ok=True)
    file_path = f'database/{data["nome"]}.json'
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return {"message": "Questionário salvo com sucesso!"}


@app.post("/calcular_perfil")
async def calcular_perfil(response: QuestionnaireResponse):
    data = response.dict()
    print("Dados recebidos do formulário:", data)
    perfil_dominante = calcular_perfil_dominante(data)
    return perfil_dominante


@app.get("/relatorio.html", response_class=HTMLResponse)
async def get_report_page(request: Request, nome: str, perfil_primario: str, descricao_primaria: str,  cursos_recomendados: str = ""):
    template = env.get_template('relatorio.html')
    report_json_path = f"database/{nome}_relatorio.json"
    
    if os.path.exists(report_json_path):
        with open(report_json_path, "r") as f:
            report_data = json.load(f)
        report_text = report_data["report_text"]
    else:
        report_text = "Relatório não encontrado."
    
    data_atual = datetime.now().strftime("%d/%m/%Y")
    cursos_recomendados_list = cursos_recomendados.split(",") if cursos_recomendados else []
    
    html_content = template.render(
        request=request, 
        nome=nome, 
        report_text=report_text, 
        data=data_atual, 
        perfil_primario=perfil_primario, 
        descricao_primaria=descricao_primaria,
        cursos_recomendados=cursos_recomendados_list
    )
    pdf_path = f"database/{nome}_relatorio.pdf"
    HTML(string=html_content).write_pdf(pdf_path)
    
    return html_content
    
    
@app.post("/agente_provoc")
async def agente_provoc_endpoint(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    nome = data.get("nome")
    perfil_dominante = data.get("perfil_dominante")
    resultado = agente_provoc(prompt, nome, perfil_dominante)
    return resultado


@app.post("/recomendar_cursos")
async def recomendar_cursos(response: QuestionnaireResponse):
    data = response.model_dump()  
    respostas_usuario = " ".join([str(resposta["resposta"]).lower() for resposta in data.get("respostas", [])])

    df = pd.read_csv('database/cursos_superiores.csv', delimiter=';', encoding='utf-8')
    detalhes_cursos = df['Detalhes'].apply(lambda x: str(x).lower()).tolist()

    # Geração de embeddings
    documentos = detalhes_cursos + [respostas_usuario]  # Último documento são as respostas do usuário
    embeddings = get_embeddings(documentos)

    # Busca de similaridade
    query_vector = embeddings[-1].reshape(1, -1)  # Vetor de consulta (respostas do usuário)
    embeddings = embeddings[:-1]  # Embeddings dos cursos

    # Cálculo da similaridade cosseno
    similarities = cosine_similarity(query_vector, embeddings)[0]

    # Gerar a lista de cursos recomendados com base na similaridade
    cursos_recomendados = [
        {"curso": df.iloc[i]['Nome do curso'], "similaridade": f"{similaridade:.2f}"}
        for i, similaridade in enumerate(similarities)
    ]

    # Ordenar os cursos recomendados pela similaridade (maior para menor)
    cursos_recomendados.sort(key=lambda x: float(x["similaridade"]), reverse=True)
    
    # Limitar a lista a três cursos
    cursos_recomendados = cursos_recomendados[:3]
    
    # Retornar a resposta com os cursos recomendados
    json_data = jsonable_encoder({
        "message": "Recomendação gerada com sucesso!",
        "cursos_recomendados": cursos_recomendados,
        "detalhes": "Os cursos estão listados em ordem de compatibilidade com suas respostas."
    })
    return JSONResponse(content=json_data)


if __name__ == "__main__":
    options = {
        'bind': '{}:{}'.format('0.0.0.0', '8000'),
        'workers': 1,
        'worker_class': 'uvicorn.workers.UvicornWorker',
        'timeout': 600
    }
    GunicornServer(app, options).run()