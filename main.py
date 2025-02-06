from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.requests import Request
from jinja2 import Environment, FileSystemLoader
from server.gunicorn_server import GunicornServer
import numpy as np
import openai
import os
import json


app = FastAPI(
    title="Provoc",
    description="Agente que funciona como Teste Vocacional",
    version="0.1.0"
)

openai_key = os.getenv('OPENAI_API_KEY')

env = Environment(loader=FileSystemLoader('templates'))

class Question(BaseModel):
    question: str
    session_id: str

class QuestionnaireResponse(BaseModel):
    nome: str
    escolaridade: str
    respostas: list

@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    template = env.get_template('provoc.html')
    return template.render(request=request)

@app.post("/ask")
async def ask_question(question: Question):
    session_id = question.session_id
    question_text = question.question

    if not question_text or not session_id:
        raise HTTPException(status_code=400, detail="Missing 'question' or 'session_id' in request body")

    memoria = Memoria(session_id)
    historico = memoria.obter_historico_formatado()

    print(f"Similar texts: {similar_texts}")

    context = "\n".join([text for text, _ in similar_texts])

    prompt = (
        f"Use o seguinte contexto para responder a pergunta: {context}\n\n"
        f"Histórico da conversa:\n{historico}\n\n"
        f"Pergunta atual: {question_text}\n\n"
        "Instrução: Responda à pergunta atual com base no contexto e no histórico da conversa."
    )
    print("Texto enviado para OpenAI:", prompt)

@app.post("/save")
async def save_questionnaire(response: QuestionnaireResponse):
    data = response.dict()
    os.makedirs('database', exist_ok=True)
    with open('database/responses.json', 'a') as f:
        json.dump(data, f)
        f.write('\n')
    return {"message": "Questionário salvo com sucesso!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)