import os
import json

class Memoria:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.file_path = os.path.join("dados", f"historico_conversas_{session_id}.json")
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.historico = self.ler_historico()

    def ler_historico(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                return json.load(file)
        return []

    def salvar_historico(self, pergunta: str, resposta: str):
        self.historico.append({"pergunta": pergunta, "resposta": resposta})
        with open(self.file_path, "w") as file:
            json.dump(self.historico, file, indent=4)

    def obter_historico_formatado(self):
        return "\n".join([f"Você: {item['pergunta']}\nBot: {item['resposta']}" for item in self.historico])