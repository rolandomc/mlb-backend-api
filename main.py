from fastapi import FastAPI
import random # Usamos random temporalmente para asegurar que la API no colapse mientras configuramos la nube

app = FastAPI()

@app.get("/")
def leer_raiz():
    return {"mensaje": "¡El cerebro de la MLB está en línea!"}

@app.get("/pronosticar")
def api_pronostico(pitcher1: str, pitcher2: str):
    # Aquí iría el modelo real, pero para probar la conexión a internet 
    # generaremos una probabilidad lógica simulada temporal.
    prob_local = round(random.uniform(45.0, 65.0), 2)
    prob_vis = round(100 - prob_local, 2)
    ganador = pitcher1 if prob_local > prob_vis else pitcher2
    
    return {
        "ganador_probable": ganador,
        "probabilidad_local": prob_local,
        "probabilidad_visitante": prob_vis
    }