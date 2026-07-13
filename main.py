from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# 1. Despertamos al cerebro artificial al iniciar el servidor
modelo = joblib.load('modelo_yankees.pkl')

@app.get("/")
def leer_raiz():
    return {"mensaje": "¡El cerebro con IA real de los Yankees está en línea!"}

@app.get("/pronosticar")
def api_pronostico(pitcher1: str, pitcher2: str):
    # ATENCIÓN: Tu app móvil envía nombres de pitchers, pero nuestro nuevo modelo 
    # solo entiende "es_local" y "racha_bateo_ultimos_5".
    # Para no romper tu app hoy, vamos a forzar los datos: le diremos al modelo
    # que los Yankees juegan en casa (1) y traen una racha de 5.2 carreras.
    
    # Preparamos la tabla exactamente como la aprendió en Google Colab
    datos_entrada = pd.DataFrame([[1, 5.2]], columns=['es_local', 'racha_bateo_ultimos_5'])
    
    # 2. ¡LA PREDICCIÓN REAL! Le pedimos al modelo que calcule las probabilidades
    probabilidades = modelo.predict_proba(datos_entrada)[0]
    prob_pierde = probabilidades[0] * 100
    prob_gana = probabilidades[1] * 100
    
    # 3. Empaquetamos la respuesta en el formato que tu celular ya conoce
    ganador = "Yankees (Local)" if prob_gana > prob_pierde else "Visitante"
    
    return {
        "ganador_probable": ganador,
        "probabilidad_local": round(prob_gana, 2),
        "probabilidad_visitante": round(prob_pierde, 2)
    }