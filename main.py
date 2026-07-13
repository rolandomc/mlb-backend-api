from fastapi import FastAPI
import joblib
import pandas as pd
import requests
from datetime import datetime

app = FastAPI()

# 1. Cargamos el súper modelo XGBoost que acabas de entrenar
modelo = joblib.load('modelo_mlb_xgboost.pkl')

@app.get("/")
def leer_raiz():
    return {"mensaje": "Servidor automatizado de la MLB en línea"}

@app.get("/partidos_hoy")
def obtener_partidos_hoy():
    # 2. Obtenemos la fecha actual en formato Año-Mes-Día (ej. 2026-07-13)
    hoy = datetime.today().strftime('%Y-%m-%d')
    
    # 3. Consultamos la API oficial de la MLB para el calendario del día
    url_mlb = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={hoy}"
    
    try:
        respuesta = requests.get(url_mlb)
        datos_json = respuesta.json()
        lista_predicciones = []
        
        # Verificamos si hay partidos programados para la fecha actual
        if 'dates' in datos_json and len(datos_json['dates']) > 0:
            for juego in datos_json['dates'][0]['games']:
                local = juego['teams']['home']['team']['name']
                visitante = juego['teams']['away']['team']['name']
                
                # NOTA TÉCNICA: Como aún no creamos la mega base de datos histórica en el servidor
                # para calcular el wOBA y FIP en vivo de cada equipo, usaremos valores estándar temporales 
                # (es_local=1, racha_ofensiva=5.0 carreras, racha_defensa=4.0 carreras) para que el modelo XGBoost opere.
                datos_entrada = pd.DataFrame([[1, 5.0, 4.0]], columns=['es_local', 'racha_ofensiva', 'racha_pitcheo_defensa'])
                
                # 4. Ejecutamos la predicción con tu archivo .pkl
                probabilidades = modelo.predict_proba(datos_entrada)[0]
                prob_gana_vis = round(probabilidades[0] * 100, 2)
                prob_gana_local = round(probabilidades[1] * 100, 2)
                
                lista_predicciones.append({
                    "id_juego": juego['gamePk'],
                    "local": local,
                    "visitante": visitante,
                    "prob_local": prob_gana_local,
                    "prob_visitante": prob_gana_vis,
                    "favorito": local if prob_gana_local > prob_gana_vis else visitante
                })
                
        return lista_predicciones
        
    except Exception as e:
        return {"error": f"No se pudieron obtener los partidos: {str(e)}"}