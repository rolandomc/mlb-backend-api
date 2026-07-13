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
from datetime import datetime, timedelta # Asegúrate de importar timedelta arriba

@app.get("/partidos_hoy")
def obtener_partidos_hoy():
    # 1. Calculamos la fecha de hoy y la fecha de dentro de 5 días
    hoy = datetime.today()
    futuro = hoy + timedelta(days=5)
    
    hoy_str = hoy.strftime('%Y-%m-%d')
    futuro_str = futuro.strftime('%Y-%m-%d')
    
    # 2. Le pedimos a la MLB un rango de fechas
    url_mlb = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={hoy_str}&endDate={futuro_str}"
    
    try:
        respuesta = requests.get(url_mlb)
        datos_json = respuesta.json()
        lista_predicciones = []
        
        # 3. Ahora tenemos que recorrer varios "días" (dates)
        if 'dates' in datos_json:
            for dia in datos_json['dates']:
                fecha_juego = dia['date'] # Extraemos la fecha del partido
                
                for juego in dia['games']:
                    local = juego['teams']['home']['team']['name']
                    visitante = juego['teams']['away']['team']['name']
                    
                    # Simulamos los datos temporalmente para que el modelo funcione
                    datos_entrada = pd.DataFrame([[1, 5.0, 4.0]], columns=['es_local', 'racha_ofensiva', 'racha_pitcheo_defensa'])
                    
                    probabilidades = modelo.predict_proba(datos_entrada)[0]
                    prob_gana_vis = round(probabilidades[0] * 100, 2)
                    prob_gana_local = round(probabilidades[1] * 100, 2)
                    
                    lista_predicciones.append({
                        "id_juego": juego['gamePk'],
                        "fecha": fecha_juego, # Agregamos la fecha al resultado
                        "local": local,
                        "visitante": visitante,
                        "prob_local": prob_gana_local,
                        "prob_visitante": prob_gana_vis,
                        "favorito": local if prob_gana_local > prob_gana_vis else visitante
                    })
                    
        return lista_predicciones
        
    except Exception as e:
        return {"error": f"No se pudieron obtener los partidos: {str(e)}"}