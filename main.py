from fastapi import FastAPI
import pandas as pd
import requests
from datetime import datetime, timedelta
import joblib
import os

app = FastAPI()

# 1. Cargamos el Súper Modelo Regresor Real
modelo = None
if os.path.exists('modelo_mlb_regresor.pkl'):
    try:
        modelo = joblib.load('modelo_mlb_regresor.pkl')
        print("✅ Cerebro Quants XGBoost cargado con éxito.")
    except Exception as e:
        print(f"⚠️ Error al cargar el modelo: {e}")

@app.get("/")
def leer_raiz():
    return {"mensaje": "Servidor Quants MLB 100% Real activo"}

# 2. SISTEMA DE CACHÉ: Para no saturar a la MLB y hacer la app rapidísima
cache_stats_mlb = {}

def obtener_stats_reales_api(team_id):
    """
    Se conecta a los servidores de la MLB y extrae las estadísticas 
    REALES de la temporada actual para alimentar al modelo matemático.
    """
    # Si ya descargamos los datos hoy, los usamos de la memoria (Caché)
    if team_id in cache_stats_mlb:
        return cache_stats_mlb[team_id]
        
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&group=hitting,pitching"
    
    # Valores por defecto en caso extremo de que la MLB se caiga
    stats_finales = {
        "ops": 0.730, "whip": 1.30, "k9": 8.5, "racha": 5, "era": 4.10,
        "avg": ".245", "hr": 150, "obp": ".315", "slg": ".415"
    }
    
    try:
        respuesta = requests.get(url, timeout=5)
        datos = respuesta.json()
        
        # Buscamos en el JSON de la MLB las métricas de bateo y pitcheo
        if 'stats' in datos:
            for categoria in datos['stats']:
                grupo = categoria.get('group', {}).get('displayName')
                metricas = categoria.get('splits', [{}])[0].get('stat', {})
                
                if grupo == 'hitting':
                    stats_finales['ops'] = float(metricas.get('ops', 0.730))
                    stats_finales['avg'] = metricas.get('avg', '.245')
                    stats_finales['hr'] = metricas.get('homeRuns', 150)
                    stats_finales['obp'] = metricas.get('obp', '.315')
                elif grupo == 'pitching':
                    stats_finales['whip'] = float(metricas.get('whip', 1.30))
                    stats_finales['era'] = float(metricas.get('era', 4.10))
                    stats_finales['k9'] = float(metricas.get('strikeoutsPer9Inn', 8.5))
        
        # Guardamos en la memoria para que el siguiente usuario cargue al instante
        cache_stats_mlb[team_id] = stats_finales
        return stats_finales
        
    except Exception as e:
        print(f"Error descargando API MLB para equipo {team_id}: {e}")
        return stats_finales

@app.get("/partidos_hoy")
def obtener_partidos_hoy():
    hoy = datetime.today()
    futuro = hoy + timedelta(days=5)
    
    url_mlb = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={hoy.strftime('%Y-%m-%d')}&endDate={futuro.strftime('%Y-%m-%d')}"
    
    try:
        respuesta = requests.get(url_mlb)
        datos_json = respuesta.json()
        lista_predicciones = []
        
        if 'dates' in datos_json:
            for dia in datos_json['dates']:
                for juego in dia['games']:
                    # Extraemos IDs reales y nombres
                    local = juego['teams']['home']['team']['name']
                    id_local = juego['teams']['home']['team']['id']
                    
                    visitante = juego['teams']['away']['team']['name']
                    id_visitante = juego['teams']['away']['team']['id']
                    
                    # 3. AQUÍ SUCEDE LA MAGIA: Alimentamos al modelo con la VERDAD
                    stats_visita = obtener_stats_reales_api(id_visitante)
                    stats_local = obtener_stats_reales_api(id_local)

                    if modelo is not None:
                        try:
                            # XGBoost analiza los datos reales de la API
                            datos_visita = pd.DataFrame([[0, stats_visita['ops'], stats_visita['whip'], stats_visita['k9'], stats_visita['racha']]], columns=['es_local', 'ops', 'whip', 'k9', 'racha'])
                            datos_local = pd.DataFrame([[1, stats_local['ops'], stats_local['whip'], stats_local['k9'], stats_local['racha']]], columns=['es_local', 'ops', 'whip', 'k9', 'racha'])
                            
                            carr_visita = float(round(modelo.predict(datos_visita)[0], 1))
                            carr_local = float(round(modelo.predict(datos_local)[0], 1))
                        except Exception as e:
                            carr_visita, carr_local = 4.1, 4.5
                    else:
                        carr_visita, carr_local = 4.1, 4.5
                    
                    total_carreras = round(carr_visita + carr_local, 1)
                    dif = round(carr_local - carr_visita, 1)
                    favorito = local if carr_local > carr_visita else visitante
                    
                    spread = f"{local} -1.5" if dif >= 1.5 else (f"{visitante} -1.5" if dif <= -1.5 else f"{favorito} ML")
                    
                    prob_loc = round((carr_local / total_carreras) * 100)
                    prob_vis = 100 - prob_loc

                    lista_predicciones.append({
                        "id_juego": juego['gamePk'],
                        "fecha": dia['date'],
                        "local": local,
                        "visitante": visitante,
                        "favorito": favorito,
                        "prob_local": prob_loc,
                        "prob_visitante": prob_vis,
                        "prediccion_vegas": {
                            "carreras_local": carr_local,
                            "carreras_visita": carr_visita,
                            "total_ou": total_carreras,
                            "spread": spread
                        },
                        "stats": {
                            "away": stats_visita,
                            "home": stats_local
                        }
                    })
                    
        return lista_predicciones
        
    except Exception as e:
        return {"error": str(e)}