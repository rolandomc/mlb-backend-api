from fastapi import FastAPI
import pandas as pd
import requests
from datetime import datetime, timedelta
import joblib
import os

app = FastAPI()

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

cache_stats_mlb = {}

def obtener_stats_reales_api(team_id):
    if team_id in cache_stats_mlb:
        return cache_stats_mlb[team_id]
        
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&group=hitting,pitching"
    
    # Valores por defecto ajustados a la baja (Evita que sumen 11 carreras)
    stats_finales = {
        "ops": 0.710, "whip": 1.35, "k9": 8.0, "racha": 5, "era": 4.20,
        "avg": ".240", "hr": 130, "obp": ".310", "slg": ".400"
    }
    
    try:
        respuesta = requests.get(url, timeout=5)
        datos = respuesta.json()
        
        if 'stats' in datos:
            for categoria in datos['stats']:
                grupo = categoria.get('group', {}).get('displayName')
                metricas = categoria.get('splits', [{}])[0].get('stat', {})
                
                if grupo == 'hitting':
                    stats_finales['ops'] = float(metricas.get('ops', 0.710))
                    stats_finales['avg'] = metricas.get('avg', '.240')
                    stats_finales['hr'] = metricas.get('homeRuns', 130)
                    stats_finales['obp'] = metricas.get('obp', '.310')
                elif grupo == 'pitching':
                    stats_finales['whip'] = float(metricas.get('whip', 1.35))
                    stats_finales['era'] = float(metricas.get('era', 4.20))
                    stats_finales['k9'] = float(metricas.get('strikeoutsPer9Inn', 8.0))
        
        cache_stats_mlb[team_id] = stats_finales
        return stats_finales
        
    except Exception as e:
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
                    local = juego['teams']['home']['team']['name']
                    id_local = juego['teams']['home']['team']['id']
                    
                    visitante = juego['teams']['away']['team']['name']
                    id_visitante = juego['teams']['away']['team']['id']
                    
                    stats_visita = obtener_stats_reales_api(id_visitante)
                    stats_local = obtener_stats_reales_api(id_local)

                    # 1. PREDICCIÓN BASE XGBOOST
                    if modelo is not None:
                        try:
                            datos_visita = pd.DataFrame([[0, stats_visita['ops'], stats_visita['whip'], stats_visita['k9'], stats_visita['racha']]], columns=['es_local', 'ops', 'whip', 'k9', 'racha'])
                            datos_local = pd.DataFrame([[1, stats_local['ops'], stats_local['whip'], stats_local['k9'], stats_local['racha']]], columns=['es_local', 'ops', 'whip', 'k9', 'racha'])
                            
                            carr_visita = float(modelo.predict(datos_visita)[0])
                            carr_local = float(modelo.predict(datos_local)[0])
                        except Exception:
                            carr_visita, carr_local = 4.1, 4.4
                    else:
                        carr_visita, carr_local = 4.1, 4.4
                    
                    # 2. FILTRO VEGAS (Calibración Estándar)
                    # Bajamos el output del modelo para alinearlo con el Over/Under promedio de 8.5 a 9.0
                    carr_visita = max(1.5, carr_visita - 1.25)
                    carr_local = max(1.5, carr_local - 1.15) # Local retiene mínima ventaja
                    
                    # 3. OVERRIDE: JUEGO DE ESTRELLAS (Exhibición)
                    # Si detecta que es el All-Star, ignora la IA y aplica las líneas oficiales de Las Vegas
                    if "All-Star" in local or "All-Star" in visitante or "AL " in local or "NL " in local:
                        carr_visita = 4.2
                        carr_local = 4.3  # Vegas abrió el O/U en 8.5 para este juego
                        
                        # Fix para logos del All-Star (ya que no los encuentra en el diccionario normal)
                        local = "National League All-Stars" if "NL" in local or "National" in local else "American League All-Stars"
                        visitante = "American League All-Stars" if "AL" in visitante or "American" in visitante else "National League All-Stars"

                    carr_visita = round(carr_visita, 1)
                    carr_local = round(carr_local, 1)
                    
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