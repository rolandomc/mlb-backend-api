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
    except Exception:
        pass

@app.get("/")
def leer_raiz():
    return {"mensaje": "Servidor Quants MLB 100% Real + LIVE activo"}

cache_stats_mlb = {}

def obtener_stats_reales_api(team_id):
    if team_id in cache_stats_mlb:
        return cache_stats_mlb[team_id]
        
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&group=hitting,pitching"
    
    stats_finales = {
        "ops": 0.720, "whip": 1.30, "k9": 8.5, "racha": 5, "era": 4.10,
        "avg": ".245", "hr": 120, "obp": ".315", "slg": ".405"
    }
    
    try:
        respuesta = requests.get(url, timeout=5)
        datos = respuesta.json()
        
        if 'stats' in datos:
            for categoria in datos['stats']:
                grupo = categoria.get('group', {}).get('displayName')
                metricas = categoria.get('splits', [{}])[0].get('stat', {})
                
                if grupo == 'hitting':
                    stats_finales['ops'] = float(metricas.get('ops', 0.720))
                    stats_finales['avg'] = metricas.get('avg', '.245')
                    stats_finales['hr'] = metricas.get('homeRuns', 120)
                elif grupo == 'pitching':
                    stats_finales['whip'] = float(metricas.get('whip', 1.30))
                    stats_finales['era'] = float(metricas.get('era', 4.10))
                    stats_finales['k9'] = float(metricas.get('strikeoutsPer9Inn', 8.5))
        
        cache_stats_mlb[team_id] = stats_finales
        return stats_finales
    except Exception:
        return stats_finales

@app.get("/partidos_hoy")
def obtener_partidos_hoy():
    # 1. EL FIX DE LA ZONA HORARIA:
    # Restamos 7 horas al servidor (UTC) para forzarlo a pensar en horario de Norteamérica (MST/PT).
    # Así garantizamos que no se salte los juegos nocturnos.
    hoy = datetime.now() - timedelta(hours=7)
    futuro = hoy + timedelta(days=5)
    
    hoy_str = hoy.strftime('%Y-%m-%d')
    futuro_str = futuro.strftime('%Y-%m-%d')
    
    # 2. EL FIX DEL ALL-STAR:
    # Agregamos 'E' (Exhibition) y 'S' (Spring/Special) por si la MLB cambió la etiqueta del All-Star
    url_mlb = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameTypes=R,A,E,S,F,D,L,W&hydrate=linescore&startDate={hoy_str}&endDate={futuro_str}"
    
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
                    
                    # --- EXTRACCIÓN DE DATOS EN VIVO ---
                    estado_abstracto = juego.get('status', {}).get('abstractGameState', 'Preview')
                    linescore = juego.get('linescore', {})
                    
                    es_en_vivo = False
                    es_final = False
                    estado_juego = "PROGRAMADO"
                    score_local = 0
                    score_visita = 0
                    
                    if estado_abstracto == "Live":
                        es_en_vivo = True
                        inning = linescore.get('currentInningOrdinal', '')
                        is_top = "Top" if linescore.get('isTopInning') else "Bot"
                        estado_juego = f"LIVE {is_top} {inning}"
                        score_local = linescore.get('teams', {}).get('home', {}).get('runs', 0)
                        score_visita = linescore.get('teams', {}).get('away', {}).get('runs', 0)
                    elif estado_abstracto == "Final":
                        es_final = True
                        estado_juego = "FINAL"
                        score_local = linescore.get('teams', {}).get('home', {}).get('runs', 0)
                        score_visita = linescore.get('teams', {}).get('away', {}).get('runs', 0)
                    else:
                        estado_juego = "PROGRAMADO"
                        
                    # --- PREDICCIONES Y LÍMITES ESTRICTOS ---
                    stats_visita = obtener_stats_reales_api(id_visitante)
                    stats_local = obtener_stats_reales_api(id_local)

                    if modelo is not None:
                        try:
                            datos_v = pd.DataFrame([[0, stats_visita['ops'], stats_visita['whip'], stats_visita['k9'], stats_visita['racha']]], columns=['es_local', 'ops', 'whip', 'k9', 'racha'])
                            datos_l = pd.DataFrame([[1, stats_local['ops'], stats_local['whip'], stats_local['k9'], stats_local['racha']]], columns=['es_local', 'ops', 'whip', 'k9', 'racha'])
                            carr_visita = float(modelo.predict(datos_v)[0])
                            carr_local = float(modelo.predict(datos_l)[0])
                        except:
                            carr_visita, carr_local = 4.2, 4.5
                    else:
                        carr_visita, carr_local = 4.2, 4.5
                    
                    # 🛡️ LIMITADOR ESTRICTO 
                    carr_visita = min(max(carr_visita, 2.5), 5.2)
                    carr_local = min(max(carr_local, 2.5), 5.5)

                    # 🌟 REGLA ESPECÍFICA ALL-STAR
                    if "All-Star" in local or "All-Star" in visitante or id_local in [159, 160, 159, 160] or "National" in local or "American" in local:
                        carr_visita, carr_local = 4.2, 4.3
                        local = "NL All-Stars" if "NL" in local or "National" in local else "AL All-Stars"
                        visitante = "AL All-Stars" if "AL" in visitante or "American" in visitante else "NL All-Stars"

                    carr_visita = round(carr_visita, 1)
                    carr_local = round(carr_local, 1)
                    total_carreras = round(carr_visita + carr_local, 1)
                    favorito = local if carr_local > carr_visita else visitante
                    dif = round(carr_local - carr_visita, 1)
                    spread = f"{local} -1.5" if dif >= 1.5 else (f"{visitante} -1.5" if dif <= -1.5 else f"{favorito} ML")
                    prob_loc = round((carr_local / total_carreras) * 100)

                    lista_predicciones.append({
                        "id_juego": juego['gamePk'],
                        "fecha": dia['date'],
                        "local": local,
                        "visitante": visitante,
                        "favorito": favorito,
                        "prob_local": prob_loc,
                        "prob_visitante": 100 - prob_loc,
                        "estado": {
                            "texto": estado_juego,
                            "en_vivo": es_en_vivo,
                            "finalizado": es_final,
                            "score_local": score_local,
                            "score_visita": score_visita
                        },
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