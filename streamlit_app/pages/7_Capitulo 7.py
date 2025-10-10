import streamlit as st
import subprocess
import os
from pathlib import Path

# ======== CONFIGURACIÓN GENERAL ========
st.set_page_config(
    page_title="Capítulo 7 - Segmentación de Imágenes",
    page_icon="🎯",
    layout="wide",
)

# ======== ESTILO PERSONALIZADO ========
st.markdown("""
    <style>
        /* Fondo general */
        .main { background-color: #f1f5f9; }
        
        /* Título principal */
        .main-title {
            text-align: center;
            color: #0f172a;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            margin-top: 1rem;
        }
        
        .subtitle {
            text-align: center;
            color: #64748b;
            font-size: 1.1rem;
            margin-bottom: 3rem;
            font-weight: 400;
        }
        
        /* Cards personalizadas */
        .detection-card {
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 
                        0 2px 4px -1px rgba(0, 0, 0, 0.06),
                        0 0 20px rgba(251, 191, 36, 0.15);
            transition: all 0.3s ease;
            margin-bottom: 1rem;
            border: 1px solid rgba(251, 191, 36, 0.2);
            position: relative;
        }
        
        .detection-card:hover {
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 
                        0 10px 10px -5px rgba(0, 0, 0, 0.04),
                        0 0 30px rgba(251, 191, 36, 0.3);
            transform: translateY(-2px);
            border: 1px solid rgba(251, 191, 36, 0.4);
        }
        
        .card-icon {
            font-size: 3rem;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .card-title {
            color: #1e293b;
            font-size: 1.3rem;
            font-weight: 600;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        
        /* Botones */
        div.stButton > button {
            width: 100%;
            border-radius: 10px;
            font-weight: 600;
            font-size: 1rem;
            padding: 0.7rem 1.5rem;
            border: none;
            transition: all 0.2s ease;
            margin-top: 0.5rem;
        }
        
        .start-button > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
        }
        
        .start-button > button:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            transform: scale(1.02);
            box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4);
        }
        
        .stop-button > button {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
        }
        
        .stop-button > button:hover {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            transform: scale(1.02);
            box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.4);
        }
        
        /* Status badge */
        .status-badge {
            position: absolute;
            top: 1rem;
            right: 1rem;
            padding: 0.35rem 0.85rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .status-running {
            background-color: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
        }
        
        .status-stopped {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            color: #94a3b8;
            font-size: 0.9rem;
            margin-top: 4rem;
            padding: 2rem;
            border-top: 1px solid #e2e8f0;
        }
        
        /* Ajustes de columnas */
        [data-testid="column"] {
            padding: 0.5rem;
        }
        
        /* Contenedor principal con ancho del 90% */
        .block-container {
            max-width: 90% !important;
            padding-left: 5rem !important;
            padding-right: 5rem !important;
        }
        
        /* Contenedor de botones */
        .button-container {
            display: flex;
            gap: 1rem;
            margin-top: 0.5rem;
        }
        
        .button-wrapper {
            flex: 1;
        }
    </style>
""", unsafe_allow_html=True)

# ======== TÍTULO ========
st.markdown("<h1 class='main-title'>🎯 Segmentación de Imágenes</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Divide y analiza imágenes mediante técnicas de segmentación avanzada</p>", unsafe_allow_html=True)

# ======== ARCHIVOS ========
files = {
    "./chapter7/prueba1.py": {"name": "Segmentar una Imagen", "icon": "🔍"},
    "./chapter7/prueba2.py": {"name": "Cuenca Hidrográfica", "icon": "💧"},
    "./chapter7/prueba3.py": {"name": "Aproximar un Contorno", "icon": "📐"},
    "./chapter7/prueba4.py": {"name": "Censurar Formas", "icon": "🚫"},
    "./chapter7/prueba5.py": {"name": "Detectar Convexidad", "icon": "⬢"}
}

# ======== CONTROL DE PROCESOS ========
if "processes" not in st.session_state:
    st.session_state.processes = {}

def start_process(script_path, label):
    """Ejecutar un script en un subproceso"""
    if label in st.session_state.processes and st.session_state.processes[label].poll() is None:
        st.warning(f"⚙️ {label} ya está en ejecución.")
        return
    try:
        process = subprocess.Popen(["python", script_path])
        st.session_state.processes[label] = process
        st.success(f"✅ {label} iniciado correctamente.")
    except Exception as e:
        st.error(f"❌ Error al ejecutar {label}: {e}")

def stop_process(label):
    """Detener un proceso activo"""
    process = st.session_state.processes.get(label)
    if process and process.poll() is None:
        try:
            process.terminate()
            st.session_state.processes[label] = None
            st.info(f"🛑 {label} detenido correctamente.")
        except Exception as e:
            st.error(f"❌ Error al detener {label}: {e}")
    else:
        st.warning(f"⚠️ {label} no se está ejecutando.")

def is_running(label):
    """Verificar si un proceso está en ejecución"""
    process = st.session_state.processes.get(label)
    return process and process.poll() is None

# ======== CREAR CARDS EN GRID ========
script_items = list(files.items())

# Crear grid de 2 columnas con espaciado
cols_per_row = 2
for i in range(0, len(script_items), cols_per_row):
    cols = st.columns(cols_per_row, gap="large")
    
    for j, col in enumerate(cols):
        idx = i + j
        if idx < len(script_items):
            script_path, info = script_items[idx]
            label = info["name"]
            icon = info["icon"]
            
            with col:
                # Card container
                with st.container():
                    # Status badge
                    running = is_running(label)
                    status_class = "status-running" if running else "status-stopped"
                    status_text = "● En ejecución" if running else "● Detenido"
                    
                    st.markdown(f"""
                        <div class="detection-card">
                            <span class="status-badge {status_class}">{status_text}</span>
                            <div class="card-icon">{icon}</div>
                            <div class="card-title">{label}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Botones debajo de la card alineados al ancho
                    st.markdown('<div class="button-container">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 1], gap="medium")
                    
                    with col1:
                        st.markdown('<div class="start-button button-wrapper">', unsafe_allow_html=True)
                        if running:
                            st.button(
                                "▶️ Iniciar",
                                key=f"start_{label}",
                                disabled=True
                            )
                        else:
                            st.button(
                                "▶️ Iniciar",
                                key=f"start_{label}",
                                on_click=start_process,
                                args=(script_path, label)
                            )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="stop-button button-wrapper">', unsafe_allow_html=True)
                        if not running:
                            st.button(
                                "⛔ Detener",
                                key=f"stop_{label}",
                                disabled=True
                            )
                        else:
                            st.button(
                                "⛔ Detener",
                                key=f"stop_{label}",
                                on_click=stop_process,
                                args=(label,)
                            )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

# ======== FOOTER ========
st.markdown("""
    <div class='footer'>
        <strong>Capítulo 7</strong> — Segmentación de Imágenes con OpenCV<br>
        Análisis y división de regiones en imágenes
    </div>
""", unsafe_allow_html=True)