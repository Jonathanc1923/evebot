from flask import Flask, request, jsonify, render_template
from llama_cpp import Llama
import os
import requests

# Configuración del modelo
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_FILENAME = "Llama-3.2-3B-Instruct-Q4_0.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Enlace del modelo en AWS S3
S3_URL = "https://mi-bucket-llama-model.s3.us-east-2.amazonaws.com/Llama-3.2-3B-Instruct-Q4_0.gguf"

def download_model():
    """Descarga el modelo desde AWS S3 si no está disponible localmente."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.isfile(MODEL_PATH):
        print("Descargando el modelo desde AWS S3...")
        response = requests.get(S3_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as model_file:
                for chunk in response.iter_content(chunk_size=8192):
                    model_file.write(chunk)
            print("Modelo descargado exitosamente.")
        else:
            print(f"Error al descargar el modelo: {response.status_code}")
            raise Exception("No se pudo descargar el modelo.")

# Descargar el modelo si no está disponible
download_model()

# Inicializa Flask y carga el modelo
app = Flask(__name__)

# Cargar el modelo
print("Cargando el modelo...")
llm = Llama(model_path=MODEL_PATH)
print("Modelo cargado correctamente.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_response():
    # Obtener el mensaje del usuario
    user_input = request.json.get('prompt', '')
    
    # Añadir contexto para respuestas amistosas y cortas
    prompt = f"Eres un asistente amigable que responde con frases cortas, motivadoras y adecuadas para niños. Responde en español: {user_input}"
    
    # Generar respuesta usando el modelo
    response = llm(
        prompt,
        max_tokens=50,  # Limita los tokens para asegurar respuestas cortas
        temperature=0.8,  # Ajusta la creatividad
        top_k=40,
        top_p=0.9
    )
    
    # Extraer y devolver el texto generado
    model_output = response['choices'][0]['text'].strip()
    return jsonify({"response": model_output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
