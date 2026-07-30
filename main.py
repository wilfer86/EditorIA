from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from openai import OpenAI
import os
import base64
import uuid

# Cargar variables de entorno
load_dotenv()

app = FastAPI(title="Editor de Imágenes IA")

# Archivos estáticos y plantillas
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configurar cliente de NVIDIA (compatible con OpenAI)
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# ======================
# RUTA PRINCIPAL
# ======================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ======================
# GENERAR IMAGEN
# ======================
@app.post("/api/generar")
async def generar_imagen(prompt: str = Form(...), ratio: str = Form("1:1")):
    try:
        # Convertir ratio a tamaño aproximado
        sizes = {
            "1:1": "1024x1024",
            "9:16": "768x1344",
            "16:9": "1344x768"
        }
        size = sizes.get(ratio, "1024x1024")

        response = client.images.generate(
            model="qwen/qwen-image",          # Modelo de generación
            prompt=prompt,
            size=size,
            n=1
        )

        # La respuesta normalmente trae una URL o base64
        image_url = response.data[0].url

        return JSONResponse({
            "success": True,
            "image": image_url
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


# ======================
# EDITAR IMAGEN
# ======================
@app.post("/api/editar")
async def editar_imagen(
    prompt: str = Form(...),
    ratio: str = Form("1:1"),
    imagen: UploadFile = File(...)
):
    try:
        # Leer la imagen subida
        contenido = await imagen.read()
        imagen_base64 = base64.b64encode(contenido).decode("utf-8")

        # Nota: Algunos modelos de edición de NVIDIA reciben la imagen en base64
        response = client.images.edit(
            model="qwen/qwen-image-edit",     # Modelo de edición
            image=contenido,
            prompt=prompt,
            n=1
        )

        image_url = response.data[0].url

        return JSONResponse({
            "success": True,
            "image": image_url
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)