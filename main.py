import os
import io
import base64
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# Cargar variables de entorno del archivo .env
load_dotenv()

app = FastAPI(title="Editor de Imágenes IA")

# Archivos estáticos y plantillas
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Leer tu API KEY unificada (la que empieza por AQ.Ab8)
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Inicializar el cliente oficial de Google GenAI forzando el canal v1beta para Imagen 3
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(api_version="v1beta")
)

# ======================
# RUTA PRINCIPAL
# ======================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )


# ======================
# GENERAR IMAGEN
# ======================
@app.post("/api/generar")
async def generar_imagen(prompt: str = Form(...), ratio: str = Form("1:1")):
    try:
        # Formatos válidos oficiales para Imagen 3: "1:1", "3:4", "4:3", "16:9", "9:16"
        formato_valido = ratio if ratio in ["1:1", "3:4", "4:3", "16:9", "9:16"] else "1:1"

        # Llamada oficial a Imagen 3 usando el entorno v1beta configurado en el cliente
        result = client.models.generate_images(
            model='imagen-3.0-generate-002',
            prompt=prompt,
            config=dict(
                number_of_images=1,
                output_mime_type="image/jpeg",
                aspect_ratio=formato_valido
            )
        )

        # Extracción correcta de bytes según la estructura de la nueva SDK google-genai
        image_bytes = result.generated_images[0].image.image_bytes
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{base64_image}"

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
        # Leer el archivo que el usuario subió desde la interfaz web
        contenido = await imagen.read()
        pil_image = Image.open(io.BytesIO(contenido))

        # Modificación de imagen mediante inpainting/outpainting
        result = client.models.edit_images(
            model='imagen-3.0-capability-002',
            prompt=prompt,
            image=pil_image,
            config=dict(
                number_of_images=1,
                output_mime_type="image/jpeg"
            )
        )

        # Convertir el resultado editado a Base64 para enviarlo al Frontend
        image_bytes = result.generated_images[0].image.image_bytes
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{base64_image}"

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
# AJUSTE OBLIGATORIO PARA RENDER
# ======================
if __name__ == '__main__':
    import uvicorn
    # Render asigna el puerto automáticamente. Si no existe, usa el 5000 por defecto.
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
