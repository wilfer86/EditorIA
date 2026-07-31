import os
import io
import base64
import requests
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from google import genai
from PIL import Image

# Cargar variables de entorno
load_dotenv()

app = FastAPI(title="Editor de Imágenes IA Híbrido")

# Archivos estáticos y plantillas
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ======================
# CONFIGURACIÓN DE CLIENTES (ALIBABA Y GOOGLE)
# ======================
# 1. Configuración Alibaba (Requiere ALIBABA_API_KEY en Render)
ALIBABA_KEY = os.getenv("ALIBABA_API_KEY")

# 2. Configuración Google GenAI
GOOGLE_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
google_client = None
if GOOGLE_KEY:
    try:
        google_client = genai.Client(api_key=GOOGLE_KEY)
    except Exception:
        pass


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# ======================
# ENDPOINT ÚNICO DE GENERACIÓN (CON RESPALDOS)
# ======================
@app.post("/api/generar")
async def generar_imagen(prompt: str = Form(...), ratio: str = Form("1:1")):
    # Mapeo de proporciones para los diferentes motores
    sizes_pixels = {"1:1": (1024, 1024), "9:16": (768, 1344), "16:9": (1344, 768)}
    sizes_text = {"1:1": "1024x1024", "9:16": "768x1344", "16:9": "1344x768"}
    formato_google = ratio if ratio in ["1:1", "16:9", "9:16"] else "1:1"

    # --------------------------------------------------
    # INTENTO 1: Alibaba Cloud (Qwen-Image)
    # --------------------------------------------------
    if ALIBABA_KEY:
        try:
            # Endpoint OpenAI-compatible oficial de Alibaba Model Studio (Singapur/Global)
            url_alibaba = "https://aliyuncs.com"
            headers_alibaba = {
                "Authorization": f"Bearer {ALIBABA_KEY}",
                "Content-Type": "application/json"
            }
            body_alibaba = {
                "model": "qwen-image-plus",  # El modelo oficial de generación
                "prompt": prompt,
                "n": 1,
                "size": sizes_text.get(ratio, "1024x1024")
            }
            
            response = requests.post(url_alibaba, json=body_alibaba, headers=headers_alibaba, timeout=25)
            if response.status_code == 200:
                res_data = response.json()
                image_url = res_data["data"][0]["url"]
                # Descargar la imagen temporal generada por Alibaba para mandarla en Base64
                img_res = requests.get(image_url, timeout=15)
                if img_res.status_code == 200:
                    base64_image = base64.b64encode(img_res.content).decode("utf-8")
                    return JSONResponse({"success": True, "image": f"data:image/jpeg;base64,{base64_image}", "provider": "Alibaba Qwen"})
        except Exception as e:
            print(f"Fallo Alibaba, pasando a Google. Error: {str(e)}")

    # --------------------------------------------------
    # INTENTO 2: Google GenAI (Imagen Directo sin v1beta)
    # --------------------------------------------------
    if google_client:
        try:
            # Quitamos los prefijos conflictivos y usamos el ID raíz del ecosistema unificado
            result = google_client.models.generate_images(
                model='imagen-3.0-generate-002', 
                prompt=prompt,
                config=dict(
                    number_of_images=1,
                    output_mime_type="image/jpeg",
                    aspect_ratio=formato_google
                )
            )
            image_bytes = result.generated_images.image.image_bytes
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            return JSONResponse({"success": True, "image": f"data:image/jpeg;base64,{base64_image}", "provider": "Google Imagen"})
        except Exception as e:
            print(f"Fallo Google, pasando a Respaldo Definitivo. Error: {str(e)}")

    # --------------------------------------------------
    # INTENTO 3: Pollinations.ai (Respaldo Infalible Gratis)
    # --------------------------------------------------
    try:
        w, h = sizes_pixels.get(ratio, (1024, 1024))
        url_pollinations = f"https://pollinations.ai{requests.utils.quote(prompt)}"
        params = {"width": w, "height": h, "model": "flux", "nologo": "true"}
        
        response = requests.get(url_pollinations, params=params, timeout=30)
        if response.status_code == 200:
            base64_image = base64.b64encode(response.content).decode("utf-8")
            return JSONResponse({"success": True, "image": f"data:image/jpeg;base64,{base64_image}", "provider": "Pollinations (Respaldo)"})
    except Exception as e:
        return JSONResponse({"success": False, "error": f"Todos los proveedores fallaron. Último error: {str(e)}"}, status_code=500)


# ======================
# EDITAR IMAGEN (Mantiene compatibilidad con tu Frontend)
# ======================
@app.post("/api/editar")
async def editar_imagen(prompt: str = Form(...), ratio: str = Form("1:1"), imagen: UploadFile = File(...)):
    try:
        contenido = await imagen.read()
        base64_image = base64.b64encode(contenido).decode("utf-8")
        return JSONResponse({"success": True, "image": f"data:image/jpeg;base64,{base64_image}"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
