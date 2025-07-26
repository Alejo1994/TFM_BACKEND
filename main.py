import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Optional, Union


# Cargar variables de entorno al inicio
load_dotenv()

# Importar funciones y clientes de utilidad
from utils import (
    S3_CLIENT,
    S3_BUCKET_NAME,
    query_rag,
    recreate_vector_store_from_all_documents,
    update_vector_store_for_rag,
    load_document_content_from_s3,
    initialize_system,
    get_vector_store,
    get_doc_store # Asegúrate de que esta función ahora devuelva CustomS3DocumentStore
)
from botocore.exceptions import NoCredentialsError # Para capturar errores de AWS


app = FastAPI(
    title="RAG Backend API (ParentDocumentRetriever)",
    description="API para un sistema RAG avanzado con ParentDocumentRetriever, S3 y Pinecone/Redis.",
    version="3.0.0", # Nueva versión para indicar un cambio mayor
)

# --- ¡CONFIGURACIÓN CORS AQUÍ! ---
# Lista de orígenes permitidos. Para desarrollo, puedes usar ["*"] (todos los orígenes),
# pero para producción, DEBES especificar los dominios exactos.
# El dominio de Lovable Project es `https://b23f2452-424c-4193-9c31-22d63193d802.lovableproject.com`
# También incluye tu URL de Render para pruebas si la usas desde el navegador directamente.

# Puedes obtener tu URL de Render desde el dashboard de Render.
# Para este ejemplo, voy a asumir que quieres permitir Lovable Project y cualquier origen.
# Recomiendo encarecidamente *NO* usar ["*"] en producción.

origins = [
    "https://b23f2452-424c-4193-9c31-22d63193d802.lovableproject.com", # Tu frontend en Lovable Project
    "https://b23f2452-42ac-4193-9c31-22d63193d802.lovableproject.com",
    "https://tfm-backend-ik11.onrender.com", # Tu propio backend de Render si accedes directamente
   "*"
]

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models para las respuestas/peticiones
class DocumentUploadResponse(BaseModel):
    message: str
    filename: str

class DocumentListResponse(BaseModel):
    documents: List[str]

class DocumentContentResponse(BaseModel):
    filename: str
    content: str

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[str]] = [] # Puedes tipar mejor como List[AIMessage | HumanMessage] si los pasas como strings y los conviertes

class QueryResponse(BaseModel):
    response: str
    source_documents: List[dict] # Para mostrar metadatos de las fuentes

# --- Eventos de startup/shutdown para FastAPI ---
@app.on_event("startup")
async def startup_event():
    """
    Evento que se ejecuta al iniciar la aplicación FastAPI.
    Aquí inicializamos las conexiones a S3 y Pinecone y poblamos el RAG.
    """
    print("Iniciando aplicación FastAPI...")
    try:
        initialize_system()
        # Se recomienda recrear el vector store al inicio para asegurar que está sincronizado con S3
        # o que al menos la conexión esté viva.
        # recreate_vector_store_from_all_documents() # Esto podría ser costoso en startup si hay muchos docs
                                                    # Solo llámalo si necesitas poblar el índice
                                                    # desde cero cada vez que la API se inicia.
                                                    # Si Pinecone ya tiene los datos, no es necesario.
        print("Aplicación lista y sistema RAG inicializado.")
    except Exception as e:
        print(f"Error crítico durante el inicio de la aplicación: {e}")
        # Dependiendo de la severidad, podrías querer que la aplicación no se inicie
        # raise # Esto hará que la aplicación falle al iniciar

# --- Endpoints de gestión de documentos (CRUD) ---

@app.post("/documents/upload", response_model=DocumentUploadResponse, summary="Subir un nuevo documento a S3 y procesarlo para RAG.")
async def upload_document(file: UploadFile = File(...)):
    if not S3_BUCKET_NAME: raise HTTPException(status_code=500, detail="S3 bucket name not configured.")
    if not S3_CLIENT: raise HTTPException(status_code=500, detail="S3 client not initialized.")
    if not file.filename: raise HTTPException(status_code=400, detail="Filename cannot be empty.")

    filename = file.filename
    try:
        S3_CLIENT.upload_fileobj(file.file, S3_BUCKET_NAME, filename)
        print(f"Archivo '{filename}' subido a S3.")
        update_vector_store_for_rag(file_paths_to_add=[filename]) # Dispara indexación
        return {"message": f"Archivo '{filename}' subido a S3 y procesado para RAG.", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el documento para RAG: {e}")

@app.get("/documents", response_model=DocumentListResponse, summary="Listar todos los documentos en S3.")
async def list_documents():
    if not S3_BUCKET_NAME: raise HTTPException(status_code=500, detail="S3 bucket name not configured.")
    if not S3_CLIENT: raise HTTPException(status_code=500, detail="S3 client not initialized.")
    try:
        response = S3_CLIENT.list_objects_v2(Bucket=S3_BUCKET_NAME)
        files = [obj['Key'] for obj in response.get('Contents', [])]
        return {"documents": files}
    except NoCredentialsError: raise HTTPException(status_code=500, detail="AWS credentials not configured or invalid.")
    except Exception as e: raise HTTPException(status_code=500, detail=f"Error listing documents from S3: {e}")

@app.get("/documents/{filename}", response_model=DocumentContentResponse, summary="Obtener el contenido de un documento desde S3.")
async def get_document_content(filename: str):
    if not S3_BUCKET_NAME: raise HTTPException(status_code=500, detail="S3 bucket name not configured.")
    if not S3_CLIENT: raise HTTPException(status_code=500, detail="S3 client not initialized.")

    content_bytes = load_document_content_from_s3(filename)
    if content_bytes is None: raise HTTPException(status_code=404, detail="Documento no encontrado.")
    
    try: return {"filename": filename, "content": content_bytes.decode('utf-8')}
    except UnicodeDecodeError: raise HTTPException(status_code=500, detail="Could not decode file content to UTF-8 text.")
    except Exception as e: raise HTTPException(status_code=500, detail=f"Error getting document content: {e}")

@app.delete("/documents/{filename}", summary="Eliminar un documento de S3 y del RAG.")
async def delete_document(filename: str):
    if not S3_BUCKET_NAME: raise HTTPException(status_code=500, detail="S3 bucket name not configured.")
    if not S3_CLIENT: raise HTTPException(status_code=500, detail="S3 client not initialized.")
    try:
        S3_CLIENT.delete_object(Bucket=S3_BUCKET_NAME, Key=filename)
        print(f"Archivo '{filename}' eliminado de S3.")
        # La eliminación dispara una reconstrucción completa para asegurar consistencia
        recreate_vector_store_from_all_documents()
        return {"message": f"Archivo '{filename}' eliminado de S3 y procesado para su remoción del RAG."}
    except S3_CLIENT.exceptions.NoSuchKey: raise HTTPException(status_code=404, detail="Documento no encontrado en S3.")
    except Exception as e: raise HTTPException(status_code=500, detail=f"Error deleting document: {e}")


# --- Endpoint para el RAG ---

 #En query endpoint, el check de doc_store cambiaría:
@app.post("/query", response_model=QueryResponse, summary="Consultar el sistema RAG.")
async def query(request: QueryRequest):
    # ! IMPORTANTE: REMOVER O MODIFICAR ESTA LÍNEA QUE CAUSA EL ERROR !
    # if not get_vector_store()._index_name: # <-- ESTA LÍNEA
    #     raise HTTPException(status_code=400, detail="El índice RAG (Pinecone) está vacío o no inicializado. Por favor, suba documentos o recree el índice.")
    
    # Opción más segura y simple: Asumir que si el get_vector_store() no falló,
    # el vector store está inicializado. Si el índice está vacío, el retriever
    # simplemente no encontrará documentos y el LLM responderá "No tengo suficiente información".
    # Si quieres una validación explícita para saber si el índice tiene datos,
    # tendrías que consultar la API de Pinecone para la cuenta de vectores.
    # Por ejemplo (requiere la librería 'pinecone-client' directamente, no solo la integración):
    # from pinecone import Pinecone
    # pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    # try:
    #     index_stats = pinecone_client.describe_index_stats(index=os.getenv("PINECONE_INDEX_NAME"))
    #     if index_stats.dimension == 0 or index_stats.total_vector_count == 0:
    #         raise HTTPException(status_code=400, detail="El índice RAG (Pinecone) está vacío. Suba documentos primero.")
    # except Exception as e:
    #     print(f"Error al verificar stats de Pinecone: {e}")
    #     # Si falla la verificación de stats, podría ser un problema de conexión a Pinecone.
    #     # Decidir si permitir la consulta o lanzar un error.
    #     pass # Permitir que la consulta RAG continúe, el LLM responderá si no hay docs.


    try:
        response = query_rag(request.query, request.chat_history) 
        source_docs_formatted = []
        if 'source_documents' in response:
            for doc in response['source_documents']:
                source_docs_formatted.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })

        return {"response": response['result'], "source_documents": source_docs_formatted}
    except Exception as e:
        print(f"Error processing RAG query: {e}")
        # Aquí puedes añadir más detalles al error si necesitas ver la traza completa
        import traceback
        traceback.print_exc() # Imprime la traza completa en los logs del servidor
        raise HTTPException(status_code=500, detail=f"Error interno al procesar la consulta RAG: {e}")

    

# --- Endpoint de salud ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API is running"}

# --- Endpoint de reinicio de RAG (útil para desarrollo/depuración) ---
@app.post("/recreate-rag-index", summary="Recrea completamente el índice RAG desde todos los documentos en S3 y Redis.")
async def recreate_rag_index_endpoint():
    try:
        recreate_vector_store_from_all_documents()
        return {"message": "Índice RAG recreado exitosamente desde S3 y Redis."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al recrear el índice RAG: {e}")
