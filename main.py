import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Optional

import boto3 # Asegúrate de importar boto3 también
from botocore.exceptions import NoCredentialsError # ¡Asegúrate de que esté aquí!

# Cargar variables de entorno al inicio de la aplicación
load_dotenv()

# Importar funciones y clientes de utilidad
from utils import (
    S3_CLIENT, # Importa el cliente S3 inicializado
    S3_BUCKET_NAME, # Importa el nombre del bucket S3
    query_rag,
    recreate_vector_store_from_all_documents,
    update_vector_store_for_rag,
    load_document_content_from_s3, # Para obtener contenido de S3
    initialize_system # Para inicializar las conexiones
)

app = FastAPI(
    title="RAG Backend API",
    description="API para un sistema RAG (Retrieval Augmented Generation) con gestión de documentos CRUD. Ahora con S3 y Pinecone.",
    version="2.0.0",
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
    """
    Sube un archivo a AWS S3 y luego lo procesa para incluirlo en el sistema RAG.
    """
    if not S3_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="Nombre del bucket S3 no configurado.")
    if not S3_CLIENT:
        raise HTTPException(status_code=500, detail="Cliente S3 no inicializado.")

    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="El nombre del archivo no puede estar vacío.")

    try:
        # Sube el archivo directamente a S3
        S3_CLIENT.upload_fileobj(file.file, S3_BUCKET_NAME, filename)
        print(f"Archivo '{filename}' subido a S3.")

        # Dispara la actualización del vector store para incluir este nuevo documento
        update_vector_store_for_rag(file_paths_to_add=[filename])
        
        return {"message": f"Archivo '{filename}' subido a S3 y procesado para RAG.", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el documento para RAG: {e}")


@app.get("/documents", response_model=DocumentListResponse, summary="Listar todos los documentos en S3.")
async def list_documents():
    """
    Lista todos los archivos presentes en el bucket S3 configurado.
    """
    if not S3_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="Nombre del bucket S3 no configurado.")
    if not S3_CLIENT:
        raise HTTPException(status_code=500, detail="Cliente S3 no inicializado.")

    try:
        response = S3_CLIENT.list_objects_v2(Bucket=S3_BUCKET_NAME)
        # Extrae solo los nombres de los archivos. `Contents` podría no existir si el bucket está vacío.
        files = [obj['Key'] for obj in response.get('Contents', [])]
        return {"documents": files}
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="Credenciales de AWS no configuradas o inválidas.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar documentos de S3: {e}")

@app.get("/documents/{filename}", response_model=DocumentContentResponse, summary="Obtener el contenido de un documento desde S3.")
async def get_document_content(filename: str):
    """
    Obtiene el contenido de un archivo específico desde AWS S3.
    """
    if not S3_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="Nombre del bucket S3 no configurado.")
    if not S3_CLIENT:
        raise HTTPException(status_code=500, detail="Cliente S3 no inicializado.")

    content_bytes = load_document_content_from_s3(filename)
    if content_bytes is None:
        raise HTTPException(status_code=404, detail="Documento no encontrado.")
    
    try:
        # Intenta decodificar a UTF-8. Podrías necesitar manejo de errores más sofisticado
        # para diferentes codificaciones.
        content_text = content_bytes.decode('utf-8')
        return {"filename": filename, "content": content_text}
    except UnicodeDecodeError:
        raise HTTPException(status_code=500, detail="No se pudo decodificar el contenido del archivo a texto UTF-8.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener contenido del documento: {e}")


@app.delete("/documents/{filename}", summary="Eliminar un documento de S3 y del RAG.")
async def delete_document(filename: str):
    """
    Elimina un archivo de AWS S3 y actualiza el sistema RAG para remover su información.
    """
    if not S3_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="Nombre del bucket S3 no configurado.")
    if not S3_CLIENT:
        raise HTTPException(status_code=500, detail="Cliente S3 no inicializado.")

    try:
        # Elimina el archivo de S3
        S3_CLIENT.delete_object(Bucket=S3_BUCKET_NAME, Key=filename)
        print(f"Archivo '{filename}' eliminado de S3.")

        # Dispara la actualización del vector store.
        # Para simplificar, una eliminación dispara una reconstrucción completa en este ejemplo.
        update_vector_store_for_rag(file_names_to_delete=[filename])
        
        return {"message": f"Archivo '{filename}' eliminado de S3 y procesado para su remoción del RAG."}
    except S3_CLIENT.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Documento no encontrado en S3.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al eliminar el documento: {e}")

# --- Endpoint para el RAG ---

@app.post("/query", response_model=QueryResponse, summary="Consultar el sistema RAG.")
async def query(request: QueryRequest):
    """
    Envía una consulta al sistema RAG y recibe una respuesta basada en los documentos.
    """
    try:
        # chat_history aquí se pasa como una lista de strings.
        # Si quisieras pasar objetos AIMessage/HumanMessage, tendrías que adaptarlo.
        response = query_rag(request.query, request.chat_history)
        
        # Formatear source_documents para la respuesta
        source_docs_formatted = []
        if 'source_documents' in response:
            for doc in response['source_documents']:
                source_docs_formatted.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })

        return {"response": response['result'], "source_documents": source_docs_formatted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta RAG: {e}")

# --- Endpoint de salud ---
@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado de la API."""
    return {"status": "ok", "message": "API is running"}

# --- Endpoint de reinicio de RAG (útil para desarrollo/depuración) ---
@app.post("/recreate-rag-index", summary="Recrea completamente el índice RAG desde todos los documentos en S3.")
async def recreate_rag_index_endpoint():
    """
    Elimina y reconstruye completamente el índice RAG en Pinecone
    basándose en todos los documentos actuales en el bucket S3.
    """
    try:
        recreate_vector_store_from_all_documents()
        return {"message": "Índice RAG recreado exitosamente desde S3."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al recrear el índice RAG: {e}")