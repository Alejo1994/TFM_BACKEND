import os
import shutil
import tempfile # Para manejar archivos temporales al procesar PDFs
from io import BytesIO
import json # Nueva importación para serializar a JSON
import uuid # Para generar IDs únicos para los documentos padre

import boto3
from botocore.exceptions import NoCredentialsError

# LangChain specific imports
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document # ¡Importante!
from langchain_text_splitters import RecursiveCharacterTextSplitter # Asegurarse de la importación correcta
from langchain_pinecone import PineconeVectorStore # Nueva importación para Pinecone
from pinecone import Pinecone, ServerlessSpec # Para inicializar Pinecone
# para qwen/qwen3-235b-a22b:free
from langchain_openai import ChatOpenAI # Se usa la clase de OpenAI porque OpenRouter imita su API

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# Para ParentDocumentRetriever
from langchain.retrievers import ParentDocumentRetriever
#from langchain.storage import InMemoryStore # Solo para testing/desarrollo local sin Redis
from langchain_core.stores import BaseStore
from typing import Optional, Iterator

from dotenv import load_dotenv

# Cargar variables de entorno (asegúrate de que esto también se llame en main.py primero)
load_dotenv()

# --- Configuración de AWS S3 ---
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_PARENT_DOCS_PREFIX = "parent_docs/" # Prefijo para los documentos padre en S3

S3_CLIENT = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

# --- Configuración de Pinecone ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Inicializa el cliente Pinecone globalmente
try:
    PINECONE_CLIENT = Pinecone(api_key=PINECONE_API_KEY)
    print("Cliente Pinecone inicializado.")
except Exception as e:
    print(f"Error al inicializar el cliente Pinecone: {e}")
    # Considera lanzar un error o manejar esto apropiadamente en un entorno de producción.

# --- Inicialización de modelos (Google Gemini) ---
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # O el modelo de embedding de Gemini que prefieras
#llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7) # O el modelo LLM de Gemini que prefieras
# Configura tu API key de OpenRouter en .env como OPENROUTER_API_KEY
# OPENROUTER_API_KEY="sk_tu_clave_de_openrouter"

llm = ChatOpenAI(
    model="qwen/qwen3-235b-a22b-2507:free", # ¡Este sería el nombre del modelo en OpenRouter! (Verificar su documentación)
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7,
    timeout=120
)

# Instancias globales
vector_store = None # Para PineconeVectorStore (hijos)
doc_store = None    # Para RedisStore (padres)

# --- CLASE CUSTOM PARA ALMACENAR DOCUMENTOS PADRE EN S3 ---
# Implementa la interfaz que ParentDocumentRetriever espera
class CustomS3DocumentStore(BaseStore):
    def __init__(self, s3_client, bucket_name, prefix):
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.prefix = prefix
        print(f"DEBUG DOCSTORE: CustomS3DocumentStore inicializado con bucket={bucket_name}, prefix={prefix}")

    # Implementar los métodos abstractos de BaseStore
    def mget(self, keys: list[str]) -> list[Document | None]:
        """Recupera múltiples documentos padre de S3."""
        print(f"DEBUG DOCSTORE: Intentando mget para {len(keys)} claves.")
        retrieved_docs = []
        for key in keys:
            s3_key = f"{self.prefix}{key}"
            try:
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
                doc_json = response['Body'].read().decode('utf-8')
                doc_dict = json.loads(doc_json)
                retrieved_docs.append(Document(
                    page_content=doc_dict.get("page_content", ""),
                    metadata=doc_dict.get("metadata", {})
                ))
            except self.s3_client.exceptions.NoSuchKey:
                retrieved_docs.append(None)
            except Exception as e:
                print(f"Error al recuperar '{s3_key}' de S3: {e}")
                retrieved_docs.append(None)
        print(f"DEBUG DOCSTORE: mget completado, recuperados {len(retrieved_docs)} docs.")
        return retrieved_docs

    def mset(self, key_value_pairs: list[tuple[str, Document]]) -> None:
        """Guarda múltiples documentos padre en S3."""
        print(f"DEBUG DOCSTORE: Intentando mset para {len(key_value_pairs)} pares clave-valor.")
        try:
            for key, doc in key_value_pairs:
                s3_key = f"{self.prefix}{key}"
                doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
                doc_json = json.dumps(doc_dict, ensure_ascii=False)
                
                # --- ¡AÑADIR ESTAS LÍNEAS DE DEPURACIÓN CRÍTICAS! ---
                print(f"DEBUG DOCSTORE: Intentando put_object para S3 Key: {s3_key}")
                print(f"DEBUG DOCSTORE: Contenido a subir (primeros 100 chars): {doc_json[:100]}...")
                
                response = self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=doc_json.encode('utf-8')
                )
                # Verifica la respuesta de S3
                if response.get('ResponseMetadata', {}).get('HTTPStatusCode') == 200:
                    print(f"DEBUG DOCSTORE: ¡ÉXITO! Objeto {s3_key} guardado en S3. ETag: {response.get('ETag')}")
                else:
                    print(f"DEBUG DOCSTORE: ADVERTENCIA: put_object para {s3_key} no devolvió 200 OK. Respuesta: {response}")
                # --- FIN DEPURACIÓN ---
                
            print(f"DEBUG DOCSTORE: mset completado para {len(key_value_pairs)} docs.")
        except Exception as e:
            print(f"ERROR DOCSTORE: ¡FALLO CRÍTICO en mset en S3!: {e}")
            raise # ¡Re-lanza esta excepción!

    def mdelete(self, keys: list[str]) -> None:
        """Elimina múltiples documentos padre de S3."""
        print(f"DEBUG DOCSTORE: Intentando mdelete para {len(keys)} claves.")
        if not keys:
            return
        objects_to_delete = [{'Key': f"{self.prefix}{key}"} for key in keys]
        self.s3_client.delete_objects(Bucket=self.bucket_name, Delete={'Objects': objects_to_delete})
        print(f"Eliminados {len(keys)} documentos padre de S3.")
        print(f"DEBUG DOCSTORE: mdelete completado.")
    
    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """
        Generador que devuelve todas las claves de los documentos padre en S3.
        Utiliza paginación para manejar muchos objetos.
        """
        print(f"DEBUG DOCSTORE: Inciando yield_keys con prefix={prefix}")
        current_prefix = f"{self.prefix}{prefix}" if prefix else self.prefix
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        try:
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=current_prefix)
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        # Eliminar el prefijo para devolver solo el ID del documento
                        yield key[len(self.prefix):]
        except Exception as e:
            print(f"ERROR DOCSTORE: Falló yield_keys al listar objetos en S3: {e}")
            # Puedes decidir si relanzar o simplemente dejar de generar claves
            # Aquí, lo dejamos para que el error se propague si es crítico
            raise

# --- Funciones de procesamiento de documentos para el CRUD y RAG ---

def get_vector_store():
    """Retorna la instancia global del vector store (Pinecone)."""
    global vector_store
    if vector_store is None:
        try:
            #embedding_dimension = len(embeddings_model.embed_query("test_text"))
            embedding_dimension = 768 # Para models/embedding-001 de Google Gemini
            if PINECONE_INDEX_NAME not in [index.name for index in PINECONE_CLIENT.list_indexes()]:
                print(f"Creando nuevo índice Pinecone: {PINECONE_INDEX_NAME} con dimensión {embedding_dimension}...")
                PINECONE_CLIENT.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=embedding_dimension,
                    metric='cosine',
                    spec = ServerlessSpec(
                        cloud="aws", # O el cloud que estés usando
                        region="us-east-1" # O la región que estés usando
                    )
                    # No spec= para Serverless Index con SDK moderna
                )

                while not PINECONE_CLIENT.describe_index(PINECONE_INDEX_NAME).status['ready']:
                    import time
                    time.sleep(1)
            
            vector_store = PineconeVectorStore(
                index_name=PINECONE_INDEX_NAME,
                embedding=embeddings_model
            )
            print("PineconeVectorStore listo.")
        except Exception as e:
            print(f"Error al obtener/crear el vector store de Pinecone: {e}")
            raise
    return vector_store

def get_doc_store():
    """Retorna la instancia global del document store (CustomS3DocumentStore)."""
    global doc_store
    if doc_store is None:
        try:
            doc_store = CustomS3DocumentStore(S3_CLIENT, S3_BUCKET_NAME, S3_PARENT_DOCS_PREFIX)
            print("DEBUG: CustomS3DocumentStore (DocumentStore) inicializado.")
        except Exception as e:
            print(f"DEBUG ERROR: Falló inicialización de CustomS3DocumentStore: {e}")
            print("Advertencia: Fallback a InMemoryStore para el DocumentStore (no persistente).")
             # Este fallback es solo para depuración si S3 falla
            from langchain.storage import InMemoryStore
            doc_store = InMemoryStore()
    return doc_store


def load_document_content_from_s3(filename: str) -> bytes | None:
    """Lee el contenido binario de un archivo desde S3."""
    try:
        response = S3_CLIENT.get_object(Bucket=S3_BUCKET_NAME, Key=filename)
        return response['Body'].read()
    except S3_CLIENT.exceptions.NoSuchKey:
        print(f"Archivo '{filename}' no encontrado en S3.")
        return None
    except NoCredentialsError:
        print("Credenciales de AWS no configuradas o inválidas para S3.")
        raise
    except Exception as e:
        print(f"Error al leer '{filename}' desde S3: {e}")
        raise

def load_documents_from_s3_bucket():
     # ... (Esta función permanece igual, pero los documentos finales serán procesados por el retriever)
    """Carga todos los documentos desde el bucket S3, procesándolos con los loaders de LangChain."""
    documents = []
    print(f"Cargando documentos desde el bucket S3: {S3_BUCKET_NAME}...")
    try:
        # Listar todos los objetos en el bucket
        response = S3_CLIENT.list_objects_v2(Bucket=S3_BUCKET_NAME)
        
        # Verificar si hay contenido en la respuesta
        if 'Contents' not in response:
            print("No se encontraron objetos en el bucket S3.")
            return documents

        for obj in response['Contents']:
            file_key = obj['Key']
            print(f"Procesando {file_key} desde S3...")

            file_content_bytes = load_document_content_from_s3(file_key)
            if not file_content_bytes:
                continue

            # Usar loaders de LangChain según el tipo de archivo
            loader = None
            if file_key.endswith(".txt"):
                # TextLoader puede tomar una cadena, así que decodificamos primero
                try:
                    text_content = file_content_bytes.decode('utf-8')
                    # Creamos un documento de LangChain directamente
                    documents.append(Document(page_content=text_content, metadata={"source": file_key, "file_type": "txt"}))
                except UnicodeDecodeError:
                    print(f"Advertencia: No se pudo decodificar el archivo TXT '{file_key}' (UTF-8). Ignorando.")
                    continue
            elif file_key.endswith(".md"):
                # UnstructuredMarkdownLoader requiere una ruta de archivo.
                # Tendremos que guardar temporalmente.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as temp_file:
                    temp_file.name # full path to temp file
                    temp_file.write(file_content_bytes)
                    temp_file_path = temp_file.name
                
                try:
                    loader = UnstructuredMarkdownLoader(temp_file_path)
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        doc.metadata["source"] = file_key # Mantener el source original de S3
                        doc.metadata["file_type"] = "md"
                    documents.extend(loaded_docs)
                except Exception as e:
                    print(f"Error al cargar el documento MD '{file_key}': {e}")
                finally:
                    os.remove(temp_file_path) # Limpiar el archivo temporal

            elif file_key.endswith(".pdf"):
                # PyPDFLoader requiere una ruta de archivo.
                # Tendremos que guardar temporalmente.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(file_content_bytes)
                    temp_file_path = temp_file.name
                
                try:
                    loader = PyPDFLoader(temp_file_path)
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        doc.metadata["source"] = file_key # Mantener el source original de S3
                        doc.metadata["file_type"] = "pdf"
                    documents.extend(loaded_docs)
                except Exception as e:
                    print(f"Error al cargar el documento PDF '{file_key}': {e}")
                finally:
                    os.remove(temp_file_path) # Limpiar el archivo temporal
            else:
                print(f"Advertencia: Tipo de archivo no soportado explícitamente: {file_key}. Ignorando.")
                continue

    except Exception as e:
        print(f"Error general al listar o cargar desde S3: {e}")
        raise # Re-lanzar para que FastAPI lo capture

    print(f"Se cargaron {len(documents)} documentos desde S3.")
    print(f"DEBUG: load_documents_from_s3_bucket cargó {len(documents)} documentos.")
    return documents

# --- Configuración de ParentDocumentRetriever y Funciones de Indexación ---

# Splitter para los "documentos padre" (chunks grandes)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

# Splitter para los "documentos hijo" (chunks pequeños para embeddings)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100) # O SentenceSplitter()

def get_parent_document_retriever():
    """Retorna una instancia del ParentDocumentRetriever."""
    vector_store_instance = get_vector_store() # Pinecone
    doc_store_instance = get_doc_store()       # CustomS3DocumentStore

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store_instance,
        docstore=doc_store_instance,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
        search_kwargs={"k": 5}, # Asegúrate de que k sea apropiado
        id_key="doc_id" # Opcional: Nombre de la clave en metadatos para el ID del documento padre
    )
    return retriever

def recreate_vector_store_from_all_documents():
    """
    Recrea el índice de Pinecone (hijos) y vacía el CustomS3DocumentStore (padres) y lo repuebla,
    cargando todos los documentos de S3.
    """
    global vector_store
    global doc_store # Actualizar la global doc_store para el nuevo CustomS3DocumentStore

    index_name = PINECONE_INDEX_NAME

    # 1. Borrar y recrear el índice de Pinecone
    print(f"Borrando y recreando el índice Pinecone: {index_name}...")
    if index_name in [idx.name for idx in PINECONE_CLIENT.list_indexes()]:
        PINECONE_CLIENT.delete_index(index_name)
    
    embedding_dimension = 768 # Hardcodeado
    PINECONE_CLIENT.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric='cosine',
        spec = ServerlessSpec(
                cloud="aws", # O el cloud que estés usando
                region="us-east-1" # O la región que estés usando
            )
    )
    while not PINECONE_CLIENT.describe_index(index_name).status['ready']:
        import time
        time.sleep(1)
    
    # 2. Re-inicializar el PineconeVectorStore para el nuevo índice
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings_model)

     # 3. Limpiar el CustomS3DocumentStore
    doc_store_instance = get_doc_store()
    if isinstance(doc_store_instance, CustomS3DocumentStore):
        print(f"DEBUG RECREATE: Intentando limpiar CustomS3DocumentStore en S3 (prefijo: {S3_PARENT_DOCS_PREFIX})...")
        try:
            paginator = S3_CLIENT.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=S3_PARENT_DOCS_PREFIX)
            objects_to_delete = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects_to_delete.append({'Key': obj['Key']})
            
            if objects_to_delete:
                for i in range(0, len(objects_to_delete), 1000):
                    batch = objects_to_delete[i:i+1000]
                    S3_CLIENT.delete_objects(Bucket=S3_BUCKET_NAME, Delete={'Objects': batch})
                print(f"DEBUG RECREATE: Eliminadas {len(objects_to_delete)} entradas de S3ByteStore.")
            else:
                print("DEBUG RECREATE: No se encontraron objetos para limpiar en S3ByteStore.")
        except Exception as e:
            print(f"ERROR RECREATE: Falló la limpieza de CustomS3DocumentStore en S3: {e}")
            # No relanzamos aquí porque queremos que la recreación continúe
    else:
        print("DEBUG RECREATE: DocStore no es CustomS3DocumentStore, no se limpiará.")


    # 4. Cargar documentos de S3 y añadirlos al retriever
    print("DEBUG RECREATE: Llamando load_documents_from_s3_bucket para obtener documentos originales.")
    documents = load_documents_from_s3_bucket()
    if documents:
        print(f"DEBUG RECREATE: Documentos cargados de S3: {len(documents)}. Asignando IDs.")
        for doc in documents:
            if 'doc_id' not in doc.metadata:
                doc.metadata['doc_id'] = str(uuid.uuid4())
        
        retriever_indexer = get_parent_document_retriever()
        print(f"DEBUG RECREATE: Llamando retriever_indexer.add_documents para indexar {len(documents)} documentos.")
        try:
            retriever_indexer.add_documents(documents) # Esto llamará a CustomS3DocumentStore.mset
            print("DEBUG RECREATE: Documentos indexados exitosamente en Pinecone y S3.")
        except Exception as e:
            print(f"ERROR RECREATE: Falló retriever_indexer.add_documents: {e}")
            raise # ¡Es crucial relanzar aquí!
    else:
        print("DEBUG RECREATE: No hay documentos en S3 para añadir al índice.")

def update_vector_store_for_rag(file_paths_to_add: list = None, file_names_to_delete: list = None):
    # ... (Esta función no cambia mucho, solo usará CustomS3DocumentStore internamente)
    """
    Actualiza el índice de Pinecone (hijos) y CustomS3DocumentStore (padres).
    Para una gestión más sencilla, las eliminaciones disparan una reconstrucción completa.
    Las adiciones se realizan directamente.
    """
    if file_names_to_delete:
        print(f"Se solicitaron eliminaciones ({file_names_to_delete}). Recreando completamente el índice.")
        recreate_vector_store_from_all_documents()
        return
    
    if file_paths_to_add:
        retriever_indexer = get_parent_document_retriever()
        new_documents = []
        for file_name_in_s3 in file_paths_to_add:
            print(f"Cargando '{file_name_in_s3}' para añadir...")
            file_content_bytes = load_document_content_from_s3(file_name_in_s3)
            if not file_content_bytes:
                print(f"Advertencia: No se pudo obtener contenido para '{file_name_in_s3}'. Saltando.")
                continue

            # Usar loaders de LangChain según el tipo de archivo
            if file_name_in_s3.endswith(".txt"):
                try:
                    text_content = file_content_bytes.decode('utf-8')
                    doc = Document(page_content=text_content, metadata={"source": file_name_in_s3, "file_type": "txt"})
                    doc.metadata['doc_id'] = str(uuid.uuid4()) # Añadir ID único
                    new_documents.append(doc)
                except UnicodeDecodeError:
                    print(f"Advertencia: No se pudo decodificar el archivo TXT '{file_name_in_s3}' (UTF-8). Saltando.")
            elif file_name_in_s3.endswith(".md"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as temp_file:
                    temp_file.write(file_content_bytes)
                    temp_file_path = temp_file.name
                try:
                    loader = UnstructuredMarkdownLoader(temp_file_path)
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        doc.metadata["source"] = file_name_in_s3
                        doc.metadata["file_type"] = "md"
                        doc.metadata['doc_id'] = str(uuid.uuid4()) # Añadir ID único
                    new_documents.extend(loaded_docs)
                except Exception as e:
                    print(f"Error al cargar el documento MD '{file_name_in_s3}': {e}")
                finally:
                    os.remove(temp_file_path)
            elif file_name_in_s3.endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(file_content_bytes)
                    temp_file_path = temp_file.name
                try:
                    loader = PyPDFLoader(temp_file_path)
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        doc.metadata["source"] = file_name_in_s3
                        doc.metadata["file_type"] = "pdf"
                        doc.metadata['doc_id'] = str(uuid.uuid4()) # Añadir ID único
                    new_documents.extend(loaded_docs)
                except Exception as e:
                    print(f"Error al cargar el documento PDF '{file_name_in_s3}': {e}")
                finally:
                    os.remove(temp_file_path)
            else:
                print(f"Advertencia: Tipo de archivo no soportado para añadir: {file_name_in_s3}. Saltando.")
                continue

        if new_documents:
            print(f"DEBUG UPDATE: Añadiendo {len(new_documents)} nuevos documentos a ParentDocumentRetriever.")
            try:
                retriever_indexer.add_documents(new_documents) # Esto llamará a CustomS3DocumentStore.mset
                print("DEBUG UPDATE: Nuevos documentos añadidos a Pinecone y CustomS3DocumentStore.")
            except Exception as e:
                print(f"ERROR UPDATE: Falló retriever_indexer.add_documents al añadir: {e}")
                raise # ¡Relanzar!
        else:
            print("DEBUG UPDATE: No se encontraron nuevos documentos válidos en S3 para añadir.")

# --- Función RAG ---
# --- Configuración de la cadena RAG ---
# Plantilla de prompt para guiar al LLM
RAG_PROMPT_TEMPLATE = """
Eres un asistente útil y preciso que responde preguntas utilizando únicamente el contexto proporcionado.
**Responde siempre en español.**
Si la respuesta no se encuentra en el contexto, simplemente di que no tienes suficiente información para responder a esta pregunta. No intentes inventar una respuesta.

Contexto:
{context}

Pregunta: {question}

Respuesta:
"""
RAG_PROMPT = PromptTemplate(template=RAG_PROMPT_TEMPLATE, input_variables=["context", "question"])


def query_rag(query_text: str, chat_history: list[AIMessage | HumanMessage]):
    """Ejecuta una consulta RAG utilizando ParentDocumentRetriever."""
    retriever = get_parent_document_retriever() # Obtener el retriever configurado
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever, # Usamos el ParentDocumentRetriever
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT}
    )

    response = qa_chain.invoke({"query": query_text})
    return response

# --- Funciones de inicialización (llamadas al inicio de la aplicación) ---
def initialize_system():
    """Función para inicializar el sistema (conectar a Pinecone, Redis, y S3)."""
    print("Inicializando sistema RAG con ParentDocumentRetriever...")
    try:
        # Asegúrate de que todos los clientes estén inicializados y conectados
        _ = get_vector_store() # Conecta a Pinecone
        _ = get_doc_store()    # Conecta a Redis
        
        # Opcional: Recrear los índices en startup si es un entorno de desarrollo
        # o si quieres asegurar que siempre estén sincronizados con S3
        # recreate_vector_store_from_all_documents()
        
        print("Sistema RAG inicializado exitosamente.")
    except Exception as e:
        print(f"Fallo al inicializar el sistema: {e}")
        raise # Lanzar la excepción para que FastAPI la capture
