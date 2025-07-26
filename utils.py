import os
import shutil
import tempfile # Para manejar archivos temporales al procesar PDFs
from io import BytesIO

import boto3
from botocore.exceptions import NoCredentialsError

# LangChain specific imports
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document # ¡Importante!
from langchain_text_splitters import RecursiveCharacterTextSplitter # Asegurarse de la importación correcta
from langchain_pinecone import PineconeVectorStore # Nueva importación para Pinecone
from pinecone import Pinecone, ServerlessSpec # Para inicializar Pinecone

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

from dotenv import load_dotenv

# Cargar variables de entorno (asegúrate de que esto también se llame en main.py primero)
load_dotenv()

# --- Configuración de AWS S3 ---
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
# Inicializa el cliente S3 una vez para reutilizarlo
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
#llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.7) # O el modelo LLM de Gemini que prefieras
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7) # O el modelo LLM de Gemini que prefieras

# Instancia global del vector store (Pinecone)
vector_store = None

# --- Funciones de procesamiento de documentos para el CRUD y RAG ---

def get_vector_store():
    """Retorna la instancia global del vector store (Pinecone), creándola si no existe."""
    global vector_store
    if vector_store is None:
        try:
            # Verificar si el índice existe en Pinecone, si no, crearlo.
            # Intenta obtener la dimensión del embedding del modelo
            try:
                # Una forma más robusta de obtener la dimensión, si el modelo lo permite
                # Asegúrate de que esta llamada a embed_query no falle si el modelo aún no está listo
                example_embedding = embeddings_model.embed_query("test_text")
                embedding_dimension = len(example_embedding)
            except Exception as e:
                print(f"Advertencia: No se pudo obtener la dimensión del embedding dinámicamente ({e}). Usando valor predeterminado 768.")
                embedding_dimension = 768 # Valor predeterminado para text-embedding-ada-002 y embedding-001 de Gemini

            if PINECONE_INDEX_NAME not in [index.name for index in PINECONE_CLIENT.list_indexes()]:
                print(f"Creando nuevo índice Pinecone: {PINECONE_INDEX_NAME} con dimensión {embedding_dimension}...")
                
                
                PINECONE_CLIENT.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=embedding_dimension,
                    metric='cosine', # O 'dotproduct' o 'euclidean'
                    spec = ServerlessSpec(
                        cloud="aws", # O el cloud que estés usando
                        region="us-east-1" # O la región que estés usando
                    )
                )
                # Esperar a que el índice esté listo
                while not PINECONE_CLIENT.describe_index(PINECONE_INDEX_NAME).status['ready']:
                    import time
                    time.sleep(1)
            
            print(f"Conectando a PineconeVectorStore con índice: {PINECONE_INDEX_NAME}...")
            vector_store = PineconeVectorStore(
                index_name=PINECONE_INDEX_NAME,
                embedding=embeddings_model
            )
            print("PineconeVectorStore listo.")
        except Exception as e:
            print(f"Error al obtener/crear el vector store de Pinecone: {e}")
            raise # Re-lanzar el error para que la aplicación lo maneje

    return vector_store

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
    return documents

def split_documents(documents: list[Document], chunk_size=1000, chunk_overlap=200):
    """Divide los documentos en chunks más pequeños."""
    print(f"Dividiendo documentos en chunks (tamaño={chunk_size}, solapamiento={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Se crearon {len(chunks)} chunks.")
    return chunks

def recreate_vector_store_from_all_documents():
    """
    Recrea la base de datos vectorial en Pinecone desde cero, cargando todos los documentos de S3.
    Esto BORRARÁ los datos existentes en el índice de Pinecone.
    """
    global vector_store
    
    # 1. Asegúrate de que el cliente Pinecone y la instancia de vector_store están listos
    vector_store_instance = get_vector_store()
    index_name = PINECONE_INDEX_NAME

    # 2. Borrar el índice si existe y volver a crearlo
    print(f"Borrando y recreando el índice Pinecone: {index_name}...")
    if index_name in [idx.name for idx in PINECONE_CLIENT.list_indexes()]:
        PINECONE_CLIENT.delete_index(index_name)
    
    # Intenta obtener la dimensión del embedding del modelo de forma dinámica
    try:
        embedding_dimension = len(embeddings_model.embed_query("test_text"))
    except Exception as e:
        print(f"Advertencia: No se pudo obtener la dimensión del embedding dinámicamente ({e}). Usando valor predeterminado 768.")
        embedding_dimension = 768

    # Define el spec basándote en tu región y cloud
    # Basado en tu captura de pantalla: Cloud: AWS, Region: us-east-1
  

    PINECONE_CLIENT.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric='cosine',
        spec = ServerlessSpec(
            cloud="aws", # O el cloud que estés usando
            region="us-east-1" # O la región que estés usando
        )
    )
    # Esperar a que el índice esté listo
    while not PINECONE_CLIENT.describe_index(index_name).status['ready']:
        import time
        time.sleep(1)

    # Re-inicializar el PineconeVectorStore para el nuevo índice
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings_model)

    # 3. Cargar documentos de S3 y añadirlos al nuevo índice
    documents = load_documents_from_s3_bucket()
    if documents:
        chunks = split_documents(documents)
        print(f"Añadiendo {len(chunks)} chunks al índice de Pinecone.")
        vector_store.add_documents(chunks)
        print("Documentos añadidos a Pinecone exitosamente.")
    else:
        print("No hay documentos en S3 para añadir al índice de Pinecone.")

def update_vector_store_for_rag(file_paths_to_add: list = None, file_names_to_delete: list = None):
    """
    Actualiza la base de datos vectorial de Pinecone.
    Para una gestión más sencilla, las eliminaciones disparan una reconstrucción completa.
    Las adiciones se realizan directamente.
    """
    vector_store_instance = get_vector_store()

    if file_names_to_delete:
        print(f"Se solicitaron eliminaciones ({file_names_to_delete}). Recreando completamente el índice de Pinecone.")
        recreate_vector_store_from_all_documents()
        return

    if file_paths_to_add:
        print(f"Procesando y añadiendo nuevos documentos a Pinecone: {file_paths_to_add}")
        new_documents = []
        for file_name_in_s3 in file_paths_to_add:
            print(f"Cargando '{file_name_in_s3}' para añadir...")
            file_content_bytes = load_document_content_from_s3(file_name_in_s3)
            if not file_content_bytes:
                print(f"Advertencia: No se pudo obtener contenido para '{file_name_in_s3}'. Saltando.")
                continue

            # Usar loaders de LangChain según el tipo de archivo (similar a load_documents_from_s3_bucket)
            if file_name_in_s3.endswith(".txt"):
                try:
                    text_content = file_content_bytes.decode('utf-8')
                    new_documents.append(Document(page_content=text_content, metadata={"source": file_name_in_s3, "file_type": "txt"}))
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
                    new_documents.extend(loaded_docs)
                except Exception as e:
                    print(f"Error al cargar el documento PDF '{file_name_in_s3}': {e}")
                finally:
                    os.remove(temp_file_path)
            else:
                print(f"Advertencia: Tipo de archivo no soportado para añadir: {file_name_in_s3}. Saltando.")
                continue

        if new_documents:
            new_chunks = split_documents(new_documents)
            print(f"Añadiendo {len(new_chunks)} nuevos chunks a Pinecone.")
            vector_store_instance.add_documents(new_chunks)
            print("Nuevos documentos añadidos a Pinecone.")
        else:
            print("No se encontraron nuevos documentos válidos para añadir a Pinecone.")

# --- Función RAG ---

def query_rag(query_text: str, chat_history: list[AIMessage | HumanMessage]):
    """Realiza una consulta RAG utilizando el vector store."""
    vector_store_instance = get_vector_store()
    
    # Configurar el retriever
    retriever = vector_store_instance.as_retriever(search_kwargs={"k": 5}) # Busca los 5 chunks más relevantes

    # Prompt Template for RAG
    # Adaptar el prompt para Gemini si es necesario, o mantenerlo general
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Eres un asistente útil que responde preguntas basadas UNICAMENTE en el contexto proporcionado. "
             "**Responde siempre en español.** " # <--- ¡AÑADE ESTA INSTRUCCIÓN CLARA!
             "Si la pregunta no puede ser respondida con la información proporcionada, "
             "responde 'No tengo suficiente información para responder a esta pregunta.'\n\n"
             "Contexto: {context}\n\n"
             "Pregunta: {question}\n"
             "Respuesta:",
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # Stuff es simple, concatena todos los chunks
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    response = qa_chain.invoke({"query": query_text})
    return response

# --- Funciones de inicialización (llamadas al inicio de la aplicación) ---
def initialize_system():
    """Función para inicializar el sistema (conectar a Pinecone y S3)."""
    print("Inicializando sistema...")
    try:
        # Asegúrate de que S3_CLIENT y PINECONE_CLIENT estén inicializados
        # y que get_vector_store() se llame para asegurar la conexión al índice.
        _ = get_vector_store()
        print("Sistema inicializado exitosamente.")
    except Exception as e:
        print(f"Fallo al inicializar el sistema: {e}")
        # Dependiendo del contexto, podrías querer que la aplicación no se inicie
        # si la conexión a la base de datos vectorial o S3 falla.
        raise # Lanzar la excepción para que FastAPI la capture
