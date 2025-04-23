from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import uvicorn
import os
from langchain_core.documents import Document
import requests
from bs4 import BeautifulSoup
import json

def get_urls_from_sitemap(sitemap_url: str, limit: int = None) -> list[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(sitemap_url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"⚠️ Error al obtener {sitemap_url}: {e}")
        return []

    soup = BeautifulSoup(response.content, "xml")

    urls = []
    if soup.find_all("sitemap"):  # es un sitemap índice
        sitemap_tags = soup.find_all("loc")
        for sitemap in sitemap_tags:
            urls += get_urls_from_sitemap(sitemap.text, limit)
            if limit and len(urls) >= limit:
                return urls[:limit]
    elif soup.find_all("url"):  # es un sitemap de URLs
        url_tags = soup.find_all("loc")
        urls = [url.text for url in url_tags]
        if limit:
            urls = urls[:limit]
    return urls


URL_TRACK_FILE = "indexed_urls.json"

def load_indexed_urls():
    if os.path.exists(URL_TRACK_FILE):
        with open(URL_TRACK_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_indexed_urls(urls):
    with open(URL_TRACK_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(list(urls)), f, ensure_ascii=False, indent=2)


# === Configuración del modelo y embeddings ===
embeddings = OllamaEmbeddings(model="nomic-embed-text")  # o "llama3-es-embed"
model = OllamaLLM(model="llama3.2")

template = """
Eres un asistente para tareas de preguntas y respuestas. Usa los siguientes fragmentos de contexto para responder la pregunta. Si no sabes la respuesta, simplemente di que no lo sabes. Máximo tres oraciones y sé conciso.
Pregunta: {question} 
Contexto: {context} 
Respuesta:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# === Inicialización de FastAPI ===
app = FastAPI(title="RAG Server")

# === Variables globales para almacenar el vector store ===
vector_store = None
INDEX_PATH = "vectorstore_index"

# === Modelos de entrada/salida ===
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# === Funciones de procesamiento ===
def load_page(url: str):
    print(f"🌐 Cargando la página: {url}")
    loader = SeleniumURLLoader(urls=[url])
    documents = loader.load()

    # Check if documents are loaded
    if not documents:
        print(f"⚠️ No se cargaron documentos para la URL: {url}")
    else:
        print(f"🚨 Se cargaron {len(documents)} documentos para la URL: {url}")

    return documents


def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return splitter.split_documents(documents)

def index_docs(docs):
    if not docs:
        raise ValueError("❌ No hay documentos para indexar.")
    return FAISS.from_documents(docs, embeddings)

def retrieve_docs(query, k=4):
    docs_and_scores = vector_store.similarity_search_with_score(query, k=k)
    return [doc for doc, _ in docs_and_scores]

def answer_question(question: str, context: str) -> str:
    return chain.invoke({"question": question, "context": context})


# === Carga inicial de la base de conocimiento ===
@app.on_event("startup")
def startup():
    global vector_store
    #sitemap_url = "https://datos.uchile.cl/sitemap.xml"
    sitemap_url = "https://saludresponde.minsal.cl/wp-sitemap.xml"
    all_urls = get_urls_from_sitemap(sitemap_url)

    # 🔧 Limitar a solo 4 URLs para pruebas
    MAX_TEST_URLS = 4
    all_urls = all_urls[:MAX_TEST_URLS]
    print(f"🔧 Modo prueba: usando solo {len(all_urls)} URLs")

    indexed_urls = load_indexed_urls()

    if not os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        print("🆕 No existe índice. Se creará uno nuevo.")
        vector_store = None
        indexed_urls.clear()  # ⬅️ Limpiamos el historial si no hay índice
    else:
        print("📂 Cargando vector store desde disco...")
        vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    print("🔍 URLs ya indexadas:", indexed_urls)

    urls_to_process = [url for url in all_urls if url not in indexed_urls]
    print("🔍 URLs para procesar:", urls_to_process)

    new_docs = []
    for url in urls_to_process:
        print(f"🌐 Procesando URL: {url}")
        try:
            documents = load_page(url)
            for i, doc in enumerate(documents):
                print(f"🧾 Documento {i+1} - ({len(doc.page_content)} caracteres):")
                print(doc.page_content[:500])


            chunks = split_text(documents)
            if not chunks:
                print(f"⚠️ No se generaron chunks para la URL: {url}")
            for chunk in chunks:
                chunk.metadata["source_url"] = url

                # ✅ Mostrar info del documento antes de guardarlo
                print("📝 Documento cargado:")
                print(f"📄 Contenido (primeros 300 caracteres): {chunk.page_content[:300]}...")
                print(f"🧾 Metadata: {chunk.metadata}")
                print("-" * 80)

            new_docs.extend(chunks)
            indexed_urls.add(url)
        except Exception as e:
            print(f"❌ Error cargando {url}: {e}")

    if new_docs:
        if vector_store:
            print("📌 Agregando nuevos documentos...")
            vector_store.add_documents(new_docs)
        else:
            print("📌 Creando nuevo índice con documentos...")
            vector_store = index_docs(new_docs)

        os.makedirs(INDEX_PATH, exist_ok=True)
        vector_store.save_local(INDEX_PATH)
        save_indexed_urls(indexed_urls)
        print("💾 Índice actualizado y guardado.")
    else:
        print("🟡 No hay nuevos documentos que agregar.")

    print("✅ RAG listo.")





# === Endpoint de salud para comprobar que el servidor funciona ===
@app.get("/ping")
def ping():
    return {"status": "ok", "message": "RAG server is alive!"}

# === Endpoint para responder preguntas ===
@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Base de conocimiento no cargada.")
    
    print("📥 Pregunta recibida:", req.question)
    
    # Retrieve documents related to the query
    docs = retrieve_docs(req.question)
    
    if docs:
        print("🔍 Documentos relevantes recuperados:")
        for i, doc in enumerate(docs):
            print(f"  {i+1}. URL: {doc.metadata.get('source_url', 'No URL')} - Content: {doc.page_content[:2000]}...")  # Show first 200 chars for brevity
    else:
        print("🟡 No se encontraron documentos relevantes.")
    
    # Combine the content of the documents into a context for the question
    context = "\n\n".join([doc.page_content for doc in docs])
    
    print("🤖 Generando respuesta...")
    answer = answer_question(req.question, context)
    
    print("✅ Respuesta generada:", answer)
    
    return AnswerResponse(answer=answer)

@app.get("/docs-indexados")
def ver_documentos_indexados(limit: int = 10):
    if vector_store is None:
        raise HTTPException(status_code=500, detail="No hay índice cargado.")

    # Accede directamente a los documentos
    try:
        # Usamos search con un término genérico que encuentre "algo"
        docs = vector_store.similarity_search("información", k=limit)
        resultado = []
        for i, doc in enumerate(docs):
            resultado.append({
                "id": i + 1,
                "source_url": doc.metadata.get("source_url", "Desconocido"),
                "contenido": doc.page_content[:300] + "..."  # muestra solo los primeros caracteres
            })
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener documentos: {e}")


# === Ejecutar con: `uvicorn rag_server:app --reload` ===
if __name__ == "__main__":
    uvicorn.run("rag_server:app", host="127.0.0.1", port=8000, reload=True)