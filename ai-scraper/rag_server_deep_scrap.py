import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
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
import xml.etree.ElementTree as ET

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
    loader = SeleniumURLLoader(urls=[url])
    return loader.load()

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


def load_site_deep(urls: List[str]):
    documents = []
    for url in urls:
        try:
            print(f"🔍 Visiting: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            documents.append(Document(page_content=text, metadata={"source": url}))
        except Exception as e:
            print(f"❌ Error accessing {url}: {e}")
            continue
    return documents


# === Carga inicial de la base de conocimiento ===
@app.on_event("startup")
def startup():
    global vector_store
    sitemap_url = "https://saludresponde.minsal.cl/wp-sitemap.xml"
    print(f"\n🧭 Cargando URLs desde el sitemap: {sitemap_url}")
    all_urls = get_all_urls_recursive(sitemap_url)

    if os.path.exists(INDEX_PATH):
        print("📂 Cargando vector store existente desde disco...")
        vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"🛠️ Construyendo nuevo vector store desde {len(all_urls)} URLs...")
        documents = load_site_deep(all_urls)
        chunks = split_text(documents)
        vector_store = index_docs(chunks)
        vector_store.save_local(INDEX_PATH)
        print("💾 Vector store guardado en disco.")

    print("✅ RAG inicializado y listo.")


# === Endpoint de salud para comprobar que el servidor funciona ===
@app.get("/ping")
def ping():
    return {"status": "ok", "message": "RAG server is alive!"}

# === Endpoint para responder preguntas ===
@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Base de conocimiento no cargada.")
    
    print("📥 Pregunta recibida")
    docs = retrieve_docs(req.question)
    print("🔍 Documentos relevantes recuperados")
    context = "\n\n".join([doc.page_content for doc in docs])
    print("🤖 Generando respuesta...")
    answer = answer_question(req.question, context)
    print("✅ Respuesta generada")
    
    return AnswerResponse(answer=answer)

def get_all_urls_recursive(sitemap_url):
    urls = []
    response = requests.get(sitemap_url)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    if root.tag.endswith('sitemapindex'):
        # It's an index of sitemaps
        sitemap_urls = [s.find('ns:loc', namespace).text for s in root.findall('ns:sitemap', namespace)]
        for sm_url in sitemap_urls:
            urls.extend(get_all_urls_recursive(sm_url))
    elif root.tag.endswith('urlset'):
        # It's a sitemap of URLs
        urls.extend([u.find('ns:loc', namespace).text for u in root.findall('ns:url', namespace)])

    return urls

all_urls = get_all_urls_recursive("https://saludresponde.minsal.cl/wp-sitemap.xml")

print(f"\n🧭 Total URLs extracted: {len(all_urls)}")


# === Ejecutar con: `uvicorn rag_server:app --reload` ===
if __name__ == "__main__":
    uvicorn.run("rag_server:app", host="127.0.0.1", port=8000, reload=True)

def crawl_site(start_url, max_pages=20):
    visited = set()
    to_visit = [start_url]
    documents = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            print(f"🔍 Visiting: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"❌ Error accessing {url}: {e}")
            continue

        visited.add(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        documents.append({"url": url, "content": text})

        base = "{0.scheme}://{0.netloc}".format(urlparse(url))
        for link_tag in soup.find_all("a", href=True):
            link = urljoin(url, link_tag['href'])
            if base in link and link not in visited:
                to_visit.append(link)

    return documents
