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

import json

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
        chunk_overlap=300,
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
    url_list = [
        "https://uchile.cl/convocatorias/226820/ingles-troncales-2025",
        "https://uchile.cl/convocatorias/184155/concurso-fondecyt-de-iniciacion-en-investigacion-2023",
        "https://www.chileatiende.gob.cl/fichas/9715-afiliacion-a-fonasa",
        "https://www.superdesalud.gob.cl/orientacion-en-salud/como-funciona-el-sistema-de-salud-en-chile/#accordion_0",
        "https://redsalud.ssmso.cl/wp-content/uploads/2020/02/11.-Diptico-Atención-Migrantes.pdf",
        "https://www.superdesalud.gob.cl/tax-materias-prestadores/ley-de-derechos-y-deberes-4185/",
        "https://saludresponde.minsal.cl/buscador-de-establecimientos-de-salud/#:~:text=Atención%20Secundaria%3A%20Corresponde%20a%20los,requiere%20atención%20de%20mayor%20complejidad.",
        "https://chile.iom.int/sites/g/files/tmzbdl906/files/documents/2024-08/folletooim-derechosaludeducacion.pdf",
        "https://www.chileatiende.gob.cl/fichas/2464-plan-auge-ges",
        "https://www.superdesalud.gob.cl/tax-materias-prestadores/ley-de-derechos-y-deberes-4185/",
        "https://www.chileatiende.gob.cl/fichas/2467-salud-responde",
        "https://www.bcn.cl/portal/leyfacil/recurso/ley-de-migracion-y-extranjeria",
        "https://www.hospitalfricke.cl/wp-content/uploads/2017/12/Cartilla-Migrantes.pdf",
        "https://www.dt.gob.cl/portal/1626/w3-article-126452.html",

    ]

    # Load indexed URLs from file, if exists
    indexed_urls = load_indexed_urls()
    print(f"🔍 URLs ya indexadas: {indexed_urls}")

    index_file = os.path.join(INDEX_PATH, "index.faiss")
    
    # If no index exists, clear the indexed URLs
    if not os.path.exists(index_file):
        print("🆕 No existe índice. Se creará uno nuevo.")
        vector_store = None
        indexed_urls = set()  # Clear the indexed URLs when no index is found
    else:
        print("📂 Cargando vector store desde disco...")
        vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    # URLs to process (always process all URLs)
    urls_to_process = [url for url in url_list if url not in indexed_urls]
    print(f"🔍 URLs para procesar: {urls_to_process}")

    new_docs = []
    for url in urls_to_process:
        print(f"🌐 Procesando nueva URL: {url}")
        documents = load_page(url)
        chunks = split_text(documents)
        for chunk in chunks:
            chunk.metadata["source_url"] = url  # 👈 Optional but useful
        new_docs.extend(chunks)
        indexed_urls.add(url)

    if new_docs:
        print(f"📑 Se han encontrado {len(new_docs)} documentos para agregar.")
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
            print(f"  {i+1}. URL: {doc.metadata.get('source_url', 'No URL')} - Content: {doc.page_content[:1000]}...")  # Show first 200 chars for brevity
    else:
        print("🟡 No se encontraron documentos relevantes.")
    
    # Combine the content of the documents into a context for the question
    context = "\n\n".join([doc.page_content for doc in docs])
    
    print("🤖 Generando respuesta...")
    answer = answer_question(req.question, context)
    
    print("✅ Respuesta generada:", answer)
    
    return AnswerResponse(answer=answer)


# === Ejecutar con: `uvicorn rag_server:app --reload` ===
if __name__ == "__main__":
    uvicorn.run("rag_server:app", host="127.0.0.1", port=8000, reload=True)
