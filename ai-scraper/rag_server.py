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



def load_site_deep(url):
    raw_docs = crawl_site(url, max_pages=30)  # Increase if needed
    return [Document(page_content=doc["content"], metadata={"source": doc["url"]}) for doc in raw_docs]

# === Carga inicial de la base de conocimiento ===
@app.on_event("startup")
def startup():
    global vector_store
    url = "https://uchile.cl/convocatorias/226820/ingles-troncales-2025"
    print(f"\n🔍 Cargando: {url}")

    if os.path.exists(INDEX_PATH):
        print("📂 Cargando vector store existente desde disco...")
        vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("🛠️ Construyendo nuevo vector store desde la URL...")
        documents = load_page(url)
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

# === Ejecutar con: `uvicorn rag_server:app --reload` ===
if __name__ == "__main__":
    uvicorn.run("rag_server:app", host="127.0.0.1", port=8000, reload=True)
