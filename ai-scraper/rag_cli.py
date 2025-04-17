from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# === PROMPT ===
template = """
Eres un asistente de respuestas concisas. Usa el siguiente contexto para responder la pregunta. 
Si no sabes la respuesta, solo di que no lo sabes. Responde en tres oraciones o menos.

Pregunta: {question}
Contexto: {context}
Respuesta:
"""

# === Embeddings y LLM en español ===
embeddings = OllamaEmbeddings(model="nomic-embed-text")
model = OllamaLLM(model="llama3.2")

# === FUNCIONES ===

def load_page(url):
    loader = SeleniumURLLoader(urls=[url])
    documents = loader.load()
    print(f"\n📃 Contenido completo del documento:\n{documents[0].page_content[:1000]}")
    print(f"\n📄 Documentos cargados: {len(documents)}")
    for i, doc in enumerate(documents):
        print(f"\n--- Documento {i+1} ---\n{doc.page_content[:300]}...\n")
    return documents

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    print(f"\n📄 Documento original 1 (preview):\n{documents[0].page_content[:300]}...\n")
    return chunks

def index_docs(docs):
    print("\n📄 Indexing the following chunks:")
    for i, doc in enumerate(docs):
        print(f"\n--- Chunk {i+1} ---\n{doc.page_content[:300]}...\n")
    print("Número de documentos a indexar:", len(docs))

    if not docs:
        raise ValueError("❌ No hay documentos para indexar.")

    return FAISS.from_documents(docs, embeddings)

def retrieve_docs(query, vector_store, k=4):
    docs_and_scores = vector_store.similarity_search_with_score(query, k=k)
    print("\n🔍 Retrieved similar chunks:")
    for i, (doc, score) in enumerate(docs_and_scores):
        print(f"\n--- Match {i+1} (Score: {score:.4f}) ---\n{doc.page_content[:300]}...\n")
    return [doc for doc, _ in docs_and_scores]

def answer_question(question, context):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# === MAIN ===

if __name__ == "__main__":
    try:
        url = input("Enter a URL to analyze: ")
        print(f"\n🔍 Loading: {url}")
        documents = load_page(url)
        chunked = split_text(documents)
        vector_store = index_docs(chunked)

        print("\n✅ Contenido cargado e indexado correctamente.\n")

        while True:
            question = input("❓ Pregunta algo (o escribe 'salir'): ")
            if question.lower() in ["salir", "exit", "quit"]:
                break

            docs = retrieve_docs(question, vector_store)
            context = "\n\n".join([doc.page_content for doc in docs])
            print("\n🤖 Preguntando...")
            answer = answer_question(question, context)

            print("\n🤖 Respuesta:")
            print(answer)
            print("\n" + "="*50 + "\n")

    except Exception as e:
        print(f"\n❌ Error durante la carga o indexación: {e}")
