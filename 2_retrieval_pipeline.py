from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

persistent_directory = "chroma_db"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings,collection_metadata={"hnsw:space":"cosine"})

query = "what model are there in from Anthropic"

retriever = db.as_retriever(search_kwargs={"k": 5})

relevant_docs = retriever.invoke(query)

print(f"👉👉👉 user query: {query}")

print("----context from the documents----")
for i, doc in enumerate(relevant_docs,1):
    print(f"\nDocument {i}:\n{doc.page_content}")
    print(f"Source: {doc.metadata['source']}")
    print(f"Content length: {len(doc.page_content)} characters")
    print(f"Content preview: {doc.page_content[:100]}...")
    print("-" * 20)
