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

#combining the user query and the context from the documents into a single input for the model
combined_input = f"""Based on the following context from the documents, answer the user query: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If the answer is not present in the documents, respond with "I don't know." Do not make up information or provide answers that are not supported by the documents.
"""

#creating an instance of the Anthropic model with specified parameters
model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.2, max_tokens=1000)


#Defining the system and human messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant that answers questions based on the provided context from documents. If the answer is not present in the documents, respond with 'I don't know.' Do not make up information or provide answers that are not supported by the documents."),
    HumanMessage(content=combined_input)
]

#invoking the model with the messages
result = model.invoke(messages)

print("\n---Generated Answer---")
print("🔥🔥🔥🔥Content only")
print(result.content)