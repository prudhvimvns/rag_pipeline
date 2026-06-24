import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

#1. load the documents
def load_documents(directory_path):
    print(f"Loading documents from directory: {directory_path}")
    try:
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={
                "encoding": "utf-8"
            },
            show_progress=True
        )
        documents = loader.load()

        if len(documents) == 0:
            print(f"No documents found in directory: {directory_path}")
            return None
        print(f"Loaded {len(documents)} documents from {directory_path}")

        for i, doc in enumerate(documents[:2]):
            print(f"\n Document {i+1}:")
            print(f" Source: {doc.metadata['source']}")
            print(f" Content length: {len(doc.page_content)} characters")
            print(f" Content preview: {doc.page_content[:100]}...")  # Print the first 100 characters as a preview
            print(f" Metadata: {doc.metadata}")
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return None

#2. split the documents into chunks
def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Splits the loaded documents into smaller chunks using CharacterTextSplitter."""
    print(f"Splitting documents into chunks with chunk size: {chunk_size} and chunk overlap: {chunk_overlap}")
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",
    )

    chunks = text_splitter.split_documents(documents)
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n Chunk {i+1}:")
            print(f" Source: {chunk.metadata['source']}")
            print(f" Chunk content length: {len(chunk.page_content)} characters")
            print(f" Chunk content preview: {chunk.page_content[:100]}...")  #
            print("-" * 20)
        if len(chunks) > 5:
            print(f"... and {len(chunks) - 5} more chunks.")
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks, db_path="chroma_db"):
    print(f"Creating vector store at: {db_path} using {len(chunks)} chunks")
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embedding_model = HuggingFaceEmbeddings( model_name="sentence-transformers/all-MiniLM-L6-v2")


    #create the vector store
    print("---Creating vector store---")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path,
        collection_metadata={"hnsw:space":"cosine"}
    )
    print("---Finished creating vector store---")

    print(f"Saved vector database to {db_path}")
    return vector_store


def main():
    print("Starting the ingestion pipeline...")

    #1. load the documents
    documents = load_documents("docs")  # Assuming you have a 'data' directory with your documents

    #2. split the documents into chunks
    chunks = split_documents(documents)

    #3. Embed the chunks and store in vector database
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()