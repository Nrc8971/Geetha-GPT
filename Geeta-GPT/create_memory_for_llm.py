import os
from typing import List, Dict
from langchain_community.document_loaders import (
    PyPDFLoader, 
    DirectoryLoader, 
    TextLoader, 
    CSVLoader, 
    JSONLoader, 
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def load_files(data_path: str) -> List[Document]:
    """
    Load documents from multiple file types in the specified directory.
    
    Args:
        data_path (str): Path to the directory containing the documents
        
    Returns:
        List[Document]: List of loaded documents
    """
    if not os.path.exists(data_path):
        raise ValueError(f"Directory {data_path} does not exist")
        
    documents = []
    processed_files = {"success": [], "failed": []}
    
    file_loaders = {
        '.pdf': lambda path: PyPDFLoader(path),
        '.txt': lambda path: TextLoader(path, encoding='utf-8'),
        '.csv': lambda path: CSVLoader(path, encoding='utf-8'),
        '.json': lambda path: JSONLoader(
            path,
            jq_schema='.[]',
            text_content=False
        ),
        '.docx': lambda path: UnstructuredWordDocumentLoader(path),
        '.doc': lambda path: UnstructuredWordDocumentLoader(path)
    }
    
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if not os.path.isfile(file_path):
            continue
            
        try:
            if file_ext in file_loaders:
                loader = file_loaders[file_ext](file_path)
                docs = loader.load()
                documents.extend(docs)
                processed_files["success"].append(filename)
            else:
                processed_files["failed"].append((filename, "Unsupported file type"))
                
        except Exception as e:
            processed_files["failed"].append((filename, str(e)))
    
    print("\nProcessing Summary:")
    print(f"Successfully processed {len(processed_files['success'])} files:")
    for file in processed_files["success"]:
        print(f"✓ {file}")
        
    if processed_files["failed"]:
        print(f"\nFailed to process {len(processed_files['failed'])} files:")
        for file, error in processed_files["failed"]:
            print(f"✗ {file}: {error}")
    
    return documents

def create_chunks(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Split documents into smaller chunks for processing.
    
    Args:
        documents (List[Document]): List of documents to split
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        List[Document]: List of document chunks
    """
    if not documents:
        raise ValueError("No documents provided for chunking")
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"\nChunking Summary:")
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    print(f"Average chunk size: {sum(len(chunk.page_content) for chunk in chunks) / len(chunks):.0f} characters")
    
    return chunks

def create_vectorstore(chunks: List[Document], output_path: str) -> FAISS:
    """
    Create and save a FAISS vectorstore from document chunks.
    
    Args:
        chunks (List[Document]): List of document chunks
        output_path (str): Path to save the vectorstore
        
    Returns:
        FAISS: The created vectorstore
    """
    if not chunks:
        raise ValueError("No chunks provided for vectorstore creation")
        
    print("\nCreating vectorstore...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    db = FAISS.from_documents(chunks, embedding_model)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    db.save_local(output_path)
    
    print(f"✓ Vectorstore saved to {output_path}")
    return db

def main():
    DATA_PATH = "data/"
    DB_FAISS_PATH = "vectorstore/db_faiss"
    
    try:
        # Step 1: Load documents
        print(f"Loading documents from {DATA_PATH}...")
        documents = load_files(DATA_PATH)
        
        if not documents:
            raise ValueError("No documents were successfully loaded")
            
        # Step 2: Create chunks
        chunks = create_chunks(documents)
        
        # Step 3: Create and save vectorstore
        db = create_vectorstore(chunks, DB_FAISS_PATH)
        
        print("\n✓ Process completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()