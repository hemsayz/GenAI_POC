import os
import shutil
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

class LocalFAISSRAGEngine:
    def __init__(self):
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
        embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT_NAME", "").strip()

        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=str(endpoint).strip(),
            api_key=str(api_key).strip(),
            api_version=str(api_version).strip(),
            azure_deployment=str(embedding_deployment).strip(),
            dimensions=1536
        )
        self.local_index_dir = "faiss_index"

    def ingest_local_document(self, pdf_file_path):
        """Extracts text, tags each chunk with the filename source metadata, and appends to FAISS."""
        loader = PyMuPDFLoader(pdf_file_path)
        documents = loader.load()
        
        # 🏷️ NEW: Grab the clean filename to use as our unique tracking stamp
        source_filename = os.path.basename(pdf_file_path)
        
        # Force the metadata dictionary to track the explicit filename source across pages
        for doc in documents:
            doc.metadata["source"] = source_filename
            if "page" not in doc.metadata:
                doc.metadata["page"] = 0
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
        chunks = splitter.split_documents(documents)
        
        # Explicit double-check verification pass to ensure chunk metadata carries the filename stamp
        for chunk in chunks:
            chunk.metadata["source"] = source_filename
        
        # Check if an index exists to append data instead of wiping it
        if os.path.exists(self.local_index_dir):
            existing_db = FAISS.load_local(
                self.local_index_dir, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            new_db = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
            existing_db.merge_from(new_db)
            existing_db.save_local(self.local_index_dir)
        else:
            vector_db = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
            vector_db.save_local(self.local_index_dir)
            
        return len(chunks)

    def search_workspace_vector_space(self, query, workspace_name, top_k=5):
        """Searches the database but isolates matching outputs strictly to the active workspace metadata attribute."""
        if not os.path.exists(self.local_index_dir):
            return []
            
        vector_db = FAISS.load_local(
            self.local_index_dir, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # 🎯 SMART METADATA FILTER: Tell FAISS to only grab chunks matching our active workspace card title string
        matched_docs_with_scores = vector_db.similarity_search_with_score(
            query=query, 
            k=top_k,
            filter={"source": workspace_name}
        )
        
        raw_inspection_data = []
        for doc, distance in matched_docs_with_scores:
            accuracy_score = 1 / (1 + distance) 
            
            try:
                doc_id = [k for k, v in vector_db.docstore._dict.items() if v == doc][0]
                faiss_id = vector_db.index_to_docstore_id[doc_id]
                raw_vector = vector_db.index.reconstruct(int(faiss_id))
            except Exception:
                raw_vector = np.zeros(1536)

            raw_inspection_data.append({
                "chunk_text": doc.page_content,
                "page": doc.metadata.get("page", 0) + 1,
                "source": doc.metadata.get("source", "Unknown Content Source"),
                "raw_embedding_sample": list(raw_vector[:5]),
                "total_dimensions": len(raw_vector),
                "raw_distance": round(float(distance), 4),
                "accuracy_percentage": round(float(accuracy_score), 4)
            })
            
        return raw_inspection_data

    def inspect_raw_vector_space(self, user_query, top_k=3):
        """Global fallback similarity lookup search that checks across all un-filtered vector rows."""
        if not os.path.exists(self.local_index_dir):
            return []
            
        vector_db = FAISS.load_local(
            self.local_index_dir, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        matched_docs_with_scores = vector_db.similarity_search_with_score(query=user_query, k=top_k)
        
        raw_inspection_data = []
        for doc, distance in matched_docs_with_scores:
            accuracy_score = 1 / (1 + distance) 
            
            try:
                doc_id = [k for k, v in vector_db.docstore._dict.items() if v == doc][0]
                faiss_id = vector_db.index_to_docstore_id[doc_id]
                raw_vector = vector_db.index.reconstruct(int(faiss_id))
            except Exception:
                raw_vector = np.zeros(1536)

            raw_inspection_data.append({
                "chunk_text": doc.page_content,
                "page": doc.metadata.get("page", 0) + 1,
                "source": doc.metadata.get("source", "Legacy Document"),
                "raw_embedding_sample": list(raw_vector[:5]),
                "total_dimensions": len(raw_vector),
                "raw_distance": round(float(distance), 4),
                "accuracy_percentage": round(float(accuracy_score), 4)
            })
            
        return raw_inspection_data

    def get_all_stored_chunks(self):
        """Reads the complete docstore matrix and returns all chunks formatted nicely."""
        if not os.path.exists(self.local_index_dir):
            return []

        vector_db = FAISS.load_local(
            self.local_index_dir, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        all_chunks_data = []
        for idx, (doc_id, doc) in enumerate(vector_db.docstore._dict.items()):
            try:
                faiss_id = vector_db.index_to_docstore_id[doc_id]
                raw_vector = vector_db.index.reconstruct(int(faiss_id))
            except Exception:
                raw_vector = np.zeros(1536)
                
            all_chunks_data.append({
                "chunk_index": idx + 1,
                "chunk_text": doc.page_content,
                "character_length": len(doc.page_content),
                "page": doc.metadata.get("page", 0) + 1,
                "source": doc.metadata.get("source", "Legacy Document"),
                "full_vector": [float(val) for val in raw_vector]
            })
            
        return all_chunks_data