"""
Build vector index from insurance documents.

This script processes documents from the documents/ folder, generates embeddings,
and builds a FAISS index for RAG retrieval.
"""

import os
import logging
import pickle
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Tuple

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexBuilder:
    """Build vector index from documents."""

    def __init__(self):
        """Initialize index builder."""
        self.documents_dir = Path('documents')
        self.rag_data_dir = Path('rag_data')
        self.rag_data_dir.mkdir(exist_ok=True)

        # Initialize embeddings model
        logger.info("Initializing embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.chunk_size = 1000
        self.chunk_overlap = 200

    def load_documents(self) -> List:
        """Load documents from the documents folder."""
        logger.info("Loading documents...")

        if not self.documents_dir.exists():
            logger.error(f"Documents directory not found: {self.documents_dir}")
            return []

        loader = DirectoryLoader(
            str(self.documents_dir),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )

        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from documents")
        return documents

    def process_documents(self, documents: List) -> List:
        """Split documents into chunks."""
        logger.info("Processing documents...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        texts = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(texts)} chunks")
        return texts

    def build_index(self, texts: List):
        """Build FAISS index from texts."""
        logger.info("Building FAISS index...")

        try:
            vectorstore = FAISS.from_documents(texts, self.embeddings)
            index_path = self.rag_data_dir / "faiss_index"

            vectorstore.save_local(str(index_path))
            logger.info(f"Index saved to {index_path}")

            # Save metadata
            metadata = {
                "num_documents": len(texts),
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }

            metadata_path = self.rag_data_dir / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

            logger.info(f"Metadata saved to {metadata_path}")

        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise

    def build(self):
        """Build the complete index."""
        documents = self.load_documents()
        if not documents:
            logger.warning("No documents loaded")
            return

        texts = self.process_documents(documents)
        if not texts:
            logger.warning("No text chunks created")
            return

        self.build_index(texts)
        logger.info("Index building completed successfully")


def main():
    """Main function."""
    builder = IndexBuilder()
    builder.build()


if __name__ == '__main__':
    main()
