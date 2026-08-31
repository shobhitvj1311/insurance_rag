"""
Query the insurance RAG system.

This script allows users to ask questions about insurance documents
using the built FAISS index and LLM.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InsuranceRAG:
    """Insurance RAG system for querying documents."""

    def __init__(self):
        """Initialize RAG system."""
        self.rag_data_dir = Path('rag_data')
        self.index_path = self.rag_data_dir / "faiss_index"

        # Initialize embeddings
        logger.info("Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load vector store
        logger.info("Loading vector store...")
        self.vectorstore = self._load_vectorstore()

        # Initialize LLM
        logger.info("Initializing LLM...")
        self.llm = self._initialize_llm()

        # Setup QA chain
        self.qa_chain = self._setup_qa_chain()

    def _load_vectorstore(self) -> FAISS:
        """Load FAISS vector store."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found at {self.index_path}")

        vectorstore = FAISS.load_local(
            str(self.index_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully")
        return vectorstore

    def _initialize_llm(self):
        """Initialize the LLM."""
        api_key = os.getenv('OPENAI_API_KEY')
        model = os.getenv('OPENAI_MODEL', 'gpt-4')

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        return ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0.1
        )

    def _setup_qa_chain(self):
        """Setup QA chain with custom prompt."""
        prompt_template = """You are an expert insurance advisor. Use the provided context 
from insurance documents to answer the user's question accurately and helpfully.

Context:
{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        return qa_chain

    def ask(self, question: str) -> Dict[str, any]:
        """Ask a question about insurance documents."""
        logger.info(f"Processing question: {question}")

        try:
            result = self.qa_chain.invoke({"query": question})

            response = {
                "question": question,
                "answer": result["result"],
                "sources": [
                    {
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "Unknown")
                    }
                    for doc in result.get("source_documents", [])
                ]
            }

            return response

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise

    def interactive_mode(self):
        """Run in interactive mode."""
        print("\n" + "="*60)
        print("Insurance RAG System - Interactive Mode")
        print("="*60)
        print("Type 'quit' or 'exit' to stop\n")

        while True:
            try:
                question = input("Ask a question about insurance: ").strip()

                if question.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break

                if not question:
                    print("Please enter a valid question.\n")
                    continue

                response = self.ask(question)

                print("\n" + "-"*60)
                print(f"Answer: {response['answer']}\n")

                if response['sources']:
                    print("Sources:")
                    for i, source in enumerate(response['sources'], 1):
                        print(f"  {i}. {source['source']} (Page {source['page']})")

                print("-"*60 + "\n")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


def main():
    """Main function."""
    try:
        rag = InsuranceRAG()
        rag.interactive_mode()

    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise


if __name__ == '__main__':
    main()
