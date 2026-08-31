# Insurance RAG (Retrieval-Augmented Generation)

A Python-based Retrieval-Augmented Generation system for insurance document analysis and Q&A.

## Architecture Overview

- **Azure**: Hosts resource groups, blob storage, and data files
- **Foundry**: Hosts embedding models and LLM endpoints
- **GitHub**: Contains Python codebase and scripts
- **Local**: Document processing pipeline

## Project Structure

```
insurance-rag-python/
├── documents/           # Local document cache (PDF, DOCX, etc.)
├── rag_data/           # Vector index and metadata
├── .env                # Environment configuration (not committed)
├── .gitignore          # Git ignore rules
├── requirements.txt    # Python dependencies
├── download_documents.py  # Download docs from Azure
├── build_index.py      # Build FAISS vector index
├── ask.py              # Interactive Q&A interface
└── README.md           # This file
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8+
- Azure account with blob storage
- OpenAI API key (or other LLM endpoint)
- Foundry account (optional, for embeddings/models)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/shobhitvj1311/insurance_rag.git
cd insurance_rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the root directory (copy from `.env` template):

```bash
# Azure Configuration
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_BLOB_CONTAINER_NAME=insurance-documents

# OpenAI Configuration
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4

# Foundry Configuration (optional)
FOUNDRY_API_KEY=your_key
FOUNDRY_EMBEDDING_MODEL=your_model
```

## Usage

### Step 1: Download Documents

Download insurance documents from Azure Blob Storage:

```bash
python download_documents.py
```

This will download all documents to the `documents/` folder.

### Step 2: Build Index

Process documents and build FAISS vector index:

```bash
python build_index.py
```

This will:
- Load all PDFs from `documents/`
- Split them into chunks
- Generate embeddings
- Build and save FAISS index to `rag_data/`

### Step 3: Ask Questions

Run the interactive Q&A interface:

```bash
python ask.py
```

Example questions:
```
Ask a question about insurance: What are the coverage limits for liability?
Ask a question about insurance: How do I file a claim?
```

## Key Features

- **Document Processing**: Supports PDF, DOCX, PPTX formats
- **Embedding Generation**: Uses Sentence Transformers for semantic understanding
- **Vector Search**: FAISS for fast similarity search
- **LLM Integration**: OpenAI GPT-4 or custom Foundry models
- **Source Tracking**: Returns source documents for each answer
- **Logging**: Comprehensive logging for debugging

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_STORAGE_CONNECTION_STRING` | Azure blob storage connection | Yes |
| `AZURE_BLOB_CONTAINER_NAME` | Azure container name | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Yes (if using OpenAI) |
| `OPENAI_MODEL` | Model name (default: gpt-4) | No |
| `FOUNDRY_API_KEY` | Foundry API key | No |
| `LOG_LEVEL` | Logging level (INFO, DEBUG) | No |

## Dependencies

See `requirements.txt` for complete list:
- `langchain`: LLM framework
- `langchain-community`: Community integrations
- `azure-storage-blob`: Azure storage access
- `faiss-cpu`: Vector search
- `sentence-transformers`: Embeddings
- `openai`: LLM API

## Performance Considerations

- **Chunking**: Documents are split into 1000-token chunks with 200-token overlap
- **Similarity Search**: Default retrieval returns top 3 similar documents
- **Embedding Model**: Uses lightweight MiniLM for speed
- **Temperature**: Set to 0.1 for consistent, factual responses

## Troubleshooting

### Azure Connection Issues
```bash
# Verify connection string
python -c "from azure.storage.blob import BlobServiceClient; \
BlobServiceClient.from_connection_string('YOUR_CONNECTION_STRING')"
```

### No Documents Found
- Ensure `documents/` folder exists
- Check that PDFs are in `documents/` folder
- Verify file permissions

### Index Build Fails
- Ensure sufficient disk space for index
- Check memory availability
- Verify document quality

## Future Enhancements

- [ ] Multi-language support
- [ ] Web API endpoint
- [ ] Document upload interface
- [ ] Advanced filtering and metadata search
- [ ] Integration with more LLM providers
- [ ] Caching and optimization

## License

MIT License

## Support

For issues or questions, create an issue in the repository.
