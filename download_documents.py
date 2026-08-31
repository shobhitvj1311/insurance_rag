"""
Download insurance documents from Azure Blob Storage.

This script downloads documents from Azure Blob Storage and stores them locally
in the documents/ folder for processing.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentDownloader:
    """Download documents from Azure Blob Storage."""

    def __init__(self):
        """Initialize Azure Blob Storage client."""
        self.connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.container_name = os.getenv('AZURE_BLOB_CONTAINER_NAME', 'insurance-documents')
        self.documents_dir = Path('documents')
        self.documents_dir.mkdir(exist_ok=True)

        if self.connection_string:
            self.blob_client = BlobServiceClient.from_connection_string(
                self.connection_string
            )
        else:
            # Use DefaultAzureCredential for managed identity
            account_name = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
            account_url = f"https://{account_name}.blob.core.windows.net"
            self.blob_client = BlobServiceClient(
                account_url=account_url,
                credential=DefaultAzureCredential()
            )

    def download_documents(self):
        """Download all documents from Azure Blob Storage."""
        try:
            container_client = self.blob_client.get_container_client(self.container_name)
            blobs = container_client.list_blobs()

            logger.info(f"Starting download from container: {self.container_name}")

            for blob in blobs:
                self._download_blob(container_client, blob.name)

            logger.info("Document download completed successfully")

        except Exception as e:
            logger.error(f"Error downloading documents: {e}")
            raise

    def _download_blob(self, container_client, blob_name):
        """Download a single blob."""
        try:
            file_path = self.documents_dir / blob_name
            file_path.parent.mkdir(parents=True, exist_ok=True)

            blob_client = container_client.get_blob_client(blob_name)
            with open(file_path, 'wb') as file:
                file.write(blob_client.download_blob().readall())

            logger.info(f"Downloaded: {blob_name}")

        except Exception as e:
            logger.error(f"Error downloading blob {blob_name}: {e}")


def main():
    """Main function."""
    downloader = DocumentDownloader()
    downloader.download_documents()


if __name__ == '__main__':
    main()
