# scripts/setup_structure.py
from google.cloud import storage
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

def create_project_structure():
    """Create project folder structure in Google Cloud Storage."""
    client = storage.Client.from_service_account_json(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    bucket = client.bucket(GCS_BUCKET_NAME)
    domains = ['domain_1', 'carbon']  # Example domains
    folders = [
        'docs/',
        'data/',
        'scripts/config/'
    ]
    for domain in domains:
        folders.extend([
            f"docs/{domain}/",
            f"data/{domain}/json/",
            f"data/{domain}/chunks/",
            f"data/{domain}/embeddings/",
            f"data/{domain}/indices/"
        ])
    for folder in folders:
        blob = bucket.blob(folder)
        blob.upload_from_string('', content_type='application/x-www-form-urlencoded')
        logging.info(f"Created folder in GCS: gs://{GCS_BUCKET_NAME}/{folder}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_project_structure()