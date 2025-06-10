# scripts/setup_structure.py
from google.cloud import storage
from dotenv import load_dotenv
import os
import json
import logging
from google.oauth2 import service_account

# Load environment variables
load_dotenv()
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

def get_gcs_client():
    """Get Google Cloud Storage client with proper credential handling"""
    
    # Check if credentials are provided as JSON string (for Render/production)
    credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if credentials_json:
        try:
            credentials_info = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            return storage.Client(credentials=credentials)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")
            raise
    
    # Check if credentials file path is provided (for local development)
    credentials_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if credentials_file and os.path.exists(credentials_file):
        return storage.Client.from_service_account_json(credentials_file)
    
    # Try default credentials (when running on GCP)
    try:
        return storage.Client()
    except Exception as e:
        logging.error(f"Failed to initialize GCS client with default credentials: {e}")
        raise Exception(
            "Unable to authenticate with Google Cloud Storage. "
            "Please set either GOOGLE_APPLICATION_CREDENTIALS_JSON (JSON string) "
            "or GOOGLE_APPLICATION_CREDENTIALS (file path) environment variable."
        )

def create_project_structure():
    """Create project folder structure in Google Cloud Storage."""
    
    # Verify environment variables
    if not GCS_BUCKET_NAME:
        logging.error("GCS_BUCKET_NAME not set in .env file")
        return
    
    # Check for either type of credentials
    if not (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")):
        logging.error("Neither GOOGLE_APPLICATION_CREDENTIALS nor GOOGLE_APPLICATION_CREDENTIALS_JSON is set in .env file")
        logging.error("Please set one of these environment variables for GCP authentication")
        return
    
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        
        # Check if bucket exists
        if not bucket.exists():
            logging.error(f"Bucket '{GCS_BUCKET_NAME}' does not exist or is not accessible")
            return
            
    except Exception as e:
        logging.error(f"Failed to initialize GCS client: {e}")
        return
    
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
    
    created_folders = []
    failed_folders = []
    
    for folder in folders:
        try:
            blob = bucket.blob(folder)
            # Check if folder already exists (has any objects with this prefix)
            existing_blobs = list(bucket.list_blobs(prefix=folder, max_results=1))
            
            if not existing_blobs:
                blob.upload_from_string('', content_type='application/x-www-form-urlencoded')
                created_folders.append(folder)
                logging.info(f"Created folder in GCS: gs://{GCS_BUCKET_NAME}/{folder}")
            else:
                logging.info(f"Folder already exists in GCS: gs://{GCS_BUCKET_NAME}/{folder}")
                
        except Exception as e:
            failed_folders.append(folder)
            logging.error(f"Failed to create folder gs://{GCS_BUCKET_NAME}/{folder}: {e}")
    
    # Summary
    logging.info(f"Structure setup completed!")
    logging.info(f"Created {len(created_folders)} new folders")
    
    if failed_folders:
        logging.warning(f"Failed to create {len(failed_folders)} folders: {failed_folders}")
    else:
        logging.info("All folders created successfully!")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    create_project_structure()