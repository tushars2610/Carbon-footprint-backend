from google.cloud import storage
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.DEBUG)
load_dotenv()

def list_domains(bucket_name):
    try:
        client = storage.Client.from_service_account_json(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix="docs/", delimiter='/')
        domains = []
        for prefix in blobs.prefixes:
            domain = prefix.replace('docs/', '').strip('/')
            if domain and not domain.startswith('.'):
                domains.append(domain)
        logging.debug(f"Prefixes found: {[p for p in blobs.prefixes]}")
        logging.info(f"Detected domains: {domains}")
        return domains
    except Exception as e:
        logging.error(f"Error listing domains: {e}")
        return []

if __name__ == "__main__":
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    print(f"Testing bucket: {bucket_name}")
    domains = list_domains(bucket_name)
    print(f"Domains: {domains}")