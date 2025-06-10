# scripts/run_pipeline.py
import fire
from pathlib import Path
import logging
import sys
from typing import Dict, List, Any
from joblib import Parallel, delayed
import psutil
from dotenv import load_dotenv
import os
import json
from google.cloud import storage
from google.oauth2 import service_account

# Load environment variables
load_dotenv()
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# Add project root to sys.path for direct execution
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from scripts.utils import load_config, merge_configs, generate_domain_config, setup_logging, gcs_list_domains
    from scripts.ingest import DocumentIngester
    from scripts.chunk import DocumentChunker
    from scripts.embed import EmbeddingGenerator
    from scripts.index import FAISSIndexer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure all required scripts (utils.py, ingest.py, chunk.py, embed.py, index.py) are present in the scripts/ directory.")
    sys.exit(1)

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

def process_domain(domain: str, base_dir: str, global_config: Dict[str, Any], stages: List[str], skip_existing: bool, force: bool):
    """Process a single domain through specified pipeline stages."""
    if not domain:
        logging.error("Empty domain name provided, skipping processing")
        return
    
    config_path = f"{base_dir}/scripts/config/{domain}.yaml"
    client = get_gcs_client()  # Use the helper function instead
    bucket = client.bucket(GCS_BUCKET_NAME)
    
    # Check if config exists in GCS
    if not bucket.blob(f"scripts/config/{domain}.yaml").exists():
        logging.info(f"Generating config for domain: {domain}")
        generate_domain_config(domain, global_config, f"gs://{GCS_BUCKET_NAME}/scripts/config")
    
    try:
        domain_config = load_config(config_path)
    except Exception as e:
        logging.error(f"Failed to load domain config for {domain}: {e}")
        return
    
    config = merge_configs(global_config, domain_config)
    
    if force:
        for folder in ['json', 'chunks', 'embeddings', 'indices']:
            prefix = f"data/{domain}/{folder}/"
            blobs = bucket.list_blobs(prefix=prefix)
            for blob in blobs:
                blob.delete()
                logging.info(f"Deleted gs://{GCS_BUCKET_NAME}/{blob.name}")
    
    if 'ingest' in stages:
        try:
            ingester = DocumentIngester(domain, config, base_dir)
            processed_files = ingester.ingest(skip_existing=skip_existing)
            logging.info(f"Ingested {len(processed_files)} files for domain {domain}")
        except Exception as e:
            logging.error(f"Error in ingest stage for {domain}: {e}")
    
    if 'chunk' in stages:
        try:
            chunker = DocumentChunker(domain, config, base_dir)
            chunk_stats = chunker.chunk_directory()
            logging.info(f"Chunked {chunk_stats['total_chunks']} chunks for domain {domain}")
        except Exception as e:
            logging.error(f"Error in chunk stage for {domain}: {e}")
    
    if 'embed' in stages:
        try:
            embedder = EmbeddingGenerator(domain, config, base_dir)
            result = embedder.process_chunks()
            if result['embeddings']:
                embedder.save_embeddings(result['embeddings'], result['statistics'])
                embedder.verify_pkl(f"gs://{GCS_BUCKET_NAME}/data/{domain}/embeddings/embeddings.pkl")
                logging.info(f"Generated embeddings for {len(result['embeddings'])} chunks in domain {domain}")
            else:
                logging.warning(f"No embeddings generated for domain {domain}")
        except Exception as e:
            logging.error(f"Error in embed stage for {domain}: {e}")
    
    if 'index' in stages:
        try:
            indexer = FAISSIndexer(domain, config, base_dir)
            data = indexer.load_embeddings()
            if data['embeddings'].size:
                result = indexer.build_index(data['embeddings'], data['metadata'])
                if result['index']:
                    indexer.save_index({**data['statistics'], **result['statistics']})
                    logging.info(f"Built index for domain {domain}")
                else:
                    logging.warning(f"No index built for domain {domain}")
            else:
                logging.warning(f"No embeddings found for indexing in domain {domain}")
        except Exception as e:
            logging.error(f"Error in index stage for {domain}: {e}")

def run_pipeline(
    domain: str = None,
    all_domains: bool = False,
    stages: str = "ingest,chunk,embed,index",
    full_pipeline: bool = True,
    parallel: int = None,
    config_override: str = None,
    skip_existing: bool = False,
    force: bool = False,
    verbose: bool = False
):
    """Orchestrate the multi-domain pipeline."""
    base_dir = f"gs://{GCS_BUCKET_NAME}"
    global_config_path = f"{base_dir}/scripts/config/global.yaml"
    
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
        global_config = load_config(global_config_path)
    except Exception as e:
        logging.error(f"Failed to load global config at {global_config_path}: {e}")
        logging.error("Ensure global.yaml exists in gs://my-document-pipeline-bucket/scripts/config/. Run 'gsutil cp global.yaml gs://my-document-pipeline-bucket/scripts/config/' to upload it.")
        return
    
    if config_override:
        try:
            override_config = load_config(config_override)
            global_config = merge_configs(global_config, override_config)
        except Exception as e:
            logging.error(f"Failed to load override config: {e}")
            return
    
    if verbose:
        global_config['logging']['level'] = 'debug'
    
    setup_logging(global_config)
    
    if parallel is not None:
        global_config['processing']['parallel_domains'] = max(1, parallel)
    
    # List domains from GCS
    domains = gcs_list_domains(GCS_BUCKET_NAME)
    if not domains:
        logging.error("No valid domains found in gs://<bucket>/docs/. Ensure domain folders like 'docs/domain_1/' exist.")
        logging.error("Run 'gsutil ls gs://my-document-pipeline-bucket/docs/' to verify folder structure.")
        return
    
    if not all_domains and domain:
        domains = [d.strip() for d in domain.split(',') if d.strip() in domains]
        if not domains:
            logging.error(f"No valid domains specified. Available domains: {', '.join(gcs_list_domains(GCS_BUCKET_NAME))}")
            return
    
    if full_pipeline:
        stages = ['ingest', 'chunk', 'embed', 'index']
    else:
        stages = [s.strip() for s in stages.split(',')]
        if not all(s in ['ingest', 'chunk', 'embed', 'index'] for s in stages):
            logging.error("Invalid stages specified")
            return
    
    logging.info(f"System resources: CPU {psutil.cpu_count()} cores, Memory {psutil.virtual_memory().total / 1024**3:.2f} GB")
    logging.info(f"Processing domains: {', '.join(domains)} with stages: {', '.join(stages)}")
    
    Parallel(n_jobs=global_config['processing']['parallel_domains'], backend='threading')(
        delayed(process_domain)(d, base_dir, global_config, stages, skip_existing, force) for d in domains
    )
    
    logging.info(f"Completed processing for domains: {', '.join(domains)}")

if __name__ == "__main__":
    fire.Fire(run_pipeline)