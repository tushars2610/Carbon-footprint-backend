import yaml
import logging
import json
import pickle
from concurrent_log_handler import ConcurrentRotatingFileHandler
from google.cloud import storage
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
import os
from jsonschema import validate, ValidationError

# Load environment variables
load_dotenv()

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

# Configure logging with thread-safe handler
def setup_logging(config: Dict[str, Any]):
    logger = logging.getLogger()
    logger.handlers = []  # Clear existing handlers
    level = getattr(logging, config['logging']['level'].upper(), logging.INFO)
    logger.setLevel(level)
    
    # File handler with rotation
    log_file = config['logging']['file']
    file_handler = ConcurrentRotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Console handler
    if config['logging']['console']:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
    
    return logger

# Configuration schema for validation
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "domain_name": {"type": "string"},
        "chunking": {
            "type": "object",
            "properties": {
                "chunk_size": {"type": "integer", "minimum": 100},
                "chunk_overlap": {"type": "integer", "minimum": 0},
                "min_chunk_size": {"type": "integer", "minimum": 50},
                "sentence_boundary": {"type": "boolean"}
            },
            "required": ["chunk_size", "chunk_overlap", "min_chunk_size", "sentence_boundary"]
        },
        "embedding": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string"},
                "batch_size": {"type": "integer", "minimum": 1},
                "normalize": {"type": "boolean"},
                "use_gpu": {"enum": ["auto", True, False]}
            },
            "required": ["model_name", "batch_size", "normalize", "use_gpu"]
        },
        "indexing": {
            "type": "object",
            "properties": {
                "index_type": {"enum": ["IndexIVFFlat", "IndexIVFPQ", "IndexFlatL2"]},
                "nlist_factor": {"type": "integer", "minimum": 1},
                "nprobe": {"type": "integer", "minimum": 1},
                "metric": {"enum": ["inner_product", "l2"]}
            },
            "required": ["index_type", "nlist_factor", "nprobe", "metric"]
        },
        "processing": {
            "type": "object",
            "properties": {
                "batch_size": {"type": "integer", "minimum": 1000},
                "parallel_domains": {"type": "integer", "minimum": 1},
                "temp_dir": {"type": "string"}
            },
            "required": ["batch_size", "parallel_domains", "temp_dir"]
        },
        "logging": {
            "type": "object",
            "properties": {
                "level": {"enum": ["debug", "info", "warning", "error"]},
                "file": {"type": "string"},
                "console": {"type": "boolean"}
            },
            "required": ["level", "file", "console"]
        },
        "metadata": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "source": {"type": "string"},
                "last_updated": {"type": "string"}
            }
        }
    }
}

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate a YAML configuration file from GCS or local."""
    try:
        if config_path.startswith('gs://'):
            bucket_name, blob_name = parse_gcs_path(config_path)
            client = get_gcs_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            config_data = blob.download_as_text(encoding='utf-8')
            config = yaml.safe_load(config_data)
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        validate(instance=config, schema=CONFIG_SCHEMA)
        return config
    except ValidationError as e:
        logging.error(f"Configuration validation error in {config_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading {config_path}: {e}")
        raise

def parse_gcs_path(gcs_path: str) -> tuple:
    """Parse GCS path into bucket name and blob name."""
    if not gcs_path.startswith('gs://'):
        raise ValueError(f"Invalid GCS path: {gcs_path}")
    parts = gcs_path.replace('gs://', '').split('/', 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid GCS path: {gcs_path}")
    return parts[0], parts[1]

def merge_configs(global_config: Dict[str, Any], domain_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge global and domain-specific configurations."""
    merged = global_config.copy()
    for key, value in domain_config.items():
        if key in merged and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged

def generate_domain_config(domain_name: str, global_config: Dict[str, Any], output_dir: str):
    """Generate a default domain-specific configuration file in GCS."""
    domain_config = {
        "domain_name": domain_name,
        "chunking": global_config["chunking"],
        "embedding": global_config["embedding"],
        "indexing": global_config["indexing"],
        "metadata": {
            "description": f"{domain_name} documents collection",
            "source": "unknown",
            "last_updated": "2025-05-20"
        }
    }
    output_path = f"{output_dir}/{domain_name}.yaml"
    try:
        if output_dir.startswith('gs://'):
            bucket_name, blob_name = parse_gcs_path(f"{output_dir}/{domain_name}.yaml")
            client = get_gcs_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_string(yaml.safe_dump(domain_config, default_flow_style=False), content_type='text/yaml')
            logging.info(f"Generated domain config in GCS: gs://{bucket_name}/{blob_name}")
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(domain_config, f, default_flow_style=False)
            logging.info(f"Generated domain config: {output_path}")
    except Exception as e:
        logging.error(f"Error generating domain config {output_path}: {e}")

def gcs_read_json(blob_path: str) -> Dict[str, Any]:
    """Read JSON file from GCS."""
    try:
        bucket_name, blob_name = parse_gcs_path(blob_path)
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        data = json.loads(blob.download_as_text(encoding='utf-8'))
        return data
    except Exception as e:
        logging.error(f"Error reading JSON from GCS {blob_path}: {e}")
        return {}

def gcs_write_json(data: Dict[str, Any], blob_path: str):
    """Write JSON file to GCS."""
    try:
        bucket_name, blob_name = parse_gcs_path(blob_path)
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(data, ensure_ascii=False, indent=2), content_type='application/json')
        logging.info(f"Saved JSON to GCS: gs://{bucket_name}/{blob_name}")
    except Exception as e:
        logging.error(f"Error writing JSON to GCS {blob_path}: {e}")

def gcs_read_pickle(blob_path: str) -> Any:
    """Read pickle file from GCS."""
    try:
        bucket_name, blob_name = parse_gcs_path(blob_path)
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        data = pickle.loads(blob.download_as_bytes())
        return data
    except Exception as e:
        logging.error(f"Error reading pickle from GCS {blob_path}: {e}")
        return None

def gcs_write_pickle(data: Any, blob_path: str):
    """Write pickle file to GCS."""
    try:
        bucket_name, blob_name = parse_gcs_path(blob_path)
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(pickle.dumps(data), content_type='application/octet-stream')
        logging.info(f"Saved pickle to GCS: gs://{bucket_name}/{blob_name}")
    except Exception as e:
        logging.error(f"Error writing pickle to GCS {blob_path}: {e}")

def gcs_read_file(blob_path: str) -> str:
    """Read text file from GCS."""
    try:
        bucket_name, blob_name = parse_gcs_path(blob_path)
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_text(encoding='utf-8')
    except Exception as e:
        logging.error(f"Error reading file from GCS {blob_path}: {e}")
        return ""

def gcs_write_file(data: str, blob_path: str):
    """Write text file to GCS."""
    try:
        bucket_name, blob_name = parse_gcs_path(blob_path)
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(data, content_type='text/plain')
        logging.info(f"Saved text file to GCS: gs://{bucket_name}/{blob_name}")
    except Exception as e:
        logging.error(f"Error writing text file to GCS {blob_path}: {e}")
        raise

def gcs_list_files(bucket_name: str, prefix: str) -> List[str]:
    """List files in GCS bucket with given prefix."""
    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        return [f"gs://{bucket_name}/{blob.name}" for blob in blobs if not blob.name.endswith('/')]
    except Exception as e:
        logging.error(f"Error listing files in GCS bucket {bucket_name} with prefix {prefix}: {e}")
        return []

def gcs_list_domains(bucket_name: str) -> list:
    """List domain folders in GCS under docs/."""
    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        # Ensure prefix ends with '/' to list folders
        prefix = "docs/"
        
        # Get all blobs under docs/ to find domain folders
        blobs = list(bucket.list_blobs(prefix=prefix))
        logging.info(f"Found {len(blobs)} blobs under {prefix}")
        
        # Extract domain names from blob paths
        domains = set()
        for blob in blobs:
            # Skip the docs/ folder itself
            if blob.name == prefix:
                continue
                
            # Extract potential domain name from path
            # Example: "docs/domain_1/file.txt" -> "domain_1"
            parts = blob.name[len(prefix):].split('/', 1)
            if parts and parts[0]:  # Ensure we have a non-empty domain name
                domains.add(parts[0])
        
        # Convert to list and sort
        domain_list = sorted(list(domains))
        
        # Log for debugging
        logging.info(f"Detected domains in GCS bucket {bucket_name}: {domain_list}")
        if not domain_list:
            logging.warning("No domains detected. Ensure files exist in folders like 'docs/domain_1/'.")
        
        return domain_list
    except Exception as e:
        logging.error(f"Error listing domains in GCS bucket {bucket_name}: {e}")
        logging.exception(e)  # Log the full stack trace for better debugging
        return []