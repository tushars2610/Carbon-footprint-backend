# scripts/index.py
from google.cloud import storage
import os
import json
import pickle
import logging
import time
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import faiss
import torch
from tqdm import tqdm
from .utils import setup_logging, gcs_read_pickle, gcs_write_pickle, gcs_write_json
from dotenv import load_dotenv
import tempfile
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

class FAISSIndexer:
    def __init__(self, domain: str, config: Dict[str, Any], base_dir: str):
        self.domain = domain
        self.config = config
        self.input_dir = f"{base_dir}/data/{domain}/embeddings"
        self.output_dir = f"{base_dir}/data/{domain}/indices"
        self.embedding_model = config['embedding']['model_name']
        self.index_type = config['indexing']['index_type']
        self.nlist_factor = config['indexing']['nlist_factor']
        self.nprobe = config['indexing']['nprobe']
        self.metric = config['indexing']['metric']
        self.device = 'cuda' if (config['embedding']['use_gpu'] == 'auto' and torch.cuda.is_available()) or config['embedding']['use_gpu'] == True else 'cpu'
        self.index = None
        self.metadata = []
        self.embedding_dim = None
        self.client = self.client = self.client = get_gcs_client()  # Use the helper function
        self.bucket = self.client.bucket(GCS_BUCKET_NAME)
        setup_logging(config)
        logging.info(f"Using device: {self.device} for {self.domain}")

    def load_embeddings(self) -> Dict[str, Any]:
        input_path = f"{self.input_dir}/embeddings.pkl"
        if not self.bucket.blob(input_path.replace(f"gs://{GCS_BUCKET_NAME}/", '')).exists():
            logging.error(f"Embeddings file not found at {input_path} for {self.domain}")
            return {"embeddings": np.array([]), "metadata": [], "statistics": {}}

        try:
            data = gcs_read_pickle(input_path)
            embeddings = np.array([item['embedding'] for item in data['embeddings']], dtype=np.float32) if data['embeddings'] else np.array([])
            metadata = [{
                'chunk_id': item['chunk_id'],
                'source_file': item['source_file'],
                'metadata': item['metadata'],
                'content': item['content'],
                'token_count': item['token_count']
            } for item in data['embeddings']] if data['embeddings'] else []
            statistics = data.get('statistics', {})
            logging.info(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1] if embeddings.size else 0} for {self.domain}")
            return {"embeddings": embeddings, "metadata": metadata, "statistics": statistics}
        except Exception as e:
            logging.error(f"Error loading {input_path} for {self.domain}: {e}")
            return {"embeddings": np.array([]), "metadata": [], "statistics": {}}

    def select_index_type(self, num_vectors: int) -> faiss.Index:
        self.embedding_dim = 384  # Matches BAAI/bge-small-en-v1.5 dimension
        metric_type = faiss.METRIC_INNER_PRODUCT if self.metric == "inner_product" else faiss.METRIC_L2

        if num_vectors < 1000:
            logging.info(f"Using IndexFlatL2 for {self.domain} (small dataset: {num_vectors} vectors)")
            index = faiss.IndexFlat(self.embedding_dim, metric_type)
            return index

        nlist = min(max(self.nlist_factor, int(np.sqrt(num_vectors))), num_vectors)
        nlist = min(nlist, num_vectors // 39) if num_vectors // 39 > 0 else 1

        if self.index_type == "IndexIVFPQ" or num_vectors > 1_000_000:
            logging.info(f"Using IndexIVFPQ for {self.domain} with nlist={nlist}")
            quantizer = faiss.IndexFlat(self.embedding_dim, metric_type)
            index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, 8, 8)
        else:
            logging.info(f"Using IndexIVFFlat for {self.domain} with nlist={nlist}")
            quantizer = faiss.IndexFlat(self.embedding_dim, metric_type)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        return index

    def normalize_vectors(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms

    def build_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        if embeddings.size == 0:
            logging.error(f"No embeddings provided for {self.domain}")
            return {"index": None, "statistics": {}}

        start_time = time.time()
        num_vectors = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
        self.metadata = metadata

        embeddings = self.normalize_vectors(embeddings)
        self.index = self.select_index_type(num_vectors)
        
        if self.device == 'cuda' and num_vectors > 1000:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logging.info(f"Using GPU for FAISS indexing in {self.domain}")
            except Exception as e:
                logging.warning(f"Failed to use GPU in {self.domain}: {e}. Falling back to CPU.")
        
        if not self.index.is_trained and num_vectors >= 39:
            logging.info(f"Training FAISS index for {self.domain}")
            self.index.train(embeddings)
        
        batch_size = self.config['processing']['batch_size']
        for i in tqdm(range(0, num_vectors, batch_size), desc=f"Adding vectors to index for {self.domain}"):
            batch = embeddings[i:i + batch_size]
            self.index.add(batch)

        self.index.nprobe = self.nprobe

        processing_time = time.time() - start_time
        stats = {
            'total_vectors': num_vectors,
            'embedding_dimension': self.embedding_dim,
            'index_type': self.index_type,
            'nlist': getattr(self.index, 'nlist', 1),
            'nprobe': self.index.nprobe,
            'processing_time_seconds': round(processing_time, 2)
        }
        return {"index": self.index, "statistics": stats}

    def validate_index(self, embeddings: np.ndarray) -> Dict[str, float]:
        if embeddings.size == 0 or not self.index:
            logging.warning(f"Cannot validate empty index or embeddings for {self.domain}")
            return {'recall_at_1': 0.0, 'average_query_time_ms': 0.0}

        np.random.seed(42)
        sample_indices = np.random.choice(len(embeddings), min(10, len(embeddings)), replace=False)
        sample_embeddings = embeddings[sample_indices]

        recall_at_1 = 0
        query_times = []
        
        for i, query_embedding in enumerate(sample_embeddings):
            start_time = time.time()
            distances, indices = self.index.search(query_embedding.reshape(1, -1), k=1)
            query_time = (time.time() - start_time) * 1000
            query_times.append(query_time)
            if indices[0][0] == sample_indices[i]:
                recall_at_1 += 1

        recall_at_1 = recall_at_1 / len(sample_embeddings) if sample_embeddings.size else 0
        avg_query_time = np.mean(query_times) if query_times else 0
        
        logging.info(f"Validation for {self.domain}: Recall@1 = {recall_at_1:.2f}, Avg query time = {avg_query_time:.2f} ms")
        return {'recall_at_1': recall_at_1, 'average_query_time_ms': round(avg_query_time, 2)}

    def save_index(self, statistics: Dict[str, Any]):
        index_path = f"{self.output_dir}/faiss_index.index"
        manifest_path = f"{self.output_dir}/index_manifest.json"

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.index') as temp_file:
                faiss.write_index(self.index, temp_file.name)
                bucket_name, blob_name = self.parse_gcs_path(index_path)
                blob = self.bucket.blob(blob_name)
                blob.upload_from_filename(temp_file.name, content_type='application/octet-stream')
                logging.info(f"Saved FAISS index to {index_path} for {self.domain}")
            os.unlink(temp_file.name)

            manifest = {
                'index_config': {
                    'index_type': self.index_type,
                    'nlist': getattr(self.index, 'nlist', 1),
                    'nprobe': self.nprobe,
                    'embedding_dimension': self.embedding_dim
                },
                'embedding_model': self.embedding_model,
                'dataset_statistics': statistics,
                'creation_timestamp': datetime.utcnow().isoformat(),
                'metadata_mapping': self.metadata,
                'validation_metrics': self.validate_index(statistics.get('embeddings', np.array([])))
            }
            gcs_write_json(manifest, manifest_path)
        except Exception as e:
            logging.error(f"Error saving index or manifest for {self.domain}: {e}")

    def parse_gcs_path(self, gcs_path: str) -> tuple:
        """Parse GCS path into bucket name and blob name."""
        if not gcs_path.startswith('gs://'):
            raise ValueError(f"Invalid GCS path: {gcs_path}")
        parts = gcs_path.replace('gs://', '').split('/', 1)
        if len(parts) < 2:
            raise ValueError(f"Invalid GCS path: {gcs_path}")
        return parts[0], parts[1]