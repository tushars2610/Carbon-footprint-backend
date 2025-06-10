# scripts/embed.py
from google.cloud import storage
import json
import logging
import pickle
import time
from typing import List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
from .utils import setup_logging, gcs_read_json, gcs_write_pickle, gcs_read_pickle
from dotenv import load_dotenv
import os
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

class EmbeddingGenerator:
    def __init__(self, domain: str, config: Dict[str, Any], base_dir: str):
        self.domain = domain
        self.input_dir = f"{base_dir}/data/{domain}/chunks"
        self.output_dir = f"{base_dir}/data/{domain}/embeddings"
        self.model_name = config['embedding']['model_name']
        self.batch_size = config['embedding']['batch_size']
        self.normalize = config['embedding']['normalize']
        self.use_gpu = config['embedding']['use_gpu']
        self.device = 'cuda' if (self.use_gpu == 'auto' and torch.cuda.is_available()) or self.use_gpu == True else 'cpu'
        self.model = None
        self.client = self.client = get_gcs_client()  # Use the helper function
        self.bucket = self.client.bucket(GCS_BUCKET_NAME)
        setup_logging(config)
        logging.info(f"Using device: {self.device} for {self.domain}")

    def initialize_model(self):
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.model.eval()
            logging.info(f"Initialized model: {self.model_name} for {self.domain}")
        except Exception as e:
            logging.error(f"Error initializing model {self.model_name} for {self.domain}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        try:
            embeddings = []
            for i in tqdm(range(0, len(texts), self.batch_size), desc=f"Generating embeddings for {self.domain}"):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=self.normalize,
                    show_progress_bar=False,
                    device=self.device
                )
                embeddings.append(batch_embeddings)
            return np.vstack(embeddings)
        except Exception as e:
            logging.error(f"Error generating embeddings for {self.domain}: {e}")
            return np.array([])

    def process_chunks(self) -> Dict[str, Any]:
        input_path = f"{self.input_dir}/chunks.json"
        if not self.bucket.blob(input_path.replace(f"gs://{GCS_BUCKET_NAME}/", '')).exists():
            logging.error(f"Chunk file not found at {input_path} for {self.domain}")
            return {"embeddings": [], "statistics": {}}

        try:
            data = gcs_read_json(input_path)
        except Exception as e:
            logging.error(f"Error reading {input_path} for {self.domain}: {e}")
            return {"embeddings": [], "statistics": {}}

        chunks = data.get('chunks', [])
        if not chunks:
            logging.warning(f"No chunks found in input file for {self.domain}")
            return {"embeddings": [], "statistics": {}}

        self.initialize_model()
        texts = [chunk['content'] for chunk in chunks]
        metadata = [{
            'chunk_id': chunk['chunk_id'],
            'source_file': chunk['source_file'],
            'metadata': chunk['metadata'],
            'content': chunk['content'],
            'token_count': chunk['token_count']
        } for chunk in chunks]

        start_time = time.time()
        embeddings = self.generate_embeddings(texts)
        processing_time = time.time() - start_time

        if len(embeddings) != len(chunks):
            logging.error(f"Mismatch: {len(embeddings)} embeddings generated for {len(chunks)} chunks in {self.domain}")
            return {"embeddings": [], "statistics": {}}

        embedding_dim = embeddings.shape[1]
        expected_dim = self.model.get_sentence_embedding_dimension()
        if embedding_dim != expected_dim:
            logging.error(f"Embedding dimension mismatch: got {embedding_dim}, expected {expected_dim} in {self.domain}")
            return {"embeddings": [], "statistics": {}}

        embedding_data = [{
            'chunk_id': meta['chunk_id'],
            'source_file': meta['source_file'],
            'metadata': meta['metadata'],
            'content': meta['content'],
            'token_count': meta['token_count'],
            'embedding': embedding.tolist()
        } for meta, embedding in zip(metadata, embeddings)]

        stats = {
            'total_chunks': len(chunks),
            'embedding_dimension': embedding_dim,
            'processing_time_seconds': round(processing_time, 2),
            'average_tokens_per_chunk': round(np.mean([chunk['token_count'] for chunk in chunks]), 2) if chunks else 0
        }

        return {"embeddings": embedding_data, "statistics": stats}

    def save_embeddings(self, embedding_data: List[Dict[str, Any]], stats: Dict[str, Any]):
        output_path = f"{self.output_dir}/embeddings.pkl"
        gcs_write_pickle({
            'embeddings': embedding_data,
            'statistics': stats
        }, output_path)

    def verify_pkl(self, pkl_path: str) -> bool:
        try:
            data = gcs_read_pickle(pkl_path)
            embeddings = data.get('embeddings', [])
            stats = data.get('statistics', {})
            logging.info(f"PKL verification: Loaded {len(embeddings)} embeddings, stats: {stats} for {self.domain}")
            return True
        except Exception as e:
            logging.error(f"PKL verification failed for {pkl_path} in {self.domain}: {e}")
            return False