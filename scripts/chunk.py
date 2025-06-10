# scripts/chunk.py
from google.cloud import storage
import json
import logging
from transformers import AutoTokenizer
import nltk
import regex as re
from tqdm import tqdm
from typing import List, Dict, Any
from .utils import setup_logging, gcs_list_files, gcs_read_json, gcs_write_json
from dotenv import load_dotenv
from google.oauth2 import service_account
import os
import nltk
nltk.download('punkt_tab')
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
class DocumentChunker:
    def __init__(self, domain: str, config: Dict[str, Any], base_dir: str):
        self.domain = domain
        self.input_dir = f"{base_dir}/data/{domain}/json"
        self.output_dir = f"{base_dir}/data/{domain}/chunks"
        self.chunk_size = min(config['chunking']['chunk_size'], 500)  # Cap at 500 tokens
        self.chunk_overlap = config['chunking']['chunk_overlap']
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.sentence_boundary = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])|(?<=\n)\s*') if config['chunking']['sentence_boundary'] else None
        self.client = get_gcs_client()  # Use the helper function
        self.bucket = self.client.bucket(GCS_BUCKET_NAME)
        setup_logging(config)

    def tokenize_text(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False, max_length=512, truncation=True)

    def detokenize_tokens(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def split_into_sentences(self, text: str) -> List[str]:
        if not self.sentence_boundary:
            return [text]
        preliminary_sentences = self.sentence_boundary.split(text)
        sentences = []
        for sent in preliminary_sentences:
            if sent.strip():
                nltk_sentences = nltk.sent_tokenize(sent)
                sentences.extend([s.strip() for s in nltk_sentences if s.strip()])
        return sentences

    def convert_excel_to_text(self, excel_data: List[Dict[str, Any]]) -> str:
        text_lines = []
        for row in excel_data:
            row_text = " | ".join([f"{key}: {value}" for key, value in row.items() if value])
            text_lines.append(row_text)
        return "\n".join(text_lines)

    def create_chunks(self, text: str, source_file: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_token_count = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_tokens = self.tokenize_text(sentence)
            sentence_token_count = len(sentence_tokens)

            if current_token_count + sentence_token_count > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "chunk_id": f"{source_file.rsplit('.', 1)[0]}_{chunk_id}",
                    "source_file": source_file,
                    "metadata": metadata,
                    "content": chunk_text,
                    "token_count": current_token_count
                })
                overlap_tokens = self.tokenize_text(" ".join(current_chunk))[-self.chunk_overlap:]
                current_chunk = [self.detokenize_tokens(overlap_tokens)]
                current_token_count = len(overlap_tokens)
                chunk_id += 1

            current_chunk.append(sentence)
            current_token_count += sentence_token_count

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if current_token_count > self.chunk_size:
                tokens = self.tokenize_text(chunk_text)
                for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
                    chunk_tokens = tokens[i:i + self.chunk_size]
                    if chunk_tokens:
                        chunk_text = self.detokenize_tokens(chunk_tokens)
                        chunks.append({
                            "chunk_id": f"{source_file.rsplit('.', 1)[0]}_{chunk_id}",
                            "source_file": source_file,
                            "metadata": metadata,
                            "content": chunk_text,
                            "token_count": len(chunk_tokens)
                        })
                        chunk_id += 1
            else:
                chunks.append({
                    "chunk_id": f"{source_file.rsplit('.', 1)[0]}_{chunk_id}",
                    "source_file": source_file,
                    "metadata": metadata,
                    "content": chunk_text,
                    "token_count": current_token_count
                })

        return chunks

    def process_json_file(self, json_path: str) -> List[Dict[str, Any]]:
        try:
            data = gcs_read_json(json_path)
        except Exception as e:
            logging.error(f"Error reading {json_path}: {e}")
            return []

        metadata = data.get('metadata', {})
        content = data.get('content', '')

        if isinstance(content, list):
            text = self.convert_excel_to_text(content)
        else:
            text = content

        if not text:
            logging.warning(f"No content extracted from {json_path}")
            return []

        return self.create_chunks(text, json_path.rsplit('/', 1)[-1], metadata)

    def chunk_directory(self, skip_existing: bool = False) -> Dict[str, Any]:
        all_chunks = []
        stats = {"total_files": 0, "total_chunks": 0, "total_tokens": 0}

        json_files = gcs_list_files(GCS_BUCKET_NAME, f"data/{self.domain}/json/")
        stats['total_files'] = len(json_files)
        output_path = f"{self.output_dir}/chunks.json"

        if skip_existing and self.bucket.blob(output_path.replace(f"gs://{GCS_BUCKET_NAME}/", '')).exists():
            logging.info(f"Skipping chunking for {self.domain}: {output_path} exists")
            return stats

        for json_file in tqdm(json_files, desc=f"Chunking JSON files for {self.domain}"):
            if json_file.endswith('.json'):
                chunks = self.process_json_file(json_file)
                all_chunks.extend(chunks)
                stats['total_chunks'] += len(chunks)
                stats['total_tokens'] += sum(chunk['token_count'] for chunk in chunks)

        gcs_write_json({
            "chunks": all_chunks,
            "statistics": stats
        }, output_path)
        return stats