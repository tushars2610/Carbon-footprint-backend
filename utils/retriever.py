from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
import torch
import re
from scripts.utils import gcs_read_json, load_config, get_gcs_client
from google.cloud import storage
from dotenv import load_dotenv
import os
import logging
from functools import lru_cache
import tempfile
from collections import Counter
import string

# Load environment variables
load_dotenv()
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

class IndexNotFoundError(Exception):
    """Raised when an index (e.g., FAISS index) is not found."""
    pass

class EnhancedBM25Tokenizer:
    """Enhanced tokenizer for better handling of structured documents."""
    
    def __init__(self):
        # Patterns for structured content
        self.field_patterns = [
            r'\*\*([^*]+):\s*([^*\n]+)\*\*',  # **Field: Value**
            r'([A-Z][A-Za-z\s]+):\s*([^\n]+)',  # Field: Value
            r'(\w+)\s*=\s*([^\n,;]+)',  # Field = Value
        ]
    
    def tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization that preserves structured information."""
        # Convert to lowercase for processing
        text_lower = text.lower()
        
        # Extract structured fields and values
        structured_tokens = []
        for pattern in self.field_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    field, value = match
                    # Add field and value as separate tokens
                    structured_tokens.extend([
                        field.strip().lower(),
                        value.strip().lower()
                    ])
                    # Also add combined tokens for better matching
                    field_clean = re.sub(r'[^\w\s]', ' ', field.strip().lower())
                    value_clean = re.sub(r'[^\w\s]', ' ', value.strip().lower())
                    structured_tokens.extend(field_clean.split())
                    structured_tokens.extend(value_clean.split())
        
        # Standard tokenization
        # Remove special characters but keep alphanumeric and spaces
        text_clean = re.sub(r'[^\w\s]', ' ', text_lower)
        standard_tokens = text_clean.split()
        
        # Combine all tokens and remove duplicates while preserving order
        all_tokens = []
        seen = set()
        for token in structured_tokens + standard_tokens:
            token = token.strip()
            if token and len(token) > 1 and token not in seen:
                all_tokens.append(token)
                seen.add(token)
        
        return all_tokens

@lru_cache(maxsize=10)
def get_domain_retriever(domain: str) -> FAISS:
    """
    Load or create a FAISS retriever for a specific domain.
    
    Args:
        domain: The domain name.
    
    Returns:
        FAISS retriever instance.
    
    Raises:
        IndexNotFoundError: If the FAISS index or manifest file is not found in GCS.
        ValueError: If the manifest file is invalid or missing required keys.
    """
    try:
        config_path = f"gs://{GCS_BUCKET_NAME}/scripts/config/{domain}.yaml"
        config = load_config(config_path)
        model_name = config.get('embedding', {}).get('model_name', 'BAAI/bge-small-en-v1.5')
        
        # Define GCS paths
        index_path = f"gs://{GCS_BUCKET_NAME}/data/{domain}/indices/faiss_index.index"
        manifest_path = f"gs://{GCS_BUCKET_NAME}/data/{domain}/indices/index_manifest.json"
        
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        
        # Check if FAISS index exists
        index_blob = bucket.get_blob(index_path.replace(f"gs://{GCS_BUCKET_NAME}/", ''))
        if not index_blob:
            raise IndexNotFoundError(f"FAISS index not found at {index_path}. Ensure documents have been processed for domain {domain}.")
        
        # Check if manifest exists
        manifest_blob = bucket.get_blob(manifest_path.replace(f"gs://{GCS_BUCKET_NAME}/", ''))
        if not manifest_blob:
            raise IndexNotFoundError(f"Manifest file not found at {manifest_path}. Ensure documents have been processed for domain {domain}.")
        
        # Download FAISS index to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.index') as temp_file:
            index_blob.download_to_filename(temp_file.name)
            faiss_index = faiss.read_index(temp_file.name)
        
        # Clean up temporary file
        os.unlink(temp_file.name)
        
        # Load manifest (JSON file)
        manifest = gcs_read_json(manifest_path)
        if not manifest or 'metadata_mapping' not in manifest:
            raise ValueError(f"Invalid manifest file at {manifest_path}. Expected 'metadata_mapping' key.")
        
        metadata = manifest.get('metadata_mapping', [])
        if not metadata:
            logging.warning(f"No metadata found in manifest for domain {domain}. Returning empty retriever.")
        
        # Initialize embedding model with proper device selection
        device = 'cuda' if torch.cuda.is_available() and config.get('embedding', {}).get('use_gpu', 'auto') != False else 'cpu'
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': config.get('embedding', {}).get('normalize', True)}
        )
        
        # Create FAISS vector store
        docstore = InMemoryDocstore({i: Document(page_content=m['content'], metadata=m) for i, m in enumerate(metadata)})
        index_to_docstore_id = {i: i for i in range(len(metadata))}
        retriever = FAISS(
            embedding_function=embedding_model,
            index=faiss_index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        
        logging.info(f"Loaded FAISS retriever for domain {domain} on device: {device}")
        return retriever
    except IndexNotFoundError as e:
        logging.error(f"Error loading FAISS retriever for domain {domain}: {e}")
        raise
    except ValueError as e:
        logging.error(f"Error loading FAISS retriever for domain {domain}: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading FAISS retriever for domain {domain}: {e}")
        raise

def deduplicate_documents(documents: List[Document]) -> List[Document]:
    """
    Deduplicate a list of Document objects based on chunk_id or page_content.
    
    Args:
        documents: List of Document objects.
    
    Returns:
        Deduplicated list of Document objects.
    """
    seen_ids = set()
    deduped = []
    for doc in documents:
        identifier = doc.metadata.get('chunk_id', doc.page_content)
        if identifier not in seen_ids:
            seen_ids.add(identifier)
            deduped.append(doc)
    return deduped

def extract_query_terms(query: str) -> List[str]:
    """Extract and expand query terms for better matching."""
    # Convert to lowercase and clean
    query_clean = re.sub(r'[^\w\s]', ' ', query.lower()).strip()
    
    # Split into terms
    terms = [term.strip() for term in query_clean.split() if len(term.strip()) > 1]
    
    # Add variations and related terms
    expanded_terms = set(terms)
    
    # Add partial matches for longer terms
    for term in terms:
        if len(term) > 4:
            # Add prefixes and suffixes
            for i in range(3, len(term)):
                expanded_terms.add(term[:i])
                expanded_terms.add(term[i:])
    
    return list(expanded_terms)

def calculate_term_overlap_score(query: str, document: Document) -> float:
    """Calculate overlap score based on term matching."""
    query_terms = set(extract_query_terms(query))
    doc_terms = set(extract_query_terms(document.page_content))
    
    if not query_terms or not doc_terms:
        return 0.0
    
    # Calculate Jaccard similarity with emphasis on query coverage
    intersection = len(query_terms.intersection(doc_terms))
    union = len(query_terms.union(doc_terms))
    
    # Give more weight to how many query terms are found
    query_coverage = intersection / len(query_terms) if query_terms else 0
    jaccard = intersection / union if union > 0 else 0
    
    # Weighted combination favoring query coverage
    return 0.7 * query_coverage + 0.3 * jaccard

def smart_similarity_threshold(query: str, num_results: int) -> float:
    """Dynamically adjust similarity threshold based on query characteristics."""
    query_length = len(query.split())
    
    # For very short queries (1-2 words), be more lenient
    if query_length <= 2:
        return 0.1
    # For medium queries (3-5 words), moderate threshold
    elif query_length <= 5:
        return 0.3
    # For longer queries, higher threshold
    else:
        return 0.5

def hybrid_search(query: str, domain: str, k: int = 5, similarity_threshold: float = 0.5) -> List[Document]:
    """
    Perform enhanced hybrid search combining FAISS (semantic), BM25 (sparse), and term overlap.
    
    Args:
        query: The user query.
        domain: The domain name.
        k: Number of documents to return.
        similarity_threshold: Minimum similarity score for returned documents.
    
    Returns:
        List of retrieved documents.
    """
    try:
        # Get FAISS retriever
        faiss_retriever = get_domain_retriever(domain)
        
        # Get documents for BM25
        metadata = gcs_read_json(f"gs://{GCS_BUCKET_NAME}/data/{domain}/indices/index_manifest.json").get('metadata_mapping', [])
        documents = [Document(page_content=m['content'], metadata=m) for m in metadata]
        
        if not documents:
            logging.info(f"No documents available for domain {domain}")
            return []
        
        # Enhanced BM25 retrieval with custom tokenizer
        tokenizer = EnhancedBM25Tokenizer()
        bm25_retriever = BM25Retriever.from_documents(
            documents, 
            preprocess_func=tokenizer.tokenize
        )
        bm25_retriever.k = min(k * 3, len(documents))  # Retrieve more for better reranking
        
        # Adjust similarity threshold dynamically
        dynamic_threshold = smart_similarity_threshold(query, k)
        effective_threshold = min(similarity_threshold, dynamic_threshold)
        
        # Perform FAISS search with more candidates
        faiss_results = faiss_retriever.similarity_search_with_score(query, k=min(k * 3, len(documents)))
        
        # Convert FAISS results to Document objects with similarity scores
        faiss_docs = []
        for doc, score in faiss_results:
            # Convert distance to similarity (FAISS typically returns distances)
            # Ensure score is converted to Python float
            score_float = float(score) if hasattr(score, 'item') else float(score)
            similarity_score = max(0.0, 1.0 - score_float) if score_float <= 1.0 else 1.0 / (1.0 + score_float)
            doc_copy = Document(
                page_content=doc.page_content, 
                metadata={**doc.metadata, 'faiss_score': float(similarity_score)}
            )
            faiss_docs.append(doc_copy)
        
        # Perform BM25 search
        try:
            bm25_results = bm25_retriever.invoke(query)
            # Add BM25 scores
            for doc in bm25_results:
                doc.metadata['bm25_score'] = 1.0  # BM25Retriever doesn't return scores directly
        except Exception as e:
            logging.warning(f"BM25 search failed: {e}")
            bm25_results = []
        
        # Calculate term overlap scores for all documents
        all_candidates = deduplicate_documents(faiss_docs + bm25_results)
        
        # Add term overlap scores
        for doc in all_candidates:
            overlap_score = calculate_term_overlap_score(query, doc)
            doc.metadata['overlap_score'] = float(overlap_score)
        
        # Combine scores with weights
        for doc in all_candidates:
            faiss_score = float(doc.metadata.get('faiss_score', 0.0))
            bm25_score = float(doc.metadata.get('bm25_score', 0.0))
            overlap_score = float(doc.metadata.get('overlap_score', 0.0))
            
            # Weighted combination - emphasize term overlap for factual queries
            combined_score = (
                0.3 * faiss_score +      # Semantic similarity
                0.3 * (1.0 if bm25_score > 0 else 0.0) +  # BM25 relevance
                0.4 * overlap_score      # Term overlap (most important for factual queries)
            )
            doc.metadata['combined_score'] = float(combined_score)
        
        # Sort by combined score
        ranked_results = sorted(all_candidates, key=lambda x: x.metadata['combined_score'], reverse=True)
        
        # Apply cross-encoder reranking to top candidates
        top_candidates = ranked_results[:min(k * 2, len(ranked_results))]
        if top_candidates:
            reranked_results = rerank_results(query, top_candidates)
        else:
            reranked_results = ranked_results
        
        # Filter by effective threshold and limit to k
        filtered_results = []
        for doc in reranked_results:
            final_score = float(doc.metadata.get('similarity_score', doc.metadata.get('combined_score', 0.0)))
            if final_score >= effective_threshold:
                doc.metadata['final_score'] = float(final_score)
                filtered_results.append(doc)
            if len(filtered_results) >= k:
                break
        
        # If no results meet threshold, return top results anyway (for very restrictive thresholds)
        if not filtered_results and reranked_results:
            filtered_results = reranked_results[:min(k, 3)]  # Return at least top 3 or k, whichever is smaller
            for doc in filtered_results:
                final_score = float(doc.metadata.get('similarity_score', doc.metadata.get('combined_score', 0.0)))
                doc.metadata['final_score'] = float(final_score)
        
        logging.info(f"Retrieved {len(filtered_results)} documents for query '{query}' in domain {domain}")
        logging.debug(f"Used effective threshold: {effective_threshold}")
        
        # Ensure all results are JSON serializable before returning
        return ensure_json_serializable(filtered_results)
        
    except Exception as e:
        logging.error(f"Error in hybrid search for domain {domain}: {e}")
        raise

def rerank_results(query: str, documents: List[Document]) -> List[Document]:
    """
    Rerank retrieved documents using a cross-encoder for improved relevance.
    
    Args:
        query: The user query.
        documents: List of retrieved documents.
    
    Returns:
        Reranked list of documents with similarity scores in metadata.
    """
    if not documents:
        return documents
        
    try:
        # Use a lighter cross-encoder for better performance
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')
        
        # Prepare pairs for cross-encoder
        pairs = []
        for doc in documents:
            # Truncate very long documents for cross-encoder
            content = doc.page_content[:512] if len(doc.page_content) > 512 else doc.page_content
            pairs.append([query, content])
        
        # Get cross-encoder scores
        scores = cross_encoder.predict(pairs)
        
        # Combine with existing scores
        for doc, ce_score in zip(documents, scores):
            existing_score = float(doc.metadata.get('combined_score', 0.0))
            # Weighted combination of existing score and cross-encoder score
            # Convert numpy types to Python float
            ce_score_float = float(ce_score) if hasattr(ce_score, 'item') else float(ce_score)
            final_score = 0.6 * ce_score_float + 0.4 * existing_score
            doc.metadata['similarity_score'] = float(final_score)
        
        # Sort by final similarity score
        return sorted(documents, key=lambda x: x.metadata['similarity_score'], reverse=True)
        
    except Exception as e:
        logging.error(f"Error reranking documents: {e}")
        # Fallback: sort by combined_score if available
        try:
            return sorted(documents, key=lambda x: x.metadata.get('combined_score', 0.0), reverse=True)
        except:
            return documents  # Return original documents if all else fails

def ensure_json_serializable(documents: List[Document]) -> List[Document]:
    """
    Ensure all document metadata is JSON serializable by converting numpy types to Python types.
    
    Args:
        documents: List of Document objects.
    
    Returns:
        List of Document objects with JSON-serializable metadata.
    """
    for doc in documents:
        # Convert all metadata values to JSON-serializable types
        clean_metadata = {}
        for key, value in doc.metadata.items():
            if hasattr(value, 'item'):  # numpy scalar
                clean_metadata[key] = float(value.item())
            elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                clean_metadata[key] = float(value)
            elif isinstance(value, np.ndarray):
                clean_metadata[key] = value.tolist()
            else:
                clean_metadata[key] = value
        
        doc.metadata = clean_metadata
    
    return documents