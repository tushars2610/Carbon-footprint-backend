import json
import re
from typing import Tuple
from google.cloud import storage
from google.oauth2 import service_account
from scripts.utils import gcs_read_json, parse_gcs_path
from dotenv import load_dotenv
import os
import logging

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

def load_validation_configs(domain: str) -> tuple[dict, dict]:
    """
    Load whitelist and blacklist configurations for a domain from GCS.
    
    Args:
        domain: The domain name.
    
    Returns:
        Tuple of whitelist and blacklist dictionaries.
    """
    try:
        client = get_gcs_client()
        whitelist = gcs_read_json(f"gs://{GCS_BUCKET_NAME}/data/{domain}/config/whitelist.json")
        blacklist = gcs_read_json(f"gs://{GCS_BUCKET_NAME}/data/{domain}/config/blacklist.json")
        
        # Add debug logging
        logging.debug(f"Loaded whitelist for {domain}: {whitelist}")
        logging.debug(f"Loaded blacklist for {domain}: {blacklist}")
        
        # Ensure we have dictionaries even if files don't exist or are empty
        whitelist_dict = whitelist if isinstance(whitelist, dict) else {}
        blacklist_dict = blacklist if isinstance(blacklist, dict) else {}
        
        return whitelist_dict, blacklist_dict
    except Exception as e:
        logging.error(f"Error loading validation configs for domain {domain}: {e}")
        return {}, {}

def validate_input(query: str, domain: str) -> Tuple[bool, str]:
    """
    Validate user input against whitelist and blacklist rules.
    
    Args:
        query: The user input query.
        domain: The domain name.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    if not query or not query.strip():
        return False, "Query cannot be empty"
    
    # Add debug logging
    logging.debug(f"Validating query: '{query}' for domain: '{domain}'")
    
    whitelist, blacklist = load_validation_configs(domain)
    
    # Check blacklist patterns and keywords
    blacklist_patterns = blacklist.get("patterns", [])
    blacklist_keywords = blacklist.get("keywords", [])
    
    for pattern in blacklist_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            logging.warning(f"Query rejected due to blacklist pattern match: {pattern}")
            return False, "Please enter a valid input. Your query contains inappropriate content."
    
    for keyword in blacklist_keywords:
        if keyword.lower() in query.lower():
            logging.warning(f"Query rejected due to blacklist keyword: {keyword}")
            return False, "Please enter a valid input. Your query contains inappropriate content."
    
    # Check whitelist (if defined, query must match at least one pattern or keyword)
    whitelist_patterns = whitelist.get("patterns", [])
    whitelist_keywords = whitelist.get("keywords", [])
    
    # IMPORTANT: The whitelist only ENFORCES special patterns or keywords if 
    # there's a whitelist category called "enforce" set to True
    enforce_whitelist = whitelist.get("enforce", False)
    
    if enforce_whitelist and (whitelist_patterns or whitelist_keywords):
        whitelist_match = False
        
        for pattern in whitelist_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                whitelist_match = True
                logging.debug(f"Query matched whitelist pattern: {pattern}")
                break
                
        for keyword in whitelist_keywords:
            if keyword.lower() in query.lower():
                whitelist_match = True
                logging.debug(f"Query matched whitelist keyword: {keyword}")
                break
                
        if not whitelist_match:
            logging.warning(f"Query rejected: does not match whitelist criteria")
            return False, "Please enter a valid input. Your query does not match allowed criteria."
    else:
        # If whitelist is not enforced or no patterns/keywords exist, we consider this a match
        logging.debug("Whitelist not enforced or no patterns/keywords defined, continuing validation")
    
    logging.debug("Query validation successful")
    return True, ""

def sanitize_input(query: str) -> str:
    """
    Sanitize user input by removing excessive whitespace and potentially harmful characters.
    
    Args:
        query: The user input query.
    
    Returns:
        Sanitized query string.
    """
    if not query:
        return ""
    
    # Remove excessive whitespace
    query = ' '.join(query.split())
    
    # Remove potentially harmful characters (e.g., control characters)
    query = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', query)
    
    return query.strip()

def process_query(query_json: dict, domain: str) -> Tuple[bool, str, dict]:
    """
    Process a query JSON object, validating and sanitizing the input.
    
    Args:
        query_json: JSON object containing the query and parameters.
        domain: The domain name.
        
    Returns:
        Tuple of (is_valid, error_message, processed_query).
    """
    try:
        if not isinstance(query_json, dict):
            return False, "Invalid request format", {}
            
        # Extract query text
        query_text = query_json.get("query", "")
        
        # Validate the query
        is_valid, error_message = validate_input(query_text, domain)
        if not is_valid:
            return False, error_message, {}
            
        # Sanitize the query
        sanitized_query = sanitize_input(query_text)
        
        # Create processed query object with sanitized input
        processed_query = {
            "query": sanitized_query,
            "num_results": query_json.get("num_results", 5),
            "similarity_threshold": query_json.get("similarity_threshold", 0.5)
        }
        
        return True, "", processed_query
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return False, f"Error processing query: {str(e)}", {}

# Add a test function to verify validation logic
def test_validation():
    """
    Test function to verify validation logic with different inputs and configurations.
    """
    # Example configurations
    test_configs = [
        {
            "name": "No restrictions",
            "whitelist": {},
            "blacklist": {},
            "query": "i want to know about documents",
            "expected": True
        },
        {
            "name": "With blacklist only",
            "whitelist": {},
            "blacklist": {"keywords": ["prohibited"]},
            "query": "i want to know about documents",
            "expected": True
        },
        {
            "name": "With blacklist rejection",
            "whitelist": {},
            "blacklist": {"keywords": ["documents"]},
            "query": "i want to know about documents",
            "expected": False
        },
        {
            "name": "With non-enforced whitelist",
            "whitelist": {"keywords": ["abc0001"], "enforce": False},
            "blacklist": {},
            "query": "i want to know about documents",
            "expected": True
        },
        {
            "name": "With enforced whitelist rejection",
            "whitelist": {"keywords": ["abc0001"], "enforce": True},
            "blacklist": {},
            "query": "i want to know about documents",
            "expected": False
        },
        {
            "name": "With enforced whitelist pass",
            "whitelist": {"keywords": ["documents"], "enforce": True},
            "blacklist": {},
            "query": "i want to know about documents",
            "expected": True
        }
    ]
    
    # Mock the load_validation_configs function
    original_load_fn = load_validation_configs
    
    for test in test_configs:
        def mock_load_configs(domain):
            return test["whitelist"], test["blacklist"]
        
        # Replace the function temporarily
        globals()["load_validation_configs"] = mock_load_configs
        
        # Run validation
        result, message = validate_input(test["query"], "test-domain")
        
        # Check result
        if result == test["expected"]:
            print(f"✅ Test '{test['name']}' passed")
        else:
            print(f"❌ Test '{test['name']}' failed. Expected {test['expected']}, got {result}. Message: {message}")
    
    # Restore original function
    globals()["load_validation_configs"] = original_load_fn

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Run tests
    test_validation()
    
    # Example query processing
    example_query = {
        "query": "i want to know about documents",
        "num_results": 5,
        "similarity_threshold": 0.5
    }
    
    domain = "example-domain"
    is_valid, error_message, processed_query = process_query(example_query, domain)
    
    if is_valid:
        print(f"Query is valid. Processed query: {processed_query}")
    else:
        print(f"Query is invalid: {error_message}")