from google.cloud import storage
from scripts.utils import gcs_read_file, gcs_write_file
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

def get_system_prompt(domain: str) -> str:
    """
    Retrieve the system prompt for a domain from GCS.
    
    Args:
        domain: The domain name.
    
    Returns:
        The system prompt text, or a default prompt if not found.
    """
    prompt_path = f"gs://{GCS_BUCKET_NAME}/data/{domain}/config/prompt.txt"
    try:
        prompt = gcs_read_file(prompt_path)
        if prompt:
            return prompt
        logging.warning(f"No system prompt found for domain {domain}, using default")
    except Exception as e:
        logging.error(f"Error reading system prompt for domain {domain}: {e}")
    
    # Default prompt
    return (
        "You are a helpful assistant providing accurate and concise answers based on the provided context. "
        "Use the context to answer the user's query clearly and professionally. If the context is insufficient, "
        "state so and provide a general answer based on your knowledge."
    )

def update_system_prompt(domain: str, prompt: str):
    """
    Update the system prompt for a domain in GCS.
    
    Args:
        domain: The domain name.
        prompt: The new system prompt.
    """
    prompt_path = f"gs://{GCS_BUCKET_NAME}/data/{domain}/config/prompt.txt"
    try:
        gcs_write_file(prompt, prompt_path)
        logging.info(f"Updated system prompt for domain {domain}")
    except Exception as e:
        logging.error(f"Error updating system prompt for domain {domain}: {e}")
        raise