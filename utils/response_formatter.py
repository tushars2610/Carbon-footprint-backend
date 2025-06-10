from langchain.docstore.document import Document
from typing import List, Dict, Any
import re
import logging

def format_response(query: str, documents: List[Document], llm_response: str) -> Dict[str, Any]:
    """
    Format the retrieval results and LLM response into a user-friendly structure.
    
    Args:
        query: The user query.
        documents: List of retrieved documents.
        llm_response: The LLM-generated response.
    
    Returns:
        Formatted response dictionary.
    """
    try:
        # Format retrieved documents
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            formatted_docs.append({
                "document_id": doc.metadata.get('chunk_id', f"doc_{i}"),
                "source_file": doc.metadata.get('source_file', 'unknown'),
                "content": doc.page_content,
                "similarity_score": doc.metadata.get('similarity_score', 0.0),
                "metadata": doc.metadata.get('metadata', {})
            })
        
        # Format LLM response (handle code blocks, lists, etc.)
        formatted_response = llm_response
        # Convert markdown-like code blocks to structured format
        if "```" in llm_response:
            formatted_response = re.sub(r'```(\w+)?\n(.*?)\n```', r'[Code Block]\n\2\n[/Code Block]', llm_response, flags=re.DOTALL)
        
        # Convert markdown lists to structured format
        formatted_response = re.sub(r'^- (.*?)$', r'[List Item] \1', formatted_response, flags=re.MULTILINE)
        
        return {
            "query": query,
            "retrieved_documents": formatted_docs,
            "response": formatted_response,
            "document_count": len(formatted_docs)
        }
    except Exception as e:
        logging.error(f"Error formatting response: {e}")
        return {
            "query": query,
            "retrieved_documents": [],
            "response": "Error formatting response.",
            "document_count": 0
        }