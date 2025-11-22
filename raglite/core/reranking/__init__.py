"""
Reranking components for improving document ranking quality in RAG systems.

This module provides cross-encoder based reranking to improve the relevance
of retrieved documents beyond initial semantic similarity search.
"""

from typing import List, Dict, Any, Optional
import logging
import os
# from sentence_transformers import CrossEncoder  # Moved to lazy import in _load_model
import numpy as np

logger = logging.getLogger(__name__)

class Reranker:
    """Base class for document reranking"""

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: The search query
            documents: List of document dictionaries with content and metadata
            top_k: Number of top documents to return

        Returns:
            Reranked list of documents
        """
        raise NotImplementedError("Subclasses must implement rerank method")


class CrossEncoderReranker(Reranker):
    """Cross-encoder based reranker using sentence-transformers"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model_name = model_name
        self.model = None
        logger.info(f"Initialized CrossEncoderReranker with model: {model_name}")

    def _load_model(self):
        """Lazy load the cross-encoder model"""
        if self.model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            hf_endpoint = os.getenv("HF_ENDPOINT") or os.getenv("HF_HUB_ENDPOINT")
            if hf_endpoint:
                os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
                logger.info(f"Using custom Hugging Face endpoint: {hf_endpoint}")
            try:
                from sentence_transformers import CrossEncoder  # Lazy import
                self.model = CrossEncoder(self.model_name)
                logger.info("Cross-encoder model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model: {e}")
                raise

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder scoring.

        Args:
            query: The search query
            documents: List of document dictionaries from Elasticsearch
            top_k: Number of top documents to return

        Returns:
            Reranked list of documents with updated scores
        """
        if not documents:
            return []

        self._load_model()

        # Extract text content from documents
        texts = []
        for doc in documents:
            content = self._extract_content(doc)
            texts.append(content)

        # Create query-document pairs for cross-encoder
        query_doc_pairs = [[query, text] for text in texts]

        try:
            # Get relevance scores from cross-encoder
            scores = self.model.predict(query_doc_pairs)

            # Add scores to documents and sort
            scored_docs = []
            for doc, score in zip(documents, scores):
                # Create a copy of the document with updated score
                reranked_doc = dict(doc)
                reranked_doc['_score'] = float(score)  # Cross-encoder score
                reranked_doc['_rerank_score'] = float(score)
                reranked_doc['_original_score'] = doc.get('_score', 0.0)
                scored_docs.append(reranked_doc)

            # Sort by rerank score (higher is better for cross-encoder)
            scored_docs.sort(key=lambda x: x['_rerank_score'], reverse=True)

            # Return top_k documents
            result = scored_docs[:top_k]

            logger.info(f"Reranked {len(documents)} documents, returning top {len(result)}")
            return result

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Return original documents if reranking fails
            return documents[:top_k]

    def _extract_content(self, document: Dict[str, Any]) -> str:
        """
        Extract text content from Elasticsearch document.

        Args:
            document: Elasticsearch document dictionary

        Returns:
            Extracted text content
        """
        # Try common content fields
        content_fields = ['content', 'text', 'chunk', 'message', 'content_with_weight']

        if '_source' in document:
            source = document['_source']
            for field in content_fields:
                if field in source and source[field]:
                    content = str(source[field])
                    # Limit content length to avoid token limits
                    return content[:2000] if len(content) > 2000 else content

        # Fallback to string representation
        return str(document)[:2000]


class NoOpReranker(Reranker):
    """No-operation reranker that returns documents unchanged"""

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Return documents unchanged"""
        return documents[:top_k]


def get_reranker(reranker_type: str = "cross_encoder", **kwargs) -> Reranker:
    """
    Factory function to create reranker instances.

    Args:
        reranker_type: Type of reranker ("cross_encoder", "none")
        **kwargs: Additional arguments for reranker initialization

    Returns:
        Configured reranker instance
    """
    if reranker_type == "cross_encoder":
        model_name = kwargs.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        return CrossEncoderReranker(model_name)
    elif reranker_type == "none":
        return NoOpReranker()
    else:
        logger.warning(f"Unknown reranker type: {reranker_type}, using no-op reranker")
        return NoOpReranker()