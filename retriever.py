"""
Retrieval Engine for Amharic IR System

Processes queries and ranks documents using TF-IDF and cosine similarity.
"""

import os
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tokenizer import Tokenizer
from stopwords import StopwordFilter
from lexicon import Lexicon
from stemmer import Stemmer
from vectorizer import TFIDFVectorizer


class Retriever:
    """Retrieves and ranks documents for Amharic queries."""
    
    def __init__(self, index_path="output/index.json", stopwords_file="data/stopwords.txt",
                 dictionary_file="data/dictionary.txt"):
        """
        Initialize the retriever.
        
        Args:
            index_path: Path to the inverted index JSON file
            stopwords_file: Path to stopwords file
            dictionary_file: Path to dictionary file
        """
        # Load index
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found at {index_path}")
        
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        self.index = index_data['index']
        self.doc_metadata = index_data['doc_metadata']
        
        # Initialize text processing modules
        self.tokenizer = Tokenizer()
        self.stopword_filter = StopwordFilter(stopwords_file)
        self.lexicon = Lexicon(dictionary_file)
        self.stemmer = Stemmer()
        
        # Initialize vectorizer
        self.vectorizer = TFIDFVectorizer(self.index, self.doc_metadata)
    
    def _get_stem(self, token):
        """
        Get stem for a token (lexicon-first approach).
        
        Args:
            token: Token to stem
            
        Returns:
            Stem of the token
        """
        # Try lexicon first
        root = self.lexicon.get_root(token)
        if root:
            return root
        
        # Fall back to rule-based stemmer
        return self.stemmer.stem(token)
    
    def process_query(self, query_text):
        """
        Process a query and return ranked documents.
        
        Args:
            query_text: Raw Amharic query text
            
        Returns:
            List of (doc_id, similarity_score) tuples, sorted by descending score
        """
        # Tokenize query
        tokens = self.tokenizer.tokenize(query_text)
        
        # Remove stopwords
        tokens = self.stopword_filter.filter(tokens)
        
        # Stem tokens
        stems = [self._get_stem(token) for token in tokens]
        
        # Build query vector
        query_vector = self.vectorizer.build_query_vector(stems)
        
        if not query_vector:
            # Query has no relevant terms
            return []
        
        # Compute similarities for all documents
        results = []
        for doc_id in self.doc_metadata.keys():
            similarity = self.vectorizer.cosine_similarity(query_vector, doc_id)
            if similarity > 0:  # Only include non-zero similarities
                results.append((doc_id, similarity))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_document_info(self, doc_id):
        """
        Get metadata for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dictionary with document info (name, length, vector)
        """
        if doc_id not in self.doc_metadata:
            return None
        
        return {
            'name': self.doc_metadata[doc_id]['name'],
            'length': self.doc_metadata[doc_id]['length'],
            'vector': self.vectorizer.get_doc_vector(doc_id)
        }
    
    def get_all_documents(self):
        """Get all document metadata."""
        return self.doc_metadata
    
    def get_index(self):
        """Get the inverted index."""
        return self.index
    
    def get_vectorizer(self):
        """Get the vectorizer for evaluation."""
        return self.vectorizer
