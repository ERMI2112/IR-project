"""
TF-IDF Vectorizer and Cosine Similarity for Amharic IR

Computes:
- IDF (Inverse Document Frequency)
- TF-IDF weights for documents
- Query vectors
- Cosine similarity between vectors
"""

import math
from collections import defaultdict


class TFIDFVectorizer:
    """Computes TF-IDF vectors and similarities."""
    
    def __init__(self, index, doc_metadata):
        """
        Initialize vectorizer with precomputed index.
        
        Args:
            index: Inverted index {term: {doc_id: tf}}
            doc_metadata: Document metadata {doc_id: {name, length}}
        """
        self.index = index
        self.doc_metadata = doc_metadata
        self.num_docs = len(doc_metadata)
        
        # Precompute IDF for all terms
        self.idf = self._compute_idf()
        
        # Precompute document vectors
        self.doc_vectors = self._compute_doc_vectors()
    
    def _compute_idf(self):
        """
        Compute IDF for all terms.
        IDF = log10(total_docs / doc_frequency)
        
        Returns:
            Dictionary {term: idf}
        """
        idf_dict = {}
        
        for term, doc_freqs in self.index.items():
            df = len(doc_freqs)  # Number of documents containing the term
            idf = math.log10(self.num_docs / df) if df > 0 else 0
            idf_dict[term] = idf
        
        return idf_dict
    
    def _compute_tf_weight(self, tf):
        """
        Compute TF weight using logarithmic scaling.
        TF_weight = 1 + log10(tf) if tf > 0, else 0
        
        Args:
            tf: Term frequency in a document
            
        Returns:
            TF weight
        """
        if tf > 0:
            return 1 + math.log10(tf)
        return 0
    
    def _compute_doc_vectors(self):
        """
        Precompute TF-IDF vectors for all documents.
        
        Returns:
            Dictionary {doc_id: {term: tfidf_weight}}
        """
        doc_vectors = {}
        
        for term, doc_freqs in self.index.items():
            idf = self.idf[term]
            
            for doc_id, tf in doc_freqs.items():
                if doc_id not in doc_vectors:
                    doc_vectors[doc_id] = {}
                
                # TF-IDF = tf_weight * idf
                tf_weight = self._compute_tf_weight(tf)
                tfidf = tf_weight * idf
                
                doc_vectors[doc_id][term] = tfidf
        
        return doc_vectors
    
    def build_query_vector(self, query_tokens):
        """
        Build TF-IDF vector for a query.
        Uses same TF and IDF computation as documents.
        
        Args:
            query_tokens: List of stemmed query tokens
            
        Returns:
            Query vector {term: tfidf_weight}
        """
        # Count term frequencies in query
        query_tf = defaultdict(int)
        for token in query_tokens:
            query_tf[token] += 1
        
        # Build query vector using same weights as documents
        query_vector = {}
        for term, tf in query_tf.items():
            if term in self.idf:
                tf_weight = self._compute_tf_weight(tf)
                tfidf = tf_weight * self.idf[term]
                query_vector[term] = tfidf
        
        return query_vector
    
    def _compute_vector_magnitude(self, vector):
        """
        Compute magnitude (L2 norm) of a vector.
        ||vec|| = sqrt(sum(weight^2))
        
        Args:
            vector: Dictionary {term: weight}
            
        Returns:
            Magnitude of the vector
        """
        sum_squares = sum(weight ** 2 for weight in vector.values())
        return math.sqrt(sum_squares) if sum_squares > 0 else 0
    
    def _compute_dot_product(self, vec1, vec2):
        """
        Compute dot product of two vectors.
        vec1 · vec2 = sum(weight1 * weight2) for common terms
        
        Args:
            vec1: Dictionary {term: weight}
            vec2: Dictionary {term: weight}
            
        Returns:
            Dot product
        """
        dot_product = 0
        for term in vec1:
            if term in vec2:
                dot_product += vec1[term] * vec2[term]
        return dot_product
    
    def cosine_similarity(self, query_vector, doc_id):
        """
        Compute cosine similarity between query and document.
        similarity = (query · doc) / (||query|| * ||doc||)
        
        Args:
            query_vector: Query TF-IDF vector
            doc_id: Document ID
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        if doc_id not in self.doc_vectors:
            return 0.0
        
        doc_vector = self.doc_vectors[doc_id]
        
        # Compute dot product
        dot_product = self._compute_dot_product(query_vector, doc_vector)
        
        # Compute magnitudes
        query_mag = self._compute_vector_magnitude(query_vector)
        doc_mag = self._compute_vector_magnitude(doc_vector)
        
        # Avoid division by zero
        if query_mag == 0 or doc_mag == 0:
            return 0.0
        
        # Cosine similarity
        similarity = dot_product / (query_mag * doc_mag)
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
    
    def get_doc_vector(self, doc_id):
        """Get precomputed TF-IDF vector for a document."""
        return self.doc_vectors.get(doc_id, {})
    
    def get_all_doc_vectors(self):
        """Get all document vectors."""
        return self.doc_vectors
    
    def get_idf(self, term):
        """Get IDF value for a term."""
        return self.idf.get(term, 0.0)
