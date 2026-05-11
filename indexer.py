"""
Inverted Index Builder for Amharic IR System

Builds an inverted index from corpus documents:
- Tokenizes text
- Removes stopwords
- Stems tokens (lexicon-based with fallback to rule-based stemmer)
- Counts term frequencies per document
- Stores document metadata
"""

import os
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tokenizer import Tokenizer
from stopwords import StopwordFilter
from lexicon import Lexicon
from stemmer import Stemmer


class Indexer:
    """Builds inverted index from Amharic corpus."""
    
    def __init__(self, corpus_dir="data/corpus", stopwords_file="data/stopwords.txt", 
                 dictionary_file="data/dictionary.txt"):
        """
        Initialize the indexer.
        
        Args:
            corpus_dir: Path to corpus directory
            stopwords_file: Path to stopwords file
            dictionary_file: Path to dictionary file for lexicon
        """
        self.corpus_dir = corpus_dir
        self.tokenizer = Tokenizer()
        self.stopword_filter = StopwordFilter(stopwords_file)
        self.lexicon = Lexicon(dictionary_file)
        self.stemmer = Stemmer()
        
        # Index structure: {term: {doc_id: tf, ...}}
        self.index = {}
        
        # Document metadata: {doc_id: {name, length}}
        self.doc_metadata = {}
    
    def _get_stem(self, token):
        """
        Get stem for a token using lexicon-first approach.
        
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
    
    def build_index(self):
        """
        Build inverted index from corpus documents.
        
        Returns:
            True if successful, False otherwise
        """
        corpus_path = Path(self.corpus_dir)
        
        if not corpus_path.exists():
            print(f"Corpus directory not found: {self.corpus_dir}")
            return False
        
        # Get all .txt files sorted
        doc_files = sorted(corpus_path.glob("*.txt"))
        
        if not doc_files:
            print(f"No .txt files found in {self.corpus_dir}")
            return False
        
        print(f"Building index from {len(doc_files)} documents...")
        print()
        
        for doc_path in doc_files:
            doc_id = doc_path.stem  # e.g., "doc1"
            doc_name = doc_path.name
            
            try:
                # Read document
                with open(doc_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                print(f"Processing {doc_name}...", end=" ")
                
                # Tokenize
                tokens = self.tokenizer.tokenize(text)
                
                # Remove stopwords
                tokens = self.stopword_filter.filter(tokens)
                
                # Stem tokens and count frequencies
                term_freq = {}
                for token in tokens:
                    stem = self._get_stem(token)
                    term_freq[stem] = term_freq.get(stem, 0) + 1
                
                # Update index
                for term, freq in term_freq.items():
                    if term not in self.index:
                        self.index[term] = {}
                    self.index[term][doc_id] = freq
                
                # Store document metadata
                self.doc_metadata[doc_id] = {
                    'name': doc_name,
                    'length': len(tokens)  # token count after stopword removal
                }
                
                print(f"✓ ({len(tokens)} tokens, {len(term_freq)} unique terms)")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                return False
        
        print()
        print(f"Index built successfully!")
        print(f"Total terms: {len(self.index)}")
        print(f"Total documents: {len(self.doc_metadata)}")
        
        return True
    
    def save_index(self, output_path="output/index.json"):
        """
        Save index to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            'index': self.index,
            'doc_metadata': self.doc_metadata,
            'total_docs': len(self.doc_metadata),
            'total_terms': len(self.index)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Index saved to {output_path}")
    
    def get_index(self):
        """Get the inverted index."""
        return self.index
    
    def get_doc_metadata(self):
        """Get document metadata."""
        return self.doc_metadata
