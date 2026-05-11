"""
Evaluation Module for Amharic IR System

Computes evaluation metrics:
- Precision, Recall, F1 Score
- Average Precision (AP)
- Mean Average Precision (MAP)
"""

import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tokenizer import Tokenizer
from stopwords import StopwordFilter
from lexicon import Lexicon
from stemmer import Stemmer


class Evaluator:
    """Evaluates IR system performance."""
    
    def __init__(self, retriever):
        """
        Initialize evaluator.
        
        Args:
            retriever: Retriever object for querying
        """
        self.retriever = retriever
        self.tokenizer = Tokenizer()
        self.stopword_filter = StopwordFilter("data/stopwords.txt")
        self.lexicon = Lexicon("data/dictionary.txt")
        self.stemmer = Stemmer()
    
    def evaluate_query(self, query_text, relevant_docs):
        """
        Evaluate a single query.
        
        Args:
            query_text: Query text
            relevant_docs: List of relevant document IDs
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get ranked results
        results = self.retriever.process_query(query_text)
        retrieved_docs = [doc_id for doc_id, _ in results]
        
        # Convert to sets for comparison
        relevant_set = set(relevant_docs)
        retrieved_set = set(retrieved_docs)
        
        # Calculate true positives, false positives, false negatives
        true_positives = len(relevant_set & retrieved_set)
        false_positives = len(retrieved_set - relevant_set)
        false_negatives = len(relevant_set - retrieved_set)
        
        total_retrieved = len(retrieved_docs)
        total_relevant = len(relevant_docs)
        
        # Precision: TP / (TP + FP)
        precision = true_positives / total_retrieved if total_retrieved > 0 else 0
        
        # Recall: TP / (TP + FN)
        recall = true_positives / total_relevant if total_relevant > 0 else 0
        
        # F1 Score: 2 * (precision * recall) / (precision + recall)
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        
        # Average Precision: sum of precision at each relevant doc / num relevant
        ap = self._compute_average_precision(results, relevant_set)
        
        return {
            'query': query_text,
            'relevant_docs': relevant_docs,
            'retrieved_docs': retrieved_docs,
            'true_positives': true_positives,
            'total_retrieved': total_retrieved,
            'total_relevant': total_relevant,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'average_precision': ap
        }
    
    def _compute_average_precision(self, ranked_results, relevant_set):
        """
        Compute average precision.
        AP = sum(precision@k for each relevant doc) / num_relevant
        
        Args:
            ranked_results: List of (doc_id, score) tuples
            relevant_set: Set of relevant document IDs
            
        Returns:
            Average precision score
        """
        if not relevant_set:
            return 0.0
        
        ap = 0.0
        num_relevant_found = 0
        
        for rank, (doc_id, _) in enumerate(ranked_results, 1):
            if doc_id in relevant_set:
                num_relevant_found += 1
                precision_at_k = num_relevant_found / rank
                ap += precision_at_k
        
        return ap / len(relevant_set) if relevant_set else 0.0
    
    def evaluate_multiple_queries(self, test_queries):
        """
        Evaluate multiple queries.
        
        Args:
            test_queries: Dictionary {query_text: [relevant_doc_ids]}
            
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        map_total = 0.0
        
        for query_text, relevant_docs in test_queries.items():
            result = self.evaluate_query(query_text, relevant_docs)
            results[query_text] = result
            map_total += result['average_precision']
        
        # Compute Mean Average Precision
        num_queries = len(test_queries)
        map_score = map_total / num_queries if num_queries > 0 else 0.0
        
        return {
            'queries': results,
            'map': map_score,
            'num_queries': num_queries
        }
    
    def print_evaluation_report(self, eval_results):
        """
        Print formatted evaluation report.
        
        Args:
            eval_results: Results from evaluate_multiple_queries
        """
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        # Print individual query results
        for query_text, result in eval_results['queries'].items():
            print(f"\nQuery: {query_text}")
            print(f"  Relevant docs: {result['relevant_docs']}")
            print(f"  Retrieved docs: {result['retrieved_docs'][:5]}"
                  f"{'...' if len(result['retrieved_docs']) > 5 else ''}")
            print(f"  Precision: {result['precision']:.4f}")
            print(f"  Recall: {result['recall']:.4f}")
            print(f"  F1 Score: {result['f1']:.4f}")
            print(f"  Average Precision: {result['average_precision']:.4f}")
        
        # Print aggregate metrics
        print("\n" + "-" * 70)
        print(f"Mean Average Precision (MAP): {eval_results['map']:.4f}")
        print(f"Total Queries Evaluated: {eval_results['num_queries']}")
        print("=" * 70)
    
    def save_evaluation_results(self, eval_results, output_path="output/evaluation_results.json"):
        """
        Save evaluation results to JSON.
        
        Args:
            eval_results: Results from evaluate_multiple_queries
            output_path: Path to save results
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert results for JSON serialization
        json_data = {
            'map': float(eval_results['map']),
            'num_queries': eval_results['num_queries'],
            'queries': {}
        }
        
        for query_text, result in eval_results['queries'].items():
            json_data['queries'][query_text] = {
                'relevant_docs': result['relevant_docs'],
                'retrieved_docs': result['retrieved_docs'],
                'true_positives': result['true_positives'],
                'total_retrieved': result['total_retrieved'],
                'total_relevant': result['total_relevant'],
                'precision': float(result['precision']),
                'recall': float(result['recall']),
                'f1': float(result['f1']),
                'average_precision': float(result['average_precision'])
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Evaluation results saved to {output_path}")
