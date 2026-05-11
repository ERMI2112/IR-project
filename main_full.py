"""
Main Runner for Complete Amharic IR System

Orchestrates the full IR workflow:
1. Builds the inverted index if needed
2. Provides interactive query interface
3. Displays ranked retrieval results
4. Runs evaluation on test queries after user exits
"""

import os
import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from indexer import Indexer
from retriever import Retriever
from evaluation import Evaluator


def build_index_if_needed():
    """
    Build the inverted index if it doesn't exist.
    
    Returns:
        True if index was built or already exists, False otherwise
    """
    index_path = "output/index.json"
    
    if os.path.exists(index_path):
        print("✓ Index already exists, skipping build...")
        return True
    
    print("Index not found. Building index...")
    print()
    
    indexer = Indexer()
    
    if not indexer.build_index():
        return False
    
    indexer.save_index(index_path)
    return True


def interactive_query_loop(retriever):
    """
    Run an interactive loop where user enters queries and views results.
    
    Args:
        retriever: Retriever object for processing queries
    """
    print("\n" + "=" * 70)
    print("INTERACTIVE RETRIEVAL INTERFACE")
    print("=" * 70)
    print("\nEnter Amharic queries to search the corpus.")
    print("Type 'quit' to exit and run evaluation.")
    print()
    
    while True:
        try:
            query = input("Enter query (or 'quit' to exit): ").strip()
            
            if query.lower() == 'quit':
                break
            
            if not query:
                print("Please enter a non-empty query.\n")
                continue
            
            # Process query
            print(f"\nProcessing query: {query}")
            results = retriever.process_query(query)
            
            if not results:
                print("No results found.\n")
                continue
            
            # Display results
            print("\nRetrieval Results (ranked by relevance):")
            print("-" * 70)
            for rank, (doc_id, score) in enumerate(results, 1):
                doc_info = retriever.get_document_info(doc_id)
                print(f"{rank:2d}. {doc_id:6s} ({doc_info['name']:15s}) "
                      f"Similarity: {score:.4f}")
            print()
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error processing query: {e}\n")


def run_evaluation(retriever):
    """
    Run evaluation on predefined test queries.
    
    Args:
        retriever: Retriever object for querying
    """
    evaluator = Evaluator(retriever)
    
    # Define test queries with relevance judgments
    # These should match the corpus content
    test_queries = {
        "ሪንዋበል ኤነርጂ ጤና": ["doc2", "doc5"],
        "አካባቢ ጥበቃ ክሊማ": ["doc1"],
        "ዛፍ ሊታመም": ["doc3"],
        "ማህበረሰብ ተሳትፎ": ["doc4"],
        "ጤና አካባቢ ብክለት": ["doc5"]
    }
    
    # Evaluate
    eval_results = evaluator.evaluate_multiple_queries(test_queries)
    evaluator.print_evaluation_report(eval_results)
    
    # Save evaluation results
    os.makedirs("output", exist_ok=True)
    eval_output_path = "output/evaluation_results.json"
    evaluator.save_evaluation_results(eval_results, eval_output_path)


def main():
    """Main entry point for the IR system."""
    print("\n" + "=" * 70)
    print("COMPLETE AMHARIC INFORMATION RETRIEVAL SYSTEM")
    print("=" * 70)
    print()
    
    # Step 1: Build index if needed
    print("STEP 1: Index Construction")
    print("-" * 70)
    if not build_index_if_needed():
        print("Failed to build index. Exiting.")
        return
    
    print()
    
    # Step 2: Initialize retriever
    print("STEP 2: Retriever Initialization")
    print("-" * 70)
    try:
        retriever = Retriever()
        print(f"✓ Retriever initialized successfully")
        
        # Print document information
        print("\nAvailable documents:")
        for doc_id, info in sorted(retriever.get_all_documents().items()):
            print(f"  - {doc_id}: {info['name']} ({info['length']} tokens)")
    except Exception as e:
        print(f"✗ Failed to initialize retriever: {e}")
        return
    
    # Step 3: Interactive query loop
    interactive_query_loop(retriever)
    
    # Step 4: Evaluation
    print("\nSTEP 3: Evaluation")
    print("-" * 70)
    run_evaluation(retriever)
    
    print("\n" + "=" * 70)
    print("SYSTEM EXECUTION COMPLETED")
    print("=" * 70)
    print("\nOutput files:")
    print("  - output/index.json: Inverted index")
    print("  - output/evaluation_results.json: Evaluation metrics")
    print()


if __name__ == "__main__":
    main()
