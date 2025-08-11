import duckdb
import pandas as pd
import json
import os
from collections import Counter
from tqdm import tqdm
import ast
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time

def analyze_cohfie_optimized(csv_path: str):
    """
    Optimized comprehensive analysis of the COHFIE dataset.
    Uses DuckDB for heavy lifting and optimized processing for detailed stats.
    """
    print(f"--- Starting optimized analysis of {csv_path} ---")
    conn = duckdb.connect()

    # --- 1. Corpus Composition (Source and Type) - DuckDB ---
    print("\n[1/5] Analyzing Corpus Composition...")
    composition_df = conn.execute(f"""
        SELECT source, source_type, COUNT(*) as frequency
        FROM read_csv_auto('{csv_path}')
        GROUP BY source, source_type
        ORDER BY frequency DESC
    """).fetchdf()
    composition_df.to_csv("cohfie_composition.csv", index=False)
    print("Corpus composition stats saved to 'cohfie_composition.csv'")
    print(composition_df.head())

    # --- 2. Temporal Distribution - DuckDB ---
    print("\n[2/5] Analyzing Temporal Distribution...")
    temporal_df = conn.execute(f"""
        SELECT year, COUNT(*) as frequency
        FROM read_csv_auto('{csv_path}')
        GROUP BY year
        ORDER BY year ASC
    """).fetchdf()
    temporal_df.to_csv("cohfie_temporal_distribution.csv", index=False)
    print("Temporal distribution stats saved to 'cohfie_temporal_distribution.csv'")
    print(temporal_df.head())

    # --- 3. Basic stats using DuckDB (much faster) ---
    print("\n[3/5] Calculating basic text statistics with DuckDB...")
    
    # Get basic text statistics using DuckDB
    basic_stats = conn.execute(f"""
        SELECT 
            AVG(LENGTH(text)) as avg_char_length,
            COUNT(*) as total_sentences,
            SUM(LENGTH(text)) as total_chars
        FROM read_csv_auto('{csv_path}')
    """).fetchone()
    
    avg_char_len, total_sentences, total_chars = basic_stats
    
    # Get word count estimates using DuckDB (approximate but fast)
    word_stats = conn.execute(f"""
        SELECT 
            AVG(LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1) as avg_word_length_approx
        FROM read_csv_auto('{csv_path}')
        WHERE LENGTH(TRIM(text)) > 0
    """).fetchone()
    
    avg_word_len_approx = word_stats[0] if word_stats[0] else 0
    
    print(f"Total sentences: {total_sentences}")
    print(f"Average Character Length: {avg_char_len:.2f}")
    print(f"Average Word Length (approx): {avg_word_len_approx:.2f}")

    # --- 4. Detailed Analysis with Sampling (for 30GB files) ---
    print("\n[4/5] Performing detailed analysis on sample...")
    
    # For very large files, analyze a representative sample
    sample_size = min(100000, total_sentences // 10)  # 10% sample or 100k max
    print(f"Analyzing sample of {sample_size} sentences for detailed stats...")
    
    # Get stratified sample using DuckDB
    sample_df = conn.execute(f"""
        SELECT text, pos_tag_2
        FROM read_csv_auto('{csv_path}')
        USING SAMPLE {sample_size}
    """).fetchdf()
    
    # Process sample for detailed stats
    detailed_stats = process_sample_optimized(sample_df)
    
    # Scale up the results
    vocab_size_estimate = int(detailed_stats['vocab_size'] * (total_sentences / sample_size) ** 0.8)  # Heaps' law approximation
    
    # --- 5. Save Results ---
    print("\n[5/5] Saving detailed statistics...")
    
    # Save sentence-level stats
    with open("cohfie_sentence_stats.txt", "w") as f:
        f.write(f"Total Sentences: {total_sentences}\n")
        f.write(f"Average Character Length: {avg_char_len:.2f}\n")
        f.write(f"Average Word Length (approx): {avg_word_len_approx:.2f}\n")
        f.write(f"Sample size used for detailed analysis: {sample_size}\n")
    
    # Save lexical stats
    total_words_estimate = int(avg_word_len_approx * total_sentences)
    ttr_estimate = vocab_size_estimate / total_words_estimate if total_words_estimate > 0 else 0
    
    with open("cohfie_lexical_stats.txt", "w") as f:
        f.write(f"Total Words (Tokens - estimated): {total_words_estimate}\n")
        f.write(f"Vocabulary Size (estimated): {vocab_size_estimate}\n")
        f.write(f"Type-Token Ratio (TTR - estimated): {ttr_estimate:.4f}\n")
        f.write(f"Based on sample of {sample_size} sentences\n")
    
    print(f"Estimated Vocabulary Size: {vocab_size_estimate}")
    print(f"Estimated TTR: {ttr_estimate:.4f}")
    
    # Save POS tag distribution from sample
    if detailed_stats['pos_counts']:
        pos_df = pd.DataFrame(detailed_stats['pos_counts'].most_common(), 
                             columns=['pos_tag', 'frequency'])
        pos_df.to_csv("cohfie_pos_distribution_sample.csv", index=False)
        print("POS tag distribution (sample) saved to 'cohfie_pos_distribution_sample.csv'")
        print(pos_df.head())
    
    # Save foreign word frequencies from sample
    if detailed_stats['fw_counts']:
        fw_df = pd.DataFrame(detailed_stats['fw_counts'].most_common(), 
                            columns=['foreign_word', 'frequency'])
        fw_df.to_csv("cohfie_foreign_word_distribution_sample.csv", index=False)
        print("Foreign word distribution (sample) saved to 'cohfie_foreign_word_distribution_sample.csv'")
        print(fw_df.head())
    
    conn.close()
    print("\n--- Optimized analysis complete. Results saved to files. ---")
    print("Note: Detailed stats are based on representative sample for efficiency.")


def safe_eval_pos_tags(pos_string):
    """Safely evaluate POS tags string."""
    if not pos_string or pos_string == '[]':
        return []
    try:
        # Try ast.literal_eval first (safer)
        return ast.literal_eval(pos_string)
    except (ValueError, SyntaxError):
        try:
            # Fallback to eval (less safe but sometimes necessary)
            return eval(pos_string)
        except:
            return []


def process_sample_optimized(sample_df):
    """Process sample data efficiently."""
    pos_tag_counts = Counter()
    foreign_word_counts = Counter()
    vocabulary = set()
    word_lengths = []
    
    print("Processing sample for detailed statistics...")
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Processing sample"):
        text = row.get('text', '')
        pos_tags_str = row.get('pos_tag_2', '[]')
        
        # Parse POS tags
        pos_list = safe_eval_pos_tags(pos_tags_str)
        
        if pos_list:
            word_lengths.append(len(pos_list))
            
            # Process each word-tag pair
            for item in pos_list:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    word, tag = item[0], item[1]
                    vocabulary.add(word.lower())
                    pos_tag_counts[tag] += 1
                    if tag == 'FW':
                        foreign_word_counts[word.lower()] += 1
        else:
            # Fallback: split by spaces
            words = text.split()
            word_lengths.append(len(words))
            for word in words:
                vocabulary.add(word.lower())
    
    return {
        'vocab_size': len(vocabulary),
        'pos_counts': pos_tag_counts,
        'fw_counts': foreign_word_counts,
        'word_lengths': word_lengths
    }


def analyze_full_detailed(csv_path: str):
    """
    Alternative: Full detailed analysis using chunked processing with multiprocessing.
    Use this only if you need exact counts (will take longer).
    """
    print(f"--- Starting FULL detailed analysis of {csv_path} ---")
    print("Warning: This will process the entire 30GB file and may take hours.")
    
    # Get total number of lines for progress tracking
    print("Counting total lines for progress tracking...")
    with tqdm(desc="Counting lines") as pbar:
        total_lines = sum(1 for _ in open(csv_path, 'r'))
        pbar.update(1)
    print(f"Total lines: {total_lines}")
    
    # Use multiprocessing for chunks
    num_processes = min(mp.cpu_count(), 8)  # Don't overwhelm the system
    chunk_size = 50000  # Larger chunks for efficiency
    total_chunks = (total_lines // chunk_size) + 1
    
    print(f"Processing {total_chunks} chunks with {num_processes} processes...")
    
    # Create progress bar for chunk processing
    chunk_progress = tqdm(total=total_chunks, desc="Processing chunks", position=0)
    
    def update_progress(result):
        chunk_progress.update(1)
        chunk_progress.set_postfix(chunk=f"{result['chunk_idx']}")
    
    # Read CSV in chunks
    chunk_iter = pd.read_csv(csv_path, chunksize=chunk_size, keep_default_na=False)
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for i, chunk in enumerate(chunk_iter):
            future = executor.submit(process_chunk, (i, chunk))
            future.add_done_callback(lambda f: update_progress(f.result()))
            futures.append(future)
        
        # Wait for all chunks to complete
        results = [future.result() for future in futures]
    
    chunk_progress.close()
    
    # Combine results with progress
    print("Combining results from all chunks...")
    with tqdm(desc="Combining results", total=1) as combine_pbar:
        combined_stats = combine_chunk_results(results)
        combine_pbar.update(1)
    
    # Save results
    print("Saving full analysis results...")
    with tqdm(desc="Saving results", total=1) as save_pbar:
        save_full_results(combined_stats)
        save_pbar.update(1)


def process_chunk(chunk_data):
    """Process a single chunk of data."""
    chunk_idx, chunk = chunk_data
    
    pos_tag_counts = Counter()
    foreign_word_counts = Counter()
    vocabulary = set()
    char_lengths = []
    word_lengths = []
    
    for _, row in chunk.iterrows():
        text = row.get('text', '')
        pos_tags_str = row.get('pos_tag_2', '[]')
        
        char_lengths.append(len(text))
        pos_list = safe_eval_pos_tags(pos_tags_str)
        
        if pos_list:
            word_lengths.append(len(pos_list))
            for item in pos_list:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    word, tag = item[0], item[1]
                    vocabulary.add(word.lower())
                    pos_tag_counts[tag] += 1
                    if tag == 'FW':
                        foreign_word_counts[word.lower()] += 1
        else:
            words = text.split()
            word_lengths.append(len(words))
            for word in words:
                vocabulary.add(word.lower())
    
    return {
        'chunk_idx': chunk_idx,
        'pos_counts': pos_tag_counts,
        'fw_counts': foreign_word_counts,
        'vocabulary': vocabulary,
        'char_lengths': char_lengths,
        'word_lengths': word_lengths
    }


def combine_chunk_results(results):
    """Combine results from all chunks."""
    combined_pos = Counter()
    combined_fw = Counter()
    combined_vocab = set()
    all_char_lengths = []
    all_word_lengths = []
    
    for result in results:
        combined_pos.update(result['pos_counts'])
        combined_fw.update(result['fw_counts'])
        combined_vocab.update(result['vocabulary'])
        all_char_lengths.extend(result['char_lengths'])
        all_word_lengths.extend(result['word_lengths'])
    
    return {
        'pos_counts': combined_pos,
        'fw_counts': combined_fw,
        'vocabulary': combined_vocab,
        'char_lengths': all_char_lengths,
        'word_lengths': all_word_lengths
    }


def save_full_results(stats):
    """Save full analysis results."""
    print("Saving full analysis results...")
    
    with tqdm(desc="Saving full results", total=5) as save_pbar:
        # Calculate and save stats
        avg_char_len = sum(stats['char_lengths']) / len(stats['char_lengths'])
        avg_word_len = sum(stats['word_lengths']) / len(stats['word_lengths'])
        
        with open("cohfie_sentence_stats_full.txt", "w") as f:
            f.write(f"Average Character Length: {avg_char_len:.2f}\n")
            f.write(f"Average Word Length: {avg_word_len:.2f}\n")
        save_pbar.update(1)
        
        total_words = sum(stats['word_lengths'])
        vocab_size = len(stats['vocabulary'])
        ttr = vocab_size / total_words if total_words > 0 else 0
        
        with open("cohfie_lexical_stats_full.txt", "w") as f:
            f.write(f"Total Words (Tokens): {total_words}\n")
            f.write(f"Vocabulary Size: {vocab_size}\n")
            f.write(f"Type-Token Ratio (TTR): {ttr:.4f}\n")
        save_pbar.update(1)
        
        # Save distributions
        pos_df = pd.DataFrame(stats['pos_counts'].most_common(), columns=['pos_tag', 'frequency'])
        pos_df.to_csv("cohfie_pos_distribution_full.csv", index=False)
        save_pbar.update(1)
        
        fw_df = pd.DataFrame(stats['fw_counts'].most_common(), columns=['foreign_word', 'frequency'])
        fw_df.to_csv("cohfie_foreign_word_distribution_full.csv", index=False)
        save_pbar.update(1)
        
        print(f"Full analysis complete!")
        print(f"Vocabulary Size: {vocab_size}")
        print(f"TTR: {ttr:.4f}")
        save_pbar.update(1)


if __name__ == "__main__":
    csv_file = 'COHFIE_V4.csv'
    if not os.path.exists(csv_file):
        print(f"Error: Required file '{csv_file}' not found in the current directory.")
    else:
        # Run optimized version (recommended for 30GB file)
        analyze_cohfie_optimized(csv_file)
        
        response = input("Do you want to run full detailed analysis? This will take much longer (y/n): ")
        if response.lower() == 'y':
             analyze_full_detailed(csv_file)
