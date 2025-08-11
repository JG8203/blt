#!/usr/bin/env python3
"""
Research-Grade COHFIE Subset Creator for Tokenizer Analysis
Creates three distinct, representative subsets optimized for empirical tokenizer comparison.

This script supports the empirical analysis of how different tokenizers handle:
1. Formal News: Standard, well-structured Tagalog (baseline)
2. Social Media: Noisy, informal, code-switched text (stress test)  
3. Literary/Encyclopedic: High information density, diverse vocabulary

Uses DuckDB for memory-efficient processing of large datasets.
"""

import duckdb
import argparse
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
import time
import json

# --- Research-Driven Corpus Definitions ---
# Based on linguistic features relevant for tokenizer analysis
CORPUS_DEFINITIONS = {
    "formal_news": {
        "sources": [
            "mb", "balita", "bandera", "gma", "abante",
            "radyoinquirer", "abscbn", "philstar"
        ],
        "description": "Clean, well-structured, grammatically standard Tagalog",
        "linguistic_features": [
            "Formal vocabulary",
            "Standard sentence structures", 
            "Minimal code-switching",
            "Professional editing"
        ],
        "tokenizer_hypothesis": "Should favor subword tokenizers (BPE, Tiktoken) due to consistent vocabulary"
    },
    "social_media": {
        "sources": [
            "youtube", "twitter", "reddit"
        ],
        "description": "Noisy, informal, mixed-language text for robustness testing",
        "linguistic_features": [
            "Heavy code-switching (Taglish)",
            "Slang and abbreviations",
            "User-generated content",
            "Non-standard grammar",
            "Emoji and special characters"
        ],
        "tokenizer_hypothesis": "Should favor byte-level tokenizers (ByT5) due to noise and mixing"
    },
    "literary_encyclopedic": {
        "sources": [
            "wikipedia", "100year", "google_books", "gutenberg", "bible"
        ],
        "description": "High information density with diverse and specialized vocabulary",
        "linguistic_features": [
            "Long-form narrative",
            "Descriptive language",
            "Specialized vocabulary",
            "Historical/archaic terms",
            "Complex sentence structures"
        ],
        "tokenizer_hypothesis": "Will reveal compression efficiency differences across tokenizers"
    }
}

def validate_input_file(input_csv_path: str) -> bool:
    """Validate that the input CSV file exists and is readable."""
    if not os.path.exists(input_csv_path):
        print(f"❌ Error: Input file not found at '{input_csv_path}'")
        return False
    
    if not os.access(input_csv_path, os.R_OK):
        print(f"❌ Error: Cannot read file '{input_csv_path}' (permission denied)")
        return False
    
    # Check file size
    file_size = os.path.getsize(input_csv_path)
    file_size_gb = file_size / (1024**3)
    print(f"📁 Input file size: {file_size_gb:.2f} GB")
    
    return True

def analyze_dataset_structure(conn: duckdb.DuckDBPyConnection, input_csv_path: str) -> Dict:
    """Analyze the dataset structure and source distribution for research purposes."""
    print("🔍 Analyzing dataset structure for research subset creation...")
    
    try:
        # Get total row count
        total_rows = conn.execute(f"""
            SELECT COUNT(*) as total_count
            FROM read_csv_auto('{input_csv_path}')
        """).fetchone()[0]
        
        # Get comprehensive source analysis
        source_analysis = conn.execute(f"""
            SELECT 
                source, 
                COUNT(*) as count,
                AVG(LENGTH(text)) as avg_text_length,
                MIN(year) as earliest_year,
                MAX(year) as latest_year,
                COUNT(DISTINCT year) as year_span
            FROM read_csv_auto('{input_csv_path}')
            GROUP BY source
            ORDER BY count DESC
        """).fetchdf()
        
        # Get column information
        columns = conn.execute(f"""
            DESCRIBE (
                SELECT * FROM read_csv_auto('{input_csv_path}') LIMIT 1
            )
        """).fetchdf()
        
        # Get sample texts for quality assessment
        sample_texts = conn.execute(f"""
            SELECT source, text, year
            FROM read_csv_auto('{input_csv_path}')
            WHERE LENGTH(text) > 50 AND LENGTH(text) < 200
            ORDER BY RANDOM()
            LIMIT 5
        """).fetchdf()
        
        analysis = {
            'total_rows': total_rows,
            'source_analysis': source_analysis,
            'columns': list(columns['column_name']),
            'available_sources': set(source_analysis['source'].tolist()),
            'sample_texts': sample_texts
        }
        
        print(f"📊 Dataset Analysis for Research:")
        print(f"   Total rows: {total_rows:,}")
        print(f"   Unique sources: {len(analysis['available_sources'])}")
        print(f"   Columns available: {analysis['columns']}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        return None

def evaluate_corpus_quality(analysis: Dict, corpus_definitions: Dict) -> Dict:
    """Evaluate the quality and representativeness of each corpus for research."""
    print("\n🔬 Evaluating corpus quality for tokenizer research...")
    
    available_sources = analysis['available_sources']
    source_stats = {}
    for _, row in analysis['source_analysis'].iterrows():
        source_stats[row['source']] = {
            'count': row['count'],
            'avg_length': row['avg_text_length'],
            'year_span': row['year_span'],
            'earliest': row['earliest_year'],
            'latest': row['latest_year']
        }
    
    corpus_evaluation = {}
    
    for corpus_name, corpus_info in corpus_definitions.items():
        print(f"\n📚 {corpus_name.upper()} CORPUS EVALUATION:")
        print(f"   Research Purpose: {corpus_info['description']}")
        print(f"   Tokenizer Hypothesis: {corpus_info['tokenizer_hypothesis']}")
        
        required_sources = corpus_info['sources']
        available_in_corpus = []
        missing_sources = []
        total_available_rows = 0
        quality_metrics = {
            'avg_text_length': 0,
            'total_year_span': 0,
            'source_diversity': 0
        }
        
        for source in required_sources:
            if source in available_sources:
                stats = source_stats[source]
                available_in_corpus.append(source)
                total_available_rows += stats['count']
                quality_metrics['avg_text_length'] += stats['avg_length'] * stats['count']
                quality_metrics['total_year_span'] = max(quality_metrics['total_year_span'], 
                                                       stats['year_span'])
                
                print(f"   ✅ {source}: {stats['count']:,} rows "
                      f"(avg {stats['avg_length']:.0f} chars, {stats['year_span']} year span)")
            else:
                missing_sources.append(source)
                print(f"   ❌ {source}: NOT FOUND")
        
        if total_available_rows > 0:
            quality_metrics['avg_text_length'] /= total_available_rows
            quality_metrics['source_diversity'] = len(available_in_corpus) / len(required_sources)
        
        corpus_evaluation[corpus_name] = {
            'available_sources': available_in_corpus,
            'missing_sources': missing_sources,
            'total_rows': total_available_rows,
            'has_sufficient_data': total_available_rows >= 1000,  # Minimum for statistical validity
            'quality_metrics': quality_metrics,
            'research_viability': len(available_in_corpus) >= len(required_sources) * 0.6  # 60% coverage
        }
        
        print(f"   📊 Total available: {total_available_rows:,} rows from {len(available_in_corpus)} sources")
        print(f"   📏 Avg text length: {quality_metrics['avg_text_length']:.0f} characters")
        print(f"   🎯 Research viability: {'✅ Good' if corpus_evaluation[corpus_name]['research_viability'] else '⚠️ Limited'}")
        
        if missing_sources:
            print(f"   ⚠️  Missing sources (may affect representativeness): {missing_sources}")
    
    return corpus_evaluation

def create_research_subset(conn: duckdb.DuckDBPyConnection, 
                          input_csv_path: str,
                          corpus_name: str,
                          corpus_info: Dict,
                          sample_size: int,
                          output_path: str,
                          total_available: int,
                          stratified: bool = True) -> Tuple[bool, Dict]:
    """Create a research-quality subset with optional stratification."""
    
    print(f"\n🎯 Creating {corpus_name} research subset...")
    print(f"   Research purpose: {corpus_info['description']}")
    print(f"   Sources: {corpus_info['available_sources']}")
    print(f"   Available rows: {total_available:,}")
    print(f"   Requested sample: {sample_size:,}")
    
    actual_sample_size = min(sample_size, total_available)
    print(f"   Final sample size: {actual_sample_size:,}")
    
    try:
        sources = corpus_info['available_sources']
        sources_sql = "', '".join(sources)
        
        start_time = time.time()
        
        if stratified and len(sources) > 1:
            # Stratified sampling: equal representation from each source
            sample_per_source = actual_sample_size // len(sources)
            remainder = actual_sample_size % len(sources)
            
            print(f"   🎲 Using stratified sampling: ~{sample_per_source} per source")
            
            # Create stratified sample
            source_samples = []
            for i, source in enumerate(sources):
                source_sample_size = sample_per_source + (1 if i < remainder else 0)
                source_samples.append(f"""
                    (SELECT * FROM read_csv_auto('{input_csv_path}')
                     WHERE source = '{source}'
                     USING SAMPLE {source_sample_size} (RESERVOIR))
                """)
            
            query = f"""
                COPY (
                    {' UNION ALL '.join(source_samples)}
                ) TO '{output_path}' (FORMAT CSV, HEADER)
            """
        else:
            # Simple random sampling
            print(f"   🎲 Using simple random sampling")
            
            if actual_sample_size >= total_available:
                query = f"""
                    COPY (
                        SELECT *
                        FROM read_csv_auto('{input_csv_path}')
                        WHERE source IN ('{sources_sql}')
                    ) TO '{output_path}' (FORMAT CSV, HEADER)
                """
            else:
                query = f"""
                    COPY (
                        SELECT *
                        FROM read_csv_auto('{input_csv_path}')
                        WHERE source IN ('{sources_sql}')
                        USING SAMPLE {actual_sample_size} (RESERVOIR)
                    ) TO '{output_path}' (FORMAT CSV, HEADER)
                """
        
        # Execute the query
        with tqdm(desc=f"Creating {corpus_name}", unit="rows") as pbar:
            conn.execute(query)
            pbar.update(actual_sample_size)
        
        duration = time.time() - start_time
        
        # Verify and analyze the output
        verification_query = f"""
            SELECT 
                source,
                COUNT(*) as count,
                AVG(LENGTH(text)) as avg_length,
                MIN(LENGTH(text)) as min_length,
                MAX(LENGTH(text)) as max_length
            FROM read_csv_auto('{output_path}')
            GROUP BY source
            ORDER BY count DESC
        """
        
        subset_analysis = conn.execute(verification_query).fetchdf()
        total_created = subset_analysis['count'].sum()
        
        print(f"   ✅ Created successfully in {duration:.2f}s")
        print(f"   📁 Output: {output_path}")
        print(f"   📊 Rows written: {total_created:,}")
        
        # Show source distribution in created subset
        print(f"   📈 Source distribution in subset:")
        for _, row in subset_analysis.iterrows():
            percentage = (row['count'] / total_created) * 100
            print(f"      {row['source']}: {row['count']:,} rows ({percentage:.1f}%)")
        
        return True, {
            'total_rows': int(total_created),
            'source_distribution': subset_analysis.to_dict('records'),
            'creation_time': duration,
            'avg_text_length': float(subset_analysis['avg_length'].mean())
        }
        
    except Exception as e:
        print(f"   ❌ Error creating subset: {e}")
        return False, {}

def create_research_subsets(input_csv_path: str, output_dir: str, sample_size: int,
                           stratified: bool = True, save_metadata: bool = True):
    """
    Create research-grade subsets optimized for tokenizer analysis.
    """
    print("🚀 COHFIE Research Subset Creator for Tokenizer Analysis")
    print("=" * 70)
    print("Purpose: Create three distinct corpora to evaluate tokenizer performance on:")
    print("1. Formal News (baseline standard Tagalog)")
    print("2. Social Media (noisy, code-switched text)")  
    print("3. Literary/Encyclopedic (high information density)")
    print("=" * 70)
    
    # Validate input
    if not validate_input_file(input_csv_path):
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Initialize DuckDB connection
    conn = duckdb.connect()
    
    try:
        # Analyze dataset structure
        analysis = analyze_dataset_structure(conn, input_csv_path)
        if analysis is None:
            return
        
        # Evaluate corpus quality for research
        corpus_evaluation = evaluate_corpus_quality(analysis, CORPUS_DEFINITIONS)
        
        # Create subsets
        print(f"\n🔨 Creating research subsets")
        print(f"   Sample size per corpus: {sample_size:,}")
        print(f"   Sampling strategy: {'Stratified' if stratified else 'Simple random'}")
        print("=" * 70)
        
        successful_subsets = 0
        subset_metadata = {}
        
        for corpus_name, corpus_definition in CORPUS_DEFINITIONS.items():
            corpus_eval = corpus_evaluation[corpus_name]
            
            if not corpus_eval['has_sufficient_data']:
                print(f"⏭️  Skipping {corpus_name} (insufficient data for research)")
                continue
            
            if not corpus_eval['research_viability']:
                print(f"⚠️  {corpus_name} has limited research viability but proceeding...")
            
            # Prepare corpus info for subset creation
            corpus_info = {
                'available_sources': corpus_eval['available_sources'],
                'description': corpus_definition['description']
            }
            
            # Create output path
            output_path = os.path.join(output_dir, f"{corpus_name}_subset.csv")
            
            # Create the subset
            success, metadata = create_research_subset(
                conn, input_csv_path, corpus_name, corpus_info,
                sample_size, output_path, corpus_eval['total_rows'], stratified
            )
            
            if success:
                successful_subsets += 1
                subset_metadata[corpus_name] = {
                    **metadata,
                    'corpus_definition': corpus_definition,
                    'evaluation_metrics': corpus_eval['quality_metrics'],
                    'file_path': output_path
                }
        
        # Save research metadata
        if save_metadata and subset_metadata:
            metadata_path = os.path.join(output_dir, "research_metadata.json")
            research_metadata = {
                'creation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'input_file': input_csv_path,
                'total_input_rows': analysis['total_rows'],
                'sample_size_per_corpus': sample_size,
                'sampling_strategy': 'stratified' if stratified else 'simple_random',
                'subsets_created': subset_metadata,
                'research_purpose': 'Tokenizer performance analysis on different text types',
                'corpus_definitions': CORPUS_DEFINITIONS
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(research_metadata, f, indent=2, ensure_ascii=False)
            
            print(f"\n📋 Research metadata saved to: {metadata_path}")
        
        # Summary
        print(f"\n✅ RESEARCH SUBSET CREATION SUMMARY")
        print("=" * 50)
        print(f"📊 Subsets created: {successful_subsets}/{len(CORPUS_DEFINITIONS)}")
        print(f"📁 Output directory: {output_dir}")
        
        if successful_subsets > 0:
            print(f"\n📋 Created research files:")
            total_subset_size = 0
            for filename in os.listdir(output_dir):
                if filename.endswith('.csv'):
                    filepath = os.path.join(output_dir, filename)
                    filesize = os.path.getsize(filepath) / (1024*1024)  # MB
                    total_subset_size += filesize
                    print(f"   • {filename} ({filesize:.1f} MB)")
            
            print(f"   Total subset size: {total_subset_size:.1f} MB")
        
        print(f"\n🎯 Next Steps for Tokenizer Research:")
        print(f"   1. Use these subsets with your tokenizer comparison scripts")
        print(f"   2. Each subset tests different linguistic challenges:")
        print(f"      • formal_news: Standard language baseline")
        print(f"      • social_media: Code-switching and noise robustness") 
        print(f"      • literary_encyclopedic: Vocabulary diversity and compression")
        print(f"   3. Compare tokenizer performance across subsets for insights")
        print(f"   4. Use metadata.json for reproducible research documentation")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        conn.close()

def main():
    """Main function with enhanced command line interface for research."""
    parser = argparse.ArgumentParser(
        description="Create research-grade subsets from COHFIE dataset for tokenizer analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Research Design:
This script creates three linguistically distinct subsets to evaluate how tokenizers 
handle different types of Tagalog text:

1. FORMAL NEWS: Clean, standard Tagalog (baseline for tokenizer comparison)
2. SOCIAL MEDIA: Noisy, code-switched text (robustness stress test)
3. LITERARY/ENCYCLOPEDIC: High information density (compression efficiency test)

Examples:
  python create_subsets.py COHFIE_V4.csv
  python create_subsets.py COHFIE_V4.csv --sample_size 15000 --stratified
  python create_subsets.py COHFIE_V4.csv --output_dir tokenizer_subsets
  python create_subsets.py COHFIE_V4.csv --list_sources
  
Performance Features:
  • Uses DuckDB for memory-efficient processing of 30GB+ files
  • Stratified sampling ensures balanced representation
  • Research metadata for reproducible analysis
  • Quality metrics for corpus evaluation
        """
    )
    
    parser.add_argument(
        "input_csv",
        help="Path to the full COHFIE dataset CSV file (e.g., COHFIE_V4.csv)"
    )
    
    parser.add_argument(
        "--output_dir",
        default="data/research_subsets",
        help="Directory to save research subset files (default: data/research_subsets)"
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10000,
        help="Number of sentences per corpus subset (default: 10000)"
    )
    
    parser.add_argument(
        "--stratified",
        action="store_true",
        default=True,
        help="Use stratified sampling for balanced source representation (default: True)"
    )
    
    parser.add_argument(
        "--simple_random",
        action="store_true",
        help="Use simple random sampling instead of stratified"
    )
    
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="Skip saving research metadata file"
    )
    
    parser.add_argument(
        "--list_sources",
        action="store_true",
        help="Analyze and list all available sources, then exit"
    )
    
    args = parser.parse_args()
    
    # Handle conflicting sampling arguments
    stratified = args.stratified and not args.simple_random
    
    if args.list_sources:
        # Analysis mode: show dataset structure and exit
        if not validate_input_file(args.input_csv):
            return
        
        conn = duckdb.connect()
        try:
            analysis = analyze_dataset_structure(conn, args.input_csv)
            if analysis:
                print(f"\n📋 Complete Source Analysis:")
                print(f"{'Source':<20} {'Rows':<10} {'Avg Length':<12} {'Year Span':<10}")
                print("-" * 55)
                for _, row in analysis['source_analysis'].iterrows():
                    print(f"{row['source']:<20} {row['count']:<10,} "
                          f"{row['avg_text_length']:<12.0f} {row['year_span']:<10}")
                
                print(f"\n📝 Sample texts for quality assessment:")
                for _, row in analysis['sample_texts'].iterrows():
                    text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
                    print(f"   {row['source']} ({row['year']}): {text_preview}")
        finally:
            conn.close()
        return
    
    # Run the main subset creation for research
    create_research_subsets(
        args.input_csv, 
        args.output_dir, 
        args.sample_size,
        stratified=stratified,
        save_metadata=not args.no_metadata
    )

if __name__ == "__main__":
    main()
