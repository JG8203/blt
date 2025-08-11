#!/usr/bin/env python3

import json
import argparse
import pandas as pd
from typing import Dict, List, Any, Optional
import statistics
from collections import defaultdict
import os
import sys

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich.prompt import Prompt, IntPrompt, Confirm
    from rich.progress import track
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("📋 Note: Install 'rich' for enhanced display: pip install rich")

class TokenizerDataViewer:
    """Interactive viewer for comprehensive tokenizer analysis data."""
    
    def __init__(self, json_file: str):
        self.json_file = json_file
        self.data = self.load_data()
        self.console = Console() if RICH_AVAILABLE else None
        
    def load_data(self) -> Dict[str, Any]:
        """Load and validate the JSON data."""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate data structure
            required_keys = ['metadata', 'samples']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Invalid data format: missing '{key}' key")
            
            print(f"✅ Loaded data for {len(data['samples'])} samples")
            return data
            
        except FileNotFoundError:
            print(f"❌ Error: File '{self.json_file}' not found")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON format: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)
    
    def print_section(self, title: str, content: str = "", style: str = "bold blue"):
        """Print a formatted section."""
        if RICH_AVAILABLE:
            if content:
                self.console.print(Panel(content, title=title, title_align="left"))
            else:
                self.console.print(f"\n[{style}]{title}[/{style}]")
        else:
            print(f"\n{'='*60}")
            print(f"{title}")
            print('='*60)
            if content:
                print(content)
    
    def show_overview(self):
        """Display overview of the entire dataset."""
        metadata = self.data['metadata']
        samples = self.data['samples']
        
        # Basic statistics
        total_samples = len(samples)
        entropy_models = metadata.get('entropy_models_loaded', [])
        traditional_tokenizers = metadata.get('traditional_tokenizers_loaded', [])
        
        overview_text = f"""
📊 Analysis Overview:
   • Timestamp: {metadata.get('analysis_timestamp', 'Unknown')}
   • Total Samples: {total_samples}
   • Entropy Models: {len(entropy_models)} ({', '.join(entropy_models)})
   • Traditional Tokenizers: {len(traditional_tokenizers)}
   
📈 Sample Sources:
"""
        
        # Count sources
        source_counts = defaultdict(int)
        total_chars = 0
        total_words = 0
        
        for sample in samples:
            if 'original_data' in sample:
                source = sample['original_data'].get('source', 'unknown')
                source_counts[source] += 1
                total_chars += sample['original_data'].get('char_length', 0)
                total_words += sample['original_data'].get('word_count', 0)
        
        for source, count in sorted(source_counts.items()):
            overview_text += f"   • {source}: {count} samples\n"
        
        overview_text += f"""
📏 Text Statistics:
   • Average characters per sample: {total_chars/total_samples:.1f}
   • Average words per sample: {total_words/total_samples:.1f}
   • Total characters analyzed: {total_chars:,}
   • Total words analyzed: {total_words:,}
"""
        
        self.print_section("📋 DATASET OVERVIEW", overview_text)
    
    def show_model_comparison(self):
        """Show comparison between all models."""
        samples = self.data['samples']
        
        # Collect statistics for each model
        model_stats = defaultdict(lambda: {
            'token_counts': [],
            'compression_ratios': [],
            'processing_times': [],
            'perfect_reconstructions': 0,
            'total_samples': 0,
            'errors': 0
        })
        
        for sample in samples:
            # Traditional tokenizers
            if 'traditional_tokenizer_results' in sample:
                for tokenizer_key, result in sample['traditional_tokenizer_results'].items():
                    if 'error' not in result and 'statistics' in result:
                        stats = model_stats[result['tokenizer_name']]
                        stats['token_counts'].append(result['statistics']['token_count'])
                        stats['compression_ratios'].append(result['statistics']['compression_ratio_chars'])
                        stats['processing_times'].append(result['processing_time_seconds'])
                        if result['reconstruction']['perfect_reconstruction']:
                            stats['perfect_reconstructions'] += 1
                        stats['total_samples'] += 1
                    else:
                        model_stats[self.data['tokenizer_metadata'].get(tokenizer_key, {}).get('name', tokenizer_key)]['errors'] += 1
            
            # Entropy models
            if 'entropy_model_results' in sample:
                for model_key, result in sample['entropy_model_results'].items():
                    if 'error' not in result and 'patch_statistics' in result:
                        stats = model_stats[result['model_name']]
                        stats['token_counts'].append(result['patch_statistics']['patch_count'])
                        stats['compression_ratios'].append(result['patch_statistics']['avg_patch_length'])
                        stats['processing_times'].append(result['processing_time_seconds'])
                        if result['reconstruction']['perfect_reconstruction']:
                            stats['perfect_reconstructions'] += 1
                        stats['total_samples'] += 1
                    else:
                        model_stats[result.get('model_name', model_key)]['errors'] += 1
        
        # Display comparison table
        if RICH_AVAILABLE:
            table = Table(title="🔀 Model Performance Comparison")
            table.add_column("Model", style="cyan", no_wrap=True)
            table.add_column("Avg Segments", justify="right")
            table.add_column("Avg Compression", justify="right")
            table.add_column("Avg Time (ms)", justify="right")
            table.add_column("Perfect Recon", justify="right")
            table.add_column("Success Rate", justify="right")
            
            for model_name, stats in sorted(model_stats.items()):
                if stats['total_samples'] > 0:
                    avg_tokens = statistics.mean(stats['token_counts'])
                    avg_compression = statistics.mean(stats['compression_ratios'])
                    avg_time_ms = statistics.mean(stats['processing_times']) * 1000
                    perfect_rate = (stats['perfect_reconstructions'] / stats['total_samples']) * 100
                    success_rate = (stats['total_samples'] / (stats['total_samples'] + stats['errors'])) * 100
                    
                    table.add_row(
                        model_name,
                        f"{avg_tokens:.1f}",
                        f"{avg_compression:.2f}",
                        f"{avg_time_ms:.2f}",
                        f"{perfect_rate:.1f}%",
                        f"{success_rate:.1f}%"
                    )
            
            self.console.print(table)
        else:
            print("\n🔀 MODEL PERFORMANCE COMPARISON")
            print("-" * 100)
            print(f"{'Model':<25} {'Avg Segments':<12} {'Avg Compression':<15} {'Avg Time (ms)':<12} {'Perfect Recon':<12} {'Success Rate'}")
            print("-" * 100)
            
            for model_name, stats in sorted(model_stats.items()):
                if stats['total_samples'] > 0:
                    avg_tokens = statistics.mean(stats['token_counts'])
                    avg_compression = statistics.mean(stats['compression_ratios'])
                    avg_time_ms = statistics.mean(stats['processing_times']) * 1000
                    perfect_rate = (stats['perfect_reconstructions'] / stats['total_samples']) * 100
                    success_rate = (stats['total_samples'] / (stats['total_samples'] + stats['errors'])) * 100
                    
                    print(f"{model_name:<25} {avg_tokens:<12.1f} {avg_compression:<15.2f} {avg_time_ms:<12.2f} {perfect_rate:<12.1f}% {success_rate:.1f}%")
    
    def show_sample_detail(self, sample_index: int):
        """Show detailed view of a specific sample."""
        if sample_index < 1 or sample_index > len(self.data['samples']):
            print(f"❌ Invalid sample index. Choose between 1 and {len(self.data['samples'])}")
            return
        
        sample = self.data['samples'][sample_index - 1]
        original_data = sample['original_data']
        
        # Basic sample info
        sample_info = f"""
📝 Original Text: "{original_data['text']}"
📚 Source: {original_data.get('source', 'Unknown')} ({original_data.get('year', 'Unknown')})
📏 Length: {original_data['char_length']} characters, {original_data['word_count']} words
💾 Bytes: {original_data['byte_length']} bytes, {original_data['unique_chars']} unique characters
"""
        
        self.print_section(f"📄 SAMPLE {sample_index} DETAILS", sample_info)
        
        # Traditional tokenizer results
        if 'traditional_tokenizer_results' in sample:
            self.print_section("🔤 Traditional Tokenizer Results")
            
            for tokenizer_key, result in sample['traditional_tokenizer_results'].items():
                if 'error' in result:
                    print(f"❌ {tokenizer_key}: {result['error']}")
                    continue
                
                tokenizer_name = result['tokenizer_name']
                tokens = result['tokens']
                stats = result['statistics']
                
                print(f"\n🔧 {tokenizer_name}:")
                print(f"   Tokens: {stats['token_count']}")
                print(f"   Compression: {stats['compression_ratio_chars']:.2f} chars/token")
                print(f"   Processing: {result['processing_time_seconds']*1000:.2f}ms")
                print(f"   Perfect reconstruction: {'✅' if result['reconstruction']['perfect_reconstruction'] else '❌'}")
                
                # Show first few tokens
                max_display = 10
                token_display = []
                for i in range(min(max_display, len(tokens['token_strings']))):
                    token_str = tokens['token_strings'][i]
                    token_id = tokens['token_ids'][i]
                    token_display.append(f"[{token_id}:'{token_str}']")
                
                tokens_text = " ".join(token_display)
                if len(tokens['token_strings']) > max_display:
                    tokens_text += f" ... (+{len(tokens['token_strings']) - max_display} more)"
                
                print(f"   Tokens: {tokens_text}")
        
        # Entropy model results
        if 'entropy_model_results' in sample:
            self.print_section("🧠 Entropy Model Results")
            
            for model_key, result in sample['entropy_model_results'].items():
                if 'error' in result:
                    print(f"❌ {model_key}: {result['error']}")
                    continue
                
                model_name = result['model_name']
                patches = result['patches']
                patch_stats = result['patch_statistics']
                entropy_analysis = result.get('entropy_analysis', {})
                
                print(f"\n🧠 {model_name}:")
                print(f"   Patches: {patch_stats['patch_count']}")
                print(f"   Avg patch length: {patch_stats['avg_patch_length']:.2f} tokens")
                print(f"   Processing: {result['processing_time_seconds']*1000:.2f}ms")
                
                if entropy_analysis.get('avg_token_entropy'):
                    print(f"   Avg token entropy: {entropy_analysis['avg_token_entropy']:.3f}")
                
                # Show patches
                max_display = 8
                patch_display = []
                for i in range(min(max_display, len(patches))):
                    patch = patches[i]
                    patch_text = patch['patch_text']
                    patch_length = patch['patch_length']
                    entropy_score = patch.get('patch_entropy_score', 'N/A')
                    if entropy_score != 'N/A':
                        patch_display.append(f"[{patch_length}:'{patch_text}':{entropy_score:.2f}]")
                    else:
                        patch_display.append(f"[{patch_length}:'{patch_text}']")
                
                patches_text = " ".join(patch_display)
                if len(patches) > max_display:
                    patches_text += f" ... (+{len(patches) - max_display} more)"
                
                print(f"   Patches: {patches_text}")
    
    def show_entropy_analysis(self):
        """Show detailed entropy analysis for entropy models."""
        samples = self.data['samples']
        entropy_data = defaultdict(lambda: {
            'token_entropies': [],
            'patch_entropies': [],
            'patch_lengths': [],
            'avg_entropies': [],
            'entropy_variances': []
        })
        
        # Collect entropy data
        for sample in samples:
            if 'entropy_model_results' not in sample:
                continue
                
            for model_key, result in sample['entropy_model_results'].items():
                if 'error' in result:
                    continue
                
                model_name = result['model_name']
                
                # Individual token entropies
                if 'entropy_analysis' in result:
                    entropy_analysis = result['entropy_analysis']
                    if entropy_analysis.get('individual_token_entropies'):
                        entropy_data[model_name]['token_entropies'].extend(
                            entropy_analysis['individual_token_entropies']
                        )
                    
                    if entropy_analysis.get('avg_token_entropy') is not None:
                        entropy_data[model_name]['avg_entropies'].append(
                            entropy_analysis['avg_token_entropy']
                        )
                
                # Patch-level data
                if 'patches' in result:
                    for patch in result['patches']:
                        if patch.get('patch_entropy_score') is not None:
                            entropy_data[model_name]['patch_entropies'].append(
                                patch['patch_entropy_score']
                            )
                        entropy_data[model_name]['patch_lengths'].append(
                            patch['patch_length']
                        )
        
        # Display entropy statistics
        if RICH_AVAILABLE:
            table = Table(title="🌡️ Entropy Analysis Summary")
            table.add_column("Model", style="cyan")
            table.add_column("Avg Token Entropy", justify="right")
            table.add_column("Entropy Std Dev", justify="right")
            table.add_column("Avg Patch Entropy", justify="right")
            table.add_column("Avg Patch Length", justify="right")
            table.add_column("Total Tokens", justify="right")
            
            for model_name, data in entropy_data.items():
                if data['token_entropies']:
                    avg_token_entropy = statistics.mean(data['token_entropies'])
                    entropy_std = statistics.stdev(data['token_entropies']) if len(data['token_entropies']) > 1 else 0
                    avg_patch_entropy = statistics.mean(data['patch_entropies']) if data['patch_entropies'] else 0
                    avg_patch_length = statistics.mean(data['patch_lengths']) if data['patch_lengths'] else 0
                    total_tokens = len(data['token_entropies'])
                    
                    table.add_row(
                        model_name,
                        f"{avg_token_entropy:.3f}",
                        f"{entropy_std:.3f}",
                        f"{avg_patch_entropy:.3f}",
                        f"{avg_patch_length:.2f}",
                        str(total_tokens)
                    )
            
            self.console.print(table)
        else:
            print("\n🌡️ ENTROPY ANALYSIS SUMMARY")
            print("-" * 80)
            print(f"{'Model':<20} {'Avg Token Entropy':<15} {'Entropy Std':<12} {'Avg Patch Entropy':<15} {'Avg Patch Len':<12} {'Total Tokens'}")
            print("-" * 80)
            
            for model_name, data in entropy_data.items():
                if data['token_entropies']:
                    avg_token_entropy = statistics.mean(data['token_entropies'])
                    entropy_std = statistics.stdev(data['token_entropies']) if len(data['token_entropies']) > 1 else 0
                    avg_patch_entropy = statistics.mean(data['patch_entropies']) if data['patch_entropies'] else 0
                    avg_patch_length = statistics.mean(data['patch_lengths']) if data['patch_lengths'] else 0
                    total_tokens = len(data['token_entropies'])
                    
                    print(f"{model_name:<20} {avg_token_entropy:<15.3f} {entropy_std:<12.3f} {avg_patch_entropy:<15.3f} {avg_patch_length:<12.2f} {total_tokens}")
    
    def search_samples(self, query: str):
        """Search samples by text content."""
        query_lower = query.lower()
        matching_samples = []
        
        for i, sample in enumerate(self.data['samples'], 1):
            if 'original_data' in sample:
                text = sample['original_data']['text'].lower()
                if query_lower in text:
                    matching_samples.append(i)
        
        if matching_samples:
            print(f"\n🔍 Found {len(matching_samples)} samples matching '{query}':")
            for sample_idx in matching_samples:
                sample = self.data['samples'][sample_idx - 1]
                text = sample['original_data']['text']
                source = sample['original_data'].get('source', 'Unknown')
                print(f"   {sample_idx}. [{source}] {text[:80]}{'...' if len(text) > 80 else ''}")
        else:
            print(f"🔍 No samples found matching '{query}'")
        
        return matching_samples
    
    def export_summary_csv(self, filename: str):
        """Export a summary of results to CSV."""
        summary_data = []
        
        for i, sample in enumerate(self.data['samples'], 1):
            base_row = {
                'sample_id': i,
                'text': sample['original_data']['text'],
                'source': sample['original_data'].get('source', 'Unknown'),
                'year': sample['original_data'].get('year', None),
                'char_length': sample['original_data']['char_length'],
                'word_count': sample['original_data']['word_count']
            }
            
            # Add traditional tokenizer results
            if 'traditional_tokenizer_results' in sample:
                for tokenizer_key, result in sample['traditional_tokenizer_results'].items():
                    if 'error' not in result:
                        row = base_row.copy()
                        row.update({
                            'tokenizer_type': 'traditional',
                            'tokenizer_name': result['tokenizer_name'],
                            'token_count': result['statistics']['token_count'],
                            'compression_ratio': result['statistics']['compression_ratio_chars'],
                            'processing_time_ms': result['processing_time_seconds'] * 1000,
                            'perfect_reconstruction': result['reconstruction']['perfect_reconstruction']
                        })
                        summary_data.append(row)
            
            # Add entropy model results
            if 'entropy_model_results' in sample:
                for model_key, result in sample['entropy_model_results'].items():
                    if 'error' not in result:
                        row = base_row.copy()
                        row.update({
                            'tokenizer_type': 'entropy',
                            'tokenizer_name': result['model_name'],
                            'token_count': result['patch_statistics']['patch_count'],
                            'compression_ratio': result['patch_statistics']['avg_patch_length'],
                            'processing_time_ms': result['processing_time_seconds'] * 1000,
                            'perfect_reconstruction': result['reconstruction']['perfect_reconstruction']
                        })
                        
                        # Add entropy-specific metrics
                        if 'entropy_analysis' in result:
                            entropy_analysis = result['entropy_analysis']
                            row['avg_token_entropy'] = entropy_analysis.get('avg_token_entropy')
                            row['entropy_variance'] = entropy_analysis.get('entropy_variance')
                        
                        summary_data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(summary_data)
        df.to_csv(filename, index=False)
        print(f"📊 Summary exported to {filename}")
        print(f"   {len(summary_data)} rows exported")
    
    def interactive_menu(self):
        """Main interactive menu."""
        while True:
            if RICH_AVAILABLE:
                self.console.print("\n[bold cyan]🔬 TOKENIZER DATA VIEWER[/bold cyan]")
                self.console.print("[dim]Choose an option:[/dim]")
                options = [
                    "1. 📋 Show dataset overview",
                    "2. 🔀 Compare all models",
                    "3. 📄 View sample details",
                    "4. 🌡️ Entropy analysis",
                    "5. 🔍 Search samples",
                    "6. 📊 Export summary to CSV",
                    "7. 🚪 Exit"
                ]
                for option in options:
                    self.console.print(f"   {option}")
                
                choice = Prompt.ask("\nEnter your choice", choices=["1", "2", "3", "4", "5", "6", "7"])
            else:
                print("\n🔬 TOKENIZER DATA VIEWER")
                print("Choose an option:")
                print("   1. 📋 Show dataset overview")
                print("   2. 🔀 Compare all models")
                print("   3. 📄 View sample details")
                print("   4. 🌡️ Entropy analysis")
                print("   5. 🔍 Search samples")
                print("   6. 📊 Export summary to CSV")
                print("   7. 🚪 Exit")
                
                choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == "1":
                self.show_overview()
            
            elif choice == "2":
                self.show_model_comparison()
            
            elif choice == "3":
                if RICH_AVAILABLE:
                    sample_idx = IntPrompt.ask(
                        f"Enter sample number (1-{len(self.data['samples'])})",
                        default=1
                    )
                else:
                    try:
                        sample_idx = int(input(f"Enter sample number (1-{len(self.data['samples'])}): "))
                    except ValueError:
                        print("❌ Invalid number")
                        continue
                
                self.show_sample_detail(sample_idx)
            
            elif choice == "4":
                self.show_entropy_analysis()
            
            elif choice == "5":
                if RICH_AVAILABLE:
                    query = Prompt.ask("Enter search query")
                else:
                    query = input("Enter search query: ")
                
                if query:
                    matching_samples = self.search_samples(query)
                    if matching_samples and len(matching_samples) > 0:
                        if RICH_AVAILABLE:
                            view_detail = Confirm.ask("View details of first matching sample?")
                        else:
                            view_detail = input("View details of first matching sample? (y/n): ").lower() == 'y'
                        
                        if view_detail:
                            self.show_sample_detail(matching_samples[0])
            
            elif choice == "6":
                if RICH_AVAILABLE:
                    filename = Prompt.ask("Enter CSV filename", default="tokenizer_summary.csv")
                else:
                    filename = input("Enter CSV filename (default: tokenizer_summary.csv): ") or "tokenizer_summary.csv"
                
                self.export_summary_csv(filename)
            
            elif choice == "7":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")

def main():
    parser = argparse.ArgumentParser(
        description="Interactive viewer for comprehensive tokenizer analysis data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool provides an interactive interface to explore comprehensive tokenizer analysis data.

Features:
• 📋 Dataset overview with statistics
• 🔀 Model performance comparisons
• 📄 Detailed sample-by-sample analysis
• 🌡️ Entropy analysis for BLT models
• 🔍 Search functionality
• 📊 CSV export capabilities

Examples:
  python data_viewer.py results.json
  python data_viewer.py comprehensive_tokenizer_data_10_samples_20250120_153045.json
        """
    )
    
    parser.add_argument(
        "json_file",
        help="Path to the comprehensive tokenizer analysis JSON file"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run in batch mode (show overview and exit)"
    )
    
    parser.add_argument(
        "--export",
        type=str,
        help="Export summary CSV and exit (provide filename)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.json_file):
        print(f"❌ Error: File '{args.json_file}' not found")
        sys.exit(1)
    
    # Initialize viewer
    viewer = TokenizerDataViewer(args.json_file)
    
    if args.batch:
        # Batch mode: show overview and exit
        viewer.show_overview()
        viewer.show_model_comparison()
    
    elif args.export:
        # Export mode: export CSV and exit
        viewer.export_summary_csv(args.export)
    
    else:
        # Interactive mode
        viewer.interactive_menu()

if __name__ == "__main__":
    main()
