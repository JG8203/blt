#!/usr/bin/env python3

import argparse
import pandas as pd
import random
import os
import torch
import logging
from typing import List, Tuple, Dict, Any, Optional
import time
import json

# Import BLT components
from bytelatent.entropy_model import load_entropy_model
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from bytelatent.data.patcher import Patcher, PatcherArgs, PatchingModeEnum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TokenizerManager:
    """Manages different types of tokenizers for comparison."""
    
    def __init__(self):
        self.tokenizers = {}
        self.load_all_tokenizers()
    
    def load_all_tokenizers(self):
        """Load all available tokenizers."""
        print("🔄 Loading tokenizers for comparison...")
        
        # 1. Tagalog GPT-2 (BPE)
        try:
            from transformers import GPT2Tokenizer
            self.tokenizers['tagalog_gpt2'] = {
                'tokenizer': GPT2Tokenizer.from_pretrained('baebee/gpt2-tagalog-80k'),
                'type': 'huggingface',
                'name': 'Tagalog GPT-2 (BPE)',
                'description': 'Tagalog-specific GPT-2 with 80k vocabulary'
            }
            print("✅ Tagalog GPT-2 loaded")
        except Exception as e:
            print(f"❌ Tagalog GPT-2 failed: {e}")
        
        # 2. Standard GPT-2 (for comparison)
        try:
            from transformers import GPT2Tokenizer
            self.tokenizers['standard_gpt2'] = {
                'tokenizer': GPT2Tokenizer.from_pretrained('gpt2'),
                'type': 'huggingface',
                'name': 'Standard GPT-2 (BPE)',
                'description': 'Standard English GPT-2 tokenizer'
            }
            print("✅ Standard GPT-2 loaded")
        except Exception as e:
            print(f"❌ Standard GPT-2 failed: {e}")
        
        # 3. Llama-3 Tokenizer
        try:
            from transformers import AutoTokenizer
            # Try different Llama-3 variants
            llama_models = [
                "meta-llama/Meta-Llama-3-8B",
                "meta-llama/Meta-Llama-3-8B-Instruct", 
                "NousResearch/Llama-2-7b-hf"  # Fallback
            ]
            
            for model_name in llama_models:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    self.tokenizers['llama3'] = {
                        'tokenizer': tokenizer,
                        'type': 'huggingface',
                        'name': f'Llama-3 ({model_name.split("/")[-1]})',
                        'description': 'Llama-3 SentencePiece tokenizer'
                    }
                    print(f"✅ Llama-3 loaded: {model_name}")
                    break
                except Exception as e:
                    print(f"⚠️ {model_name} failed: {e}")
                    continue
        except Exception as e:
            print(f"❌ Llama-3 failed: {e}")
        
        # 4. Tiktoken tokenizers
        try:
            import tiktoken
            
            # cl100k_base
            self.tokenizers['tiktoken_cl100k'] = {
                'tokenizer': tiktoken.get_encoding("cl100k_base"),
                'type': 'tiktoken',
                'name': 'Tiktoken (cl100k_base)',
                'description': 'OpenAI GPT-4 tokenizer'
            }
            print("✅ Tiktoken cl100k loaded")
            
            # o200k_base
            self.tokenizers['tiktoken_o200k'] = {
                'tokenizer': tiktoken.get_encoding("o200k_base"),
                'type': 'tiktoken', 
                'name': 'Tiktoken (o200k_base)',
                'description': 'OpenAI GPT-4o tokenizer'
            }
            print("✅ Tiktoken o200k loaded")
            
            # o200k_harmony
            try:
                self.tokenizers['tiktoken_harmony'] = {
                    'tokenizer': tiktoken.get_encoding("o200k_harmony"),
                    'type': 'tiktoken',
                    'name': 'Tiktoken (o200k_harmony)',
                    'description': 'OpenAI GPT-OSS harmony tokenizer'
                }
                print("✅ Tiktoken o200k_harmony loaded")
            except Exception as e:
                print(f"⚠️ o200k_harmony not available: {e}")
                
        except Exception as e:
            print(f"❌ Tiktoken failed: {e}")
        
        # 5. ByT5 (byte-level)
        try:
            from transformers import AutoTokenizer
            self.tokenizers['byt5'] = {
                'tokenizer': AutoTokenizer.from_pretrained('google/byt5-small'),
                'type': 'byt5',
                'name': 'ByT5 (byte-level)',
                'description': 'Google ByT5 byte-level tokenizer'
            }
            print("✅ ByT5 loaded")
        except Exception as e:
            print(f"❌ ByT5 failed: {e}")
        
        print(f"📊 Loaded {len(self.tokenizers)} tokenizers for comparison")
    
    def tokenize_text(self, text: str, tokenizer_key: str) -> Tuple[List[Any], List[str], Dict[str, Any]]:
        """
        Tokenize text and return tokens, token strings, and metadata.
        Returns: (token_ids, token_strings, metadata)
        """
        if tokenizer_key not in self.tokenizers:
            return [], [], {"error": "Tokenizer not available"}
        
        tokenizer_info = self.tokenizers[tokenizer_key]
        tokenizer = tokenizer_info['tokenizer']
        tokenizer_type = tokenizer_info['type']
        
        try:
            start_time = time.time()
            
            if tokenizer_type == 'tiktoken':
                # Tiktoken tokenizers
                token_ids = tokenizer.encode(text)
                token_strings = []
                
                for token_id in token_ids:
                    try:
                        token_bytes = tokenizer.decode_single_token_bytes(token_id)
                        token_str = token_bytes.decode('utf-8', errors='replace')
                        token_strings.append(token_str)
                    except:
                        token_strings.append(f"[{token_id}]")
                
                decoded = tokenizer.decode(token_ids)
                
            elif tokenizer_type == 'byt5':
                # ByT5 tokenizer
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                token_strings = []
                
                for token_id in token_ids:
                    token_str = tokenizer.decode([token_id], skip_special_tokens=True)
                    token_strings.append(token_str)
                
                decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
                
            else:
                # Hugging Face tokenizers (GPT-2, Llama, etc.)
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                token_strings = []
                
                for token_id in token_ids:
                    token_str = tokenizer.decode([token_id])
                    token_strings.append(token_str)
                
                decoded = tokenizer.decode(token_ids)
            
            duration = time.time() - start_time
            
            metadata = {
                'token_count': len(token_ids),
                'compression_ratio': len(text) / len(token_ids) if token_ids else 0,
                'processing_time': duration,
                'perfect_reconstruction': text == decoded,
                'decoded_text': decoded,
                'tokenizer_name': tokenizer_info['name'],
                'tokenizer_type': tokenizer_type
            }
            
            return token_ids, token_strings, metadata
            
        except Exception as e:
            return [], [], {"error": str(e)}


def find_latest_checkpoint(base_checkpoint_dir: str) -> str:
    """Finds the latest numerical checkpoint directory or consolidated entropy model."""
    import glob

    # Check if this is HF weights structure (has entropy_model subdirectory)
    hf_entropy_model_path = os.path.join(base_checkpoint_dir, "entropy_model")
    if os.path.isdir(hf_entropy_model_path):
        logging.info(f"Found HF weights entropy model at: {hf_entropy_model_path}")
        return hf_entropy_model_path

    # Handle the full path to models/my_entropy_model_checkpoints/checkpoints
    actual_checkpoint_dir = os.path.join(base_checkpoint_dir, "checkpoints")
    if not os.path.isdir(actual_checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found at: {actual_checkpoint_dir}")

    # Find all numbered subdirectories
    dirs = glob.glob(os.path.join(actual_checkpoint_dir, "[0-9]"*10))
    if not dirs:
        raise FileNotFoundError(f"No numerical checkpoint subdirectories found in {actual_checkpoint_dir}")

    # Return the one with the highest number (latest step)
    latest_dir = max(dirs)
    logging.info(f"Found latest checkpoint at: {latest_dir}")
    return latest_dir


def load_model_and_patcher(model_dir: str, threshold: float, device: str, model_name: str = "") -> Tuple[object, object, Dict]:
    """Load entropy model and create patcher."""
    # Find checkpoint directory
    latest_checkpoint_dir = find_latest_checkpoint(model_dir)

    # Check if this is already a consolidated directory (HF weights)
    direct_state_dict_path = os.path.join(latest_checkpoint_dir, "consolidated.pth")
    direct_params_path = os.path.join(latest_checkpoint_dir, "params.json")

    if os.path.exists(direct_state_dict_path) and os.path.exists(direct_params_path):
        # HF weights case - files are directly in the directory
        consolidated_dir = latest_checkpoint_dir
        state_dict_path = direct_state_dict_path
        params_path = direct_params_path
        logging.info(f"Using consolidated files directly from: {consolidated_dir}")
    else:
        # Training checkpoint case - need consolidated subdirectory
        consolidated_dir = os.path.join(latest_checkpoint_dir, "consolidated")
        if not os.path.exists(consolidated_dir):
            logging.info("Consolidated checkpoint not found. Running consolidation script...")
            from bytelatent.checkpoint import main as consolidate_main
            consolidate_main(command="consolidate", model_checkpoint_dir=latest_checkpoint_dir)
            logging.info("Consolidation complete.")

        state_dict_path = os.path.join(consolidated_dir, "consolidated.pth")
        params_path = os.path.join(consolidated_dir, "params.json")

    if not os.path.exists(state_dict_path) or not os.path.exists(params_path):
        raise FileNotFoundError(f"Could not find 'consolidated.pth' or 'params.json' in '{consolidated_dir}'")

    # Load model parameters to get model info
    import json
    with open(params_path, 'r') as f:
        model_params = json.load(f)

    # Load the entropy model
    entropy_model, _ = load_entropy_model(consolidated_dir, state_dict_path, device=device)
    entropy_model.eval()

    # Setup the patcher
    patcher_args = PatcherArgs(
        patching_mode=PatchingModeEnum.entropy,
        threshold=threshold,
        device=device,
        patching_device=device,
        realtime_patching=True,
        entropy_model_checkpoint_dir=consolidated_dir
    )
    patcher = patcher_args.build()
    patcher.entropy_model = entropy_model

    # Get model info
    model_info = {
        'name': model_name,
        'params': model_params,
        'checkpoint_dir': consolidated_dir,
        'threshold': threshold,
        'type': 'entropy_model'
    }

    return entropy_model, patcher, model_info


def get_patches_for_sentence(sentence: str, patcher: object, tokenizer: object, device: str) -> Tuple[List[int], List[str], Dict]:
    """Get patch lengths and patch strings for a sentence."""
    start_time = time.time()
    
    # Encode the sentence
    tokens_list = tokenizer.encode(sentence, add_bos=True, add_eos=True)
    tokens_tensor = torch.tensor([tokens_list], device=device)

    # Get patches
    patch_lengths_tensor, scores_tensor = patcher.patch(tokens_tensor)
    patch_lengths = patch_lengths_tensor.squeeze().tolist()

    # Convert patches to strings
    patch_strings = []
    current_pos = 0
    for length in patch_lengths:
        patch_token_ids = tokens_list[current_pos : current_pos + length]
        patch_strings.append(tokenizer.decode(patch_token_ids))
        current_pos += length

    duration = time.time() - start_time
    
    metadata = {
        'patch_count': len(patch_lengths),
        'total_tokens': sum(patch_lengths),
        'compression_ratio': sum(patch_lengths) / len(patch_lengths) if patch_lengths else 0,
        'processing_time': duration,
        'perfect_reconstruction': True,  # BLT should always reconstruct perfectly
        'decoded_text': ''.join(patch_strings),
        'type': 'entropy_patches'
    }

    return patch_lengths, patch_strings, metadata


def comprehensive_comparison(csv_file: str, num_samples: int, my_model_dir: str, hf_model_dir: str, 
                           threshold: float, save_to_file: bool = False, source_filter: str = None,
                           include_entropy_models: bool = True, include_traditional_tokenizers: bool = True):
    """
    Comprehensive comparison of entropy models and traditional tokenizers.
    """
    print(f"\n🚀 COMPREHENSIVE TOKENIZER & ENTROPY MODEL COMPARISON")
    print("=" * 80)
    print(f"📊 Samples: {num_samples}")
    if include_entropy_models:
        print(f"🧠 My Model: {my_model_dir}")
        print(f"🤗 HF Model: {hf_model_dir}")
        print(f"🎯 Threshold: {threshold}")
    print(f"🔤 Traditional Tokenizers: {'✅ Enabled' if include_traditional_tokenizers else '❌ Disabled'}")
    print(f"🧠 Entropy Models: {'✅ Enabled' if include_entropy_models else '❌ Disabled'}")
    print("=" * 80)

    # Load CSV and sample sentences
    df = pd.read_csv(csv_file)

    # Filter by source if specified
    if source_filter:
        df = df[df['source'] == source_filter]
        if df.empty:
            raise ValueError(f"No rows found with source '{source_filter}'. Available sources: {list(df['source'].unique())}")
        print(f"📋 Filtered to {len(df)} rows with source '{source_filter}'")

    sampled_rows = df.sample(n=min(num_samples, len(df)), random_state=42)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Initialize components
    models = {}
    
    # Load entropy models if requested
    if include_entropy_models:
        print("\n🧠 Loading entropy models...")
        
        # Initialize BLT tokenizer
        blt_tokenizer = BltTokenizer()
        
        try:
            logging.info("Loading my entropy model...")
            my_model, my_patcher, my_model_info = load_model_and_patcher(my_model_dir, threshold, device, "My Model")
            models['my_entropy'] = {
                'patcher': my_patcher,
                'tokenizer': blt_tokenizer,
                'info': my_model_info,
                'type': 'entropy_model'
            }
            print("✅ My entropy model loaded")
        except Exception as e:
            print(f"❌ My entropy model failed: {e}")

        try:
            logging.info("Loading HF entropy model...")
            hf_model, hf_patcher, hf_model_info = load_model_and_patcher(hf_model_dir, threshold, device, "HF Model")
            models['hf_entropy'] = {
                'patcher': hf_patcher,
                'tokenizer': blt_tokenizer,
                'info': hf_model_info,
                'type': 'entropy_model'
            }
            print("✅ HF entropy model loaded")
        except Exception as e:
            print(f"❌ HF entropy model failed: {e}")
    
    # Load traditional tokenizers if requested
    tokenizer_manager = None
    if include_traditional_tokenizers:
        print("\n🔤 Loading traditional tokenizers...")
        tokenizer_manager = TokenizerManager()

    # Print model information
    if include_entropy_models:
        print(f"\n📋 ENTROPY MODEL INFORMATION:")
        for key, model_data in models.items():
            if model_data['type'] == 'entropy_model':
                info = model_data['info']
                print(f"  {info['name']}:")
                print(f"    Checkpoint: {info['checkpoint_dir']}")
                print(f"    Parameters: {info['params'].get('n_params', 'Unknown')}")
                print(f"    Dimensions: {info['params'].get('dim', 'Unknown')}")
                print(f"    Threshold: {info['threshold']}")

    if include_traditional_tokenizers and tokenizer_manager:
        print(f"\n📋 TRADITIONAL TOKENIZER INFORMATION:")
        for key, tokenizer_data in tokenizer_manager.tokenizers.items():
            print(f"  {tokenizer_data['name']}: {tokenizer_data['description']}")

    # Collect statistics
    all_results = []
    segment_statistics = {}

    # Process each sample
    for i, (idx, row) in enumerate(sampled_rows.iterrows(), 1):
        sentence = row['text']
        print(f"\n{'='*20} SAMPLE {i}/{num_samples} {'='*20}")
        print(f"📝 Sentence: {sentence}")
        print(f"📚 Source: {row['source']} ({row['year']})")
        print(f"📏 Length: {len(sentence)} characters, {len(sentence.split())} words")
        
        sample_results = {
            'sample_id': i,
            'sentence': sentence,
            'source': row['source'],
            'year': row['year'],
            'char_length': len(sentence),
            'word_count': len(sentence.split()),
            'segmentations': {}
        }

        # Test entropy models
        if include_entropy_models:
            print(f"\n🧠 ENTROPY MODEL RESULTS:")
            for model_key, model_data in models.items():
                if model_data['type'] != 'entropy_model':
                    continue
                    
                try:
                    patch_lengths, patch_strings, metadata = get_patches_for_sentence(
                        sentence, model_data['patcher'], model_data['tokenizer'], device
                    )
                    
                    model_name = model_data['info']['name']
                    print(f"\n  {model_name}:")
                    print(f"    Patches: {len(patch_strings)} (avg length: {metadata['compression_ratio']:.2f})")
                    print(f"    Segments: {' | '.join([f'[{patch}]' for patch in patch_strings])}")
                    print(f"    Lengths: {patch_lengths}")
                    print(f"    Processing time: {metadata['processing_time']:.4f}s")
                    
                    sample_results['segmentations'][model_key] = {
                        'segments': patch_strings,
                        'segment_lengths': patch_lengths,
                        'segment_count': len(patch_strings),
                        'total_tokens': metadata['total_tokens'],
                        'avg_segment_length': metadata['compression_ratio'],
                        'processing_time': metadata['processing_time'],
                        'type': 'entropy_patches',
                        'model_name': model_name
                    }
                    
                    # Collect for statistics
                    if model_key not in segment_statistics:
                        segment_statistics[model_key] = {'lengths': [], 'counts': [], 'times': []}
                    segment_statistics[model_key]['lengths'].extend(patch_lengths)
                    segment_statistics[model_key]['counts'].append(len(patch_strings))
                    segment_statistics[model_key]['times'].append(metadata['processing_time'])
                    
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    sample_results['segmentations'][model_key] = {'error': str(e)}

        # Test traditional tokenizers
        if include_traditional_tokenizers and tokenizer_manager:
            print(f"\n🔤 TRADITIONAL TOKENIZER RESULTS:")
            for tokenizer_key, tokenizer_data in tokenizer_manager.tokenizers.items():
                try:
                    token_ids, token_strings, metadata = tokenizer_manager.tokenize_text(sentence, tokenizer_key)
                    
                    if 'error' in metadata:
                        print(f"\n  {tokenizer_data['name']}: ❌ {metadata['error']}")
                        continue
                    
                    print(f"\n  {tokenizer_data['name']}:")
                    print(f"    Tokens: {metadata['token_count']} (avg length: {metadata['compression_ratio']:.2f})")
                    print(f"    Segments: {' | '.join([f'[{token}]' for token in token_strings[:10]])}{'...' if len(token_strings) > 10 else ''}")
                    print(f"    Perfect reconstruction: {'✅' if metadata['perfect_reconstruction'] else '❌'}")
                    print(f"    Processing time: {metadata['processing_time']:.4f}s")
                    
                    sample_results['segmentations'][tokenizer_key] = {
                        'segments': token_strings,
                        'segment_lengths': [1] * len(token_strings),  # Traditional tokens are length 1
                        'segment_count': metadata['token_count'],
                        'total_tokens': metadata['token_count'],
                        'avg_segment_length': metadata['compression_ratio'],
                        'processing_time': metadata['processing_time'],
                        'perfect_reconstruction': metadata['perfect_reconstruction'],
                        'type': 'traditional_tokens',
                        'model_name': tokenizer_data['name']
                    }
                    
                    # Collect for statistics
                    if tokenizer_key not in segment_statistics:
                        segment_statistics[tokenizer_key] = {'lengths': [], 'counts': [], 'times': []}
                    segment_statistics[tokenizer_key]['lengths'].extend([1] * metadata['token_count'])
                    segment_statistics[tokenizer_key]['counts'].append(metadata['token_count'])
                    segment_statistics[tokenizer_key]['times'].append(metadata['processing_time'])
                    
                except Exception as e:
                    print(f"  {tokenizer_data['name']}: ❌ Error: {e}")
                    sample_results['segmentations'][tokenizer_key] = {'error': str(e)}

        all_results.append(sample_results)
        print("-" * 80)

    # Print comprehensive statistics
    print(f"\n📊 COMPREHENSIVE STATISTICS")
    print("=" * 80)
    
    print(f"\n📈 SEGMENTATION EFFICIENCY COMPARISON:")
    print(f"{'Method':<30} {'Avg Segments':<12} {'Avg Length':<12} {'Avg Time (ms)':<12} {'Type':<15}")
    print("-" * 85)
    
    efficiency_ranking = []
    
    for method_key, stats in segment_statistics.items():
        if not stats['counts']:
            continue
            
        avg_segments = sum(stats['counts']) / len(stats['counts'])
        avg_length = sum(stats['lengths']) / len(stats['lengths']) if stats['lengths'] else 0
        avg_time_ms = (sum(stats['times']) / len(stats['times'])) * 1000
        
        # Determine method name and type
        if method_key in models:
            method_name = models[method_key]['info']['name']
            method_type = "Entropy"
        elif tokenizer_manager and method_key in tokenizer_manager.tokenizers:
            method_name = tokenizer_manager.tokenizers[method_key]['name']
            method_type = "Traditional"
        else:
            method_name = method_key
            method_type = "Unknown"
        
        print(f"{method_name:<30} {avg_segments:<12.1f} {avg_length:<12.2f} {avg_time_ms:<12.2f} {method_type:<15}")
        
        efficiency_ranking.append({
            'method': method_name,
            'avg_segments': avg_segments,
            'avg_length': avg_length,
            'compression_efficiency': avg_length,  # Higher is better (fewer segments needed)
            'speed': 1/avg_time_ms if avg_time_ms > 0 else 0  # Higher is better
        })

    # Efficiency analysis
    print(f"\n🏆 EFFICIENCY RANKINGS:")
    
    # Sort by compression efficiency (higher avg segment length = fewer segments = better compression)
    compression_ranking = sorted(efficiency_ranking, key=lambda x: x['compression_efficiency'], reverse=True)
    print(f"\n📦 Best Compression (fewest segments):")
    for i, method in enumerate(compression_ranking[:5], 1):
        print(f"  {i}. {method['method']}: {method['compression_efficiency']:.2f} chars/segment")
    
    # Sort by speed
    speed_ranking = sorted(efficiency_ranking, key=lambda x: x['speed'], reverse=True)
    print(f"\n⚡ Fastest Processing:")
    for i, method in enumerate(speed_ranking[:5], 1):
        if method['speed'] > 0:
            print(f"  {i}. {method['method']}: {1/method['speed']:.2f}ms avg")

    # Save results if requested
    if save_to_file:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        output_file = f"comprehensive_comparison_{num_samples}_samples_{timestamp}.json"
        
        results_data = {
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_samples': num_samples,
                'source_filter': source_filter,
                'threshold': threshold if include_entropy_models else None,
                'include_entropy_models': include_entropy_models,
                'include_traditional_tokenizers': include_traditional_tokenizers
            },
            'samples': all_results,
            'statistics': segment_statistics,
            'efficiency_ranking': efficiency_ranking
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        # Save summary CSV
        csv_file = f"comprehensive_summary_{num_samples}_samples_{timestamp}.csv"
        summary_data = []
        
        for sample in all_results:
            for method_key, result in sample['segmentations'].items():
                if 'error' not in result:
                    summary_data.append({
                        'sample_id': sample['sample_id'],
                        'sentence': sample['sentence'][:100] + '...' if len(sample['sentence']) > 100 else sample['sentence'],
                        'source': sample['source'],
                        'method': result['model_name'],
                        'segment_count': result['segment_count'],
                        'avg_segment_length': result['avg_segment_length'],
                        'processing_time_ms': result['processing_time'] * 1000,
                        'type': result['type']
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(csv_file, index=False)
        print(f"📊 Summary CSV saved to: {csv_file}")

    print(f"\n✅ COMPREHENSIVE COMPARISON COMPLETE!")
    print("=" * 80)
    print("🎯 Key Insights:")
    print("• Entropy models provide adaptive segmentation based on information content")
    print("• Traditional tokenizers offer consistent, predictable segmentation")
    print("• Tagalog-specific models should outperform generic ones on Filipino text")
    print("• ByT5 provides language-agnostic robustness at the cost of longer sequences")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive comparison of entropy models and traditional tokenizers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script provides a comprehensive comparison between:

ENTROPY MODELS (BLT approach):
• Your trained entropy model
• HuggingFace reference entropy model

TRADITIONAL TOKENIZERS:
• Tagalog GPT-2 (baebee/gpt2-tagalog-80k)
• Standard GPT-2 (OpenAI)
• Llama-3 SentencePiece
• Tiktoken (cl100k_base, o200k_base, o200k_harmony)
• ByT5 (byte-level)

Examples:
  python comprehensive_comparison.py --csv_file COHFIE_V4.csv --num_samples 10
  python comprehensive_comparison.py --csv_file subset.csv --save_to_file --source_filter "twitter"
  python comprehensive_comparison.py --csv_file data.csv --no_entropy --traditional_only
        """
    )
    
    parser.add_argument(
        "--csv_file",
        type=str,
        default="lang_dis_dataset.csv",
        help="Path to the CSV file containing sentences."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of random sentences to sample and compare."
    )
    parser.add_argument(
        "--my_model_dir",
        type=str,
        default="models/my_entropy_model_checkpoints",
        help="Path to your trained entropy model directory."
    )
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        default="hf-weights",
        help="Path to the HuggingFace weights directory."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.33,
        help="The entropy threshold to use for creating patches."
    )
    parser.add_argument(
        "--save_to_file",
        action="store_true",
        help="Save the comparison results to JSON and CSV files."
    )
    parser.add_argument(
        "--source_filter",
        type=str,
        default=None,
        help="Filter sentences by source (e.g., 'twitter', 'wikipedia')."
    )
    parser.add_argument(
        "--no_entropy",
        action="store_true",
        help="Skip entropy model comparison, only test traditional tokenizers."
    )
    parser.add_argument(
        "--traditional_only",
        action="store_true",
        help="Only test traditional tokenizers (same as --no_entropy)."
    )
    parser.add_argument(
        "--entropy_only",
        action="store_true",
        help="Only test entropy models, skip traditional tokenizers."
    )

    args = parser.parse_args()
    
    # Handle argument logic
    include_entropy = not (args.no_entropy or args.traditional_only)
    include_traditional = not args.entropy_only
    
    if not include_entropy and not include_traditional:
        print("❌ Error: Cannot disable both entropy models and traditional tokenizers!")
        exit(1)

    comprehensive_comparison(
        args.csv_file,
        args.num_samples,
        args.my_model_dir,
        args.hf_model_dir,
        args.threshold,
        args.save_to_file,
        args.source_filter,
        include_entropy,
        include_traditional
    )
