#!/usr/bin/env python3

import argparse
import pandas as pd
import random
import os
import torch
import logging
from typing import List, Tuple, Dict

# Import necessary components from the bytelatent library
from bytelatent.entropy_model import load_entropy_model
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from bytelatent.data.patcher import Patcher, PatcherArgs, PatchingModeEnum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        'threshold': threshold
    }
    
    return entropy_model, patcher, model_info

def get_patches_for_sentence(sentence: str, patcher: object, tokenizer: object, device: str) -> Tuple[List[int], List[str]]:
    """Get patch lengths and patch strings for a sentence."""
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
    
    return patch_lengths, patch_strings

def compare_entropy_models(csv_file: str, num_samples: int, my_model_dir: str, hf_model_dir: str, threshold: float, save_to_file: bool = False, source_filter: str = None):
    """Compare patching results between two entropy models on random sentences."""
    print(f"\n=== Entropy Model Comparison ===")
    print(f"Samples: {num_samples}")
    print(f"My Model: {my_model_dir}")
    print(f"HF Model: {hf_model_dir}")
    print(f"Threshold: {threshold}")
    print("=" * 50)
    
    # Load CSV and sample sentences
    df = pd.read_csv(csv_file)
    
    # Filter by source if specified
    if source_filter:
        df = df[df['source'] == source_filter]
        if df.empty:
            raise ValueError(f"No rows found with source '{source_filter}'. Available sources: {list(df['source'].unique())}")
        print(f"Filtered to {len(df)} rows with source '{source_filter}'")
    
    sampled_rows = df.sample(n=min(num_samples, len(df)), random_state=42)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = BltTokenizer()
    
    # Load both models
    print("\nLoading models...")
    logging.info("Loading my entropy model...")
    my_model, my_patcher, my_model_info = load_model_and_patcher(my_model_dir, threshold, device, "My Model")
    
    logging.info("Loading HF entropy model...")
    hf_model, hf_patcher, hf_model_info = load_model_and_patcher(hf_model_dir, threshold, device, "HF Model")
    
    print("Models loaded successfully!\n")
    
    # Print model information
    print(f"My Model Info:")
    print(f"  Checkpoint: {my_model_info['checkpoint_dir']}")
    print(f"  Parameters: {my_model_info['params'].get('n_params', 'Unknown')}")
    print(f"  Dimensions: {my_model_info['params'].get('dim', 'Unknown')}")
    
    print(f"\nHF Model Info:")
    print(f"  Checkpoint: {hf_model_info['checkpoint_dir']}")
    print(f"  Parameters: {hf_model_info['params'].get('n_params', 'Unknown')}")
    print(f"  Dimensions: {hf_model_info['params'].get('dim', 'Unknown')}")
    print()
    
    # Collect statistics
    my_all_lengths = []
    hf_all_lengths = []
    results = []
    
    # Setup output
    output_lines = []
    
    # Process each sample
    for i, (idx, row) in enumerate(sampled_rows.iterrows(), 1):
        sentence = row['text']
        
        output = f"\n--- Sample {i}/{num_samples} ---\n"
        output += f"Sentence: {sentence}\n"
        output += f"Source: {row['source']} ({row['year']})\n"
        
        try:
            # Get patches from my model
            my_lengths, my_patches = get_patches_for_sentence(sentence, my_patcher, tokenizer, device)
            
            # Get patches from HF model
            hf_lengths, hf_patches = get_patches_for_sentence(sentence, hf_patcher, tokenizer, device)
            
            # Collect lengths for statistics
            my_all_lengths.extend(my_lengths)
            hf_all_lengths.extend(hf_lengths)
            
            # Calculate average lengths for this sentence
            my_avg = sum(my_lengths) / len(my_lengths) if my_lengths else 0
            hf_avg = sum(hf_lengths) / len(hf_lengths) if hf_lengths else 0
            
            output += f"\nMy Model Patches ({len(my_patches)} patches, avg length: {my_avg:.2f}):\n"
            output += " | ".join([f"[{patch}]" for patch in my_patches]) + "\n"
            output += f"Lengths: {my_lengths}\n"
            
            output += f"\nHF Model Patches ({len(hf_patches)} patches, avg length: {hf_avg:.2f}):\n"
            output += " | ".join([f"[{patch}]" for patch in hf_patches]) + "\n"
            output += f"Lengths: {hf_lengths}\n"
            
            # Enhanced comparison analysis
            if my_lengths == hf_lengths:
                output += "✅ Identical patching: Same number and lengths\n"
            else:
                # Analyze differences
                total_tokens_my = sum(my_lengths)
                total_tokens_hf = sum(hf_lengths)
                
                if total_tokens_my != total_tokens_hf:
                    output += f"⚠️  Token count mismatch: My={total_tokens_my}, HF={total_tokens_hf}\n"
                else:
                    output += f"📊 Different segmentation strategies:\n"
                    output += f"   My Model: {len(my_patches)} patches (avg: {my_avg:.2f})\n"
                    output += f"   HF Model: {len(hf_patches)} patches (avg: {hf_avg:.2f})\n"
                    
                    # Compression efficiency comparison
                    my_compression = total_tokens_my / len(my_patches) if my_patches else 0
                    hf_compression = total_tokens_hf / len(hf_patches) if hf_patches else 0
                    
                    if my_compression > hf_compression:
                        output += f"   📈 My Model is more efficient (fewer patches)\n"
                    elif hf_compression > my_compression:
                        output += f"   📈 HF Model is more efficient (fewer patches)\n"
                    else:
                        output += f"   ⚖️  Similar efficiency\n"
            
            results.append({
                'sentence': sentence,
                'my_patches': len(my_patches),
                'hf_patches': len(hf_patches),
                'my_avg_length': my_avg,
                'hf_avg_length': hf_avg,
                'my_total_tokens': sum(my_lengths),
                'hf_total_tokens': sum(hf_lengths),
                'identical_patching': my_lengths == hf_lengths,
                'my_patch_strings': " | ".join([f"[{patch}]" for patch in my_patches]),
                'hf_patch_strings': " | ".join([f"[{patch}]" for patch in hf_patches])
            })
                
        except Exception as e:
            output += f"Error processing sentence {i}: {e}\n"
            logging.error(f"Error processing sentence {i}: {e}")
            continue
        
        output += "-" * 80 + "\n"
        output_lines.append(output)
        print(output)
    
    # Print enhanced overall statistics
    overall_stats = f"\n=== Overall Statistics ===\n"
    if my_all_lengths and hf_all_lengths:
        my_overall_avg = sum(my_all_lengths) / len(my_all_lengths)
        hf_overall_avg = sum(hf_all_lengths) / len(hf_all_lengths)
        my_total_tokens = sum(my_all_lengths)
        hf_total_tokens = sum(hf_all_lengths)
        
        overall_stats += f"\nModel Comparison Summary:\n"
        overall_stats += f"My Model:\n"
        overall_stats += f"  - Total patches: {len(my_all_lengths)}\n"
        overall_stats += f"  - Total tokens: {my_total_tokens}\n"
        overall_stats += f"  - Average patch length: {my_overall_avg:.2f}\n"
        overall_stats += f"  - Compression ratio: {my_total_tokens / len(my_all_lengths):.2f} tokens/patch\n"
        
        overall_stats += f"\nHF Model:\n"
        overall_stats += f"  - Total patches: {len(hf_all_lengths)}\n"
        overall_stats += f"  - Total tokens: {hf_total_tokens}\n"
        overall_stats += f"  - Average patch length: {hf_overall_avg:.2f}\n"
        overall_stats += f"  - Compression ratio: {hf_total_tokens / len(hf_all_lengths):.2f} tokens/patch\n"
        
        overall_stats += f"\nComparison:\n"
        overall_stats += f"  - Patch count difference: {abs(len(my_all_lengths) - len(hf_all_lengths))} patches\n"
        overall_stats += f"  - Average length difference: {abs(my_overall_avg - hf_overall_avg):.2f}\n"
        
        # Efficiency analysis
        if len(my_all_lengths) < len(hf_all_lengths):
            efficiency_diff = ((len(hf_all_lengths) - len(my_all_lengths)) / len(hf_all_lengths)) * 100
            overall_stats += f"  - My Model uses {efficiency_diff:.1f}% fewer patches\n"
        elif len(hf_all_lengths) < len(my_all_lengths):
            efficiency_diff = ((len(my_all_lengths) - len(hf_all_lengths)) / len(my_all_lengths)) * 100
            overall_stats += f"  - HF Model uses {efficiency_diff:.1f}% fewer patches\n"
        else:
            overall_stats += f"  - Both models use the same number of patches\n"
            
        
    overall_stats += "=" * 50 + "\n"
    
    print(overall_stats)
    
    # Save to file if requested
    if save_to_file:
        output_file = f"entropy_comparison_{num_samples}_samples.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Entropy Model Comparison ===\n")
            f.write(f"Samples: {num_samples}\n")
            f.write(f"My Model: {my_model_dir}\n")
            f.write(f"HF Model: {hf_model_dir}\n")
            f.write(f"Threshold: {threshold}\n")
            f.write("=" * 50 + "\n")
            
            # Write model info
            f.write(f"\nMy Model Info:\n")
            f.write(f"  Checkpoint: {my_model_info['checkpoint_dir']}\n")
            f.write(f"  Parameters: {my_model_info['params'].get('n_params', 'Unknown')}\n")
            f.write(f"  Dimensions: {my_model_info['params'].get('dim', 'Unknown')}\n")
            
            f.write(f"\nHF Model Info:\n")
            f.write(f"  Checkpoint: {hf_model_info['checkpoint_dir']}\n")
            f.write(f"  Parameters: {hf_model_info['params'].get('n_params', 'Unknown')}\n")
            f.write(f"  Dimensions: {hf_model_info['params'].get('dim', 'Unknown')}\n\n")
            
            for line in output_lines:
                f.write(line)
            f.write(overall_stats)
        print(f"Results saved to: {output_file}")
        
        # Also save a CSV summary
        csv_file = f"entropy_comparison_{num_samples}_samples.csv"
        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_file, index=False)
        print(f"CSV summary saved to: {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare entropy model patching results on random sentences from CSV.")
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
        help="Save the comparison results to a text file."
    )
    parser.add_argument(
        "--source_filter",
        type=str,
        default=None,
        help="Filter sentences by source document (e.g., 'abante', 'inquirer'). If not specified, samples from all sources."
    )
    
    args = parser.parse_args()
    
    compare_entropy_models(
        args.csv_file,
        args.num_samples,
        args.my_model_dir,
        args.hf_model_dir,
        args.threshold,
        args.save_to_file,
        args.source_filter
    )