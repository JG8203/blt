#!/usr/bin/env python3
"""
Preprocess raw JSONL data to add entropy calculations for BLT training.
Uses the existing entropy model checkpoints to compute entropies and converts to Arrow format.
"""

import os
import glob
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Preprocess data for BLT training")
    parser.add_argument("--input-dir", default="data/my_dataset_raw_chunks", 
                       help="Directory containing raw JSONL files")
    parser.add_argument("--output-dir", default="data/my_dataset_preprocessed", 
                       help="Directory to save preprocessed files")
    parser.add_argument("--entropy-model-checkpoint", 
                       default="models/my_entropy_model_checkpoints/checkpoints/0000100000/consolidated",
                       help="Path to entropy model checkpoint")
    parser.add_argument("--log-step", type=int, default=1000,
                       help="Log progress every N steps")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all JSONL files in the input directory
    jsonl_files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files to process")
    
    for jsonl_file in jsonl_files:
        # Get the base filename without extension
        base_name = Path(jsonl_file).stem
        output_file = os.path.join(args.output_dir, f"{base_name}.arrow")
        
        print(f"\nProcessing: {jsonl_file}")
        print(f"Output: {output_file}")
        
        # Run the preprocessing command
        cmd = [
            "python", "-m", "bytelatent.preprocess.preprocess_entropies",
            jsonl_file,
            output_file,
            "--entropy-model-checkpoint-dir", args.entropy_model_checkpoint,
            "--entropy-model-state-dict-path", os.path.join(args.entropy_model_checkpoint, "consolidated.pth"),
            "--bpe-tokenizer-path", "",  # Use empty string to bypass BPE tokenizer
            "--log-step", str(args.log_step),
            "--patching-device", "cuda"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Execute the command
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully processed {jsonl_file}")
        else:
            print(f"❌ Error processing {jsonl_file}")
            print(f"Error: {result.stderr}")
            continue
    
    print("\n🎉 All files processed successfully!")
    print(f"Preprocessed data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()