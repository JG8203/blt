#!/usr/bin/env python3
"""
Batch preprocessing script to convert JSONL files to Arrow format using entropy model.
"""

import os
import subprocess
import sys
import json
from pathlib import Path
from tqdm import tqdm

def count_lines_in_jsonl(file_path):
    """Count the number of lines in a JSONL file."""
    try:
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Warning: Could not count lines in {file_path}: {e}")
        return 0

def run_preprocessing(input_file, output_file, entropy_checkpoint_dir, entropy_state_dict_path, pbar=None):
    """Run the preprocessing command for a single file."""
    cmd = [
        "python", "bytelatent/preprocess/preprocess_entropies.py",
        input_file,
        output_file,
        "--entropy-model-checkpoint-dir", entropy_checkpoint_dir,
        "--entropy-model-state-dict-path", entropy_state_dict_path,
        "--bpe-tokenizer-path", ""
    ]
    
    print(f"Processing {input_file} -> {output_file}")
    
    # Create a subprocess that we can monitor
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    
    # Monitor the output and update progress bar
    stdout_lines = []
    stderr_lines = []
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            stdout_lines.append(output.strip())
            # Update progress bar if we see processing indicators
            if pbar and ("--- Processing document" in output or "Completed steps:" in output):
                pbar.update(1)
    
    # Get any remaining output
    remaining_stdout, remaining_stderr = process.communicate()
    if remaining_stdout:
        stdout_lines.extend(remaining_stdout.strip().split('\n'))
    if remaining_stderr:
        stderr_lines.extend(remaining_stderr.strip().split('\n'))
    
    if process.returncode == 0:
        print(f"✓ Successfully processed {input_file}")
        return True
    else:
        print(f"✗ Failed to process {input_file}")
        if stderr_lines:
            print(f"Error: {chr(10).join(stderr_lines)}")
        return False

def main():
    # Configuration
    entropy_checkpoint_dir = "models/my_entropy_model_checkpoints/checkpoints/0000100000"
    entropy_state_dict_path = "models/my_entropy_model_checkpoints/checkpoints/0000100000/consolidated/consolidated.pth"
    
    input_dir = Path("data/my_dataset_raw_chunks")
    output_dir = Path("data/my_dataset_preprocessed")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSONL files in the input directory
    jsonl_files = list(input_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to process")
    
    # Count total lines across all files for progress tracking
    print("Counting total lines in all JSONL files...")
    total_lines = 0
    file_line_counts = {}
    for jsonl_file in sorted(jsonl_files):
        line_count = count_lines_in_jsonl(jsonl_file)
        file_line_counts[jsonl_file] = line_count
        total_lines += line_count
        print(f"  {jsonl_file.name}: {line_count:,} lines")
    
    print(f"Total lines across all files: {total_lines:,}")
    
    success_count = 0
    total_count = len(jsonl_files)
    
    # Create overall progress bar for individual lines
    with tqdm(total=total_lines, desc="Processing lines", unit="lines") as overall_pbar:
        for jsonl_file in sorted(jsonl_files):
            print(f"\n{'='*60}")
            print(f"Processing file: {jsonl_file.name}")
            print(f"Lines in this file: {file_line_counts[jsonl_file]:,}")
            
            # Generate output filename
            output_file = output_dir / f"{jsonl_file.stem}.arrow"
            
            # Process the file with progress tracking
            if run_preprocessing(str(jsonl_file), str(output_file), entropy_checkpoint_dir, entropy_state_dict_path, overall_pbar):
                success_count += 1
                print(f"✓ Completed {jsonl_file.name}")
            else:
                print(f"✗ Failed {jsonl_file.name}")
                # Still update progress bar for failed files to keep it accurate
                remaining_lines = file_line_counts[jsonl_file]
                overall_pbar.update(remaining_lines)
    
    print(f"\n{'='*60}")
    print(f"Processing complete: {success_count}/{total_count} files processed successfully")
    print(f"Total lines processed: {total_lines:,}")

if __name__ == "__main__":
    main()