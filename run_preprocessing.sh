#!/bin/bash

# This script will exit immediately if any command fails.
set -e
set -o pipefail

# --- User Configuration ---
# Directory with your raw .jsonl chunks
INPUT_DIR="data/my_dataset_raw_chunks"

# Directory for the final preprocessed .arrow files
OUTPUT_DIR="data/my_dataset_preprocessed"

# CRITICAL: This must point to the *consolidated* checkpoint directory of your trained entropy model.
# This directory should contain 'consolidated.pth' and 'params.json'.
ENTROPY_MODEL_DIR="models/my_entropy_model_checkpoints/checkpoints/0000100000/consolidated"

# --- Performance Configuration ---
# Set how many files to process in parallel. This should ideally match your number of GPUs.
MAX_JOBS=4

# --- NEW: GPU Configuration ---
# Set the number of GPUsyou want to use
NUM_GPUS=1
# ----------------------------

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting entropy calculation for all chunks in: $INPUT_DIR"
echo "Preprocessed files will be saved to: $OUTPUT_DIR"
echo "Using entropy model from: $ENTROPY_MODEL_DIR"
echo "Running up to $MAX_JOBS jobs in parallel across $NUM_GPUS GPUs."
echo "================================================================="

# --- Main Processing Loop ---
job_counter=0
skipped_count=0
processed_count=0

# Loop through every .jsonl file in the input directory
#
# vvvvvv THIS IS THE CORRECTED LINE vvvvvv
for input_file in "$INPUT_DIR"/*.jsonl; do
# ^^^^^^ THIS IS THE CORRECTED LINE ^^^^^^
#
  # Exit if no files are found to prevent errors
  [ -e "$input_file" ] || { echo "No .jsonl files found in $INPUT_DIR. Exiting."; exit 1; }

  # Get the base filename without the .jsonl extension for cleaner output names
  base_name=$(basename "$input_file" .jsonl)
  output_file="$OUTPUT_DIR/${base_name}.arrow"
  complete_marker="${output_file}.complete" # Success marker file

  # --- Check if already processed ---
  if [ -f "$complete_marker" ]; then
    echo "--> SKIPPING: $base_name (already processed)"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  # --- Run the preprocessing job in the background ---

  # --- NEW: Assign a GPU to this job ---
  # This calculates which GPU to use (0, 1, 2, 3, 0, 1, ...)
  gpu_id=$((job_counter % NUM_GPUS))
  echo "--> PROCESSING: $base_name on GPU:$gpu_id"

  ( # Start a subshell for the background job
    # This ensures set -e in the subshell doesn't kill the main script
    set -e
    
    # --- MODIFIED: Set CUDA_VISIBLE_DEVICES for this specific job ---
    CUDA_VISIBLE_DEVICES=$gpu_id python -m bytelatent.preprocess.preprocess_entropies \
      "$input_file" \
      "$output_file" \
      --entropy-model-checkpoint-dir "$ENTROPY_MODEL_DIR" \
      --entropy-model-state-dict-path "$ENTROPY_MODEL_DIR/consolidated.pth" \
      --patching-device "cuda" \
      --log-step 1000

    # If the python script succeeds, create the success marker file
    touch "$complete_marker"
    echo "--> FINISHED: $base_name (from GPU:$gpu_id)"
  ) & # The '&' runs this subshell in the background

  # --- Manage Parallel Jobs ---
  processed_count=$((processed_count + 1))
  job_counter=$((job_counter + 1)) # Increment the job counter for GPU assignment

  # If we've hit the max number of parallel jobs, wait for one to finish
  if [ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]; then
    wait -n # Waits for the next background job to terminate
  fi
done

# Wait for all remaining background jobs to complete
wait

echo "================================================================="
echo "All processing complete."
printf "Total Files Processed: %d\n" "$processed_count"
printf "Total Files Skipped:   %d\n" "$skipped_count"
echo "Your preprocessed data is ready in: $OUTPUT_DIR" 
