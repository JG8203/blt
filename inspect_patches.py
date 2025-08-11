# ~/blt/inspect_patches.py

import argparse
import os
import torch
import glob
import logging

# Import necessary components from the bytelatent library
from bytelatent.entropy_model import load_entropy_model
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from bytelatent.data.patcher import Patcher, PatcherArgs, PatchingModeEnum

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_latest_checkpoint(base_checkpoint_dir: str) -> str:
    """Finds the latest numerical checkpoint directory or consolidated entropy model."""
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

def visualize_patching(sentence: str, model_base_dir: str, threshold: float):
    """
    Loads a trained entropy model, calculates patch boundaries for a given
    sentence, and prints a detailed visualization.
    """
    print("\n--- Initializing ---")

    try:
        # --- 1. Locate and Consolidate the Checkpoint ---
        latest_checkpoint_dir = find_latest_checkpoint(model_base_dir)

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

        # --- 2. Load the Entropy Model and Tokenizer ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        logging.info(f"Loading entropy model from: {consolidated_dir}")
        entropy_model, _ = load_entropy_model(consolidated_dir, state_dict_path, device=device)
        entropy_model.eval()

        logging.info("Initializing byte-level tokenizer...")
        tokenizer = BltTokenizer()

        # --- 3. Setup the Patcher ---
        logging.info("Setting up entropy patcher...")
        patcher_args = PatcherArgs(
            patching_mode=PatchingModeEnum.entropy,
            threshold=threshold,
            device=device,
            patching_device=device,
            realtime_patching=True, # Important for on-the-fly patching
            entropy_model_checkpoint_dir=consolidated_dir # Needed by the Patcher's __init__
        )
        # We manually build the patcher and assign our loaded model
        patcher = patcher_args.build()
        patcher.entropy_model = entropy_model

        # --- 4. Process the Sentence ---
        print("\n--- Processing Sentence ---")
        logging.info(f"Input Sentence: '{sentence}'")

        # Encode the sentence into byte tokens (as a batch of 1)
        tokens_list = tokenizer.encode(sentence, add_bos=True, add_eos=True)
        tokens_tensor = torch.tensor([tokens_list], device=device)
        logging.info(f"Sentence encoded into {tokens_tensor.shape[1]} byte tokens.")

        # The patcher's .patch() method does all the work: it uses the entropy_model
        # to calculate scores and then determines patch lengths based on the threshold.
        patch_lengths_tensor, scores_tensor = patcher.patch(tokens_tensor)
        patch_lengths = patch_lengths_tensor.squeeze().tolist()

        # --- 5. Visualize the Results ---
        print("\n--- Patching Visualization ---")
        print(f"Entropy Threshold Used: {threshold}")
        print(f"Found {len(patch_lengths)} patches with lengths: {patch_lengths}")
        print("-" * 60)
        print(f"{'Index':<7}{'Char':<6}{'ByteID':<8}{'Entropy':<12}{'Patch Start?':<15}")
        print("-" * 60)

        patch_start_indices = {0}
        current_index = 0
        for length in patch_lengths[:-1]:
            current_index += length
            patch_start_indices.add(current_index)

        # The entropy at scores[i] is for predicting the token at tokens[i+1].
        # We add a placeholder for the first token which has no preceding entropy.
        entropy_list = [None] + scores_tensor.squeeze().tolist()

        for i, token_id in enumerate(tokens_list):
            # Attempt to decode the byte for a readable character representation
            if token_id >= tokenizer.offsetting_special_char:
                char_repr = repr(bytes([token_id - tokenizer.offsetting_special_char]).decode('utf-8', errors='replace'))
            else:
                char_repr = f"<ID:{token_id}>"

            entropy_val = entropy_list[i]
            is_start_marker = "✅" if i in patch_start_indices else ""
            entropy_str = f"{entropy_val:.4f}" if entropy_val is not None else "N/A"

            print(f"{i:<7}{char_repr:<6}{token_id:<8}{entropy_str:<12}{is_start_marker:<15}")

        print("-" * 60)
        print("\nFinal Patched Sentence:")
        patched_sentence_parts = []
        current_pos = 0
        for length in patch_lengths:
            patch_token_ids = tokens_list[current_pos : current_pos + length]
            patched_sentence_parts.append(tokenizer.decode(patch_token_ids))
            current_pos += length

        print(" | ".join([f"[{part}]" for part in patched_sentence_parts]))
        print("\n")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize BLT entropy patching for a sentence.")
    parser.add_argument(
        "sentence",
        type=str,
        help="The sentence you want to patch."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the BASE checkpoint directory of your trained entropy model (e.g., 'models/my_entropy_model_checkpoints')."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.33,
        help="The entropy threshold to use for creating patches."
    )
    args = parser.parse_args()

    visualize_patching(args.sentence, args.model_dir, args.threshold)
