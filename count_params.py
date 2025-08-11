# ~/blt/count_params.py (Updated Version)

import argparse
import os
import logging

# Import the necessary functions from the bytelatent library
from bytelatent.entropy_model import load_entropy_model
from bytelatent.metrics import get_num_params

# Setup basic logging to suppress verbose outputs from the library
logging.basicConfig(level=logging.WARNING)

def count_model_parameters(model_dir: str):
    """
    Loads a trained model from a consolidated checkpoint directory
    and prints its parameter count.
    """
    print("\n--- Model Parameter Counter ---")

    try:
        # --- 1. Locate the consolidated model files (REVISED LOGIC) ---
        print(f"Analyzing directory: {model_dir}")

        # This variable will hold the path to the directry containing the model files.
        files_dir = model_dir

        # Check if a 'consolidated' subdirectory exists (for local training checkpoints)
        potential_consolidated_dir = os.path.join(model_dir, "consolidated")
        if os.path.isdir(potential_consolidated_dir):
            print("Found 'consolidated' subfolder, using it.")
            files_dir = potential_consolidated_dir
        else:
            print("No 'consolidated' subfolder found. Assuming model files are directly in the provided directory.")

        state_dict_path = os.path.join(files_dir, "consolidated.pth")
        params_path = os.path.join(files_dir, "params.json")

        if not os.path.exists(state_dict_path) or not os.path.exists(params_path):
            raise FileNotFoundError(
                f"Could not find 'consolidated.pth' or 'params.json' in '{files_dir}'.\n"
                "Please ensure you have provided the correct model directory."
            )

        # --- 2. Load the Model ---
        print(f"Loading model from: {files_dir}")
        model, _ = load_entropy_model(files_dir, state_dict_path, device="cpu")
        print("Model loaded successfully.")

        # --- 3. Count the Parameters ---
        total_params = get_num_params(model)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("\n--- Parameter Count ---")
        print(f"Total Parameters: {total_params:,}")
        print(f"  - Trainable:    {trainable_params:,}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count the parameters of a consolidated BLT model."
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to the model directory. This can be the base directory (e.g., 'hf-weights/entropy_model') "
             "or a specific FSDP checkpoint directory containing a 'consolidated' subfolder."
    )
    args = parser.parse_args()
    count_model_parameters(args.model_dir)
