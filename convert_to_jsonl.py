# convert_to_jsonl.py
import csv
import json
import argparse
from tqdm import tqdm

def convert_csv_to_jsonl(input_csv_path: str, output_jsonl_path: str):
    print(f"Starting conversion of '{input_csv_path}' to '{output_jsonl_path}'...")
    try:
        with open(input_csv_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f) - 1
        with open(input_csv_path, 'r', encoding='utf-8') as infile, \
             open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
            reader = csv.DictReader(infile)
            if 'text' not in reader.fieldnames:
                raise ValueError("CSV file must contain a 'text' column.")
            for i, row in enumerate(tqdm(reader, total=total_lines, desc="Converting rows")):
                output_data = {"id": f"doc_{i+1}", "text": row['text']}
                outfile.write(json.dumps(output_data) + '\n')
        print(f"\nConversion complete. Output saved to: {output_jsonl_path}")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_csv_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to JSONL.")
    parser.add_argument("input_csv", help="Path to input CSV.")
    parser.add_argument("output_jsonl", help="Path for output JSONL.")
    args = parser.parse_args()
    convert_csv_to_jsonl(args.input_csv, args.output_jsonl)
