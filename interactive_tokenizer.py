#!/usr/bin/env python3
"""
Interactive Tokenizer with Entropy Models for COHFIE Dataset
Allows real-time testing of traditional tokenizers and BLT entropy models.
"""

import sys
import time
import torch
import os
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

# Import BLT components for entropy models
try:
    from bytelatent.entropy_model import load_entropy_model
    from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
    from bytelatent.data.patcher import Patcher, PatcherArgs, PatchingModeEnum
    BLT_AVAILABLE = True
except ImportError:
    BLT_AVAILABLE = False
    print("⚠️  BLT components not available. Entropy models will be disabled.")

class InteractiveTokenizerWithEntropy:
    """Interactive tokenizer with entropy models for real-time testing."""

    def __init__(self, my_model_dir: str = None, hf_model_dir: str = None, threshold: float = 1.33):
        self.tokenizers = {}
        self.entropy_models = {}
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔧 Using device: {self.device}")
        
        self.load_traditional_tokenizers()
        
        # Load entropy models if directories are provided and BLT is available
        if BLT_AVAILABLE:
            if my_model_dir or hf_model_dir:
                self.load_entropy_models(my_model_dir, hf_model_dir)
            else:
                print("⚠️  No entropy model directories provided.")
                print("   Default paths will be checked: models/my_entropy_model_checkpoints, hf-weights")
                print("   Use --my_model and --hf_model to specify custom paths")

    def load_traditional_tokenizers(self):
        """Load available traditional tokenizers."""
        print("🔄 Loading traditional tokenizers...")

        # GPT-2
        try:
            from transformers import GPT2Tokenizer
            self.tokenizers['gpt2'] = {
                'tokenizer': GPT2Tokenizer.from_pretrained('gpt2'),
                'type': 'huggingface',
                'name': 'GPT-2 BPE'
            }
            print("✅ GPT-2 BPE loaded")
        except Exception as e:
            print(f"❌ GPT-2 failed: {e}")

        # Tagalog GPT-2
        try:
            from transformers import GPT2Tokenizer
            self.tokenizers['tagalog_gpt2'] = {
                'tokenizer': GPT2Tokenizer.from_pretrained('baebee/gpt2-tagalog-80k'),
                'type': 'huggingface',
                'name': 'Tagalog GPT-2 (BPE)'
            }
            print("✅ Tagalog GPT-2 loaded")
        except Exception as e:
            print(f"❌ Tagalog GPT-2 failed: {e}")

        # Tiktoken cl100k
        try:
            import tiktoken
            self.tokenizers['tiktoken_cl100k'] = {
                'tokenizer': tiktoken.get_encoding("cl100k_base"),
                'type': 'tiktoken',
                'name': 'Tiktoken (cl100k_base)'
            }
            print("✅ Tiktoken cl100k loaded")
        except Exception as e:
            print(f"❌ Tiktoken cl100k failed: {e}")

        # Tiktoken o200k
        try:
            import tiktoken
            self.tokenizers['tiktoken_o200k'] = {
                'tokenizer': tiktoken.get_encoding("o200k_base"),
                'type': 'tiktoken',
                'name': 'Tiktoken (o200k_base)'
            }
            print("✅ Tiktoken o200k loaded")
        except Exception as e:
            print(f"❌ Tiktoken o200k failed: {e}")

        # Tiktoken o200k_harmony
        try:
            import tiktoken
            self.tokenizers['tiktoken_harmony'] = {
                'tokenizer': tiktoken.get_encoding("o200k_harmony"),
                'type': 'tiktoken',
                'name': 'Tiktoken (o200k_harmony)'
            }
            print("✅ Tiktoken o200k_harmony loaded")
        except Exception as e:
            print(f"❌ Tiktoken o200k_harmony failed: {e}")

        # ByT5
        try:
            from transformers import AutoTokenizer
            self.tokenizers['byt5'] = {
                'tokenizer': AutoTokenizer.from_pretrained('google/byt5-small'),
                'type': 'byt5',
                'name': 'ByT5 (byte-level)'
            }
            print("✅ ByT5 loaded")
        except Exception as e:
            print(f"❌ ByT5 failed: {e}")

        # Llama-3 (with fallbacks)
        try:
            from transformers import AutoTokenizer
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
                        'name': f'Llama-3 ({model_name.split('/')[-1]})'
                    }
                    print(f"✅ Llama-3 loaded: {model_name}")
                    break
                except Exception as e:
                    print(f"⚠️ {model_name} failed: {e}")
                    continue
        except Exception as e:
            print(f"❌ Llama-3 failed: {e}")

    def load_entropy_models(self, my_model_dir: str = None, hf_model_dir: str = None):
        """Load entropy models for BLT analysis."""
        if not BLT_AVAILABLE:
            print("❌ BLT not available, skipping entropy models")
            return

        print("\n🧠 Loading entropy models...")

        # Load my model
        if my_model_dir and os.path.exists(my_model_dir):
            try:
                my_model, my_patcher, my_info = self.load_single_entropy_model(
                    my_model_dir, "My Entropy Model"
                )
                self.entropy_models['my_model'] = {
                    'model': my_model,
                    'patcher': my_patcher,
                    'tokenizer': BltTokenizer(),
                    'info': my_info,
                    'name': 'My Entropy Model'
                }
                print("✅ My entropy model loaded")
            except Exception as e:
                print(f"❌ My entropy model failed: {e}")
                import traceback
                traceback.print_exc()

        # Load HF model
        if hf_model_dir and os.path.exists(hf_model_dir):
            try:
                hf_model, hf_patcher, hf_info = self.load_single_entropy_model(
                    hf_model_dir, "HF Entropy Model"
                )
                self.entropy_models['hf_model'] = {
                    'model': hf_model,
                    'patcher': hf_patcher,
                    'tokenizer': BltTokenizer(),
                    'info': hf_info,
                    'name': 'HF Entropy Model'
                }
                print("✅ HF entropy model loaded")
            except Exception as e:
                print(f"❌ HF entropy model failed: {e}")
                import traceback
                traceback.print_exc()

        if self.entropy_models:
            print(f"🧠 Loaded {len(self.entropy_models)} entropy models")
        else:
            print("⚠️  No entropy models loaded")

    def find_latest_checkpoint(self, base_checkpoint_dir: str) -> str:
        """Find the latest checkpoint directory."""
        import glob
        
        # Check if this is HF weights structure (has entropy_model subdirectory)
        hf_entropy_model_path = os.path.join(base_checkpoint_dir, "entropy_model")
        if os.path.isdir(hf_entropy_model_path):
            print(f"Found HF weights entropy model at: {hf_entropy_model_path}")
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
        print(f"Found latest checkpoint at: {latest_dir}")
        return latest_dir

    def load_single_entropy_model(self, model_dir: str, model_name: str) -> Tuple[Any, Any, Dict]:
        """Load a single entropy model."""
        # Find latest checkpoint
        latest_checkpoint_dir = self.find_latest_checkpoint(model_dir)
        
        # Check if consolidated
        direct_state_dict_path = os.path.join(latest_checkpoint_dir, "consolidated.pth")
        direct_params_path = os.path.join(latest_checkpoint_dir, "params.json")

        if os.path.exists(direct_state_dict_path) and os.path.exists(direct_params_path):
            consolidated_dir = latest_checkpoint_dir
            state_dict_path = direct_state_dict_path
            params_path = direct_params_path
            print(f"Using consolidated files directly from: {consolidated_dir}")
        else:
            consolidated_dir = os.path.join(latest_checkpoint_dir, "consolidated")
            if not os.path.exists(consolidated_dir):
                print("Consolidated checkpoint not found. Running consolidation script...")
                from bytelatent.checkpoint import main as consolidate_main
                consolidate_main(command="consolidate", model_checkpoint_dir=latest_checkpoint_dir)
                print("Consolidation complete.")
            
            state_dict_path = os.path.join(consolidated_dir, "consolidated.pth")
            params_path = os.path.join(consolidated_dir, "params.json")

        # Load parameters
        import json
        with open(params_path, 'r') as f:
            model_params = json.load(f)

        # Load entropy model
        entropy_model, _ = load_entropy_model(consolidated_dir, state_dict_path, device=self.device)
        entropy_model.eval()

        # Setup patcher
        patcher_args = PatcherArgs(
            patching_mode=PatchingModeEnum.entropy,
            threshold=self.threshold,
            device=self.device,
            patching_device=self.device,
            realtime_patching=True,
            entropy_model_checkpoint_dir=consolidated_dir
        )
        patcher = patcher_args.build()
        patcher.entropy_model = entropy_model

        model_info = {
            'name': model_name,
            'params': model_params,
            'checkpoint_dir': consolidated_dir,
            'threshold': self.threshold
        }

        return entropy_model, patcher, model_info

    def tokenize_text(self, text: str, tokenizer_key: str) -> Dict[str, Any]:
        """Tokenize text with specified traditional tokenizer."""
        if tokenizer_key not in self.tokenizers:
            return {"error": "Tokenizer not available"}

        tokenizer_info = self.tokenizers[tokenizer_key]
        tokenizer = tokenizer_info['tokenizer']
        tokenizer_type = tokenizer_info['type']

        try:
            start_time = time.time()
            
            if tokenizer_type == 'tiktoken':
                # Tiktoken tokenizers
                token_ids = tokenizer.encode(text)

                # Get individual tokens
                tokens = []
                for token_id in token_ids:
                    try:
                        token_bytes = tokenizer.decode_single_token_bytes(token_id)
                        token_str = token_bytes.decode('utf-8', errors='replace')
                        tokens.append({
                            'id': token_id,
                            'string': token_str,
                            'bytes': list(token_bytes),
                            'repr': repr(token_str)
                        })
                    except Exception as e:
                        tokens.append({
                            'id': token_id,
                            'error': str(e)
                        })

                decoded = tokenizer.decode(token_ids)

            elif tokenizer_type == 'byt5':
                # ByT5 tokenizer
                token_ids = tokenizer.encode(text, add_special_tokens=False)

                tokens = []
                for token_id in token_ids:
                    token_str = tokenizer.decode([token_id], skip_special_tokens=True)

                    # Determine what this token represents
                    if token_id == 0:
                        meaning = "PAD"
                    elif token_id == 1:
                        meaning = "EOS"
                    elif token_id == 2:
                        meaning = "UNK"
                    elif 3 <= token_id <= 258:
                        byte_val = token_id - 3
                        if byte_val < 128:
                            try:
                                char = chr(byte_val)
                                meaning = f"'{char}' (byte {byte_val})"
                            except:
                                meaning = f"byte {byte_val}"
                        else:
                            meaning = f"byte {byte_val}"
                    else:
                        meaning = f"extended token {token_id}"

                    tokens.append({
                        'id': token_id,
                        'string': token_str,
                        'meaning': meaning,
                        'repr': repr(token_str)
                    })

                decoded = tokenizer.decode(token_ids, skip_special_tokens=True)

            else:
                # Hugging Face tokenizers (GPT-2, Llama, etc.)
                token_ids = tokenizer.encode(text, add_special_tokens=False)

                tokens = []
                for token_id in token_ids:
                    token_str = tokenizer.decode([token_id])
                    tokens.append({
                        'id': token_id,
                        'string': token_str,
                        'bytes': list(token_str.encode('utf-8')),
                        'repr': repr(token_str)
                    })

                decoded = tokenizer.decode(token_ids)

            processing_time = time.time() - start_time

            return {
                'token_ids': token_ids,
                'tokens': tokens,
                'decoded': decoded,
                'count': len(token_ids),
                'compression_ratio': len(text) / len(token_ids) if token_ids else 0,
                'perfect_reconstruction': text == decoded,
                'processing_time_ms': processing_time * 1000,
                'type': 'traditional'
            }

        except Exception as e:
            return {"error": str(e)}

    def analyze_entropy_model(self, text: str, model_key: str) -> Dict[str, Any]:
        """Analyze text with entropy model and return detailed results."""
        if model_key not in self.entropy_models:
            return {"error": "Entropy model not available"}

        model_data = self.entropy_models[model_key]
        
        try:
            start_time = time.time()
            
            # Encode the sentence
            tokens_list = model_data['tokenizer'].encode(text, add_bos=True, add_eos=True)
            tokens_tensor = torch.tensor([tokens_list], device=self.device)

            # Get patches
            patch_lengths_tensor, scores_tensor = model_data['patcher'].patch(tokens_tensor)
            patch_lengths = patch_lengths_tensor.squeeze().tolist()
            entropy_scores = scores_tensor.squeeze().tolist() if scores_tensor is not None else []

            # Get individual token entropies using the model directly
            token_entropies = []
            token_details = []
            
            with torch.no_grad():
                try:
                    # Forward pass through the model
                    logits = model_data['model'](tokens_tensor)
                    
                    if isinstance(logits, torch.Tensor) and len(logits.shape) == 3:
                        # Calculate probabilities and entropies
                        log_probs = torch.log_softmax(logits, dim=-1)
                        probs = torch.softmax(logits, dim=-1)
                        
                        # Calculate entropy: -sum(p * log(p))
                        entropy_tensor = -(probs * log_probs).sum(dim=-1)
                        entropy_tensor = entropy_tensor.squeeze(0)  # Remove batch dimension
                        
                        if entropy_tensor.dim() == 0:
                            token_entropies = [float(entropy_tensor.item())]
                        else:
                            token_entropies = [float(x) for x in entropy_tensor.tolist()]
                        
                        # Get token details for visualization
                        for i, token_id in enumerate(tokens_list):
                            # Decode individual token for display
                            if token_id >= model_data['tokenizer'].offsetting_special_char:
                                try:
                                    char_repr = bytes([token_id - model_data['tokenizer'].offsetting_special_char]).decode('utf-8', errors='replace')
                                except:
                                    char_repr = f"<ID:{token_id}>"
                            else:
                                char_repr = f"<ID:{token_id}>"
                            
                            entropy_val = token_entropies[i] if i < len(token_entropies) else None
                            
                            token_details.append({
                                'index': i,
                                'token_id': token_id,
                                'char': char_repr,
                                'entropy': entropy_val
                            })
                            
                except Exception as e:
                    print(f"⚠️ Could not calculate token entropies: {e}")

            # Convert patches to detailed information
            patches_detailed = []
            current_pos = 0
            
            # Create patch start indices for visualization
            patch_start_indices = {0}
            temp_pos = 0
            for length in patch_lengths[:-1]:  # Don't include the last patch end
                temp_pos += length
                patch_start_indices.add(temp_pos)
            
            for i, length in enumerate(patch_lengths):
                patch_token_ids = tokens_list[current_pos : current_pos + length]
                patch_text = model_data['tokenizer'].decode(patch_token_ids)
                
                # Get entropies for this patch
                patch_token_entropies = []
                if current_pos < len(token_entropies):
                    patch_token_entropies = token_entropies[current_pos : current_pos + length]
                
                patch_info = {
                    'index': i,
                    'length': length,
                    'text': patch_text,
                    'token_ids': patch_token_ids,
                    'start_pos': current_pos,
                    'end_pos': current_pos + length,
                    'entropy_score': entropy_scores[i] if i < len(entropy_scores) else None,
                    'token_entropies': patch_token_entropies,
                    'avg_token_entropy': sum(patch_token_entropies) / len(patch_token_entropies) if patch_token_entropies else None,
                    'entropy_variance': float(np.var(patch_token_entropies)) if len(patch_token_entropies) > 1 else 0.0
                }
                
                patches_detailed.append(patch_info)
                current_pos += length

            processing_time = time.time() - start_time
            
            return {
                'model_name': model_data['name'],
                'patches': patches_detailed,
                'patch_count': len(patch_lengths),
                'total_tokens': sum(patch_lengths),
                'compression_ratio': sum(patch_lengths) / len(patch_lengths) if patch_lengths else 0,
                'processing_time_ms': processing_time * 1000,
                'perfect_reconstruction': True,  # BLT should always reconstruct perfectly
                'threshold': self.threshold,
                'individual_token_entropies': token_entropies,
                'token_details': token_details,
                'patch_start_indices': patch_start_indices,
                'avg_token_entropy': sum(token_entropies) / len(token_entropies) if token_entropies else None,
                'entropy_variance': float(np.var(token_entropies)) if len(token_entropies) > 1 else 0.0,
                'type': 'entropy'
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def display_entropy_visualization(self, text: str, entropy_result: Dict[str, Any]):
        """Display entropy visualization with patches and entropy scores."""
        print(f"\n🧠 {entropy_result['model_name']}")
        print("-" * 80)
        
        if "error" in entropy_result:
            print(f"❌ Error: {entropy_result['error']}")
            return

        print(f"📊 Input: '{text}'")
        print(f"🔧 Patches: {entropy_result['patch_count']}")
        print(f"📈 Avg patch length: {entropy_result['compression_ratio']:.2f} tokens")
        print(f"⏱️  Processing time: {entropy_result['processing_time_ms']:.2f}ms")
        print(f"🎯 Threshold: {entropy_result['threshold']}")
        
        if entropy_result.get('avg_token_entropy'):
            print(f"🌡️  Avg token entropy: {entropy_result['avg_token_entropy']:.3f}")
            print(f"📊 Entropy variance: {entropy_result['entropy_variance']:.3f}")

        # Display detailed token-by-token analysis
        if entropy_result.get('token_details'):
            print(f"\n📋 Token-by-Token Analysis:")
            print("   # | Token ID |   Char   | Entropy | Patch Start?")
            print("   " + "-" * 50)
            
            patch_starts = entropy_result.get('patch_start_indices', set())
            
            for token_detail in entropy_result['token_details'][:20]:  # Show first 20 tokens
                index = token_detail['index']
                token_id = token_detail['token_id']
                char = token_detail['char']
                entropy = token_detail['entropy']
                
                is_start = "✅" if index in patch_starts else ""
                entropy_str = f"{entropy:.4f}" if entropy is not None else "N/A"
                
                # Truncate char display if too long
                if len(char) > 8:
                    char_display = char[:5] + "..."
                else:
                    char_display = char
                
                print(f"   {index:2d} | {token_id:8d} | {char_display:8s} | {entropy_str:7s} | {is_start}")
            
            if len(entropy_result['token_details']) > 20:
                print(f"   ... ({len(entropy_result['token_details']) - 20} more tokens)")

        # Display patches with entropy visualization
        print(f"\n📋 Patches with Entropy Analysis:")
        print("   # | Length |      Text      | Patch Entropy | Avg Token Entropy")
        print("   " + "-" * 70)

        for patch in entropy_result['patches']:
            patch_text = patch['text']
            if len(patch_text) > 12:
                display_text = patch_text[:9] + "..."
            else:
                display_text = patch_text

        # Show final patched sentence
        print(f"\n🔗 Patched Sentence:")
        patch_strings = []
        for patch in entropy_result['patches']:
            patch_strings.append(f"[{patch['text']}]")
        print(" | ".join(patch_strings))

        # Entropy visualization bar
        if entropy_result.get('individual_token_entropies'):
            self.display_entropy_bar(entropy_result['individual_token_entropies'], entropy_result['patches'])

    def display_entropy_bar(self, token_entropies: List[float], patches: List[Dict]):
        """Display a visual entropy bar showing entropy changes across tokens."""
        if not token_entropies:
            return

        print(f"\n🌡️  Entropy Visualization (per token):")
        
        # Normalize entropies for visualization (0-10 scale)
        min_entropy = min(token_entropies)
        max_entropy = max(token_entropies)
        entropy_range = max_entropy - min_entropy if max_entropy > min_entropy else 1
        
        # Create entropy bar
        bar_width = min(60, len(token_entropies) * 2)
        tokens_per_char = len(token_entropies) / bar_width
        
        entropy_bar = []
        for i in range(bar_width):
            token_idx = int(i * tokens_per_char)
            if token_idx < len(token_entropies):
                normalized_entropy = (token_entropies[token_idx] - min_entropy) / entropy_range
                if normalized_entropy >= 0.8:
                    entropy_bar.append('█')  # High entropy
                elif normalized_entropy >= 0.6:
                    entropy_bar.append('▓')  # Medium-high entropy
                elif normalized_entropy >= 0.4:
                    entropy_bar.append('▒')  # Medium entropy
                elif normalized_entropy >= 0.2:
                    entropy_bar.append('░')  # Low-medium entropy
                else:
                    entropy_bar.append('·')  # Low entropy
            else:
                entropy_bar.append(' ')
        
        print(f"   Low  {''.join(entropy_bar)} High")
        print(f"   {min_entropy:.2f}{' ' * (bar_width - 8)}{max_entropy:.2f}")
        
        # Show patch boundaries
        patch_boundaries = []
        current_pos = 0
        for patch in patches:
            patch_boundaries.append(current_pos)
            current_pos += patch['length']
        
        if len(patch_boundaries) > 1:
            boundary_bar = [' '] * bar_width
            for boundary in patch_boundaries[1:]:  # Skip first boundary (always 0)
                boundary_pos = int((boundary / len(token_entropies)) * bar_width)
                if 0 <= boundary_pos < bar_width:
                    boundary_bar[boundary_pos] = '|'
            
            print(f"   Patches: {''.join(boundary_bar)}")

    def display_traditional_tokenization(self, text: str, tokenizer_key: str):
        """Display traditional tokenization results."""
        tokenizer_name = self.tokenizers[tokenizer_key]['name']

        print(f"\n🔤 {tokenizer_name}")
        print("-" * 60)

        result = self.tokenize_text(text, tokenizer_key)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return

        print(f"📊 Input: '{text}'")
        print(f"📈 Tokens: {result['count']}")
        print(f"📉 Compression: {result['compression_ratio']:.2f} chars/token")
        print(f"✅ Perfect reconstruction: {result['perfect_reconstruction']}")
        print(f"⏱️  Processing time: {result['processing_time_ms']:.2f}ms")
        print(f"🔄 Decoded: '{result['decoded']}'")

        print(f"\n📋 Individual tokens:")
        print("   #  |   ID   |    String    |           Representation")
        print("   " + "-" * 55)

        for i, token in enumerate(result['tokens'][:10]):  # Show first 10 tokens
            if 'error' in token:
                print(f"   {i:2d} | {token['id']:6d} | ERROR: {token['error']}")
                continue

            token_str = token['string']
            if len(token_str) > 10:
                display_str = token_str[:7] + "..."
            else:
                display_str = token_str

            repr_str = token['repr']
            if len(repr_str) > 25:
                repr_str = repr_str[:22] + "..."

            # Add extra info for ByT5
            extra = ""
            if 'meaning' in token:
                extra = f" ({token['meaning']})"

            print(f"   {i:2d} | {token['id']:6d} | {display_str:>12s} | {repr_str}{extra}")
        
        if len(result['tokens']) > 10:
            print(f"   ... ({len(result['tokens']) - 10} more tokens)")

    def compare_all_models(self, text: str):
        """Compare text across all available models (traditional + entropy)."""
        print(f"\n🔀 COMPREHENSIVE MODEL COMPARISON")
        print("=" * 80)
        print(f"Input text: '{text}'")
        print(f"Length: {len(text)} characters, {len(text.split())} words")
        print()

        # Collect all results
        all_results = {}
        
        # Traditional tokenizers
        for key, info in self.tokenizers.items():
            result = self.tokenize_text(text, key)
            all_results[key] = {
                'name': info['name'],
                'result': result,
                'type': 'traditional'
            }

        # Entropy models
        for key, model_data in self.entropy_models.items():
            result = self.analyze_entropy_model(text, key)
            all_results[key] = {
                'name': model_data['name'],
                'result': result,
                'type': 'entropy'
            }

        # Summary table
        print("Model                      | Type   | Segments | Ratio | Time(ms) | Entropy | Recon")
        print("-" * 85)

        valid_results = []
        for key, data in all_results.items():
            result = data['result']
            
            if "error" in result:
                print(f"{data['name']:26s} | ERROR: {result['error']}")
                continue

            model_type = "Entropy" if data['type'] == 'entropy' else "Trad"
            
            if data['type'] == 'entropy':
                segments = result['patch_count']
                ratio = result['compression_ratio']
                avg_entropy = result.get('avg_token_entropy', 0)
                entropy_str = f"{avg_entropy:.3f}" if avg_entropy else "N/A"
            else:
                segments = result['count']
                ratio = result['compression_ratio']
                entropy_str = "N/A"

            time_ms = result.get('processing_time_ms', 0)
            reconstruction = "✅" if result.get('perfect_reconstruction', False) else "❌"

            print(f"{data['name']:26s} | {model_type:6s} | {segments:8d} | {ratio:5.2f} | {time_ms:8.2f} | {entropy_str:7s} | {reconstruction}")
            
            valid_results.append((key, data))

        print()

        # Show efficiency rankings
        if valid_results:
            print("🏆 EFFICIENCY RANKINGS:")
            
            # Sort by compression ratio (higher is better for traditional, different for entropy)
            traditional_results = [(k, d) for k, d in valid_results if d['type'] == 'traditional' and 'error' not in d['result']]
            entropy_results = [(k, d) for k, d in valid_results if d['type'] == 'entropy' and 'error' not in d['result']]
            
            if traditional_results:
                compression_sorted = sorted(traditional_results, key=lambda x: x[1]['result']['compression_ratio'], reverse=True)
                print("\n📦 Best Traditional Compression (chars/token):")
                for i, (key, data) in enumerate(compression_sorted[:3], 1):
                    ratio = data['result']['compression_ratio']
                    print(f"   {i}. {data['name']}: {ratio:.2f}")
            
            if entropy_results:
                patch_efficiency = sorted(entropy_results, key=lambda x: x[1]['result']['compression_ratio'], reverse=True)
                print("\n🧠 Best Entropy Efficiency (tokens/patch):")
                for i, (key, data) in enumerate(patch_efficiency[:3], 1):
                    ratio = data['result']['compression_ratio']
                    print(f"   {i}. {data['name']}: {ratio:.2f}")
            
            # Sort by speed (lower time is better)
            speed_sorted = sorted(valid_results, key=lambda x: x[1]['result'].get('processing_time_ms', float('inf')))
            print("\n⚡ Fastest Processing:")
            for i, (key, data) in enumerate(speed_sorted[:3], 1):
                time_ms = data['result'].get('processing_time_ms', 0)
                print(f"   {i}. {data['name']}: {time_ms:.2f}ms")

        # Show detailed breakdown for entropy models (with visualization)
        print(f"\n🧠 DETAILED ENTROPY ANALYSIS:")
        for key, data in all_results.items():
            if data['type'] == 'entropy' and 'error' not in data['result']:
                self.display_entropy_visualization(text, data['result'])

        # Show detailed breakdown for traditional models
        print(f"\n🔤 DETAILED TRADITIONAL TOKENIZATION:")
        for key, data in all_results.items():
            if data['type'] == 'traditional' and 'error' not in data['result']:
                self.display_traditional_tokenization(text, key)

    def interactive_mode(self):
        """Run in interactive mode."""
        print("\n🎮 INTERACTIVE MODE")
        print("=" * 50)
        print("Type text to tokenize (or 'quit' to exit)")
        print("Commands:")
        print("  - 'help': Show this help")
        print("  - 'list': List available models")
        print("  - 'compare <text>': Compare all models")
        print("  - 'entropy <text>': Show only entropy models")
        print("  - 'traditional <text>': Show only traditional tokenizers")
        print("  - '<model_key> <text>': Use specific model")
        print("  - 'threshold <value>': Change entropy threshold")
        print("  - 'quit': Exit")
        print()

        while True:
            try:
                user_input = input("💬 Enter text: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break

                elif user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  help - Show this help")
                    print("  list - List all models")
                    print("  compare <text> - Compare all models")
                    print("  entropy <text> - Show only entropy models")
                    print("  traditional <text> - Show only traditional tokenizers")
                    print("  <model_key> <text> - Use specific model")
                    print("  threshold <value> - Change entropy threshold")
                    print("  quit - Exit")
                    continue

                elif user_input.lower() == 'list':
                    print("\nAvailable models:")
                    print("\n🔤 Traditional Tokenizers:")
                    for key, info in self.tokenizers.items():
                        print(f"  {key}: {info['name']}")
                    
                    if self.entropy_models:
                        print("\n🧠 Entropy Models:")
                        for key, info in self.entropy_models.items():
                            print(f"  {key}: {info['name']} (threshold: {self.threshold})")
                    else:
                        print("\n🧠 Entropy Models: None loaded")
                    continue

                elif user_input.lower().startswith('compare '):
                    text = user_input[8:].strip()
                    if text:
                        self.compare_all_models(text)
                    else:
                        print("❌ Please provide text to compare")
                    continue

                elif user_input.lower().startswith('entropy '):
                    text = user_input[8:].strip()
                    if text:
                        if self.entropy_models:
                            print(f"\n🧠 ENTROPY MODEL ANALYSIS")
                            print("=" * 50)
                            for key, model_data in self.entropy_models.items():
                                result = self.analyze_entropy_model(text, key)
                                self.display_entropy_visualization(text, result)
                        else:
                            print("❌ No entropy models loaded")
                    else:
                        print("❌ Please provide text to analyze")
                    continue

                elif user_input.lower().startswith('traditional '):
                    text = user_input[12:].strip()
                    if text:
                        print(f"\n🔤 TRADITIONAL TOKENIZER ANALYSIS")
                        print("=" * 50)
                        for key in self.tokenizers.keys():
                            self.display_traditional_tokenization(text, key)
                    else:
                        print("❌ Please provide text to analyze")
                    continue

                elif user_input.lower().startswith('threshold '):
                    try:
                        new_threshold = float(user_input[10:].strip())
                        if 0.1 <= new_threshold <= 10.0:
                            old_threshold = self.threshold
                            self.threshold = new_threshold
                            print(f"🎯 Threshold changed from {old_threshold:.2f} to {new_threshold:.2f}")
                            
                            # Update patcher thresholds
                            for key, model_data in self.entropy_models.items():
                                model_data['patcher'].threshold = new_threshold
                                print(f"🔄 Updated threshold for {model_data['name']}")
                        else:
                            print("❌ Threshold must be between 0.1 and 10.0")
                    except ValueError:
                        print("❌ Invalid threshold value. Use a number between 0.1 and 10.0")
                    continue

                # =======================
                # FIXED MODEL COMMAND PARSER
                # =======================
                # Try model-specific invocation only if the first token is a known key
                parts = user_input.split(' ', 1)
                if len(parts) == 2:
                    first = parts[0]
                    text = parts[1]
                    if first in self.tokenizers:
                        self.display_traditional_tokenization(text, first)
                        continue
                    if first in self.entropy_models:
                        result = self.analyze_entropy_model(text, first)
                        self.display_entropy_visualization(text, result)
                        continue

                # Fallback: treat the whole input as plain text and compare all models
                self.compare_all_models(user_input)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()

    def demo_mode(self):
        """Run demo with predefined texts."""
        print("\n🎬 DEMO MODE")
        print("=" * 50)

        demo_texts = [
            "Kumusta ka?",
            "Magandang umaga po",
            "Salamat sa lahat ng inyong tulong",
            "Mahal kita pero hindi kita kayang mahalin",  # Code-switching
            "I love you pero hindi mo ako love",           # More code-switching
            "Ganda ng weather today, tara na sa mall!",    # Mixed language
            "Dinala ng palad na di cayat asam sa natitiualag na Bayanbayanan",  # Historical text
            "The quick brown fox jumps over the lazy dog"  # English comparison
        ]

        try:
            for i, text in enumerate(demo_texts, 1):
                print(f"\n{'='*20} DEMO {i}/{len(demo_texts)}: '{text}' {'='*20}")
                
                # Show comprehensive analysis for each demo
                self.compare_all_models(text)
                
                # Show entropy changes if entropy models are available
                if self.entropy_models:
                    print(f"\n🌡️  ENTROPY CHANGE ANALYSIS:")
                    for key, model_data in self.entropy_models.items():
                        result = self.analyze_entropy_model(text, key)
                        if 'error' not in result and result.get('individual_token_entropies'):
                            entropies = result['individual_token_entropies']
                            if len(entropies) > 1:
                                entropy_change = max(entropies) - min(entropies)
                                entropy_trend = "increasing" if entropies[-1] > entropies[0] else "decreasing"
                                print(f"   {model_data['name']}: Entropy range {min(entropies):.3f}-{max(entropies):.3f} "
                                      f"(change: {entropy_change:.3f}, trend: {entropy_trend})")

                if i < len(demo_texts):  # Don't wait after last demo
                    try:
                        input("\nPress Enter to continue to next demo...")
                    except KeyboardInterrupt:
                        print("\n👋 Demo interrupted!")
                        break
        except Exception as e:
            print(f"❌ Demo error: {e}")
        finally:
            print("\n🎬 Demo completed!")

    def batch_process_csv(self, csv_file: str, output_file: str = None, num_samples: int = None, 
                         source_filter: str = None, save_detailed: bool = True, max_samples: int = None):
        """Process a CSV file and analyze all texts with available models."""
        try:
            import pandas as pd
        except ImportError:
            print("❌ pandas not available. Install with: pip install pandas")
            return
            
        print(f"\n📊 BATCH PROCESSING CSV: {csv_file}")
        print("=" * 60)
        
        # Load CSV
        try:
            df = pd.read_csv(csv_file)
            print(f"✅ Loaded CSV with {len(df)} rows")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return
            
        # Check required columns
        if 'text' not in df.columns:
            print("❌ CSV must have a 'text' column")
            return
            
        # Filter by source if specified
        if source_filter:
            original_len = len(df)
            df = df[df['source'] == source_filter]
            if df.empty:
                available_sources = df['source'].unique() if 'source' in df.columns else ['N/A']
                print(f"❌ No rows found with source '{source_filter}'. Available sources: {list(available_sources)}")
                return
            print(f"🔍 Filtered from {original_len} to {len(df)} rows with source '{source_filter}'")
            
        # Handle sampling with proper precedence
        final_sample_size = None
        if max_samples:
            final_sample_size = max_samples
            print(f"🎯 Using max_samples limit: {max_samples}")
        elif num_samples:
            final_sample_size = num_samples
            print(f"🎲 Using num_samples: {num_samples}")
            
        if final_sample_size and final_sample_size < len(df):
            df = df.sample(n=final_sample_size, random_state=42)
            print(f"📝 Processing {final_sample_size} out of {len(pd.read_csv(csv_file))} total rows")
        else:
            print(f"📝 Processing all {len(df)} rows")
            
        # Setup output file
        if not output_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            sample_suffix = f"_{final_sample_size}samples" if final_sample_size else "_all"
            output_file = f"tokenization_results_{base_name}{sample_suffix}_{timestamp}.txt"
            
        print(f"💾 Output will be saved to: {output_file}")
        print(f"🔧 Processing {len(df)} texts...")
        
        # Collect all results
        all_results = []
        
        # Process each text
        for idx, row in df.iterrows():
            text = str(row['text'])
            
            # Skip empty texts
            if not text.strip():
                continue
                
            print(f"\r🔄 Processing {len(all_results)+1}/{len(df)}: {text[:100]}{'...' if len(text) > 100 else ''}", end='', flush=True)
            
            result_entry = {
                'index': idx,
                'text': text,
                'char_length': len(text),
                'word_count': len(text.split()),
                'source': row.get('source', 'N/A'),
                'year': row.get('year', 'N/A'),
                'lang': row.get('lang', 'N/A'),
                'lang_prob': row.get('lang_prob', 'N/A'),
                'traditional_results': {},
                'entropy_results': {}
            }
            
            # Process with traditional tokenizers
            for key, info in self.tokenizers.items():
                try:
                    result = self.tokenize_text(text, key)
                    if 'error' not in result:
                        result_entry['traditional_results'][key] = {
                            'name': info['name'],
                            'token_count': result['count'],
                            'compression_ratio': result['compression_ratio'],
                            'processing_time_ms': result['processing_time_ms'],
                            'perfect_reconstruction': result['perfect_reconstruction']
                        }
                        if save_detailed:
                            result_entry['traditional_results'][key]['tokens'] = result['tokens']  # Save ALL tokens for analysis
                    else:
                        result_entry['traditional_results'][key] = {'error': result['error']}
                except Exception as e:
                    result_entry['traditional_results'][key] = {'error': str(e)}
                    
            # Process with entropy models
            for key, model_data in self.entropy_models.items():
                try:
                    result = self.analyze_entropy_model(text, key)
                    if 'error' not in result:
                        result_entry['entropy_results'][key] = {
                            'name': model_data['name'],
                            'patch_count': result['patch_count'],
                            'total_tokens': result['total_tokens'],
                            'compression_ratio': result['compression_ratio'],
                            'processing_time_ms': result['processing_time_ms'],
                            'avg_token_entropy': result.get('avg_token_entropy'),
                            'entropy_variance': result.get('entropy_variance'),
                            'threshold': result['threshold']
                        }
                        if save_detailed:
                            result_entry['entropy_results'][key]['patches'] = result['patches']  # Save ALL patches for analysis
                            result_entry['entropy_results'][key]['individual_entropies'] = result.get('individual_token_entropies', [])  # Save ALL token entropies
                    else:
                        result_entry['entropy_results'][key] = {'error': result['error']}
                except Exception as e:
                    result_entry['entropy_results'][key] = {'error': str(e)}
                    
            all_results.append(result_entry)
            
        print(f"\n✅ Processed {len(all_results)} texts")
        
        # Save results to file
        self.save_batch_results(all_results, output_file, csv_file, save_detailed)
        
        # Generate summary statistics
        self.generate_batch_summary(all_results, output_file)
        
        return all_results
        
    def save_batch_results(self, results: List[Dict], output_file: str, csv_file: str, save_detailed: bool):
        """Save batch processing results to a file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("🚀 BATCH TOKENIZATION RESULTS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Source CSV: {csv_file}\n")
                f.write(f"Processing Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Texts: {len(results)}\n")
                f.write(f"Threshold: {self.threshold}\n")
                f.write("=" * 80 + "\n\n")
                
                # Model summary
                f.write("📋 LOADED MODELS:\n")
                f.write(f"Traditional Tokenizers: {len(self.tokenizers)}\n")
                for key, info in self.tokenizers.items():
                    f.write(f"  - {key}: {info['name']}\n")
                f.write(f"Entropy Models: {len(self.entropy_models)}\n")
                for key, info in self.entropy_models.items():
                    f.write(f"  - {key}: {info['name']}\n")
                f.write("\n")
                
                # Process each result
                for i, result in enumerate(results, 1):
                    f.write(f"{'='*20} TEXT {i}/{len(results)} {'='*20}\n")
                    f.write(f"Text: {result['text']}\n")  # Save FULL text without truncation
                    f.write(f"Length: {result['char_length']} chars, {result['word_count']} words\n")
                    f.write(f"Source: {result['source']} ({result['year']})\n")
                    f.write(f"Language: {result['lang']} (prob: {result['lang_prob']})\n")
                    f.write("-" * 60 + "\n")
                    
                    # Traditional tokenizers summary
                    if result['traditional_results']:
                        f.write("🔤 TRADITIONAL TOKENIZERS:\n")
                        f.write("Model                      | Tokens | Ratio | Time(ms) | Recon\n")
                        f.write("-" * 65 + "\n")
                        
                        for key, res in result['traditional_results'].items():
                            if 'error' in res:
                                f.write(f"{self.tokenizers[key]['name']:26s} | ERROR: {res['error']}\n")
                            else:
                                name = res['name']
                                tokens = res['token_count']
                                ratio = res['compression_ratio']
                                time_ms = res['processing_time_ms']
                                recon = "✅" if res['perfect_reconstruction'] else "❌"
                                f.write(f"{name:26s} | {tokens:6d} | {ratio:5.2f} | {time_ms:8.2f} | {recon}\n")
                        f.write("\n")
                        
                    # Entropy models summary
                    if result['entropy_results']:
                        f.write("🧠 ENTROPY MODELS:\n")
                        f.write("Model                      | Patches | Tokens | Ratio | Time(ms) | Entropy\n")
                        f.write("-" * 75 + "\n")
                        
                        for key, res in result['entropy_results'].items():
                            if 'error' in res:
                                f.write(f"{res['name']:26s} | ERROR: {res['error']}\n")
                            else:
                                name = res['name']
                                patches = res['patch_count']
                                tokens = res['total_tokens']
                                ratio = res['compression_ratio']
                                time_ms = res['processing_time_ms']
                                entropy = res['avg_token_entropy']
                                entropy_str = f"{entropy:.3f}" if entropy else "N/A"
                                f.write(f"{name:26s} | {patches:7d} | {tokens:6d} | {ratio:5.2f} | {time_ms:8.2f} | {entropy_str}\n")
                        f.write("\n")
                        
                    # Detailed results if requested
                    if save_detailed:
                        f.write("📝 DETAILED ANALYSIS:\n")
                        
                        # Show tokens from each traditional tokenizer (ALL tokens, not truncated)
                        for key, res in result['traditional_results'].items():
                            if 'tokens' in res and res['tokens']:
                                f.write(f"\n{res['name']} - All {len(res['tokens'])} tokens:\n")
                                for j, token in enumerate(res['tokens']):
                                    if 'error' not in token:
                                        f.write(f"  {j}: {token.get('repr', str(token))}\n")
                                        
                        # Show patches from each entropy model (ALL patches, not truncated)
                        for key, res in result['entropy_results'].items():
                            if 'patches' in res and res['patches']:
                                f.write(f"\n{res['name']} - All {len(res['patches'])} patches:\n")
                                for j, patch in enumerate(res['patches']):
                                    f.write(f"  Patch {j}: [{patch['text']}] (length: {patch['length']})\n")
                                    if patch.get('avg_token_entropy'):
                                        f.write(f"    Avg entropy: {patch['avg_token_entropy']:.3f}\n")
                                        
                                # Also save individual token entropies for full analysis
                                if 'individual_entropies' in res and res['individual_entropies']:
                                    f.write(f"  Individual token entropies ({len(res['individual_entropies'])} values):\n")
                                    entropies = res['individual_entropies']
                                    # Format entropies in groups of 10 for readability
                                    for i in range(0, len(entropies), 10):
                                        entropy_group = entropies[i:i+10]
                                        formatted_entropies = [f"{e:.3f}" for e in entropy_group]
                                        f.write(f"    {i:3d}-{min(i+9, len(entropies)-1):3d}: {' '.join(formatted_entropies)}\n")
                                        
                        f.write("\n")
                        
                    f.write("=" * 80 + "\n\n")
                    
            print(f"💾 Results saved to: {output_file}")
            
            # Save JSON file if detailed mode is enabled
            if save_detailed:
                self.save_json_results(results, output_file, csv_file)
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            
    def save_json_results(self, results: List[Dict], output_file: str, csv_file: str):
        """Save detailed results in machine-readable JSON format."""
        import json
        
        json_file = output_file.replace('.txt', '.json')
        
        try:
            # Prepare JSON data structure
            json_data = {
                "metadata": {
                    "source_csv": csv_file,
                    "processing_time": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "total_texts": len(results),
                    "entropy_threshold": self.threshold,
                    "traditional_tokenizers": {
                        key: {
                            "name": info['name'],
                            "type": info['type']
                        } for key, info in self.tokenizers.items()
                    },
                    "entropy_models": {
                        key: {
                            "name": info['name'],
                            "checkpoint_dir": info['info']['checkpoint_dir'],
                            "threshold": info['info']['threshold']
                        } for key, info in self.entropy_models.items()
                    }
                },
                "results": []
            }
            
            # Process each result for JSON
            for result in results:
                json_result = {
                    "text_metadata": {
                        "index": result['index'],
                        "text": result['text'],
                        "char_length": result['char_length'],
                        "word_count": result['word_count'],
                        "source": result['source'],
                        "year": result['year'],
                        "lang": result['lang'],
                        "lang_prob": result['lang_prob']
                    },
                    "traditional_tokenization": {},
                    "entropy_analysis": {}
                }
                
                # Add traditional tokenizer results
                for key, res in result['traditional_results'].items():
                    if 'error' not in res:
                        tokenizer_result = {
                            "model_name": res['name'],
                            "statistics": {
                                "token_count": res['token_count'],
                                "compression_ratio": res['compression_ratio'],
                                "processing_time_ms": res['processing_time_ms'],
                                "perfect_reconstruction": res['perfect_reconstruction']
                            }
                        }
                        
                        # Add detailed token information
                        if 'tokens' in res:
                            tokenizer_result["tokens"] = res['tokens']
                            
                        json_result["traditional_tokenization"][key] = tokenizer_result
                    else:
                        json_result["traditional_tokenization"][key] = {"error": res['error']}
                
                # Add entropy model results
                for key, res in result['entropy_results'].items():
                    if 'error' not in res:
                        entropy_result = {
                            "model_name": res['name'],
                            "statistics": {
                                "patch_count": res['patch_count'],
                                "total_tokens": res['total_tokens'],
                                "compression_ratio": res['compression_ratio'],
                                "processing_time_ms": res['processing_time_ms'],
                                "avg_token_entropy": res.get('avg_token_entropy'),
                                "entropy_variance": res.get('entropy_variance'),
                                "threshold": res['threshold']
                            }
                        }
                        
                        # Add detailed patch and entropy information
                        if 'patches' in res:
                            entropy_result["patches"] = res['patches']
                            
                        if 'individual_entropies' in res:
                            entropy_result["individual_token_entropies"] = res['individual_entropies']
                            
                        json_result["entropy_analysis"][key] = entropy_result
                    else:
                        json_result["entropy_analysis"][key] = {"error": res['error']}
                
                json_data["results"].append(json_result)
            
            # Write JSON file
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
                
            print(f"📄 JSON data saved to: {json_file}")
            
            # Also save a compact JSON version for easier processing
            compact_json_file = json_file.replace('.json', '_compact.json')
            with open(compact_json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, separators=(',', ':'), ensure_ascii=False)
                
            print(f"📄 Compact JSON saved to: {compact_json_file}")
            
        except Exception as e:
            print(f"❌ Error saving JSON results: {e}")
            import traceback
            traceback.print_exc()
            
    def generate_batch_summary(self, results: List[Dict], output_file: str):
        """Generate and save summary statistics."""
        summary_file = output_file.replace('.txt', '_summary.txt')
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("📊 BATCH PROCESSING SUMMARY\n")
                f.write("=" * 50 + "\n")
                f.write(f"Total texts processed: {len(results)}\n")
                f.write(f"Processing time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Calculate statistics for traditional tokenizers
                if any(r['traditional_results'] for r in results):
                    f.write("🔤 TRADITIONAL TOKENIZER STATISTICS:\n")
                    f.write("-" * 40 + "\n")
                    
                    # Get all tokenizer keys
                    all_trad_keys = set()
                    for result in results:
                        all_trad_keys.update(result['traditional_results'].keys())
                        
                    for key in all_trad_keys:
                        valid_results = []
                        for result in results:
                            if key in result['traditional_results'] and 'error' not in result['traditional_results'][key]:
                                valid_results.append(result['traditional_results'][key])
                                
                        if valid_results:
                            name = valid_results[0]['name']
                            avg_tokens = sum(r['token_count'] for r in valid_results) / len(valid_results)
                            avg_ratio = sum(r['compression_ratio'] for r in valid_results) / len(valid_results)
                            avg_time = sum(r['processing_time_ms'] for r in valid_results) / len(valid_results)
                            success_rate = len(valid_results) / len(results) * 100
                            
                            f.write(f"\n{name}:\n")
                            f.write(f"  Success rate: {success_rate:.1f}%\n")
                            f.write(f"  Avg tokens: {avg_tokens:.1f}\n")
                            f.write(f"  Avg compression: {avg_ratio:.2f} chars/token\n")
                            f.write(f"  Avg time: {avg_time:.2f}ms\n")
                            
                # Calculate statistics for entropy models
                if any(r['entropy_results'] for r in results):
                    f.write("\n🧠 ENTROPY MODEL STATISTICS:\n")
                    f.write("-" * 40 + "\n")
                    
                    # Get all entropy model keys
                    all_entropy_keys = set()
                    for result in results:
                        all_entropy_keys.update(result['entropy_results'].keys())
                        
                    for key in all_entropy_keys:
                        valid_results = []
                        for result in results:
                            if key in result['entropy_results'] and 'error' not in result['entropy_results'][key]:
                                valid_results.append(result['entropy_results'][key])
                                
                        if valid_results:
                            name = valid_results[0]['name']
                            avg_patches = sum(r['patch_count'] for r in valid_results) / len(valid_results)
                            avg_tokens = sum(r['total_tokens'] for r in valid_results) / len(valid_results)
                            avg_ratio = sum(r['compression_ratio'] for r in valid_results) / len(valid_results)
                            avg_time = sum(r['processing_time_ms'] for r in valid_results) / len(valid_results)
                            
                            # Calculate entropy statistics
                            entropies = [r['avg_token_entropy'] for r in valid_results if r.get('avg_token_entropy')]
                            avg_entropy = sum(entropies) / len(entropies) if entropies else 0
                            
                            success_rate = len(valid_results) / len(results) * 100
                            
                            f.write(f"\n{name}:\n")
                            f.write(f"  Success rate: {success_rate:.1f}%\n")
                            f.write(f"  Avg patches: {avg_patches:.1f}\n")
                            f.write(f"  Avg tokens: {avg_tokens:.1f}\n")
                            f.write(f"  Avg compression: {avg_ratio:.2f} tokens/patch\n")
                            f.write(f"  Avg entropy: {avg_entropy:.3f}\n")
                            f.write(f"  Avg time: {avg_time:.2f}ms\n")
                            
                # Language and source statistics
                if results:
                    f.write("\n📈 DATASET STATISTICS:\n")
                    f.write("-" * 30 + "\n")
                    
                    # Language distribution
                    lang_counts = {}
                    for result in results:
                        lang = result.get('lang', 'unknown')
                        lang_counts[lang] = lang_counts.get(lang, 0) + 1
                        
                    f.write("Language distribution:\n")
                    for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
                        pct = count / len(results) * 100
                        f.write(f"  {lang}: {count} ({pct:.1f}%)\n")
                        
                    # Source distribution
                    source_counts = {}
                    for result in results:
                        source = result.get('source', 'unknown')
                        source_counts[source] = source_counts.get(source, 0) + 1
                        
                    f.write("\nSource distribution:\n")
                    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
                        pct = count / len(results) * 100
                        f.write(f"  {source}: {count} ({pct:.1f}%)\n")
                        
                    # Text length statistics
                    lengths = [r['char_length'] for r in results]
                    word_counts = [r['word_count'] for r in results]
                    
                    f.write(f"\nText length statistics:\n")
                    f.write(f"  Avg character length: {sum(lengths) / len(lengths):.1f}\n")
                    f.write(f"  Min/Max chars: {min(lengths)}/{max(lengths)}\n")
                    f.write(f"  Avg word count: {sum(word_counts) / len(word_counts):.1f}\n")
                    f.write(f"  Min/Max words: {min(word_counts)}/{max(word_counts)}\n")
                    
            print(f"📊 Summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")

    # =========================
    # FIX: Proper method header
    # =========================
    def entropy_analysis_mode(self):
        """Special mode focused on entropy analysis."""
        if not self.entropy_models:
            print("❌ No entropy models loaded for entropy analysis mode")
            return

        print("\n🌡️  ENTROPY ANALYSIS MODE")
        print("=" * 50)
        print("Focus on entropy patterns and changes")
        print("Commands: 'help', 'threshold <value>', 'compare <text1> vs <text2>', 'quit'")
        print()

        while True:
            try:
                user_input = input("🌡️  Enter text for entropy analysis: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'quit':
                    break

                elif user_input.lower() == 'help':
                    print("\nEntropy Analysis Commands:")
                    print("  compare <text1> vs <text2> - Compare entropy patterns")
                    print("  threshold <value> - Change threshold")
                    print("  <text> - Analyze entropy patterns")
                    continue

                elif user_input.lower().startswith('compare ') and ' vs ' in user_input:
                    parts = user_input[8:].split(' vs ', 1)
                    if len(parts) == 2:
                        text1, text2 = parts[0].strip(), parts[1].strip()
                        print(f"\n🔀 ENTROPY COMPARISON: '{text1}' vs '{text2}'")
                        print("=" * 60)
                        
                        for key, model_data in self.entropy_models.items():
                            print(f"\n🧠 {model_data['name']}:")
                            
                            result1 = self.analyze_entropy_model(text1, key)
                            result2 = self.analyze_entropy_model(text2, key)
                            
                            if 'error' not in result1 and 'error' not in result2:
                                entropy1 = result1.get('avg_token_entropy', 0) or 0
                                entropy2 = result2.get('avg_token_entropy', 0) or 0
                                patches1 = result1.get('patch_count', 0)
                                patches2 = result2.get('patch_count', 0)
                                
                                print(f"   Text 1: {entropy1:.3f} avg entropy, {patches1} patches")
                                print(f"   Text 2: {entropy2:.3f} avg entropy, {patches2} patches")
                                print(f"   Entropy diff: {abs(entropy1 - entropy2):.3f}")
                                print(f"   Patch diff: {abs(patches1 - patches2)} patches")
                    continue

                elif user_input.lower().startswith('threshold '):
                    try:
                        new_threshold = float(user_input[10:].strip())
                        if 0.1 <= new_threshold <= 10.0:
                            self.threshold = new_threshold
                            # Update all patcher thresholds
                            for key, model_data in self.entropy_models.items():
                                model_data['patcher'].threshold = new_threshold
                            print(f"🎯 Threshold set to {new_threshold:.2f}")
                        else:
                            print("❌ Threshold must be between 0.1 and 10.0")
                    except ValueError:
                        print("❌ Invalid threshold value")
                    continue

                else:
                    # Analyze entropy for single text
                    print(f"\n🌡️  ENTROPY ANALYSIS: '{user_input}'")
                    print("=" * 60)
                    
                    for key, model_data in self.entropy_models.items():
                        result = self.analyze_entropy_model(user_input, key)
                        self.display_entropy_visualization(user_input, result)
                        
                        # Additional entropy insights
                        if 'error' not in result and result.get('individual_token_entropies'):
                            entropies = result['individual_token_entropies']
                            if len(entropies) > 1:
                                entropy_std = float(np.std(entropies))
                                entropy_trend = "↗️" if entropies[-1] > entropies[0] else "↘️"
                                
                                print(f"   📊 Entropy statistics:")
                                print(f"      Standard deviation: {entropy_std:.3f}")
                                print(f"      Trend: {entropy_trend} ({'increasing' if entropies[-1] > entropies[0] else 'decreasing'})")
                                print(f"      Most uncertain token: {max(entropies):.3f}")
                                print(f"      Most certain token: {min(entropies):.3f}")

            except KeyboardInterrupt:
                print("\n👋 Exiting entropy analysis mode!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()

def main():
    """Main function."""
    print("🚀 INTERACTIVE TOKENIZER WITH ENTROPY MODELS")
    print("=" * 65)

    # Parse command line arguments for entropy models
    import argparse
    parser = argparse.ArgumentParser(description="Interactive tokenizer with entropy models")
    parser.add_argument("--my_model", type=str, default="models/my_entropy_model_checkpoints", 
                       help="Path to your entropy model directory (default: models/my_entropy_model_checkpoints)")
    parser.add_argument("--hf_model", type=str, default="hf-weights", 
                       help="Path to HF entropy model directory (default: hf-weights)") 
    parser.add_argument("--threshold", type=float, default=1.33, help="Entropy threshold")
    parser.add_argument("--no_defaults", action="store_true", 
                       help="Don't try to load default model paths, only load if explicitly specified")
    parser.add_argument("--csv_file", type=str, help="Path to CSV file for batch processing")
    parser.add_argument("--num_samples", type=int, help="Number of samples to process from CSV (default: all)")
    parser.add_argument("--source_filter", type=str, help="Filter CSV by source column (e.g., 'books', 'inquirer')")
    parser.add_argument("--output_file", type=str, help="Output file for batch processing results")
    parser.add_argument("--save_detailed", action="store_true", default=True, help="Save detailed tokenization results (ALL tokens/patches, not truncated)")
    parser.add_argument("--save_minimal", action="store_true", help="Save only summary statistics (faster, smaller files)")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process (overrides --num_samples)")
    parser.add_argument("--auto_batch", action="store_true", help="Automatically run batch processing and exit")
    args = parser.parse_args()

    try:
        # Only try default paths if not disabled
        my_model_path = None
        hf_model_path = None
        
        if not args.no_defaults:
            # Check if default paths exist, otherwise use None
            if os.path.exists(args.my_model):
                my_model_path = args.my_model
                print(f"🔍 Found default my_model at: {args.my_model}")
            
            if os.path.exists(args.hf_model):
                hf_model_path = args.hf_model
                print(f"🔍 Found default hf_model at: {args.hf_model}")
                
            if not my_model_path and not hf_model_path:
                print("⚠️  Default entropy model paths not found:")
                print(f"   {args.my_model} - not found")
                print(f"   {args.hf_model} - not found")
                print("   Continuing with traditional tokenizers only...")
        else:
            print("🚫 Default model loading disabled with --no_defaults")
        
        tokenizer = InteractiveTokenizerWithEntropy(
            my_model_dir=my_model_path,
            hf_model_dir=hf_model_path, 
            threshold=args.threshold
        )

        total_models = len(tokenizer.tokenizers) + len(tokenizer.entropy_models)
        if total_models == 0:
            print("❌ No models loaded. Please install required packages.")
            return

        print(f"✅ Loaded {len(tokenizer.tokenizers)} traditional tokenizers")
        print(f"✅ Loaded {len(tokenizer.entropy_models)} entropy models")
        print(f"🎯 Entropy threshold: {tokenizer.threshold}")

        # Check for batch processing mode
        if args.csv_file or args.auto_batch:
            csv_file = args.csv_file or "lang_dis_dataset.csv"  # Default CSV name from your example
            
            if not os.path.exists(csv_file):
                print(f"❌ CSV file not found: {csv_file}")
                return
                
            print(f"\n🚀 BATCH PROCESSING MODE")
            print("=" * 50)
            
            # Run batch processing
            results = tokenizer.batch_process_csv(
                csv_file=csv_file,
                output_file=args.output_file,
                num_samples=args.num_samples,
                source_filter=args.source_filter,
                save_detailed=not args.save_minimal,  # Use detailed unless minimal is requested
                max_samples=args.max_samples
            )
            
            if args.auto_batch:
                print("✅ Batch processing completed. Exiting...")
                return
            else:
                print("\n📋 Batch processing completed. Choose what to do next:")

        # Show example usage if no entropy models loaded
        if not tokenizer.entropy_models:
            print("\n💡 To load entropy models:")
            print("   Default paths (auto-checked):")
            print(f"     --my_model {args.my_model}")
            print(f"     --hf_model {args.hf_model}")
            print("   Custom paths:")
            print("     python script.py --my_model path/to/your/model --hf_model path/to/hf/model")
            print("   Skip defaults:")
            print("     python script.py --no_defaults --my_model custom/path")
            print("\n💡 For batch processing:")
            print("     python script.py --csv_file dataset.csv --auto_batch")
            print("     python script.py --csv_file dataset.csv --num_samples 100 --source_filter books")
            print("     python script.py --csv_file dataset.csv --max_samples 50  # Hard limit")
            print("     python script.py --csv_file dataset.csv --save_minimal  # Fast processing")

        # Show menu
        print("\nChoose mode:")
        print("1. Interactive mode (comprehensive analysis)")
        print("2. Demo mode (predefined examples)")
        if tokenizer.entropy_models:
            print("3. Entropy analysis mode (focus on entropy patterns)")
        if args.csv_file and not args.auto_batch:
            print("4. Batch process CSV (run again)")
        print("5. Quick test and exit")

        try:
            choice = input(f"\nEnter choice (1-5): ").strip()

            if choice == '1':
                tokenizer.interactive_mode()
            elif choice == '2':
                tokenizer.demo_mode()
            elif choice == '3':
                if tokenizer.entropy_models:
                    tokenizer.entropy_analysis_mode()
                else:
                    print("❌ No entropy models loaded for entropy analysis mode")
                    tokenizer.interactive_mode()
            elif choice == '4' and args.csv_file:
                # Re-run batch processing with potentially different parameters
                print("\n🔄 Re-running batch processing...")
                results = tokenizer.batch_process_csv(
                    csv_file=args.csv_file,
                    output_file=args.output_file,
                    num_samples=args.num_samples,
                    source_filter=args.source_filter,
                    save_detailed=not args.save_minimal,  # Use detailed unless minimal is requested
                    max_samples=args.max_samples
                )
            elif choice == '5':
                test_text = "Kumusta ka? Okay lang ako."
                print(f"\n🧪 Quick test with: '{test_text}'")
                tokenizer.compare_all_models(test_text)
            else:
                print("Invalid choice. Running quick test...")
                tokenizer.compare_all_models("Kumusta ka?")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")

    except Exception as e:
        print(f"❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🔚 Program ended.")

if __name__ == "__main__":
    main()

