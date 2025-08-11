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
import numpy as np

# Import BLT components
from bytelatent.entropy_model import load_entropy_model
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from bytelatent.data.patcher import Patcher, PatcherArgs, PatchingModeEnum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ComprehensiveTokenizerAnalyzer:
    """Comprehensive analyzer that captures ALL tokenization details."""
    
    def __init__(self):
        self.tokenizers = {}
        self.entropy_models = {}
        self.load_all_components()
    
    def load_all_components(self):
        """Load all tokenizers and prepare for entropy models."""
        print("🔄 Loading all tokenizers and components...")
        
        # 1. Tagalog GPT-2 (BPE)
        try:
            from transformers import GPT2Tokenizer
            self.tokenizers['tagalog_gpt2'] = {
                'tokenizer': GPT2Tokenizer.from_pretrained('baebee/gpt2-tagalog-80k'),
                'type': 'huggingface',
                'name': 'Tagalog GPT-2 (BPE)',
                'description': 'Tagalog-specific GPT-2 with 80k vocabulary',
                'vocab_size': 80000,
                'model_id': 'baebee/gpt2-tagalog-80k'
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
                'description': 'Standard English GPT-2 tokenizer',
                'vocab_size': 50257,
                'model_id': 'gpt2'
            }
            print("✅ Standard GPT-2 loaded")
        except Exception as e:
            print(f"❌ Standard GPT-2 failed: {e}")
        
        # 3. Llama-3 Tokenizer
        try:
            from transformers import AutoTokenizer
            llama_models = [
                "meta-llama/Meta-Llama-3-8B",
                "meta-llama/Meta-Llama-3-8B-Instruct", 
                "NousResearch/Llama-2-7b-hf"
            ]
            
            for model_name in llama_models:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    vocab_size = len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab') else 'Unknown'
                    self.tokenizers['llama3'] = {
                        'tokenizer': tokenizer,
                        'type': 'huggingface',
                        'name': f'Llama-3 ({model_name.split("/")[-1]})',
                        'description': 'Llama-3 SentencePiece tokenizer',
                        'vocab_size': vocab_size,
                        'model_id': model_name
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
            encoding = tiktoken.get_encoding("cl100k_base")
            self.tokenizers['tiktoken_cl100k'] = {
                'tokenizer': encoding,
                'type': 'tiktoken',
                'name': 'Tiktoken (cl100k_base)',
                'description': 'OpenAI GPT-4 tokenizer',
                'vocab_size': encoding.n_vocab,
                'model_id': 'cl100k_base'
            }
            print("✅ Tiktoken cl100k loaded")
            
            # o200k_base
            encoding = tiktoken.get_encoding("o200k_base")
            self.tokenizers['tiktoken_o200k'] = {
                'tokenizer': encoding,
                'type': 'tiktoken', 
                'name': 'Tiktoken (o200k_base)',
                'description': 'OpenAI GPT-4o tokenizer',
                'vocab_size': encoding.n_vocab,
                'model_id': 'o200k_base'
            }
            print("✅ Tiktoken o200k loaded")
            
            # o200k_harmony
            try:
                encoding = tiktoken.get_encoding("o200k_harmony")
                self.tokenizers['tiktoken_harmony'] = {
                    'tokenizer': encoding,
                    'type': 'tiktoken',
                    'name': 'Tiktoken (o200k_harmony)',
                    'description': 'OpenAI GPT-OSS harmony tokenizer',
                    'vocab_size': encoding.n_vocab,
                    'model_id': 'o200k_harmony'
                }
                print("✅ Tiktoken o200k_harmony loaded")
            except Exception as e:
                print(f"⚠️ o200k_harmony not available: {e}")
                
        except Exception as e:
            print(f"❌ Tiktoken failed: {e}")
        
        # 5. ByT5 (byte-level)
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
            self.tokenizers['byt5'] = {
                'tokenizer': tokenizer,
                'type': 'byt5',
                'name': 'ByT5 (byte-level)',
                'description': 'Google ByT5 byte-level tokenizer',
                'vocab_size': len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab') else 'byte-level',
                'model_id': 'google/byt5-small'
            }
            print("✅ ByT5 loaded")
        except Exception as e:
            print(f"❌ ByT5 failed: {e}")
        
        print(f"📊 Loaded {len(self.tokenizers)} tokenizers for comprehensive analysis")
    
    def comprehensive_tokenize(self, text: str, tokenizer_key: str) -> Dict[str, Any]:
        """
        Perform comprehensive tokenization analysis capturing ALL details.
        """
        if tokenizer_key not in self.tokenizers:
            return {"error": "Tokenizer not available"}
        
        tokenizer_info = self.tokenizers[tokenizer_key]
        tokenizer = tokenizer_info['tokenizer']
        tokenizer_type = tokenizer_info['type']
        
        try:
            start_time = time.time()
            
            # Initialize result structure
            result = {
                'tokenizer_key': tokenizer_key,
                'tokenizer_name': tokenizer_info['name'],
                'tokenizer_type': tokenizer_type,
                'tokenizer_description': tokenizer_info['description'],
                'vocab_size': tokenizer_info['vocab_size'],
                'model_id': tokenizer_info['model_id'],
                'input_text': text,
                'input_length_chars': len(text),
                'input_length_words': len(text.split()),
                'input_bytes': list(text.encode('utf-8')),
                'processing_time_seconds': 0,
                'tokens': {
                    'token_ids': [],
                    'token_strings': [],
                    'token_bytes': [],
                    'token_lengths_chars': [],
                    'token_lengths_bytes': [],
                    'token_positions': [],
                    'token_representations': []
                },
                'reconstruction': {
                    'decoded_text': '',
                    'perfect_reconstruction': False,
                    'reconstruction_bytes': [],
                    'char_diff_count': 0,
                    'byte_diff_count': 0
                },
                'statistics': {
                    'token_count': 0,
                    'compression_ratio_chars': 0,
                    'compression_ratio_bytes': 0,
                    'avg_token_length_chars': 0,
                    'avg_token_length_bytes': 0,
                    'unique_tokens': 0,
                    'oov_tokens': 0,
                    'special_tokens': 0
                }
            }
            
            if tokenizer_type == 'tiktoken':
                # Tiktoken tokenizers
                token_ids = tokenizer.encode(text)
                
                for i, token_id in enumerate(token_ids):
                    try:
                        token_bytes = tokenizer.decode_single_token_bytes(token_id)
                        token_str = token_bytes.decode('utf-8', errors='replace')
                        
                        result['tokens']['token_ids'].append(int(token_id))
                        result['tokens']['token_strings'].append(token_str)
                        result['tokens']['token_bytes'].append(list(token_bytes))
                        result['tokens']['token_lengths_chars'].append(len(token_str))
                        result['tokens']['token_lengths_bytes'].append(len(token_bytes))
                        result['tokens']['token_positions'].append(i)
                        result['tokens']['token_representations'].append(repr(token_str))
                        
                    except Exception as e:
                        result['tokens']['token_ids'].append(int(token_id))
                        result['tokens']['token_strings'].append(f"[ERROR:{token_id}]")
                        result['tokens']['token_bytes'].append([])
                        result['tokens']['token_lengths_chars'].append(0)
                        result['tokens']['token_lengths_bytes'].append(0)
                        result['tokens']['token_positions'].append(i)
                        result['tokens']['token_representations'].append(f"[ERROR:{e}]")
                
                decoded = tokenizer.decode(token_ids)
                
            elif tokenizer_type == 'byt5':
                # ByT5 tokenizer
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                
                for i, token_id in enumerate(token_ids):
                    token_str = tokenizer.decode([token_id], skip_special_tokens=True)
                    
                    # Determine byte value and meaning for ByT5
                    if token_id == 0:
                        meaning = "PAD"
                        byte_val = None
                    elif token_id == 1:
                        meaning = "EOS"
                        byte_val = None
                    elif token_id == 2:
                        meaning = "UNK"
                        byte_val = None
                    elif 3 <= token_id <= 258:
                        byte_val = token_id - 3
                        if byte_val < 128:
                            try:
                                char = chr(byte_val)
                                meaning = f"byte_{byte_val}_char_{char}"
                            except:
                                meaning = f"byte_{byte_val}"
                        else:
                            meaning = f"byte_{byte_val}"
                    else:
                        meaning = f"extended_{token_id}"
                        byte_val = None
                    
                    result['tokens']['token_ids'].append(int(token_id))
                    result['tokens']['token_strings'].append(token_str)
                    result['tokens']['token_bytes'].append([byte_val] if byte_val is not None else [])
                    result['tokens']['token_lengths_chars'].append(len(token_str))
                    result['tokens']['token_lengths_bytes'].append(1 if byte_val is not None else 0)
                    result['tokens']['token_positions'].append(i)
                    result['tokens']['token_representations'].append(meaning)
                
                decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
                
            else:
                # Hugging Face tokenizers (GPT-2, Llama, etc.)
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                
                for i, token_id in enumerate(token_ids):
                    token_str = tokenizer.decode([token_id])
                    token_bytes = token_str.encode('utf-8')
                    
                    # Check if it's a special token
                    is_special = False
                    if hasattr(tokenizer, 'special_tokens_map'):
                        special_token_ids = set()
                        for special_tokens in tokenizer.special_tokens_map.values():
                            if isinstance(special_tokens, list):
                                special_token_ids.update(tokenizer.convert_tokens_to_ids(special_tokens))
                            else:
                                special_token_ids.add(tokenizer.convert_tokens_to_ids(special_tokens))
                        is_special = token_id in special_token_ids
                    
                    result['tokens']['token_ids'].append(int(token_id))
                    result['tokens']['token_strings'].append(token_str)
                    result['tokens']['token_bytes'].append(list(token_bytes))
                    result['tokens']['token_lengths_chars'].append(len(token_str))
                    result['tokens']['token_lengths_bytes'].append(len(token_bytes))
                    result['tokens']['token_positions'].append(i)
                    result['tokens']['token_representations'].append(repr(token_str))
                    
                    if is_special:
                        result['statistics']['special_tokens'] += 1
                
                decoded = tokenizer.decode(token_ids)
            
            # Calculate processing time
            result['processing_time_seconds'] = time.time() - start_time
            
            # Reconstruction analysis
            result['reconstruction']['decoded_text'] = decoded
            result['reconstruction']['perfect_reconstruction'] = (text == decoded)
            result['reconstruction']['reconstruction_bytes'] = list(decoded.encode('utf-8'))
            
            # Character and byte differences
            if text != decoded:
                result['reconstruction']['char_diff_count'] = abs(len(text) - len(decoded))
                original_bytes = text.encode('utf-8')
                decoded_bytes = decoded.encode('utf-8')
                result['reconstruction']['byte_diff_count'] = abs(len(original_bytes) - len(decoded_bytes))
            
            # Calculate comprehensive statistics
            token_count = len(result['tokens']['token_ids'])
            result['statistics']['token_count'] = token_count
            
            if token_count > 0:
                result['statistics']['compression_ratio_chars'] = len(text) / token_count
                result['statistics']['compression_ratio_bytes'] = len(text.encode('utf-8')) / token_count
                result['statistics']['avg_token_length_chars'] = sum(result['tokens']['token_lengths_chars']) / token_count
                result['statistics']['avg_token_length_bytes'] = sum(result['tokens']['token_lengths_bytes']) / token_count
                result['statistics']['unique_tokens'] = len(set(result['tokens']['token_ids']))
            
            # Additional analysis for vocabulary coverage
            if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
                result['statistics']['oov_tokens'] = result['tokens']['token_ids'].count(tokenizer.unk_token_id)
            
            return result
            
        except Exception as e:
            return {
                'tokenizer_key': tokenizer_key,
                'tokenizer_name': tokenizer_info['name'],
                'error': str(e),
                'error_type': type(e).__name__
            }

    def load_entropy_model(self, model_dir: str, threshold: float, device: str, model_name: str = ""):
        """Load and store entropy model for analysis."""
        try:
            latest_checkpoint_dir = self.find_latest_checkpoint(model_dir)
            
            # Check if this is already a consolidated directory (HF weights)
            direct_state_dict_path = os.path.join(latest_checkpoint_dir, "consolidated.pth")
            direct_params_path = os.path.join(latest_checkpoint_dir, "params.json")

            if os.path.exists(direct_state_dict_path) and os.path.exists(direct_params_path):
                consolidated_dir = latest_checkpoint_dir
                state_dict_path = direct_state_dict_path
                params_path = direct_params_path
            else:
                consolidated_dir = os.path.join(latest_checkpoint_dir, "consolidated")
                if not os.path.exists(consolidated_dir):
                    from bytelatent.checkpoint import main as consolidate_main
                    consolidate_main(command="consolidate", model_checkpoint_dir=latest_checkpoint_dir)
                
                state_dict_path = os.path.join(consolidated_dir, "consolidated.pth")
                params_path = os.path.join(consolidated_dir, "params.json")

            # Load model parameters
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

            # Store in entropy models
            model_key = model_name.lower().replace(' ', '_')
            self.entropy_models[model_key] = {
                'model': entropy_model,
                'patcher': patcher,
                'tokenizer': BltTokenizer(),
                'info': {
                    'name': model_name,
                    'params': model_params,
                    'checkpoint_dir': consolidated_dir,
                    'threshold': threshold,
                    'device': device
                }
            }
            
            print(f"✅ Entropy model '{model_name}' loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load entropy model '{model_name}': {e}")
            return False

    def find_latest_checkpoint(self, base_checkpoint_dir: str) -> str:
        """Find the latest checkpoint directory."""
        import glob
        
        hf_entropy_model_path = os.path.join(base_checkpoint_dir, "entropy_model")
        if os.path.isdir(hf_entropy_model_path):
            return hf_entropy_model_path

        actual_checkpoint_dir = os.path.join(base_checkpoint_dir, "checkpoints")
        if not os.path.isdir(actual_checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found at: {actual_checkpoint_dir}")

        dirs = glob.glob(os.path.join(actual_checkpoint_dir, "[0-9]"*10))
        if not dirs:
            raise FileNotFoundError(f"No numerical checkpoint subdirectories found in {actual_checkpoint_dir}")

        return max(dirs)

    def comprehensive_entropy_analysis(self, text: str, model_key: str, device: str) -> Dict[str, Any]:
        """
        Perform comprehensive entropy analysis capturing ALL entropy details.
        """
        if model_key not in self.entropy_models:
            return {"error": "Entropy model not available"}
        
        model_data = self.entropy_models[model_key]
        
        try:
            start_time = time.time()
            
            # Encode the sentence
            tokens_list = model_data['tokenizer'].encode(text, add_bos=True, add_eos=True)
            tokens_tensor = torch.tensor([tokens_list], device=device)

            # Get patches with detailed entropy information
            patch_lengths_tensor, scores_tensor = model_data['patcher'].patch(tokens_tensor)
            patch_lengths = patch_lengths_tensor.squeeze().tolist()
            entropy_scores = scores_tensor.squeeze().tolist() if scores_tensor is not None else []

            # Get individual token entropies if possible
            with torch.no_grad():
                try:
                    # Get logits from entropy model
                    logits = model_data['model'](tokens_tensor)
                    
                    # Debug: Check logits shape and type
                    if not isinstance(logits, torch.Tensor):
                        raise ValueError(f"Model output is not a tensor, got {type(logits)}")
                    
                    # Ensure logits has the right shape [batch_size, seq_len, vocab_size]
                    if len(logits.shape) != 3:
                        raise ValueError(f"Expected 3D logits tensor, got shape {logits.shape}")
                    
                    # Calculate per-token entropies using proper tensor operations
                    log_probs = torch.log_softmax(logits, dim=-1)
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Calculate entropy: H = -sum(p * log(p))
                    entropy_tensor = -(probs * log_probs).sum(dim=-1)  # Shape: [batch_size, seq_len]
                    
                    # Remove batch dimension and convert to list
                    entropy_tensor = entropy_tensor.squeeze(0)  # Shape: [seq_len]
                    
                    # Handle case where we have a single token (0-dimensional tensor)
                    if entropy_tensor.dim() == 0:
                        token_entropies = [float(entropy_tensor.item())]
                    else:
                        token_entropies = [float(x) for x in entropy_tensor.tolist()]
                        
                except Exception as e:
                    token_entropies = []
                    print(f"⚠️ Could not calculate individual token entropies: {e}")
                    print(f"   Debug info: tokens_tensor shape: {tokens_tensor.shape}")
                    try:
                        logits = model_data['model'](tokens_tensor)
                        print(f"   Debug info: logits type: {type(logits)}, shape: {getattr(logits, 'shape', 'N/A')}")
                    except Exception as debug_e:
                        print(f"   Debug info: Error getting logits: {debug_e}")

            # Convert patches to detailed information
            patches_detailed = []
            current_pos = 0
            
            for i, length in enumerate(patch_lengths):
                patch_token_ids = tokens_list[current_pos : current_pos + length]
                patch_text = model_data['tokenizer'].decode(patch_token_ids)
                
                # Get entropies for this patch
                patch_token_entropies = []
                if current_pos < len(token_entropies):
                    patch_token_entropies = token_entropies[current_pos : current_pos + length]
                
                patch_info = {
                    'patch_index': i,
                    'patch_length': length,
                    'patch_token_ids': patch_token_ids,
                    'patch_text': patch_text,
                    'patch_bytes': list(patch_text.encode('utf-8')),
                    'patch_char_length': len(patch_text),
                    'patch_byte_length': len(patch_text.encode('utf-8')),
                    'start_position': current_pos,
                    'end_position': current_pos + length,
                    'patch_entropy_score': entropy_scores[i] if i < len(entropy_scores) else None,
                    'token_entropies': patch_token_entropies,
                    'avg_token_entropy': sum(patch_token_entropies) / len(patch_token_entropies) if patch_token_entropies else None,
                    'entropy_variance': float(np.var(patch_token_entropies)) if len(patch_token_entropies) > 1 else 0.0
                }
                
                patches_detailed.append(patch_info)
                current_pos += length

            processing_time = time.time() - start_time
            
            # Reconstruct text
            reconstructed_text = ''.join([patch['patch_text'] for patch in patches_detailed])
            
            # Create comprehensive result
            result = {
                'model_key': model_key,
                'model_name': model_data['info']['name'],
                'model_info': {
                    'checkpoint_dir': model_data['info']['checkpoint_dir'],
                    'threshold': model_data['info']['threshold'],
                    'device': model_data['info']['device'],
                    'model_params': model_data['info']['params']
                },
                'input_text': text,
                'input_length_chars': len(text),
                'input_length_words': len(text.split()),
                'input_bytes': list(text.encode('utf-8')),
                'processing_time_seconds': processing_time,
                'tokenization': {
                    'blt_token_ids': tokens_list,
                    'blt_token_count': len(tokens_list),
                    'blt_token_strings': [model_data['tokenizer'].decode([tid]) for tid in tokens_list]
                },
                'entropy_analysis': {
                    'individual_token_entropies': token_entropies,
                    'patch_entropy_scores': entropy_scores,
                    'avg_token_entropy': sum(token_entropies) / len(token_entropies) if token_entropies else None,
                    'entropy_variance': float(np.var(token_entropies)) if len(token_entropies) > 1 else 0.0,
                    'threshold_used': model_data['info']['threshold']
                },
                'patches': patches_detailed,
                'patch_statistics': {
                    'patch_count': len(patch_lengths),
                    'total_tokens_in_patches': sum(patch_lengths),
                    'avg_patch_length': sum(patch_lengths) / len(patch_lengths) if patch_lengths else 0,
                    'patch_length_variance': float(np.var(patch_lengths)) if len(patch_lengths) > 1 else 0.0,
                    'min_patch_length': min(patch_lengths) if patch_lengths else 0,
                    'max_patch_length': max(patch_lengths) if patch_lengths else 0
                },
                'reconstruction': {
                    'reconstructed_text': reconstructed_text,
                    'perfect_reconstruction': text == reconstructed_text,
                    'char_diff_count': abs(len(text) - len(reconstructed_text)),
                    'reconstruction_bytes': list(reconstructed_text.encode('utf-8'))
                },
                'compression_analysis': {
                    'compression_ratio_patches': len(tokens_list) / len(patch_lengths) if patch_lengths else 0,
                    'compression_ratio_chars': len(text) / len(patch_lengths) if patch_lengths else 0,
                    'compression_efficiency': (len(tokens_list) - len(patch_lengths)) / len(tokens_list) if tokens_list else 0
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'model_key': model_key,
                'model_name': model_data['info']['name'],
                'error': str(e),
                'error_type': type(e).__name__
            }

    def analyze_samples_comprehensive(self, samples_df: pd.DataFrame, include_entropy: bool = True, 
                                    include_traditional: bool = True) -> Dict[str, Any]:
        """
        Analyze all samples with comprehensive data capture.
        """
        print(f"\n🔬 COMPREHENSIVE ANALYSIS OF {len(samples_df)} SAMPLES")
        print("=" * 80)
        
        # Prepare results structure
        comprehensive_results = {
            'metadata': {
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_samples': len(samples_df),
                'include_entropy_models': include_entropy,
                'include_traditional_tokenizers': include_traditional,
                'entropy_models_loaded': list(self.entropy_models.keys()),
                'traditional_tokenizers_loaded': list(self.tokenizers.keys())
            },
            'tokenizer_metadata': {},
            'entropy_model_metadata': {},
            'samples': []
        }
        
        # Store tokenizer metadata
        for key, tokenizer_info in self.tokenizers.items():
            comprehensive_results['tokenizer_metadata'][key] = {
                'name': tokenizer_info['name'],
                'type': tokenizer_info['type'],
                'description': tokenizer_info['description'],
                'vocab_size': tokenizer_info['vocab_size'],
                'model_id': tokenizer_info['model_id']
            }
        
        # Store entropy model metadata
        for key, model_data in self.entropy_models.items():
            comprehensive_results['entropy_model_metadata'][key] = model_data['info']
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Process each sample
        for sample_idx, (_, row) in enumerate(samples_df.iterrows(), 1):
            text = row['text']
            print(f"\n📝 Processing sample {sample_idx}/{len(samples_df)}: {text[:50]}...")
            
            sample_result = {
                'sample_index': sample_idx,
                'original_data': {
                    'text': text,
                    'source': row['source'] if 'source' in row else 'unknown',
                    'year': int(row['year']) if 'year' in row and pd.notna(row['year']) else None,
                    'char_length': len(text),
                    'word_count': len(text.split()),
                    'byte_length': len(text.encode('utf-8')),
                    'unique_chars': len(set(text)),
                    'text_bytes': list(text.encode('utf-8'))
                },
                'traditional_tokenizer_results': {},
                'entropy_model_results': {}
            }
            
            # Traditional tokenizer analysis
            if include_traditional:
                print(f"  🔤 Analyzing with traditional tokenizers...")
                for tokenizer_key in self.tokenizers.keys():
                    try:
                        result = self.comprehensive_tokenize(text, tokenizer_key)
                        sample_result['traditional_tokenizer_results'][tokenizer_key] = result
                        
                        if 'error' not in result:
                            print(f"    ✅ {result['tokenizer_name']}: {result['statistics']['token_count']} tokens")
                        else:
                            print(f"    ❌ {self.tokenizers[tokenizer_key]['name']}: {result['error']}")
                            
                    except Exception as e:
                        print(f"    ❌ {tokenizer_key}: Unexpected error: {e}")
                        sample_result['traditional_tokenizer_results'][tokenizer_key] = {
                            'tokenizer_key': tokenizer_key,
                            'error': str(e),
                            'error_type': type(e).__name__
                        }
            
            # Entropy model analysis
            if include_entropy:
                print(f"  🧠 Analyzing with entropy models...")
                for model_key in self.entropy_models.keys():
                    try:
                        result = self.comprehensive_entropy_analysis(text, model_key, device)
                        sample_result['entropy_model_results'][model_key] = result
                        
                        if 'error' not in result:
                            print(f"    ✅ {result['model_name']}: {result['patch_statistics']['patch_count']} patches")
                        else:
                            print(f"    ❌ {result['model_name']}: {result['error']}")
                            
                    except Exception as e:
                        print(f"    ❌ {model_key}: Unexpected error: {e}")
                        sample_result['entropy_model_results'][model_key] = {
                            'model_key': model_key,
                            'error': str(e),
                            'error_type': type(e).__name__
                        }
            
            comprehensive_results['samples'].append(sample_result)
        
        return comprehensive_results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive tokenizer data collection - captures ALL tokenization details.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script performs the most comprehensive tokenization analysis possible, capturing:

TRADITIONAL TOKENIZERS:
• Complete token information (IDs, strings, bytes, positions)
• Reconstruction quality analysis
• Vocabulary coverage statistics
• Processing time measurements

ENTROPY MODELS (BLT):
• Individual token entropies
• Patch-level entropy scores
• Detailed patch composition
• Compression efficiency metrics
• Complete reconstruction verification

OUTPUT:
• Comprehensive JSON file with ALL data
• Perfect for research analysis and paper figures
• Includes metadata for full reproducibility

Examples:
  python comprehensive_data_collector.py --csv_file subset.csv --num_samples 10
  python comprehensive_data_collector.py --csv_file data.csv --my_model models/my_model --hf_model hf-weights
  python comprehensive_data_collector.py --csv_file data.csv --traditional_only --num_samples 50
        """
    )
    
    parser.add_argument("--csv_file", type=str, default="lang_dis_dataset.csv",
                       help="Path to the CSV file containing sentences.")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to analyze comprehensively.")
    parser.add_argument("--my_model_dir", type=str, default="models/my_entropy_model_checkpoints",
                       help="Path to your trained entropy model.")
    parser.add_argument("--hf_model_dir", type=str, default="hf-weights",
                       help="Path to HuggingFace entropy model.")
    parser.add_argument("--threshold", type=float, default=1.33,
                       help="Entropy threshold for patching.")
    parser.add_argument("--source_filter", type=str, default=None,
                       help="Filter by source (e.g., 'twitter', 'wikipedia').")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output JSON filename (auto-generated if not specified).")
    parser.add_argument("--traditional_only", action="store_true",
                       help="Only analyze traditional tokenizers.")
    parser.add_argument("--entropy_only", action="store_true",
                       help="Only analyze entropy models.")

    args = parser.parse_args()
    
    # Handle inclusion flags
    include_entropy = not args.traditional_only
    include_traditional = not args.entropy_only
    
    if not include_entropy and not include_traditional:
        print("❌ Error: Cannot disable both entropy and traditional analysis!")
        exit(1)
    
    # Load data
    print(f"📊 Loading data from {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    
    if args.source_filter:
        df = df[df['source'] == args.source_filter]
        if df.empty:
            print(f"❌ No data found for source '{args.source_filter}'")
            exit(1)
        print(f"📋 Filtered to {len(df)} rows for source '{args.source_filter}'")
    
    # Sample data
    sample_size = min(args.num_samples, len(df))
    sampled_df = df.sample(n=sample_size, random_state=42)
    print(f"🎯 Analyzing {sample_size} samples")
    
    # Initialize analyzer
    analyzer = ComprehensiveTokenizerAnalyzer()
    
    # Load entropy models if requested
    if include_entropy:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {device}")
        
        if analyzer.load_entropy_model(args.my_model_dir, args.threshold, device, "My Model"):
            print("✅ My entropy model loaded")
        
        if analyzer.load_entropy_model(args.hf_model_dir, args.threshold, device, "HF Model"):
            print("✅ HF entropy model loaded")
    
    # Perform comprehensive analysis
    results = analyzer.analyze_samples_comprehensive(
        sampled_df, 
        include_entropy=include_entropy,
        include_traditional=include_traditional
    )
    
    # Save results
    if args.output_file:
        output_filename = args.output_file
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"comprehensive_tokenizer_data_{sample_size}_samples_{timestamp}.json"
    
    print(f"\n💾 Saving comprehensive results to {output_filename}...")
    
    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    # Calculate file size
    file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
    
    print(f"✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: {output_filename}")
    print(f"📊 File size: {file_size_mb:.2f} MB")
    print(f"🔬 Contains complete tokenization data for research analysis")
    
    # Print summary statistics
    entropy_models = len(results['entropy_model_metadata'])
    traditional_models = len(results['tokenizer_metadata'])
    total_analyses = sample_size * (entropy_models + traditional_models)
    
    print(f"\n📈 ANALYSIS SUMMARY:")
    print(f"   Samples analyzed: {sample_size}")
    print(f"   Entropy models: {entropy_models}")
    print(f"   Traditional tokenizers: {traditional_models}")
    print(f"   Total analyses performed: {total_analyses}")
    print(f"   Data points captured: Tens of thousands per sample")

if __name__ == "__main__":
    main()
