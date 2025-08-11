#!/usr/bin/env python3
"""
Simple tokenizer test script for COHFIE dataset.
Tests readily available tokenizers without authentication requirements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import time
import duckdb

# Test sample Filipino texts
SAMPLE_TEXTS = [
    "Dinala ng palad na di cayat asam sa natitiualag na Bayanbayanan",
    "Masigla cong isip na casalucuyang nagcusang cumilos sa cau uculan lungcot",
    "Ang anyong mapanglao iiral pag-guiit ng oras na dapat ipantuyong pauis",
    "The quick brown fox jumps over the lazy dog",  # English comparison
    "¡Hola! ¿Cómo estás? Muy bien, gracias."  # Spanish comparison
]

def test_gpt2_tokenizer():
    """Test GPT-2 BPE tokenizer."""
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('baebee/gpt2-tagalog-80k')
        
        print("🤖 GPT-2 BPE Tokenizer")
        print("-" * 40)
        
        for i, text in enumerate(SAMPLE_TEXTS, 1):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(tokens)
            
            # Get individual token strings
            token_strings = []
            for token_id in tokens:
                token_str = tokenizer.decode([token_id])
                token_strings.append(f"'{token_str}'")
            
            print(f"{i}. Original: {text}")
            print(f"   Token IDs ({len(tokens)}): {tokens}")
            print(f"   Token Strings: {' | '.join(token_strings)}")
            print(f"   Decoded: {decoded}")
            print(f"   Chars/Token: {len(text)/len(tokens):.2f}")
            
            # Show byte representation for first few tokens
            if len(tokens) > 0:
                print(f"   First token details:")
                first_token = tokenizer.decode([tokens[0]])
                print(f"     '{first_token}' -> bytes: {first_token.encode('utf-8')}")
            print()
            
        return True
    except Exception as e:
        print(f"❌ GPT-2 tokenizer failed: {e}")
        return False

def test_tiktoken():
    """Test Tiktoken tokenizers."""
    try:
        import tiktoken
        
        # Test different encodings
        encodings = ["cl100k_base", "o200k_base"]
        
        for encoding_name in encodings:
            try:
                enc = tiktoken.get_encoding(encoding_name)
                print(f"🔤 Tiktoken ({encoding_name})")
                print("-" * 40)
                
                for i, text in enumerate(SAMPLE_TEXTS, 1):
                    tokens = enc.encode(text)
                    decoded = enc.decode(tokens)
                    
                    # Get individual token strings using decode_single_token_bytes
                    token_strings = []
                    for token_id in tokens:
                        try:
                            token_bytes = enc.decode_single_token_bytes(token_id)
                            token_str = token_bytes.decode('utf-8', errors='replace')
                            token_strings.append(f"'{token_str}'")
                        except:
                            token_strings.append(f"'[{token_id}]'")
                    
                    print(f"{i}. Original: {text}")
                    print(f"   Token IDs ({len(tokens)}): {tokens}")
                    print(f"   Token Strings: {' | '.join(token_strings)}")
                    print(f"   Decoded: {decoded}")
                    print(f"   Chars/Token: {len(text)/len(tokens):.2f}")
                    
                    # Show byte details for first token
                    if len(tokens) > 0:
                        first_token_bytes = enc.decode_single_token_bytes(tokens[0])
                        print(f"   First token bytes: {first_token_bytes}")
                    print()
                    
            except Exception as e:
                print(f"❌ {encoding_name} not available: {e}")
                
        return True
    except Exception as e:
        print(f"❌ Tiktoken failed: {e}")
        return False

def test_byt5_tokenizer():
    """Test ByT5 byte-level tokenizer."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
        
        print("🔢 ByT5 Byte-Level Tokenizer")
        print("-" * 40)
        
        for i, text in enumerate(SAMPLE_TEXTS, 1):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            
            # Get individual token strings - ByT5 works on bytes
            token_strings = []
            char_representations = []
            
            for token_id in tokens:
                # Decode individual token
                token_str = tokenizer.decode([token_id], skip_special_tokens=True)
                token_strings.append(f"'{token_str}'")
                
                # Show what character/byte this represents
                if token_id < 259:  # Regular byte tokens (0-255 + special tokens)
                    if token_id >= 3 and token_id <= 258:  # UTF-8 bytes (shifted by 3)
                        byte_val = token_id - 3
                        try:
                            char = chr(byte_val) if byte_val < 128 else f"\\x{byte_val:02x}"
                            char_representations.append(f"{token_id}→'{char}'")
                        except:
                            char_representations.append(f"{token_id}→byte{byte_val}")
                    else:
                        char_representations.append(f"{token_id}→special")
                else:
                    char_representations.append(f"{token_id}→extended")
            
            print(f"{i}. Original: {text}")
            print(f"   Original bytes: {text.encode('utf-8')}")
            print(f"   Token IDs ({len(tokens)}): {tokens}")
            print(f"   Token meanings: {' | '.join(char_representations[:10])}{'...' if len(tokens) > 10 else ''}")
            print(f"   Decoded: {decoded}")
            print(f"   Chars/Token: {len(text)/len(tokens):.2f}")
            print()
            
        return True
    except Exception as e:
        print(f"❌ ByT5 tokenizer failed: {e}")
        return False

def test_simple_baseline():
    """Test simple character and word-level baselines."""
    print("📝 Simple Baselines")
    print("-" * 40)
    
    for i, text in enumerate(SAMPLE_TEXTS, 1):
        # Character-level
        char_tokens = list(text)
        # Word-level
        word_tokens = text.split()
        # Byte-level
        byte_tokens = list(text.encode('utf-8'))
        
        print(f"{i}. Original: {text}")
        print(f"   Characters: {len(char_tokens)} tokens")
        print(f"   Words: {len(word_tokens)} tokens")
        print(f"   Bytes: {len(byte_tokens)} tokens")
        print(f"   Chars/Word: {len(char_tokens)/len(word_tokens):.2f}")
        print()

def test_cohfie_sample():
    """Test tokenizers on actual COHFIE sample if available."""
    csv_file = 'COHFIE_V4.csv'
    
    try:
        print("📊 Testing on COHFIE Sample")
        print("-" * 40)
        
        # Try to load a small sample
        conn = duckdb.connect()
        sample_df = conn.execute(f"""
            SELECT text 
            FROM read_csv_auto('{csv_file}')
            LIMIT 5
        """).fetchdf()
        conn.close()
        
        if len(sample_df) == 0:
            print("No data found in COHFIE file")
            return
            
        print(f"Loaded {len(sample_df)} texts from COHFIE")
        
        # Test with available tokenizers
        tokenizers_to_test = []
        
        # GPT-2
        try:
            from transformers import GPT2Tokenizer
            gpt2_tok = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizers_to_test.append(('GPT-2', gpt2_tok))
        except:
            pass
            
        # Tiktoken
        try:
            import tiktoken
            tiktoken_tok = tiktoken.get_encoding("cl100k_base")
            tokenizers_to_test.append(('Tiktoken', tiktoken_tok))
        except:
            pass
            
        # ByT5
        try:
            from transformers import AutoTokenizer
            byt5_tok = AutoTokenizer.from_pretrained('google/byt5-small')
            tokenizers_to_test.append(('ByT5', byt5_tok))
        except:
            pass
        
        # Compare tokenizers on COHFIE texts
        for idx, row in sample_df.iterrows():
            text = str(row['text'])[:200]  # Truncate for display
            
            print(f"\nText {idx+1}: {text}...")
            print(f"Length: {len(text)} chars, {len(text.split())} words")
            
            for name, tokenizer in tokenizers_to_test:
                try:
                    start_time = time.time()
                    
                    if name == 'Tiktoken':
                        tokens = tokenizer.encode(text)
                    else:
                        tokens = tokenizer.encode(text, add_special_tokens=False)
                    
                    duration = time.time() - start_time
                    
                    print(f"  {name:10}: {len(tokens):4d} tokens "
                          f"({len(text)/len(tokens):4.2f} chars/token) "
                          f"({duration*1000:5.1f}ms)")
                except Exception as e:
                    print(f"  {name:10}: Error - {e}")
        
    except Exception as e:
        print(f"Could not test COHFIE sample: {e}")
        print("Make sure COHFIE_V4.csv is in the current directory")

def performance_benchmark():
    """Quick performance benchmark."""
    print("\n⚡ Performance Benchmark")
    print("-" * 40)
    
    # Create test text
    long_text = " ".join(SAMPLE_TEXTS * 100)  # Repeat for longer text
    
    tokenizers_to_test = []
    
    # GPT-2
    try:
        from transformers import GPT2Tokenizer
        gpt2_tok = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizers_to_test.append(('GPT-2', lambda x: gpt2_tok.encode(x, add_special_tokens=False)))
    except:
        pass
        
    # Tiktoken
    try:
        import tiktoken
        tiktoken_tok = tiktoken.get_encoding("cl100k_base")
        tokenizers_to_test.append(('Tiktoken', lambda x: tiktoken_tok.encode(x)))
    except:
        pass
        
    # ByT5
    try:
        from transformers import AutoTokenizer
        byt5_tok = AutoTokenizer.from_pretrained('google/byt5-small')
        tokenizers_to_test.append(('ByT5', lambda x: byt5_tok.encode(x, add_special_tokens=False)))
    except:
        pass
    
    print(f"Testing with text of {len(long_text)} characters")
    print(f"Repeating each test 10 times for average timing")
    print()
    
    for name, tokenize_func in tokenizers_to_test:
        times = []
        token_counts = []
        
        for _ in range(10):
            start_time = time.time()
            tokens = tokenize_func(long_text)
            duration = time.time() - start_time
            
            times.append(duration)
            token_counts.append(len(tokens))
        
        avg_time = np.mean(times)
        avg_tokens = np.mean(token_counts)
        tokens_per_sec = avg_tokens / avg_time
        
        print(f"{name:12}: {avg_time*1000:6.1f}ms avg, "
              f"{avg_tokens:5.0f} tokens, "
              f"{tokens_per_sec:8.0f} tokens/sec")

def main():
    """Main function to run all tests."""
    print("🚀 COHFIE Tokenizer Quick Test")
    print("=" * 50)
    print("Testing various tokenizers on sample Filipino texts")
    print()
    
    # Test each tokenizer
    print("1️⃣ Testing GPT-2 BPE Tokenizer")
    test_gpt2_tokenizer()
    print()
    
    print("2️⃣ Testing Tiktoken")
    test_tiktoken()
    print()
    
    print("3️⃣ Testing ByT5 Byte-Level Tokenizer")
    test_byt5_tokenizer()
    print()
    
    print("4️⃣ Testing Simple Baselines")
    test_simple_baseline()
    print()
    
    print("5️⃣ Testing on COHFIE Sample")
    test_cohfie_sample()
    
    print("6️⃣ Performance Benchmark")
    performance_benchmark()
    
    print("\n✅ Testing complete!")
    print("\n📝 Summary:")
    print("- GPT-2: Standard BPE, good for English, may struggle with Filipino")
    print("- Tiktoken: Fast BPE, optimized for OpenAI models")
    print("- ByT5: Byte-level, language-agnostic, good for any language")
    print("- For Filipino/Tagalog text, ByT5 might be most robust")
    print("\n💡 Next steps:")
    print("- Run the full comparison script for comprehensive analysis")
    print("- Try different sample sizes to see scaling behavior")
    print("- Consider training custom tokenizers for Filipino")

if __name__ == "__main__":
    main()
