# tokenizer_utils.py
import torch
import torch.nn.functional as F
import re
from typing import List, Dict, Tuple
import string

class SimpleTokenizer:
    """Simple word-level tokenizer for demonstration purposes"""
    
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"
        self.bos_token = "<BOS>"
        
        # Initialize with special tokens
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
        
        # Add common words and characters
        self._build_vocab()
    
    def _build_vocab(self):
        """Build a simple vocabulary with common words and characters"""
        # Add single characters
        chars = list(string.ascii_letters + string.digits + string.punctuation + " \n\t")
        current_id = len(self.word_to_id)
        
        for char in chars:
            if char not in self.word_to_id and current_id < self.vocab_size:
                self.word_to_id[char] = current_id
                self.id_to_word[current_id] = char
                current_id += 1
        
        # Add common words
        common_words = [
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "can", "may", "might",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
            "this", "that", "these", "those", "here", "there", "where", "when", "why", "how",
            "what", "who", "which", "whose", "all", "any", "some", "many", "much", "few",
            "one", "two", "three", "first", "last", "new", "old", "good", "bad", "big", "small",
            "intelligence", "artificial", "machine", "learning", "deep", "neural", "network",
            "model", "training", "data", "algorithm", "computer", "technology", "future",
            "science", "research", "development", "system", "process", "information",
            "language", "natural", "processing", "generation", "text", "word", "sentence"
        ]
        
        for word in common_words:
            if word not in self.word_to_id and current_id < self.vocab_size:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
        
        # Fill remaining vocab with dummy tokens if needed
        while current_id < self.vocab_size:
            dummy_token = f"<UNUSED_{current_id}>"
            self.word_to_id[dummy_token] = current_id
            self.id_to_word[current_id] = dummy_token
            current_id += 1
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization - split by spaces and punctuation"""
        # Clean text
        text = text.lower().strip()
        
        # Simple regex to split words and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        # Convert to IDs
        ids = []
        for token in tokens:
            if token in self.word_to_id:
                ids.append(self.word_to_id[token])
            else:
                # If word not in vocab, try character-level fallback
                if len(token) == 1:
                    ids.append(self.word_to_id.get(token, self.word_to_id[self.unk_token]))
                else:
                    # Split unknown word into characters
                    for char in token:
                        ids.append(self.word_to_id.get(char, self.word_to_id[self.unk_token]))
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text"""
        tokens = []
        
        for id_val in ids:
            if id_val < len(self.id_to_word):
                token = self.id_to_word[id_val]
                
                if skip_special_tokens and token in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
                    continue
                
                tokens.append(token)
            else:
                if not skip_special_tokens:
                    tokens.append(self.unk_token)
        
        # Simple reconstruction
        result = ""
        for i, token in enumerate(tokens):
            if len(token) == 1 and token in string.punctuation:
                result += token
            elif len(token) == 1 and token in [' ', '\n', '\t']:
                result += token
            elif i == 0:
                result += token
            elif len(token) == 1 and (token.isalnum() or token in string.punctuation):
                result += token
            else:
                result += " " + token
        
        return result.strip()

def generate_text_improved(model, tokenizer, prompt: str, max_length: int = 50, 
                         temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9) -> str:
    """Improved text generation function with better sampling"""
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    generated_ids = input_ids.copy()
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            # Limit sequence length to model's max
            if len(generated_ids) >= model.cfg.max_seq_len:
                break
            
            # Get model predictions
            current_input = torch.tensor([generated_ids], dtype=torch.long)
            
            try:
                if hasattr(model, 'forward'):
                    logits, _, aux = model(current_input)
                    # For MoR models, we can access ponder cost from aux
                    ponder_cost = aux.get('expected_updates', 0) if aux else 0
                else:
                    logits = model(current_input)
                    ponder_cost = 0
                
                # Get logits for next token
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < top_k_values[-1]] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Check for end token or invalid token
                if next_token >= tokenizer.vocab_size:
                    break
                
                if tokenizer.id_to_word.get(next_token) == tokenizer.eos_token:
                    break
                
                generated_ids.append(next_token)
                
                # Stop at sentence boundaries for better output
                current_token = tokenizer.id_to_word.get(next_token, "")
                if current_token in [".", "!", "?"] and len(generated_ids) > len(input_ids) + 10:
                    break
                    
            except Exception as e:
                print(f"Generation error: {e}")
                break
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text, getattr(model, 'ponder_cost', ponder_cost)

# Usage example
if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = SimpleTokenizer(vocab_size=50257)
    
    test_text = "The future of artificial intelligence is bright."
    print(f"Original: {test_text}")
    
    # Encode
    ids = tokenizer.encode(test_text)
    print(f"Encoded: {ids}")
    
    # Decode
    decoded = tokenizer.decode(ids)
    print(f"Decoded: {decoded}")
    
    print(f"Vocabulary size: {len(tokenizer.word_to_id)}")
    print(f"Sample tokens: {list(tokenizer.word_to_id.keys())[:20]}")