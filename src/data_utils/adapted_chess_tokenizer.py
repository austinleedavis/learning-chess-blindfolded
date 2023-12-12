# This is an adapter class that enables the ChessTokenizer class to be initialized 
# as a PreTrainedTokenizer. This is required for us to feed the tokenizer to the
# Transformer Lens library.
import torch
from data_utils.chess_tokenizer import ChessTokenizer
from transformers import PreTrainedTokenizer

class AdaptedChessTokenizer(PreTrainedTokenizer):
    def __init__(self, 
                 vocab_file = "learning-chess-blindfolded/sample_data/lm_chess/vocab/uci/vocab.txt", 
                 notation='uci',
                 merges_file=None,
                 tokenizer_file=None,
                 pad_token="<pad>", 
                 unk_token="</s>", # unknown token
                 bos_token="<s>", # beginning of sequence
                 eos_token="</s>", # end of sequence
                 add_prefix_space=False, 
                 *args, 
                 **kwargs):
        
        self.chess_tokenizer = ChessTokenizer(
            notation=notation,
            vocab_file=vocab_file, 
            pad_token=pad_token,
            unk_token=unk_token,
            eos_token=eos_token, 
            bos_token=bos_token,
            *args, **kwargs)
        
        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            eos_token=eos_token, 
            bos_token=bos_token,
            *args, **kwargs)

    def encode(self, text, *args, **kwargs):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, token_ids, *args, **kwargs):
        return self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))
        
    @property
    def vocab_size(self) -> int:
        return len(self.get_vocab())
    
    def get_vocab(self):
        return self.chess_tokenizer.get_vocab()

    def tokenize(self, text, add_special_tokens=True, *args, **kwargs):
        instance = []
        for part in self.chess_tokenizer.move_pattern.split(text.strip()):
            if part == '':
                continue
            elif part == ' ':
                continue
                # instance.append(part)
            else:
                instance.append(part.strip())

        if add_special_tokens:
            instance = [self.bos_token] + instance + [self.eos_token]   # Empty string denotes end of move
        return instance

    def _convert_token_to_id(self, token:str) -> int:
        """Converts a token (str) in an id using the vocab."""
        return self.chess_tokenizer.encode_token(token)

    def _convert_id_to_token(self, index:int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.chess_tokenizer.decode_token(index)

    def __call__(self, text, max_length=None, **kwargs):
        return torch.tensor(self.chess_tokenizer.__call__(lines=text, max_length=max_length, **kwargs))
    
    @property
    def bos_token_id(self):
        return self.chess_tokenizer.bos_token_id
    
    @property
    def eos_token_id(self):
        return self.chess_tokenizer.eos_token_id
    
    @property
    def pad_token_id(self):
        return self.chess_tokenizer.pad_token_id
    