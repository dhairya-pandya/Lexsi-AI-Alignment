"""Fixtures: a tiny random cohere2 model and a byte-level tokenizer carrying Tiny Aya's real chat template."""
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import Cohere2Config, Cohere2ForCausalLM, PreTrainedTokenizerFast

FIXTURES = Path(__file__).parent / "fixtures"
SPECIAL_TOKENS = [
    "<PAD>", "<UNK>", "<BOS_TOKEN>", "<|END_OF_TURN_TOKEN|>", "<|START_OF_TURN_TOKEN|>",
    "<|SYSTEM_TOKEN|>", "<|USER_TOKEN|>", "<|CHATBOT_TOKEN|>", "<|START_RESPONSE|>", "<|END_RESPONSE|>",
]
TINY_WINDOW = 16


def _byte_level_tokenizer(template: str) -> PreTrainedTokenizerFast:
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for ch in sorted(alphabet):
        vocab[ch] = len(vocab)
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<UNK>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tk.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, bos_token="<BOS_TOKEN>", eos_token="<|END_OF_TURN_TOKEN|>",
        pad_token="<PAD>", unk_token="<UNK>",
        additional_special_tokens=SPECIAL_TOKENS[4:],
    )
    tok.chat_template = template
    return tok


@pytest.fixture(scope="session")
def tiny_tok():
    return _byte_level_tokenizer((FIXTURES / "tiny_aya_chat_template.jinja").read_text())


@pytest.fixture(scope="session")
def expanse_tok():
    return _byte_level_tokenizer((FIXTURES / "aya_expanse_chat_template.jinja").read_text())


@pytest.fixture(scope="session")
def tiny_model(tiny_tok):
    torch.manual_seed(0)
    n_layers = 8
    cfg = Cohere2Config(
        vocab_size=len(tiny_tok), hidden_size=64, intermediate_size=128, num_hidden_layers=n_layers,
        num_attention_heads=4, num_key_value_heads=2, sliding_window=TINY_WINDOW,
        max_position_embeddings=4096,
        layer_types=["full_attention" if (i + 1) % 4 == 0 else "sliding_attention" for i in range(n_layers)],
        pad_token_id=tiny_tok.pad_token_id, bos_token_id=tiny_tok.bos_token_id,
        eos_token_id=tiny_tok.eos_token_id,
    )
    model = Cohere2ForCausalLM(cfg).eval()
    return model
