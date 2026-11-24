r"""Exercise Hugging Face model import, local checkpoint loading and CPU inference without c10d.

Windows PowerShell from the checkout:
  & "$env:LOCALAPPDATA\Programs\LTXDesktopAMD\application\resources\python\python.exe" .\checks\check_gemma_loading.py

Runs offline with random tiny Gemma 3 weights in a temporary directory.
"""
import tempfile
import torch
from transformers import Gemma3TextConfig, Gemma3ForCausalLM

if torch.distributed.is_available():
    raise SystemExit("This regression check requires a single-process PyTorch build without native c10d.")
config = Gemma3TextConfig(
    vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=64,
    sliding_window=8, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    layer_types=['sliding_attention', 'full_attention'],
)
torch.set_num_threads(1)
torch.manual_seed(17)
inputs = torch.tensor([[1, 7, 2]])
with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
    model = Gemma3ForCausalLM(config).eval()
    expected = model(inputs, use_cache=False).logits
    model.save_pretrained(directory)
    loaded = Gemma3ForCausalLM.from_pretrained(directory, local_files_only=True).eval()
    actual = loaded(inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    print('PASS local Gemma 3 save, pretrained weight load and forward, logits', tuple(actual.shape))
