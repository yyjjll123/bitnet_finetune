from dataclasses import dataclass
from typing import Optional, Tuple, Union

import argparse
import json
import os
import random
import contextlib
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import math
from tqdm import tqdm

from xformers.ops import fmha, rope_padded
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
)

from tokenizer import Tokenizer

import ctypes
bitnet_lib = ctypes.CDLL('bitnet_kernels/libbitnet.so')


def bitnet_int8xint2_linear(input0, input1, s, ws):
    out_shape = list(input0.shape)
    out_shape[-1] = input1.shape[0]

    stream = torch.cuda.current_stream()

    M = input0.shape[0]
    if len(out_shape) == 3: 
        M *= input0.shape[1]
    N = input1.shape[0]
    K = input1.shape[1] * 4

    ret = torch.zeros(*out_shape, dtype=torch.bfloat16, device=input0.device)

    # Redirect stdout to /dev/null to suppress kernel prints
    orig_stdout_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    
    try:
        bitnet_lib.bitlinear_int8xint2(*[ctypes.c_void_p(input0.data_ptr()), ctypes.c_void_p(input1.data_ptr()), ctypes.c_void_p(ret.data_ptr()), ctypes.c_void_p(s.data_ptr()), ctypes.c_void_p(ws.data_ptr()), ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_void_p(stream.cuda_stream)])
    finally:
        # Restore stdout
        os.dup2(orig_stdout_fd, 1)
        os.close(devnull_fd)
        os.close(orig_stdout_fd)

    return ret

@dataclass
class ModelArgs:
    dim: int = 2560
    n_layers: int = 30
    n_heads: int = 20
    n_kv_heads: int = 5
    vocab_size: int = 128256
    ffn_dim: int = 6912
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_kernel: bool = False

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

LayerCache = Tuple[torch.Tensor, torch.Tensor]

class BitLinearKernel(nn.Module):
    in_features: int
    out_features: int
    weight: torch.Tensor
    weight_scale: torch.Tensor
    bias: Optional[torch.Tensor]

    def __init__(self, in_features: int, out_features: int, bias: bool = False, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(torch.zeros(out_features, in_features//4, dtype=torch.int8), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16), requires_grad=False)
        
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter('bias', None)

    @torch.compile
    def quant_input(self, input):
        s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (input * s).round().clamp(-128, 127).to(torch.int8), s

    def forward(self, input):
        input, s = self.quant_input(input)
        output = bitnet_int8xint2_linear(input, self.weight, s, self.weight_scale)
        if self.bias is not None:
            output = output + self.bias
        return output

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        norm_eps: float,
        use_kernel: bool,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_local_heads = n_heads
        self.n_local_kv_heads = n_kv_heads

        Linear = BitLinearKernel if use_kernel else nn.Linear

        self.wqkv = Linear(
            dim,
            (self.n_local_heads + 2 * self.n_local_kv_heads) * head_dim,
            bias=False,
        )
        self.wo = Linear(
            self.n_local_heads * head_dim,
            dim,
            bias=False,
        )

        self.attn_sub_norm = RMSNorm(dim, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache,
        attn_bias: AttnBias,
    ) -> torch.Tensor:

        xqkv = self.wqkv(x)

        q_size = self.n_local_heads * self.head_dim
        xq = xqkv[..., :q_size]
        xkv = xqkv[..., q_size:]
        xk, xv = xkv.chunk(2, dim=-1)

        output_shape = xq.shape
        heads_per_group = self.n_local_heads // self.n_local_kv_heads
        
        batch_size, seq_len, _ = x.shape

        xq = xq.view(
            batch_size, seq_len, self.n_local_kv_heads, heads_per_group, self.head_dim
        )
        xk = xk.view(batch_size, seq_len, self.n_local_kv_heads, 1, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_local_kv_heads, 1, self.head_dim)
        
        cache_k, cache_v = cache

        xq = rope_padded(
            xq=xq,
            xk=xk,
            xv=xv,
            cache_k=cache_k,
            cache_v=cache_v,
            attn_bias=attn_bias,
            theta=self.rope_theta,
        )

        output = fmha.memory_efficient_attention_forward(
            xq, cache_k, cache_v, attn_bias, op = fmha.flash.FwOp
        )

        output = output.reshape(output_shape)
        output = self.attn_sub_norm(output)
        output = self.wo(output)

        return output

@torch.compile
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) ** 2

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        norm_eps: float,
        use_kernel: bool,
    ):
        super().__init__()

        Linear = BitLinearKernel if use_kernel else nn.Linear

        self.w13 = Linear(
            dim,
            2 * hidden_dim,
            bias=False,
        )
        self.w2 = Linear(
            hidden_dim,
            dim,
            bias=False,
        )
        self.ffn_sub_norm = RMSNorm(hidden_dim, norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x13 = self.w13(x)
        x1, x3 = x13.chunk(2, -1)
        inner = self.ffn_sub_norm(squared_relu(x1) * x3)
        output = self.w2(inner)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.dim % args.n_heads == 0
        head_dim = args.dim // args.n_heads
        if args.n_kv_heads is not None:
            n_kv_heads = args.n_kv_heads
        else:
            n_kv_heads = args.n_heads

        assert args.n_heads % n_kv_heads == 0

        self.attention = Attention(
            dim=args.dim,
            head_dim=head_dim,
            n_heads=args.n_heads,
            n_kv_heads=n_kv_heads,
            rope_theta=args.rope_theta,
            norm_eps=args.norm_eps,
            use_kernel=args.use_kernel,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.ffn_dim,
            norm_eps=args.norm_eps,
            use_kernel=args.use_kernel,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache,
        attn_bias: AttnBias,
    ) -> torch.Tensor:
        h = x + self.attention.forward(
            self.attention_norm(x),
            cache,
            attn_bias,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size > 0

        self.tok_embeddings = nn.Embedding(
            num_embeddings=args.vocab_size,
            embedding_dim=args.dim,
        )

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False,
        )

    def forward_with_attn_bias(
        self,
        token_values: torch.Tensor,
        attn_bias: AttnBias,
        cache: list[LayerCache],
    ) -> torch.Tensor:
        h = self.tok_embeddings(token_values)
        for i, layer in enumerate(self.layers):
            h = layer(h, cache[i], attn_bias)

        logits = self.output(self.norm(h))
        return logits.float()

    def forward(
        self,
        token_values: torch.Tensor,
        token_lengths: torch.Tensor,
        start_pos: torch.Tensor,
        cache: list[LayerCache],
        kv_padding: int,
    ) -> torch.Tensor:
        attn_bias = AttnBias.from_seqlens(
            q_seqlen=token_lengths.tolist(),
            kv_seqlen=(start_pos + token_lengths).tolist(),
            kv_padding=kv_padding,
        )
        return self.forward_with_attn_bias(token_values, attn_bias, cache)


def make_cache(
    args: ModelArgs,
    length: int,
    device: Optional[Union[str, torch.device]] = None,
    n_layers: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> list[LayerCache]:
    head_dim = args.dim // args.n_heads
    n_kv_heads = args.n_kv_heads
    if n_kv_heads is None:
        n_kv_heads = args.n_heads
    n_local_kv_heads = n_kv_heads

    if n_layers is None:
        n_layers = args.n_layers

    shape = (1, length, n_local_kv_heads, 1, head_dim)
    heads_per_group = args.n_heads // n_kv_heads
    expansion = (-1, -1, -1, heads_per_group, -1)
    return [
        (
            torch.zeros(shape, device=device, dtype=dtype).expand(expansion),
            torch.zeros(shape, device=device, dtype=dtype).expand(expansion),
        )
        for _ in range(n_layers)
    ]


def cache_prefix(cache: list[LayerCache], length: int) -> list[LayerCache]:
    if len(cache) > 0:
        assert cache[0][0].shape[1] >= length

    return [(ck[:, :length], cv[:, :length]) for ck, cv in cache]

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\\n\\n"
    "### Instruction:\\n{instruction}\\n\\n### Input:\\n{input}\\n\\n### Response:"
)

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = data
        print(f"Initialized dataset with {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format prompt
        if item.get("input", ""):
            prompt = PROMPT_TEMPLATE.format(instruction=item["instruction"], input=item["input"])
        else:
            prompt = PROMPT_TEMPLATE.format(instruction=item["instruction"], input="") # Handle cases with no input
        
        response = item["output"]
        
        prompt_tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        response_tokens = self.tokenizer.encode(response, bos=False, eos=True)
        
        # Combine and truncate
        full_tokens = prompt_tokens + response_tokens
        if len(full_tokens) > self.max_seq_len:
            full_tokens = full_tokens[:self.max_seq_len]
            
        # Create labels, masking prompt part
        labels = [-100] * len(prompt_tokens) + response_tokens
        if len(labels) > self.max_seq_len:
            labels = labels[:self.max_seq_len]

        # Ensure tokens and labels are same length
        assert len(full_tokens) == len(labels)

        return torch.tensor(full_tokens), torch.tensor(labels)

def collate_fn(batch):
    tokens, labels = zip(*batch)
    tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return tokens, labels

def validate(model, dataloader, criterion, device, args):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    pbar = tqdm(dataloader, desc="Validating", bar_format='{l_bar}{r_bar}')
    with torch.no_grad():
        for input_ids, labels in pbar:
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            # Prepare inputs for the model's forward pass
            batch_size, seq_len = input_ids.shape
            token_lengths = torch.tensor([seq_len] * batch_size, device=device, dtype=torch.long)
            start_pos = torch.zeros(batch_size, device=device, dtype=torch.long)
            
            cache = make_cache(
                args,
                length=seq_len,
                device=device,
                n_layers=args.n_layers,
                dtype=torch.bfloat16,
            )
            kv_padding = seq_len

            # Forward pass
            logits = model(input_ids, token_lengths, start_pos, cache, kv_padding)
            
            # Calculate loss
            loss = criterion(logits.view(-1, args.vocab_size), labels.view(-1))
            total_loss += loss.item()
            
            # Calculate accuracy
            predicted = torch.argmax(logits, dim=-1)
            mask = (labels != -100) # Ignore padding and prompt tokens
            total_correct += (predicted == labels)[mask].sum().item()
            total_tokens += mask.sum().item()

            avg_batch_loss = total_loss / (pbar.n + 1)
            pbar.set_postfix_str(f"Loss={avg_batch_loss:.4f}")

    avg_loss = total_loss / len(dataloader)
    accuracy = (total_correct / total_tokens) * 100 if total_tokens > 0 else 0
    model.train() # Switch back to training mode
    return avg_loss, accuracy

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available, running on CPU. This will be very slow.")

    args = ModelArgs(use_kernel=True)
    model = Transformer(args).to(device, dtype=torch.bfloat16)

    if cli_args.model_path:
        try:
            model.load_state_dict(torch.load(cli_args.model_path, map_location=device, weights_only=True))
            print(f"Model weights loaded from {cli_args.model_path}")
        except FileNotFoundError:
            print(f"Weight file not found at {cli_args.model_path}. Starting from scratch.")
        except Exception as e:
            print(f"Error loading weights: {e}. Starting from scratch.")
    else:
        print("No model path provided, starting from scratch.")

    tokenizer = Tokenizer(cli_args.tokenizer_path)
    
    # Load, shuffle, and split the dataset ONCE.
    print("Loading and splitting dataset...")
    with open(cli_args.dataset_path, 'r') as f:
        full_data = json.load(f)
    
    random.shuffle(full_data)
    
    # Apply dataset_fraction to the whole dataset before splitting
    if cli_args.dataset_fraction < 1.0:
        full_data = full_data[:int(len(full_data) * cli_args.dataset_fraction)]
        print(f"Using {cli_args.dataset_fraction*100:.0f}% of the dataset: {len(full_data)} samples.")

    val_split_index = int(len(full_data) * (1 - cli_args.validation_split_fraction))
    train_data = full_data[:val_split_index]
    val_data = full_data[val_split_index:]

    print(f"Dataset split: {len(train_data)} training samples, {len(val_data)} validation samples.")

    # Create datasets and dataloaders with pre-split data
    train_dataset = InstructionDataset(train_data, tokenizer, cli_args.seq_len)
    val_dataset = InstructionDataset(val_data, tokenizer, cli_args.seq_len)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cli_args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=cli_args.num_workers,
        pin_memory=True if cli_args.num_workers > 0 and device == "cuda" else False
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cli_args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cli_args.num_workers,
        pin_memory=True if cli_args.num_workers > 0 and device == "cuda" else False
    )


    optimizer = torch.optim.AdamW(model.parameters(), lr=cli_args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("\nStarting training...")
    global_step = 0
    latest_val_accuracy = 0.0  # Initialize accuracy
    latest_val_loss = 0.0
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(cli_args.epochs):
        print(f"--- Epoch {epoch+1}/{cli_args.epochs} ---")
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_dataloader, desc="Training", bar_format='{l_bar}{r_bar}')
        for i, (input_ids, labels) in enumerate(pbar):
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            # Prepare inputs for the model's forward pass
            batch_size, seq_len = input_ids.shape
            token_lengths = torch.tensor([seq_len] * batch_size, device=device, dtype=torch.long)
            start_pos = torch.zeros(batch_size, device=device, dtype=torch.long)
            
            cache = make_cache(
                args,
                length=seq_len,
                device=device,
                n_layers=args.n_layers,
                dtype=torch.bfloat16,
            )
            kv_padding = seq_len

            # Forward pass
            # Redirect stdout to /dev/null to suppress kernel messages
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                logits = model(input_ids, token_lengths, start_pos, cache, kv_padding)
            
            loss = criterion(logits.view(-1, args.vocab_size), labels.view(-1))
            loss = loss / cli_args.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()

            if (i + 1) % cli_args.gradient_accumulation_steps == 0:
                # Gradient Clipping
                if cli_args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cli_args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # --- Validation Step ---
                if global_step > 0 and global_step % cli_args.eval_interval == 0:
                    print(f"\n--- Starting validation at step {global_step} ---")
                    val_loss, val_accuracy = validate(model, val_dataloader, criterion, device, args)
                    latest_val_accuracy = val_accuracy # Update accuracy
                    latest_val_loss = val_loss
                    print(f"--- Validation complete ---")
                    print(f"Step {global_step}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

                    # Early stopping check
                    if val_loss < best_val_loss - cli_args.early_stopping_min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Optionally save the best model here
                        if cli_args.save_path:
                            best_model_path = cli_args.save_path.replace('.pt', '_best.pt')
                            torch.save(model.state_dict(), best_model_path)
                            print(f"New best model saved to {best_model_path}")
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= cli_args.early_stopping_patience:
                        print(f"\n--- Early stopping triggered after {patience_counter} evaluations without improvement. ---")
                        break # Exit inner loop

            total_loss += loss.item() * cli_args.gradient_accumulation_steps
            pbar.set_postfix_str(f"Loss={total_loss / (i+1):.4f}, Val Loss={latest_val_loss:.4f}, Val Acc={latest_val_accuracy:.2f}%")
    
    print("\n--- Training finished ---")
    if cli_args.save_path:
        torch.save(model.state_dict(), cli_args.save_path)
        print(f"Finetuned model saved to {cli_args.save_path}")


def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune a BitNet model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model state dictionary.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the JSON dataset file.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer model.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the fine-tuned model.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--seq_len', type=int, default=512, help='Maximum sequence length.')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate for the optimizer.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Number of steps to accumulate gradients before updating weights.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading.')
    parser.add_argument('--dataset_fraction', type=float, default=1.0, help='Fraction of the dataset to use.')
    parser.add_argument('--validation_split_fraction', type=float, default=0.1, help='Fraction of the data to use for validation.')
    parser.add_argument('--eval_interval', type=int, default=200, help='Run validation every N steps.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping. Set to 0 to disable.')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Patience for early stopping. Stops if val loss does not improve for this many evaluations. Set to -1 to disable.')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001, help='Minimum change in val loss to be considered an improvement for early stopping.')
    return parser.parse_args()

if __name__ == '__main__':
    cli_args = get_args()
    main()