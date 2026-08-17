"""
hf_adapter.py — HuggingFace Transformers model adapters for real inference.

Wraps real HuggingFace causal language models (GPT-2, DistilGPT-2, etc.)
behind the ModelAdapter protocol so the entire observability pipeline —
consistency scoring, drift detection, OTel tracing, SQLite storage,
and dashboard — works with actual model outputs.

IMPORTANT ARCHITECTURAL NOTE:
    The Introspective Consistency Score (ICS) compares a "diffusion model"
    against an "autoregressive anchor". True diffusion language models
    (LLaDA, MDLM, Dream) are not available as pre-trained HuggingFace
    checkpoints and require significant compute (multiple A100 GPUs).

    To prove the observability platform works end-to-end with real models,
    this adapter provides two strategies:
    1. HuggingFaceAutoregressiveAdapter — wraps a real AR model (e.g. GPT-2)
       as the causal anchor.
    2. HuggingFacePseudoDiffusionAdapter — wraps a real AR model but generates
       tokens in a simulated parallel/masked fashion, introducing controlled
       inconsistency to exercise the ICS scorer with real logits/embeddings.

    This is explicitly NOT a real diffusion model. It is a real model whose
    outputs are restructured to exercise the diffusion-oriented observability
    pipeline. This limitation is documented throughout.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from introspect.core.models import (
    GenerationResult,
    GenerationStep,
    ModelAdapter,
    ModelConfig,
    TokenState,
)


class HuggingFaceAutoregressiveAdapter:
    """Wraps a real HuggingFace causal LM as an autoregressive anchor.

    Loads a pretrained model (e.g. 'distilgpt2', 'gpt2') and implements
    the ModelAdapter protocol by running actual forward passes to produce
    real logits and embeddings.

    Usage:
        adapter = HuggingFaceAutoregressiveAdapter("distilgpt2")
        result = adapter.generate(prompt_ids=tokenizer.encode("Hello"))
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_new_tokens: int = 64,
        device: str = "cpu",
        seed: int | None = 42,
    ) -> None:
        """Initialize with a HuggingFace model identifier.

        Args:
            model_name: HuggingFace model ID (e.g. 'distilgpt2', 'gpt2').
            max_new_tokens: Maximum tokens to generate beyond the prompt.
            device: Device for inference ('cpu', 'cuda', 'mps').
            seed: Random seed for reproducible generation.
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._device = device

        # Load tokenizer and model.
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        ).to(device)
        self._model.eval()

        # Set seed for reproducibility.
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Build config from actual model parameters.
        model_config = self._model.config
        self._config = ModelConfig(
            vocab_size=model_config.vocab_size,
            embed_dim=model_config.n_embd if hasattr(model_config, 'n_embd') else model_config.hidden_size,
            seq_len=max_new_tokens,
            num_steps=1,  # AR models generate in 1 "step" (all tokens sequentially).
            confidence_mean=1.0,
            confidence_std=0.0,
            inconsistency_rate=0.0,
            seed=seed,
        )

        self._torch = torch

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def generate(
        self, 
        prompt_ids: NDArray[np.int64] | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate tokens autoregressively using the real model.

        Args:
            prompt_ids: Optional prompt token IDs. If None, uses the
                model's BOS token.
            max_new_tokens: Optional override for tokens to generate.

        Returns:
            GenerationResult with real model logits and embeddings.
        """
        torch = self._torch
        max_new_tokens = max_new_tokens or self._max_new_tokens

        # Prepare prompt.
        if prompt_ids is None:
            prompt_text = "The meaning of life is"
            input_ids = self._tokenizer.encode(prompt_text, return_tensors="pt").to(self._device)
        else:
            input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(self._device)

        total_start = time.perf_counter()
        steps: list[GenerationStep] = []
        generated_ids = input_ids.clone()

        with torch.no_grad():
            for step_idx in range(max_new_tokens):
                step_start = time.perf_counter()

                # Forward pass to get logits.
                outputs = self._model(generated_ids)
                next_token_logits = outputs.logits[:, -1, :]  # [1, vocab_size]

                # Greedy selection.
                next_token_id = torch.argmax(next_token_logits, dim=-1)
                generated_ids = torch.cat(
                    [generated_ids, next_token_id.unsqueeze(-1)], dim=-1
                )

                step_elapsed = (time.perf_counter() - step_start) * 1000

                # Build step snapshot.
                seq_len = generated_ids.shape[1]
                logits_np = next_token_logits.cpu().numpy().astype(np.float32)
                # Expand to full sequence for compatibility.
                full_logits = np.zeros((seq_len, self._config.vocab_size), dtype=np.float32)
                full_logits[-1] = logits_np[0]

                # Compute confidence from softmax of last position.
                exp_logits = np.exp(logits_np[0] - logits_np[0].max())
                softmax = exp_logits / exp_logits.sum()
                confidence = np.full(seq_len, softmax.max(), dtype=np.float32)

                steps.append(GenerationStep(
                    step_index=step_idx,
                    total_steps=self._max_new_tokens,
                    token_ids=generated_ids[0].cpu().numpy().astype(np.int64),
                    states=[TokenState.UNMASKED] * seq_len,
                    logits=full_logits,
                    confidence=confidence,
                    elapsed_ms=step_elapsed,
                    memory_bytes=generated_ids.nelement() * 4 + logits_np.nbytes,
                    tokens_unmasked=1,
                ))

                # Stop at EOS.
                if next_token_id.item() == self._tokenizer.eos_token_id:
                    break

        total_elapsed = (time.perf_counter() - total_start) * 1000

        # Final token IDs — trim to max_new_tokens from the end of generated sequence.
        final_ids = generated_ids[0].cpu().numpy().astype(np.int64)

        # Get final logits for the complete sequence.
        with torch.no_grad():
            final_outputs = self._model(generated_ids)
            final_logits = final_outputs.logits[0].cpu().numpy().astype(np.float32)

        # Get embeddings from the model's embedding layer.
        embeddings = self.get_embeddings(final_ids)

        return GenerationResult(
            token_ids=final_ids,
            logits=final_logits,
            embeddings=embeddings,
            steps=steps,
            total_elapsed_ms=total_elapsed,
        )

    def get_logits(self, token_ids: NDArray[np.int64]) -> NDArray[np.float32]:
        """Compute logits for a given token sequence using the real model."""
        torch = self._torch

        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self._device)
        with torch.no_grad():
            outputs = self._model(input_ids)
        return outputs.logits[0].cpu().numpy().astype(np.float32)

    def get_embeddings(self, token_ids: NDArray[np.int64]) -> NDArray[np.float32]:
        """Extract embeddings from the model's embedding layer."""
        torch = self._torch

        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self._device)
        with torch.no_grad():
            # Use the transformer's word embedding layer directly.
            embeddings = self._model.get_input_embeddings()(input_ids)
        return embeddings[0].cpu().numpy().astype(np.float32)


class HuggingFacePseudoDiffusionAdapter:
    """Wraps a real HuggingFace model to simulate diffusion-style generation.

    IMPORTANT LIMITATION:
        This is NOT a real diffusion language model. Real DLMs (LLaDA, MDLM,
        Dream) require specialized architectures and pre-trained weights that
        are not available as lightweight HuggingFace checkpoints.

        This adapter uses a real autoregressive model but restructures its
        generation to mimic the masked-diffusion process:
        1. First generates a complete sequence autoregressively (ground truth).
        2. Then simulates T denoising steps by progressively "revealing"
           tokens, with controllable inconsistency injection.

        The logits and embeddings are REAL (from actual model forward passes).
        The denoising trajectory is SIMULATED.
        This is sufficient to prove the observability platform works with
        real model outputs.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_new_tokens: int = 64,
        num_steps: int = 8,
        inconsistency_rate: float = 0.1,
        device: str = "cpu",
        seed: int | None = 42,
    ) -> None:
        """Initialize the pseudo-diffusion adapter.

        Args:
            model_name: HuggingFace model ID.
            max_new_tokens: Tokens to generate.
            num_steps: Number of simulated denoising steps.
            inconsistency_rate: Fraction of tokens to intentionally corrupt
                (to exercise the ICS scorer).
            device: Inference device.
            seed: Random seed.
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._num_steps = num_steps
        self._inconsistency_rate = inconsistency_rate
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        ).to(device)
        self._model.eval()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self._rng = np.random.default_rng(seed)

        model_config = self._model.config
        self._config = ModelConfig(
            vocab_size=model_config.vocab_size,
            embed_dim=model_config.n_embd if hasattr(model_config, 'n_embd') else model_config.hidden_size,
            seq_len=max_new_tokens,
            num_steps=num_steps,
            confidence_mean=0.7,
            confidence_std=0.15,
            inconsistency_rate=inconsistency_rate,
            seed=seed,
        )

        self._torch = torch

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def generate(self, prompt_ids: NDArray[np.int64] | None = None) -> GenerationResult:
        """Simulate diffusion-style generation using real model outputs.

        Process:
        1. Generate a complete sequence autoregressively (the "ground truth").
        2. Select positions to corrupt (inconsistency injection).
        3. Simulate T denoising steps by progressively unmasking positions.
        4. At each step, compute real logits via model forward pass.

        Returns:
            GenerationResult with real logits/embeddings and simulated steps.
        """
        torch = self._torch

        # Step 1: Generate the full sequence autoregressively.
        if prompt_ids is None:
            prompt_text = "The meaning of life is"
            input_ids = self._tokenizer.encode(prompt_text, return_tensors="pt").to(self._device)
        else:
            input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(self._device)

        total_start = time.perf_counter()

        with torch.no_grad():
            generated = self._model.generate(
                input_ids,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        full_sequence = generated[0].cpu().numpy().astype(np.int64)
        gen_len = len(full_sequence)

        # Step 2: Select inconsistent positions and corrupt them.
        num_gen_tokens = gen_len - input_ids.shape[1]
        gen_start = input_ids.shape[1]
        num_inconsistent = int(num_gen_tokens * self._inconsistency_rate)

        corrupted_sequence = full_sequence.copy()
        if num_inconsistent > 0 and num_gen_tokens > 0:
            inconsistent_positions = self._rng.choice(
                num_gen_tokens, size=min(num_inconsistent, num_gen_tokens), replace=False
            ) + gen_start
            for pos in inconsistent_positions:
                # Replace with a random different token.
                wrong_token = self._rng.integers(0, self._config.vocab_size)
                while wrong_token == full_sequence[pos]:
                    wrong_token = self._rng.integers(0, self._config.vocab_size)
                corrupted_sequence[pos] = wrong_token

        # Step 3: Simulate denoising steps.
        mask = np.ones(gen_len, dtype=bool)
        # Prompt tokens are always unmasked.
        mask[:gen_start] = False

        # Divide generated positions into step groups.
        gen_positions = np.arange(gen_start, gen_len)
        self._rng.shuffle(gen_positions)
        step_groups = np.array_split(gen_positions, self._num_steps)

        current_ids = np.zeros(gen_len, dtype=np.int64)
        current_ids[:gen_start] = full_sequence[:gen_start]

        steps: list[GenerationStep] = []

        for step_idx, positions_to_unmask in enumerate(step_groups):
            step_start = time.perf_counter()

            # Unmask this step's positions.
            for pos in positions_to_unmask:
                current_ids[pos] = corrupted_sequence[pos]
                mask[pos] = False

            # Get real logits from model for the current (partial) sequence.
            ids_for_model = current_ids.copy()
            # Replace still-masked positions with pad token for forward pass.
            pad_id = self._tokenizer.pad_token_id or 0
            ids_for_model[mask] = pad_id

            with torch.no_grad():
                model_input = torch.tensor(ids_for_model, dtype=torch.long).unsqueeze(0).to(self._device)
                outputs = self._model(model_input)
                logits = outputs.logits[0].cpu().numpy().astype(np.float32)

            # Compute confidence.
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            softmax = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            confidence = softmax.max(axis=1).astype(np.float32)

            step_elapsed = (time.perf_counter() - step_start) * 1000

            states = [
                TokenState.MASKED if m else TokenState.UNMASKED
                for m in mask
            ]

            steps.append(GenerationStep(
                step_index=step_idx,
                total_steps=self._num_steps,
                token_ids=current_ids.copy(),
                states=states,
                logits=logits,
                confidence=confidence,
                elapsed_ms=step_elapsed,
                memory_bytes=logits.nbytes + current_ids.nbytes,
                tokens_unmasked=len(positions_to_unmask),
            ))

        total_elapsed = (time.perf_counter() - total_start) * 1000

        # Final forward pass for complete logits.
        with torch.no_grad():
            final_input = torch.tensor(corrupted_sequence, dtype=torch.long).unsqueeze(0).to(self._device)
            final_outputs = self._model(final_input)
            final_logits = final_outputs.logits[0].cpu().numpy().astype(np.float32)

        embeddings = self.get_embeddings(corrupted_sequence)

        return GenerationResult(
            token_ids=corrupted_sequence,
            logits=final_logits,
            embeddings=embeddings,
            steps=steps,
            total_elapsed_ms=total_elapsed,
        )

    def get_logits(self, token_ids: NDArray[np.int64]) -> NDArray[np.float32]:
        """Compute logits for a given token sequence."""
        torch = self._torch

        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self._device)
        with torch.no_grad():
            outputs = self._model(input_ids)
        return outputs.logits[0].cpu().numpy().astype(np.float32)

    def get_embeddings(self, token_ids: NDArray[np.int64]) -> NDArray[np.float32]:
        """Extract embeddings from the model's embedding layer."""
        torch = self._torch

        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self._device)
        with torch.no_grad():
            embeddings = self._model.get_input_embeddings()(input_ids)
        return embeddings[0].cpu().numpy().astype(np.float32)


class BD3Adapter:
    """Wraps a BD3-LM (Block Discrete Denoising Diffusion Language Model).
    
    This is a true discrete diffusion model that works natively on CPU
    by falling back to standard PyTorch attention when flash_attn is missing.
    """
    
    def __init__(
        self,
        model_name: str = "kuleshov-group/bd3lm-owt-block_size4",
        max_new_tokens: int = 32,
        num_steps: int = 16,
        device: str = "cpu",
        seed: int | None = 42,
    ) -> None:
        """Initialize the BD3 adapter.
        
        Args:
            model_name: HuggingFace model ID for BD3.
            max_new_tokens: Tokens to generate beyond prompt.
            num_steps: Number of denoising steps for the global schedule.
            device: Inference device (supports 'cpu' natively).
            seed: Random seed.
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._num_steps = num_steps
        self._device = device
        
        # BD3 uses GPT-2 tokenizer and requires a custom MASK token
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self._tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        
        # Patch for transformers 4.49.0 issue with custom models missing all_tied_weights_keys
        import transformers
        patched_tied = False
        if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
            transformers.PreTrainedModel.all_tied_weights_keys = property(lambda self: {})
            patched_tied = True

        # Patch torch.compile for Python 3.12+ / PyTorch < 2.5 incompatibility.
        # BD3-LM's modeling code uses @torch.compile decorators which require
        # Dynamo. On Python 3.12 + PyTorch < 2.5, Dynamo is unsupported.
        # We replace torch.compile with a no-op identity decorator for loading.
        # This is safe: we run inference-only on CPU where torch.compile
        # provides no benefit anyway.
        original_compile = torch.compile
        compile_patched = False
        try:
            # Test if compile works.
            torch.compile(lambda x: x)
        except RuntimeError:
            torch.compile = lambda fn=None, *args, **kwargs: fn if fn is not None else (lambda f: f)
            compile_patched = True

        # Load BD3 model. The custom code gracefully falls back to SDPA if flash_attn is missing.
        self._model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_backend="sdpa",
        ).to(device)
        self._model.config.attn_backend = "sdpa"
        self._model.eval()
        
        # Restore patches.
        if compile_patched:
            torch.compile = original_compile
        if patched_tied:
            del transformers.PreTrainedModel.all_tied_weights_keys

        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Fix meta tensor issue by regenerating the mask on CPU
        if hasattr(self._model, "backbone") and hasattr(self._model.backbone, "gen_mask"):
            self._model.backbone.gen_mask(
                self._model.config.model_length, 
                self._model.config.block_size, 
                attn_backend="sdpa"
            )

        self._rng = np.random.default_rng(seed)
        self._torch = torch
        
        # Build config
        model_config = self._model.config
        self._config = ModelConfig(
            vocab_size=len(self._tokenizer),
            embed_dim=model_config.hidden_dim if hasattr(model_config, 'hidden_dim') else model_config.hidden_size,
            seq_len=max_new_tokens,
            num_steps=num_steps,
            confidence_mean=0.8,
            confidence_std=0.1,
            inconsistency_rate=0.0,
            seed=seed,
        )

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer
        
    def generate(
        self,
        prompt_ids: NDArray[np.int64] | None = None,
        max_new_tokens: int | None = None,
        num_steps: int | None = None,
    ) -> GenerationResult:
        """Execute the discrete diffusion generation loop.
        
        Process:
        1. Initialize sequence with prompt tokens + [MASK] tokens.
        2. Progressively unmask tokens over T steps based on model predictions.
        
        Returns:
            GenerationResult with real BD3 logits/embeddings and denoising steps.
        """
        torch = self._torch
        max_new_tokens = max_new_tokens or self._max_new_tokens
        num_steps = num_steps or self._num_steps
        
        if prompt_ids is None:
            prompt_text = "The meaning of life is"
            input_ids = self._tokenizer.encode(prompt_text, return_tensors="pt").to(self._device)
        else:
            input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(self._device)
            
        prompt_len = input_ids.shape[1]
        seq_len = prompt_len + max_new_tokens
        
        mask_token_id = self._tokenizer.mask_token_id
        
        # Initialize full sequence: prompt followed by MASK tokens
        current_ids = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=self._device)
        current_ids[0, :prompt_len] = input_ids[0]
        
        # Track which positions are currently masked
        is_masked = torch.ones(seq_len, dtype=torch.bool, device=self._device)
        is_masked[:prompt_len] = False
        
        total_start = time.perf_counter()
        steps: list[GenerationStep] = []
        
        # Determine number of tokens to unmask at each step (linear schedule)
        tokens_to_unmask_per_step = np.full(num_steps, max_new_tokens // num_steps)
        for i in range(max_new_tokens % num_steps):
            tokens_to_unmask_per_step[i] += 1
            
        for step_idx in range(num_steps):
            step_start = time.perf_counter()
            
            # Forward pass
            with torch.no_grad():
                # BD3 expects timesteps in [0, 1] if time_conditioning=True, otherwise ignored.
                t_val = 1.0 - (step_idx / num_steps)
                timesteps = torch.tensor([t_val], dtype=torch.float32, device=self._device)
                
                # NOTE: For standard AutoModelForMaskedLM, we pass input_ids. BD3 takes sigma/timesteps implicitly or explicitly if handled in **kwargs. 
                # Their configuration_bd3lm.py doesn't strictly require timesteps for unconditioned unmasking if time_conditioning=False.
                # However, we pass it just in case.
                outputs = self._model(input_ids=current_ids, timesteps=timesteps, sample_mode=True)
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                
            masked_indices = torch.where(is_masked)[0]
            unmasked_count = 0
            if len(masked_indices) > 0:
                masked_logits = logits[0, masked_indices, :]
                
                # Predict tokens
                pred_tokens = torch.argmax(masked_logits, dim=-1)
                
                # Compute confidence for unmasking schedule
                exp_logits = torch.exp(masked_logits - masked_logits.max(dim=-1, keepdim=True).values)
                probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
                confidences = probs.max(dim=-1).values
                
                # Choose top K most confident tokens to unmask this step
                k = min(len(masked_indices), tokens_to_unmask_per_step[step_idx].item())
                if k > 0:
                    topk_indices = torch.topk(confidences, k).indices
                    indices_to_unmask = masked_indices[topk_indices]
                    tokens_to_unmask = pred_tokens[topk_indices]
                    
                    # Update sequence and mask
                    current_ids[0, indices_to_unmask] = tokens_to_unmask
                    is_masked[indices_to_unmask] = False
                    unmasked_count = k
                    
            step_elapsed = (time.perf_counter() - step_start) * 1000
            
            logits_np = logits[0].cpu().numpy().astype(np.float32)
            ids_np = current_ids[0].cpu().numpy().astype(np.int64)
            mask_np = is_masked.cpu().numpy().astype(bool)
            
            exp_log = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
            softmax = exp_log / exp_log.sum(axis=1, keepdims=True)
            conf = softmax.max(axis=1).astype(np.float32)
            
            states = [TokenState.MASKED if m else TokenState.UNMASKED for m in mask_np]
            
            steps.append(GenerationStep(
                step_index=step_idx,
                total_steps=self._num_steps,
                token_ids=ids_np,
                states=states,
                logits=logits_np,
                confidence=conf,
                elapsed_ms=step_elapsed,
                memory_bytes=logits_np.nbytes + ids_np.nbytes,
                tokens_unmasked=unmasked_count,
            ))
            
        total_elapsed = (time.perf_counter() - total_start) * 1000
        
        # Get embeddings
        embeddings = self.get_embeddings(current_ids[0].cpu().numpy())
        
        return GenerationResult(
            token_ids=current_ids[0].cpu().numpy().astype(np.int64),
            logits=steps[-1].logits, # Last step logits
            embeddings=embeddings,
            steps=steps,
            total_elapsed_ms=total_elapsed,
        )

    def get_logits(self, token_ids: NDArray[np.int64]) -> NDArray[np.float32]:
        """Compute logits for a given token sequence."""
        torch = self._torch
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self._device)
        timesteps = torch.zeros((1,), dtype=torch.float32, device=self._device)
        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, timesteps=timesteps, sample_mode=True)
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        return logits[0].cpu().numpy().astype(np.float32)

    def get_embeddings(self, token_ids: NDArray[np.int64]) -> NDArray[np.float32]:
        """Extract embeddings from the model's vocab embedding layer."""
        torch = self._torch
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self._device)
        with torch.no_grad():
            embeddings = self._model.backbone.vocab_embed(input_ids)
        return embeddings[0].cpu().numpy().astype(np.float32)


