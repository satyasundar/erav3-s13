# SmolLMv2 Based Text Generator

## Input

The model takes a steamimg input from https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus

    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", streaming=True, split="train"
    )

## Model Print output

```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(49152, 576)
    (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=192, bias=False)
          (v_proj): Linear(in_features=576, out_features=192, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((576,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((576,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((576,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)
```

## Model Config

```
Config: LlamaConfig {
  "_name_or_path": "HuggingFaceTB/SmolLM2-135M",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 576,
  "initializer_range": 0.041666666666666664,
  "intermediate_size": 1536,
  "is_llama_config": true,
  "max_position_embeddings": 2048,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 9,
  "num_hidden_layers": 30,
  "num_key_value_heads": 3,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_interleaved": false,
  "rope_scaling": null,
  "rope_theta": 100000,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.48.0",
  "use_cache": true,
  "vocab_size": 49152
}

```

## Total Parameters Calculations

### Embedding Layer (embed_tokens)

    Size: 49152 (vocab) × 576 (dim)
    Parameters: 49152 × 576 = 28,311,552

### LlamaDecoderLayers (30 layers)

1.  **Attention Subtotal = [884,736]**

        Component Calculation Parameters
        q_proj (576 → 576) 576 × 576 = 331,776
        k_proj (576 → 192) 576 × 192 = 110,592
        v_proj (576 → 192) 576 × 192 = 110,592
        o_proj (576 → 576) 576 × 576 = 331,776

2.  **MLP Subtotal = [2,654,208]**

        gate_proj (576 → 1536) 576 × 1536 = 884,736
        up_proj (576 → 1536) 576 × 1536 = 884,736
        down_proj (1536 → 576) 1536 × 576 = 884,736

3.  RMSNorm: **(×2) 576 (weights) × 2 = 1,152**
4.  Total per Layer: **884,736 + 2,654,208 + 1,152 = 3,540,096**
5.  For 30 Layers:
    **3,540,096 × 30 = 106,202,880**

6.  Final LayerNorm (norm)
    Parameters: 576 (RMSNorm weights) = 576

7.  LM Head (lm_head)
    Size: 576 → 49152 (shared with embed_tokens)
    Parameters: 0 (weight-tied with embed_tokens)

**Total Parameters**

- Component Parameters
  - Embedding 28,311,552
  - 30 Decoder Layers 106,202,880
  - Final LayerNorm 576

**Total 134,515,008 (~135M)**

## Traing Logs

- **First Run till 5000 steps :** [base_run_step_5000.log](base_run_step_5000.log)
- **Next Run :** [next_run_from_step_500.log](next_run_from_step_500.log)
