---
title: Execution Platforms
kind: module
status: draft
owners:
  - "@gcanlin"
  - "@tjtanaa"
  - "@xuechendi"
primary_code_paths:
  - vllm_omni/platforms/**
  - vllm_omni/attention/**
primary_code_path_owners:
  # Additional owners scoped to attention kernels (quantized/custom attention)
  - paths:
      - vllm_omni/attention/**
    owners:
      - "@david6666666"
      - "@lishunyang12"
related_code_paths: []
depends_on:
  - ar_runtime.md
  - diffusion/index.md
validation_paths:
  - tests/platforms/**
upstream_refs:
  - vllm.platforms
last_reviewed: 2026-07-16
---

# Execution platforms

Execution platforms isolate hardware-specific capability detection, worker
selection, patches, kernels, and configuration adjustments.

## Candidate invariants

### PLATFORM-INV-001: Capabilities are explicit

**Rule:** Hardware-dependent behavior MUST be guarded by platform selection or
capability detection and MUST fail clearly when unsupported.

### PLATFORM-INV-002: Portable code remains portable

**Rule:** Platform-neutral modules MUST NOT import a vendor implementation
directly when a common selection interface exists.

### PLATFORM-INV-003: Overrides are minimal

**Rule:** A platform implementation SHOULD override only behavior that differs
from the common or upstream implementation.

## Safe-change guide

Validate common import paths without the target accelerator and run focused
tests in a fresh `uv` environment on every affected device platform.

## External operators

Register an operator subclass with `Operator.register_oot(Implementation)` before
model construction. Override `forward_oot`; `forward_native` remains the reference
and the default fallback. Preserve the operator's constructor, parameters, and
output contract. Registration is keyed by class identity, rejects duplicates,
and applies only to external platforms; built-in dispatch is unchanged.

Qwen3 predictors expose `CodePredictorRMSNorm` and
`CodePredictorRotaryEmbedding` from
`vllm_omni.model_executor.models.common.qwen3_code_predictor`.
The rotary operator's `lookup_cos_sin(x, position_ids)` returns cached tables
in `x.dtype`, shaped `[batch, seq_len, head_dim]`.
