# Reverse Concurrency Estimator Design

**Date:** 2026-06-26  
**Project:** LLM GPU VRAM Calculator  
**Author:** Claude Code  
**Status:** Approved by user

## 1. Problem Statement

The existing VRAM calculator answers: *"Given a model, precision, batch size, and sequence length, how much VRAM do I need?"*

Production inference engines (SGLang, vLLM) work differently. They allocate a static memory fraction (`--mem-fraction-static` / `--gpu-memory-utilization`), load the model weights, and then use **all remaining space as a KV Cache pool**. The practical question operators ask is:

> *"With my GPU, this model, this precision, and this memory fraction, how many concurrent requests can I serve given an average tokens-per-request budget?"*

This design adds a **reverse concurrency estimator** that computes the answer and feeds it directly into generated deployment commands.

## 2. Goals

- Let users configure `memoryFraction` as a first-class input.
- Let users configure `avgTokensPerRequest` as a first-class input.
- Compute per-GPU KV Cache pool after subtracting static weights and system overhead.
- Compute maximum total tokens that fit in that pool using the existing per-token KV Cache formula.
- Compute `maxConcurrentRequests = floor(maxTokens / avgTokensPerRequest)`.
- Display the result in the UI.
- Emit the correct engine flag in generated commands:
  - SGLang: `--max-running-requests {N}`
  - vLLM: `--max-num-seqs {N}`

## 3. Non-Goals

- Does not change the existing forward VRAM calculation (`calculateInferenceVRAM`).
- Does not add a server-side runtime or query a live engine.
- Does not attempt to model attention logit memory, CUDA fragmentation, or speculative decoding.
- Does not add unit tests (project has no test framework yet).

## 4. Architecture

```
┌─────────────────────────────────────────┐
│  InferenceParams.tsx                    │
│  - memoryFraction slider                │
│  - avgTokensPerRequest input            │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  App.tsx                                │
│  - holds new state fields               │
│  - computes concurrencyEstimate useMemo │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  ConcurrencyEstimator.tsx               │
│  - displays KV pool / max tokens / max  │
│    concurrent requests                  │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  DeploymentScriptGenerator.tsx          │
│  - appends --max-running-requests (SGL) │
│  - appends --max-num-seqs (vLLM)        │
└─────────────────────────────────────────┘
```

## 5. Type Changes

### `src/types.ts`

Add to `InferenceConfig`:

```ts
export interface InferenceConfig {
  // ...existing fields...
  memoryFraction: number;        // 0.70 - 0.95, default 0.85
  avgTokensPerRequest: number;   // > 0, default 8192
}
```

Add new interface:

```ts
export interface ConcurrencyEstimate {
  kvPoolPerGPU_GB: number;
  perTokenKV_GB: number;
  maxTokensTotal: number;
  maxConcurrentRequests: number;
  limitingFactor: 'weight' | 'kv' | 'fit';
  isFeasible: boolean;
  message?: string;
}
```

## 6. Calculation Engine Changes

### `src/data.ts`

#### 6.1 Extract per-token KV helper

Refactor the existing inference KV calculation into a reusable helper:

```ts
export function calculatePerTokenKV(
  model: ModelPreset,
  kvCachePrecision: 'fp16' | 'fp8' | 'int8' | 'none',
  useMLACompression: boolean,
  tensorParallelism: number
): number {
  // Returns GB per token per GPU after TP split.
}
```

This avoids duplicating the MLA vs GQA branching logic.

#### 6.2 Concurrency estimator

```ts
export function estimateConcurrency(
  model: ModelPreset,
  precision: PrecisionDetails,
  inferenceConfig: InferenceConfig,
  selectedGPU: GPUType,
  envGPUCount: number
): ConcurrencyEstimate {
  const tp = Math.max(1, inferenceConfig.tensorParallelism);
  const weightsGB = (model.totalParams * (precision.bitsPerWeight / 8) * precision.overheadFactor) / tp;
  const overheadGB = inferenceConfig.systemOverheadGB;
  const usablePoolGB = selectedGPU.vram * inferenceConfig.memoryFraction;

  if (usablePoolGB <= weightsGB + overheadGB) {
    return { ..., isFeasible: false, limitingFactor: 'weight', message: '...' };
  }

  const kvPoolPerGPU_GB = usablePoolGB - weightsGB - overheadGB;
  const perTokenKV_GB = calculatePerTokenKV(model, inferenceConfig.kvCachePrecision, useMLACompression, tp);
  const maxTokensTotal = Math.floor(kvPoolPerGPU_GB / perTokenKV_GB);
  const maxConcurrentRequests = Math.floor(maxTokensTotal / inferenceConfig.avgTokensPerRequest);

  return {
    kvPoolPerGPU_GB,
    perTokenKV_GB,
    maxTokensTotal,
    maxConcurrentRequests,
    limitingFactor: maxConcurrentRequests > 0 ? 'kv' : 'fit',
    isFeasible: maxConcurrentRequests > 0,
  };
}
```

### Notes

- We use `selectedGPU.vram` (single GPU), not total cluster VRAM, because the KV pool is per-GPU after TP.
- Weights are divided by `tp` to match the existing `calculateInferenceVRAM` behavior.
- `perTokenKV_GB` is also divided by `tp` because KV heads are split across TP ranks under standard GQA/MLA setups.

## 7. UI Changes

### 7.1 `InferenceParams.tsx`

Add a new collapsible section labeled:

> **反向并发估算 / Reverse Concurrency Estimate**

Controls:
- `memoryFraction`: slider from 0.70 to 0.95, step 0.01, default 0.85.
- `avgTokensPerRequest`: number input, min 1, default 8192.

### 7.2 New `ConcurrencyEstimator.tsx`

Display card showing:
- KV Pool per GPU (GB)
- Per-token KV (KB)
- Max total tokens
- **Max concurrent requests** (highlighted)
- Limiting factor / feasibility badge

Style: match existing `VRAMGauge` dark card aesthetic (`bg-slate-900`, `text-slate-100`).

### 7.3 Placement

Place `ConcurrencyEstimator` in the right-hand dashboard column, below `VRAMGauge` and above `MathFormulaConsole`.

## 8. Deployment Command Changes

### `DeploymentScriptGenerator.tsx`

Accept new prop:

```ts
concurrencyEstimate: ConcurrencyEstimate;
```

In generated commands:

**SGLang:**
```bash
--max-running-requests ${concurrencyEstimate.maxConcurrentRequests}
```

**vLLM:**
```bash
--max-num-seqs ${concurrencyEstimate.maxConcurrentRequests}
```

Only emit when `isFeasible` is true. If not feasible, show a warning instead of the flag.

## 9. State Changes in `App.tsx`

Add to `inferenceConfig` initial state:

```ts
const [inferenceConfig, setInferenceConfig] = useState<InferenceConfig>({
  // ...existing fields...
  memoryFraction: 0.85,
  avgTokensPerRequest: 8192,
});
```

Add new `useMemo`:

```ts
const concurrencyEstimate = useMemo(() =>
  estimateConcurrency(
    selectedModel,
    selectedPrecision,
    inferenceConfig,
    selectedGPU,
    envGPUCount
  ),
  [selectedModel, selectedPrecision, inferenceConfig, selectedGPU, envGPUCount]
);
```

Pass `concurrencyEstimate` to `ConcurrencyEstimator` and `DeploymentScriptGenerator`.

## 10. Edge Cases

| Case | Behavior |
|------|----------|
| `memoryFraction` too low to fit weights + overhead | Mark `isFeasible: false`, show warning, do not emit engine flag |
| `avgTokensPerRequest` <= 0 | Clamp to 1 or show validation error |
| `maxConcurrentRequests` = 0 | Show "至少需 1 个 token/请求" warning |
| DeepSeek MLA enabled | Use compressed per-token KV path |
| `kvCachePrecision === 'none'` | Per-token KV = 0, estimator returns infinite/very large; cap display or warn |

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| GQA/MQA with odd TP splits under-estimate/over-estimate KV | Document as approximation; keep existing divide-by-tp behavior |
| Users confuse `avgTokensPerRequest` with `sequenceLength` | Add tooltip: "输入 + 输出的预期平均 token 数" |
| vLLM older versions use different flag names | Use `--max-num-seqs` which is widely supported; document vLLM version caveat |

## 12. Success Criteria

- [ ] User can adjust `memoryFraction` and `avgTokensPerRequest`.
- [ ] UI displays KV Pool, max tokens, and max concurrent requests in real time.
- [ ] Generated SGLang command contains `--max-running-requests {N}`.
- [ ] Generated vLLM command contains `--max-num-seqs {N}`.
- [ ] MathFormulaConsole explains the reverse-concurrency formula.
- [ ] `npm run lint` passes.
- [ ] `npm run build` passes.

## 13. Out of Scope

- Unit tests (no test framework in repo).
- Runtime verification against actual SGLang/vLLM instance.
- Modeling pipeline parallelism or sequence parallelism.
