# Reverse Concurrency Estimator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reverse concurrency estimator that, given a GPU, model, precision, memory fraction, and average tokens per request, computes the maximum number of concurrent requests the system can serve and emits the correct `--max-running-requests` (SGLang) / `--max-num-seqs` (vLLM) flags.

**Architecture:** Extend `InferenceConfig` with two new inputs, refactor `data.ts` to expose a per-token KV helper and a new `estimateConcurrency` function, wire the result through `App.tsx`, display it in a new `ConcurrencyEstimator` component, and feed it into `DeploymentScriptGenerator`.

**Tech Stack:** React 19, TypeScript 5.8, Tailwind CSS 4.1, Vite 6.2

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/types.ts` | Add `memoryFraction`, `avgTokensPerRequest` to `InferenceConfig`; add `ConcurrencyEstimate` interface | Modify |
| `src/data.ts` | Add `calculatePerTokenKV()` helper; add `estimateConcurrency()`; keep existing `calculateInferenceVRAM` unchanged | Modify |
| `src/App.tsx` | Add new state defaults; compute `concurrencyEstimate` via `useMemo`; pass props down | Modify |
| `src/components/InferenceParams.tsx` | Add UI controls for memory fraction and avg tokens per request | Modify |
| `src/components/ConcurrencyEstimator.tsx` | New component displaying KV pool, max tokens, max concurrent requests | Create |
| `src/components/DeploymentScriptGenerator.tsx` | Receive `concurrencyEstimate`; append engine-specific flags | Modify |
| `src/components/MathFormulaConsole.tsx` | Add reverse-concurrency formula explanation | Modify |

---

## Task 1: Extend TypeScript types

**Files:**
- Modify: `src/types.ts`

- [ ] **Step 1: Add fields to `InferenceConfig`**

```ts
export interface InferenceConfig {
  batchSize: number;
  sequenceLength: number;
  kvCachePrecision: 'fp16' | 'fp8' | 'int8' | 'none';
  chunkPrefillSize: 'off' | 512 | 1024 | 2048 | 4096;
  systemOverheadGB: number;
  tensorParallelism: number;
  memoryFraction: number;        // NEW
  avgTokensPerRequest: number;   // NEW
}
```

- [ ] **Step 2: Add `ConcurrencyEstimate` interface**

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

- [ ] **Step 3: Run type check**

```bash
npm run lint
```

Expected: FAIL because `InferenceConfig` usages are missing new fields.

---

## Task 2: Add calculation helpers in `data.ts`

**Files:**
- Modify: `src/data.ts`

- [ ] **Step 1: Extract `calculatePerTokenKV()`**

Insert after `calculateInferenceVRAM`:

```ts
/**
 * Compute KV Cache memory per token per GPU (GB).
 * Mirrors the KV logic in calculateInferenceVRAM but without batch/seq.
 */
export function calculatePerTokenKV(
  model: ModelPreset,
  kvCachePrecision: 'fp16' | 'fp8' | 'int8' | 'none',
  useMLACompression: boolean,
  tensorParallelism: number
): number {
  if (kvCachePrecision === 'none') return 0;

  const tp = Math.max(1, tensorParallelism);
  const kvBytes = kvCachePrecision === 'fp16' ? 2 : 1;
  const isDeepSeekWithMLA = (model.id.includes('deepseek') || model.id.includes('r1')) && useMLACompression !== false;

  let rawKVSizeGB = 0;

  if (isDeepSeekWithMLA) {
    const mlaEffectiveKVHeads = 4.5;
    const headDim = 128;
    const totalKeysValues = mlaEffectiveKVHeads * headDim;
    rawKVSizeGB = (2 * model.numLayers * totalKeysValues * kvBytes) / 1e9;
  } else {
    const d_head = model.numHeads > 0 ? (model.hiddenSize / model.numHeads) : 128;
    const kvHeadsUsed = model.numKVHeads || model.numHeads;
    rawKVSizeGB = (2 * model.numLayers * kvHeadsUsed * d_head * kvBytes) / 1e9;
  }

  return rawKVSizeGB / tp;
}
```

- [ ] **Step 2: Add `estimateConcurrency()`**

Insert after `calculatePerTokenKV`:

```ts
export function estimateConcurrency(
  model: ModelPreset,
  precision: PrecisionDetails,
  inferenceConfig: InferenceConfig,
  selectedGPU: GPUType,
  useMLACompression: boolean
): ConcurrencyEstimate {
  const tp = Math.max(1, inferenceConfig.tensorParallelism);

  const rawWeightsSizeGB = (model.totalParams * (precision.bitsPerWeight / 8)) * precision.overheadFactor;
  const weightsGB = rawWeightsSizeGB / tp;
  const overheadGB = inferenceConfig.systemOverheadGB;
  const usablePoolGB = selectedGPU.vram * inferenceConfig.memoryFraction;

  if (usablePoolGB <= weightsGB + overheadGB) {
    return {
      kvPoolPerGPU_GB: 0,
      perTokenKV_GB: 0,
      maxTokensTotal: 0,
      maxConcurrentRequests: 0,
      limitingFactor: 'weight',
      isFeasible: false,
      message: '显存比例过低：静态权重与系统开销已超过可用池 / Memory fraction too low: weights + overhead exceed usable pool',
    };
  }

  const kvPoolPerGPU_GB = usablePoolGB - weightsGB - overheadGB;
  const perTokenKV_GB = calculatePerTokenKV(
    model,
    inferenceConfig.kvCachePrecision,
    useMLACompression,
    tp
  );

  if (perTokenKV_GB <= 0) {
    return {
      kvPoolPerGPU_GB,
      perTokenKV_GB: 0,
      maxTokensTotal: Number.MAX_SAFE_INTEGER,
      maxConcurrentRequests: Number.MAX_SAFE_INTEGER,
      limitingFactor: 'kv',
      isFeasible: true,
      message: 'KV Cache 已禁用，并发仅受权重限制 / KV cache disabled, concurrency limited by weights only',
    };
  }

  const maxTokensTotal = Math.floor(kvPoolPerGPU_GB / perTokenKV_GB);
  const maxConcurrentRequests = Math.floor(maxTokensTotal / inferenceConfig.avgTokensPerRequest);

  return {
    kvPoolPerGPU_GB: parseFloat(kvPoolPerGPU_GB.toFixed(2)),
    perTokenKV_GB: parseFloat(perTokenKV_GB.toFixed(6)),
    maxTokensTotal,
    maxConcurrentRequests,
    limitingFactor: maxConcurrentRequests > 0 ? 'kv' : 'fit',
    isFeasible: maxConcurrentRequests > 0,
    message: maxConcurrentRequests > 0
      ? undefined
      : '单请求平均 token 数过大，无法容纳任何并发 / Avg tokens per request too large to fit any concurrency',
  };
}
```

- [ ] **Step 3: Import `ConcurrencyEstimate` type at top of `data.ts`**

```ts
import { ModelPreset, PrecisionMode, PrecisionDetails, GPUType, VRAMBreakdown, InferenceConfig, ConcurrencyEstimate } from './types';
```

- [ ] **Step 4: Run type check**

```bash
npm run lint
```

Expected: FAIL because `App.tsx` and `InferenceParams.tsx` still use old `InferenceConfig`.

---

## Task 3: Wire state and computation in `App.tsx`

**Files:**
- Modify: `src/App.tsx`

- [ ] **Step 1: Update `inferenceConfig` initial state**

```ts
const [inferenceConfig, setInferenceConfig] = useState<InferenceConfig>({
  batchSize: 1,
  sequenceLength: 4096,
  kvCachePrecision: 'fp16',
  chunkPrefillSize: 'off',
  systemOverheadGB: 2.0,
  tensorParallelism: 1,
  memoryFraction: 0.85,        // NEW
  avgTokensPerRequest: 8192,   // NEW
});
```

- [ ] **Step 2: Import `estimateConcurrency` and `ConcurrencyEstimate`**

```ts
import { MODEL_PRESETS, PRECISION_OPTS, GPU_PRESETS, calculateInferenceVRAM, calculateTrainingVRAM, estimateConcurrency, COMPONENT_IDS } from './data';
import { ModelPreset, PrecisionDetails, GPUType, CalcMode, InferenceConfig, TrainingConfig, ConcurrencyEstimate } from './types';
```

- [ ] **Step 3: Add `concurrencyEstimate` memo**

Insert after `vramBreakdown` memo:

```ts
const concurrencyEstimate = useMemo(() => {
  if (selectedMode !== 'inference') {
    return null;
  }
  return estimateConcurrency(
    selectedModel,
    selectedPrecision,
    inferenceConfig,
    selectedGPU,
    useMLACompression
  );
}, [
  selectedMode,
  selectedModel,
  selectedPrecision,
  inferenceConfig,
  selectedGPU,
  useMLACompression,
]);
```

- [ ] **Step 4: Pass `concurrencyEstimate` to `DeploymentScriptGenerator`**

```tsx
{selectedMode === 'inference' && (
  <DeploymentScriptGenerator
    selectedModel={selectedModel}
    selectedPrecision={selectedPrecision}
    selectedGPU={selectedGPU}
    inferenceConfig={inferenceConfig}
    vramBreakdown={vramBreakdown}
    concurrencyEstimate={concurrencyEstimate}
  />
)}
```

- [ ] **Step 5: Render `ConcurrencyEstimator` in dashboard column**

Add after `VRAMGauge`:

```tsx
{selectedMode === 'inference' && concurrencyEstimate && (
  <ConcurrencyEstimator
    estimate={concurrencyEstimate}
    selectedGPU={selectedGPU}
    inferenceConfig={inferenceConfig}
  />
)}
```

- [ ] **Step 6: Run type check**

```bash
npm run lint
```

Expected: FAIL because `ConcurrencyEstimator` and updated `DeploymentScriptGenerator` props don't exist yet.

---

## Task 4: Add controls to `InferenceParams.tsx`

**Files:**
- Modify: `src/components/InferenceParams.tsx`

- [ ] **Step 1: Extend props interface**

```ts
interface InferenceParamsProps {
  config: InferenceConfig;
  onConfigChange: React.Dispatch<React.SetStateAction<InferenceConfig>>;
  useMLACompression: boolean;
  onMLACompressionChange: (value: boolean) => void;
  isDeepSeekModel: boolean;
  selectedModelMaxContext: number;
}
```

(Interface likely already exists; verify it matches.)

- [ ] **Step 2: Add helper handlers**

```ts
const handleMemoryFractionChange = (e: React.ChangeEvent<HTMLInputElement>) => {
  const value = parseFloat(e.target.value);
  onConfigChange(prev => ({ ...prev, memoryFraction: Math.max(0.70, Math.min(0.95, value)) }));
};

const handleAvgTokensChange = (e: React.ChangeEvent<HTMLInputElement>) => {
  const value = parseInt(e.target.value, 10);
  onConfigChange(prev => ({ ...prev, avgTokensPerRequest: Math.max(1, value || 1) }));
};
```

- [ ] **Step 3: Add UI section before closing tag**

Insert a new `details/summary` block at the bottom of the component:

```tsx
<div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm flex flex-col gap-4">
  <h3 className="font-bold text-sm text-slate-850 flex items-center gap-2">
    <Zap className="w-4 h-4 text-amber-500" />
    反向并发估算 / Reverse Concurrency Estimate
  </h3>

  <div className="flex flex-col gap-2">
    <label className="text-xs font-semibold text-slate-600 flex justify-between">
      <span>显存借用比例 / Memory Fraction</span>
      <span className="font-mono text-slate-900">{(config.memoryFraction * 100).toFixed(0)}%</span>
    </label>
    <input
      type="range"
      min={0.70}
      max={0.95}
      step={0.01}
      value={config.memoryFraction}
      onChange={handleMemoryFractionChange}
      className="w-full accent-indigo-600"
    />
    <p className="text-[11px] text-slate-500">
      对应 SGLang <code>--mem-fraction-static</code> / vLLM <code>--gpu-memory-utilization</code>
    </p>
  </div>

  <div className="flex flex-col gap-2">
    <label className="text-xs font-semibold text-slate-600">
      单请求平均 Token 数 / Avg Tokens per Request
    </label>
    <input
      type="number"
      min={1}
      step={128}
      value={config.avgTokensPerRequest}
      onChange={handleAvgTokensChange}
      className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
    />
    <p className="text-[11px] text-slate-500">
      输入 + 输出的预期平均长度，用于反推最大并发数
    </p>
  </div>
</div>
```

- [ ] **Step 4: Import `Zap` icon**

```ts
import { Thermometer, Layers, Cpu, Zap } from 'lucide-react';
```

(Adjust based on existing imports.)

- [ ] **Step 5: Run type check**

```bash
npm run lint
```

Expected: FAIL because `ConcurrencyEstimator` still missing.

---

## Task 5: Create `ConcurrencyEstimator.tsx`

**Files:**
- Create: `src/components/ConcurrencyEstimator.tsx`

- [ ] **Step 1: Create file with content**

```tsx
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { ConcurrencyEstimate, GPUType, InferenceConfig } from '../types';
import { AlertTriangle, CheckCircle, Database, Layers, Users } from 'lucide-react';

interface ConcurrencyEstimatorProps {
  estimate: ConcurrencyEstimate;
  selectedGPU: GPUType;
  inferenceConfig: InferenceConfig;
}

export const ConcurrencyEstimator: React.FC<ConcurrencyEstimatorProps> = ({
  estimate,
  selectedGPU,
  inferenceConfig,
}) => {
  const isWarning = !estimate.isFeasible;

  return (
    <div className="bg-slate-900 text-slate-100 rounded-2xl p-6 shadow-xl border border-slate-800 flex flex-col gap-5">
      <div className="flex items-center gap-2.5">
        <div className="w-9 h-9 bg-emerald-500/10 border border-emerald-500/20 rounded-xl flex items-center justify-center text-emerald-400">
          <Users className="w-5 h-5" />
        </div>
        <div>
          <h3 className="text-base font-bold text-slate-100">
            并发上限估算 / Concurrency Limit
          </h3>
          <p className="text-xs text-slate-400">
            基于显存蓄水池模型反推 / Reverse estimate from memory pool
          </p>
        </div>
      </div>

      {isWarning ? (
        <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-3.5 flex gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
          <div className="flex flex-col gap-1">
            <span className="text-xs font-semibold text-amber-300">配置不可行 / Configuration infeasible</span>
            <span className="text-[11px] text-amber-200/80">{estimate.message}</span>
          </div>
        </div>
      ) : (
        <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-4 flex items-center justify-between">
          <div className="flex flex-col">
            <span className="text-[11px] text-emerald-300/80 uppercase tracking-wide font-semibold">最大并发请求数 / Max Concurrent</span>
            <span className="text-3xl font-extrabold text-emerald-400">{estimate.maxConcurrentRequests.toLocaleString()}</span>
          </div>
          <CheckCircle className="w-8 h-8 text-emerald-500/50" />
        </div>
      )}

      <div className="grid grid-cols-2 gap-3">
        <div className="bg-slate-850 border border-slate-800 rounded-xl p-3 flex flex-col gap-1">
          <span className="text-[10px] text-slate-500 uppercase tracking-wide flex items-center gap-1">
            <Database className="w-3 h-3" /> KV Pool / GPU
          </span>
          <span className="text-sm font-bold text-slate-200">{estimate.kvPoolPerGPU_GB.toFixed(1)} GB</span>
        </div>

        <div className="bg-slate-850 border border-slate-800 rounded-xl p-3 flex flex-col gap-1">
          <span className="text-[10px] text-slate-500 uppercase tracking-wide flex items-center gap-1">
            <Layers className="w-3 h-3" /> Per-Token KV
          </span>
          <span className="text-sm font-bold text-slate-200">{(estimate.perTokenKV_GB * 1024).toFixed(2)} KB</span>
        </div>

        <div className="bg-slate-850 border border-slate-800 rounded-xl p-3 flex flex-col gap-1 col-span-2">
          <span className="text-[10px] text-slate-500 uppercase tracking-wide">最大容纳 Token 数 / Max Total Tokens</span>
          <span className="text-sm font-bold text-slate-200">{estimate.maxTokensTotal.toLocaleString()}</span>
        </div>
      </div>

      <div className="text-[11px] text-slate-500 leading-relaxed">
        以 <strong>{selectedGPU.name}</strong> 单卡 {selectedGPU.vram}GB × {(inferenceConfig.memoryFraction * 100).toFixed(0)}% 可用池，
        扣除静态权重与系统开销后，剩余空间可缓存约 {estimate.maxTokensTotal.toLocaleString()} 个 Token。
        按平均每请求 {inferenceConfig.avgTokensPerRequest.toLocaleString()} Token 计算，
        理论最大并发为 <strong className="text-slate-300">{estimate.maxConcurrentRequests.toLocaleString()}</strong>。
      </div>
    </div>
  );
};
```

- [ ] **Step 2: Run type check**

```bash
npm run lint
```

Expected: FAIL because `DeploymentScriptGenerator` props still missing `concurrencyEstimate`.

---

## Task 6: Update `DeploymentScriptGenerator.tsx`

**Files:**
- Modify: `src/components/DeploymentScriptGenerator.tsx`

- [ ] **Step 1: Extend props interface**

```ts
interface DeploymentScriptGeneratorProps {
  selectedModel: ModelPreset;
  selectedPrecision: PrecisionDetails;
  selectedGPU: GPUType;
  inferenceConfig: InferenceConfig;
  vramBreakdown: VRAMBreakdown;
  concurrencyEstimate: ConcurrencyEstimate | null;
}
```

- [ ] **Step 2: Destructure prop**

```ts
export const DeploymentScriptGenerator: React.FC<DeploymentScriptGeneratorProps> = ({
  selectedModel,
  selectedPrecision,
  selectedGPU,
  inferenceConfig,
  vramBreakdown,
  concurrencyEstimate,
}) => {
```

- [ ] **Step 3: Add concurrency flag to SGLang command**

After `--mem-fraction-static` line:

```ts
if (concurrencyEstimate?.isFeasible) {
  args.push(`  --max-running-requests ${concurrencyEstimate.maxConcurrentRequests}`);
}
```

- [ ] **Step 4: Add concurrency flag to vLLM command**

After `--gpu-memory-utilization` line:

```ts
if (concurrencyEstimate?.isFeasible) {
  args.push(`  --max-num-seqs ${concurrencyEstimate.maxConcurrentRequests}`);
}
```

- [ ] **Step 5: Add warning when not feasible**

In the "GPU Matching Info" card, add a conditional block:

```tsx
{concurrencyEstimate && !concurrencyEstimate.isFeasible && (
  <p className="text-[11px] text-amber-400 mt-2">
    {concurrencyEstimate.message}
  </p>
)}
```

- [ ] **Step 6: Run type check**

```bash
npm run lint
```

Expected: PASS.

---

## Task 7: Update `MathFormulaConsole.tsx`

**Files:**
- Modify: `src/components/MathFormulaConsole.tsx`

- [ ] **Step 1: Import `ConcurrencyEstimate` and add prop**

```ts
import { ModelPreset, PrecisionDetails, CalcMode, InferenceConfig, TrainingConfig, VRAMBreakdown, ConcurrencyEstimate } from '../types';
```

```ts
interface MathFormulaConsoleProps {
  selectedModel: ModelPreset;
  selectedPrecision: PrecisionDetails;
  selectedMode: CalcMode;
  inferenceConfig: InferenceConfig;
  trainingConfig: TrainingConfig;
  useMLACompression: boolean;
  isDeepSeekModel: boolean;
  vramBreakdown: VRAMBreakdown;
  concurrencyEstimate: ConcurrencyEstimate | null;
}
```

- [ ] **Step 2: Add reverse concurrency explanation section**

Add a new `details/summary` block for inference mode:

```tsx
{selectedMode === 'inference' && concurrencyEstimate && (
  <details className="group">
    <summary className="flex items-center justify-between cursor-pointer text-sm font-semibold text-slate-700 py-2 border-b border-slate-100">
      <span>反向并发估算 / Reverse Concurrency</span>
      <ChevronDown className="w-4 h-4 text-slate-400 group-open:rotate-180 transition-transform" />
    </summary>
    <div className="pt-3 pb-1 text-xs text-slate-600 space-y-2">
      <p>
        <strong>可用 KV 池:</strong> <code>GPU_VRAM × memoryFraction − weight − overhead</code>
      </p>
      <p>
        <strong>单 Token KV:</strong> <code>2 × layers × kvHeads × d_head × kvBytes / TP / 1e9</code> (GB)
      </p>
      <p>
        <strong>最大 Token 数:</strong> <code>floor(KV_Pool / perTokenKV)</code>
      </p>
      <p>
        <strong>最大并发:</strong> <code>floor(Max_Tokens / avgTokensPerRequest)</code>
      </p>
    </div>
  </details>
)}
```

- [ ] **Step 3: Pass prop from `App.tsx`**

```tsx
<MathFormulaConsole
  selectedModel={selectedModel}
  selectedPrecision={selectedPrecision}
  selectedMode={selectedMode}
  inferenceConfig={inferenceConfig}
  trainingConfig={trainingConfig}
  useMLACompression={useMLACompression}
  isDeepSeekModel={isDeepSeekModel}
  vramBreakdown={vramBreakdown}
  concurrencyEstimate={concurrencyEstimate}
/>
```

- [ ] **Step 4: Run type check**

```bash
npm run lint
```

Expected: PASS.

---

## Task 8: Build and verify

- [ ] **Step 1: Run type check**

```bash
npm run lint
```

Expected: PASS with no errors.

- [ ] **Step 2: Run production build**

```bash
npm run build
```

Expected: PASS, `dist/` folder generated.

- [ ] **Step 3: Manual smoke test**

```bash
npm run dev
```

Open the local URL. Verify:
1. Inference mode shows "反向并发估算" section with slider and input.
2. Adjusting `memoryFraction` changes the Concurrency Limit card.
3. Selecting a DeepSeek model uses MLA-compressed per-token KV.
4. Generated SGLang command contains `--max-running-requests {N}`.
5. Generated vLLM command contains `--max-num-seqs {N}`.
6. Setting `memoryFraction` too low shows infeasible warning and removes engine flags.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: add reverse concurrency estimator with SGLang/vLLM flags

- Add memoryFraction and avgTokensPerRequest to InferenceConfig
- Add calculatePerTokenKV and estimateConcurrency to data.ts
- Add ConcurrencyEstimator dashboard component
- Wire concurrency estimate into DeploymentScriptGenerator
- Show reverse concurrency formulas in MathFormulaConsole

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** Every requirement in `2026-06-26-reverse-concurrency-design.md` maps to a task above.
- [ ] **No placeholders:** All code blocks contain real code; no "TODO" or "implement later".
- [ ] **Type consistency:** `InferenceConfig`, `ConcurrencyEstimate`, and prop names match across all files.
- [ ] **Engine flags:** SGLang uses `--max-running-requests`; vLLM uses `--max-num-seqs`.
- [ ] **Feasibility handling:** Infeasible configs show warnings and do not emit flags.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-26-reverse-concurrency-estimator.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks.
2. **Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`.

Which approach do you prefer?
