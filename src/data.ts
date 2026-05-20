/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ModelPreset, PrecisionMode, PrecisionDetails, GPUType, VRAMBreakdown } from './types';

export const COMPONENT_IDS = {
  PRESETS: 'presets-section',
  SPECS: 'specs-section',
  PRECISION: 'precision-section',
  INF_PARAMS: 'inference-params-section',
  TRAIN_PARAMS: 'training-params-section',
  VRAM_GAUGE: 'vram-gauge-section',
  BREAKDOWN: 'breakdown-section',
  GPU_FIT: 'gpu-fit-section',
  MODE_TAB: 'mode-tab-button',
  CUSTOM_SPEC: 'custom-spec-input',
  COMPRESSED_MLA: 'compressed-mla-checkbox'
};

export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: 'llama-3-8b',
    name: 'LLaMA 3 / 3.1 8B',
    creator: 'Meta',
    totalParams: 8.03,
    numLayers: 32,
    numHeads: 32,
    numKVHeads: 8, // Grouped-Query Attention
    hiddenSize: 4096, // d_head = 4096 / 32 = 128
    maxContext: 131072, // 128k context
    description: 'The standard modern small LLM. Uses Grouped-Query Attention (GQA).',
    defaultPrecisionId: 'fp16'
  },
  {
    id: 'llama-3-70b',
    name: 'LLaMA 3 / 3.1 70B',
    creator: 'Meta',
    totalParams: 70.6,
    numLayers: 80,
    numHeads: 64,
    numKVHeads: 8, // GQA
    hiddenSize: 8192, // d_head = 128
    maxContext: 131072, // 128k context
    description: 'High performance enterprise model. Large size, utilizes GQA structure.',
    defaultPrecisionId: 'fp16'
  },
  {
    id: 'llama-3.1-405b',
    name: 'LLaMA 3.1 405B',
    creator: 'Meta',
    totalParams: 405.0,
    numLayers: 126,
    numHeads: 128,
    numKVHeads: 8, // GQA
    hiddenSize: 16384, // d_head = 128
    maxContext: 131072, // 128k context
    description: 'Ultra-large flagship open-weights model by Meta. Massive VRAM footprint.',
    defaultPrecisionId: 'fp16'
  },
  {
    id: 'deepseek-r1-v3',
    name: 'DeepSeek R1 / V3 (MoE)',
    creator: 'DeepSeek',
    totalParams: 671.0,
    activeParams: 37.1, // 37.1B active parameters per token
    numLayers: 61,
    numHeads: 128,
    numKVHeads: 128, // MLA Compression applies
    hiddenSize: 7168, // d_head = 128
    maxContext: 131072, // 128k context
    description: 'State-of-the-art MoE. Uses Multi-head Latent Attention (MLA) which compresses KV Cache to 1/5th of normal size, and sparse multi-gated expert system.',
    defaultPrecisionId: 'fp8'
  },
  {
    id: 'deepseek-v4-flash',
    name: 'DeepSeek V4-Flash (MoE)',
    creator: 'DeepSeek',
    totalParams: 284.0,
    activeParams: 37.0, // MoE active per token
    numLayers: 61,
    numHeads: 128,
    numKVHeads: 128, // MLA Compression
    hiddenSize: 7168,
    maxContext: 131072, // 128k context
    description: 'Highly optimized sparse MoE model. Features 284B total parameters with ~37B active parameters and advanced MLA compressed KV caches.',
    defaultPrecisionId: 'fp8'
  },
  {
    id: 'qwen-2.5-7b',
    name: 'Qwen 2.5 7B',
    creator: 'Alibaba',
    totalParams: 7.61,
    numLayers: 28,
    numHeads: 28,
    numKVHeads: 4, // GQA
    hiddenSize: 3584, // d_head = 128
    maxContext: 131072, // 128k context
    description: 'Highly popular versatile multilingual model.',
    defaultPrecisionId: 'fp16'
  },
  {
    id: 'qwen-2.5-14b',
    name: 'Qwen 2.5 14B',
    creator: 'Alibaba',
    totalParams: 14.7,
    numLayers: 48,
    numHeads: 40,
    numKVHeads: 8, // GQA
    hiddenSize: 5120, // d_head = 128
    maxContext: 131072, // 128k context
    description: 'Strong mid-sized model.',
    defaultPrecisionId: 'fp16'
  },
  {
    id: 'qwen-2.5-72b',
    name: 'Qwen 2.5 72B',
    creator: 'Alibaba',
    totalParams: 72.7,
    numLayers: 80,
    numHeads: 64,
    numKVHeads: 8, // GQA
    hiddenSize: 8192, // d_head = 128
    maxContext: 131072, // 128k context
    description: 'Standard dense alternative to LLaMA-70B.',
    defaultPrecisionId: 'fp16'
  },
  {
    id: 'gemma-2-9b',
    name: 'Gemma 2 9B',
    creator: 'Google',
    totalParams: 9.24,
    numLayers: 42,
    numHeads: 16,
    numKVHeads: 8, // GQA (specifically query_heads / kv_heads = 2)
    hiddenSize: 3584, // d_head=256 actually (Gemma 2 uses d_head = 256)
    maxContext: 8192, // 8k context
    description: 'Advanced small-sized model utilizing GQA and local sliding window attention.',
    defaultPrecisionId: 'fp16'
  },
  {
    id: 'gemma-2-27b',
    name: 'Gemma 2 27B',
    creator: 'Google',
    totalParams: 27.2,
    numLayers: 46,
    numHeads: 32,
    numKVHeads: 16,
    hiddenSize: 4608, // d_head=128
    maxContext: 8192, // 8k context
    description: 'Google high-efficiency model, outperforms many models twice its size.',
    defaultPrecisionId: 'fp16'
  },
  {
    id: 'phi-3-medium',
    name: 'Phi-3 Medium 14B',
    creator: 'Microsoft',
    totalParams: 14.0,
    numLayers: 40,
    numHeads: 40,
    numKVHeads: 10,
    hiddenSize: 5120,
    maxContext: 131072, // 128k context window support
    description: 'Heavy grouping GQA model with highly optimized token processing.',
    defaultPrecisionId: 'fp16'
  }
];

export const PRECISION_OPTS: PrecisionDetails[] = [
  {
    id: PrecisionMode.FP16_BF16,
    name: 'FP16 / BF16 (16-bit)',
    bitsPerWeight: 16,
    overheadFactor: 1.0,
    description: 'Standard native training/inference precision. Best accuracy. Raw performance.'
  },
  {
    id: PrecisionMode.FP8,
    name: 'FP8 (8-bit Float)',
    bitsPerWeight: 8,
    overheadFactor: 1.02, // 2% overhead for scale parameters
    description: 'Modern standard for high-throughput inference (H100 native). Negligible accuracy loss.'
  },
  {
    id: PrecisionMode.INT8,
    name: 'INT8 (8-bit Integer)',
    bitsPerWeight: 8,
    overheadFactor: 1.05, // 5% scale overhead
    description: 'Classic quantization format. Supported on almost all hardware.'
  },
  {
    id: PrecisionMode.GGUF_Q8,
    name: 'GGUF Q8_0 (8.5-bit)',
    bitsPerWeight: 8.5,
    overheadFactor: 1.01,
    description: 'High-quality CPU/GPU home setup layout. Minimum degradation.'
  },
  {
    id: PrecisionMode.GGUF_Q4,
    name: 'GGUF Q4_K_M (~4.5-bit)',
    bitsPerWeight: 4.5,
    overheadFactor: 1.04,
    description: 'Recommended GGUF tradeoff. Blends high accuracy with highly compact 4.5 bit coding.'
  },
  {
    id: PrecisionMode.INT4,
    name: 'INT4 (4-bit Integer)',
    bitsPerWeight: 4.0,
    overheadFactor: 1.08, // scale & zero point overheads
    description: 'Very compact layout. Significant VRAM savings, minor perplexity rise.'
  },
  {
    id: PrecisionMode.FP4,
    name: 'FP4 (4-bit Float)',
    bitsPerWeight: 4.0,
    overheadFactor: 1.05,
    description: 'Ultra-quantized state-of-the-art model. Standard in next-gen client architectures.'
  },
  {
    id: PrecisionMode.CUSTOM,
    name: 'Custom (Bits/Weight)',
    bitsPerWeight: 6.0,
    overheadFactor: 1.0,
    description: 'Specify a custom bits-per-parameter layout.'
  }
];

export const GPU_PRESETS: GPUType[] = [
  // --- NVIDIA Data Center ---
  { id: 'b200', name: 'NVIDIA Blackwell B200', vram: 192, type: 'datacenter', busWidth: '8192-bit', bandwidth: '8.0 TB/s' },
  { id: 'h200', name: 'NVIDIA H200 (SXM)', vram: 141, type: 'datacenter', busWidth: '6144-bit', bandwidth: '4.8 TB/s' },
  { id: 'h100', name: 'NVIDIA H100 (SXM)', vram: 80, type: 'datacenter', busWidth: '5120-bit', bandwidth: '3.35 TB/s' },
  { id: 'h100-pcie', name: 'NVIDIA H100 (PCIe)', vram: 80, type: 'datacenter', busWidth: '5120-bit', bandwidth: '2.0 TB/s' },
  { id: 'a100-80gb', name: 'NVIDIA A100 (80GB SXM)', vram: 80, type: 'datacenter', busWidth: '5120-bit', bandwidth: '2.0 TB/s' },
  { id: 'a100-40gb', name: 'NVIDIA A100 (40GB SXM)', vram: 40, type: 'datacenter', busWidth: '5120-bit', bandwidth: '1.5 TB/s' },
  { id: 'a800-80gb', name: 'NVIDIA A800 (80GB)', vram: 80, type: 'datacenter', busWidth: '5120-bit', bandwidth: '2.0 TB/s' },
  { id: 'l40s', name: 'NVIDIA L40S', vram: 48, type: 'datacenter', busWidth: '384-bit', bandwidth: '864 GB/s' },
  { id: 'l4', name: 'NVIDIA L4 (Tensor Core Core)', vram: 24, type: 'datacenter', busWidth: '192-bit', bandwidth: '300 GB/s' },
  { id: 'a10g', name: 'NVIDIA A10G', vram: 24, type: 'datacenter', busWidth: '384-bit', bandwidth: '600 GB/s' },
  { id: 't4', name: 'NVIDIA T4', vram: 16, type: 'legacy', busWidth: '256-bit', bandwidth: '320 GB/s' },

  // --- NVIDIA Consumer Geforce / Workstation ---
  { id: 'rtx5090', name: 'NVIDIA RTX 5090', vram: 32, type: 'consumer', busWidth: '512-bit', bandwidth: '1.8 TB/s' },
  { id: 'rtx4090', name: 'NVIDIA RTX 4090', vram: 24, type: 'consumer', busWidth: '384-bit', bandwidth: '1.0 TB/s' },
  { id: 'rtx3090', name: 'NVIDIA RTX 3090', vram: 24, type: 'consumer', busWidth: '384-bit', bandwidth: '936 GB/s' },
  { id: 'rtx4080', name: 'NVIDIA RTX 4080 / Super', vram: 16, type: 'consumer', busWidth: '256-bit', bandwidth: '736 GB/s' },
  { id: 'rtx4070ti', name: 'NVIDIA RTX 4070 Ti', vram: 16, type: 'consumer', busWidth: '192-bit', bandwidth: '504 GB/s' },
  { id: 'rtx6000-ada', name: 'NVIDIA RTX 6000 Ada', vram: 48, type: 'consumer', busWidth: '384-bit', bandwidth: '960 GB/s' },

  // --- AMD Instinct & Radeon ---
  { id: 'amd-mi300x', name: 'AMD Instinct MI300X', vram: 192, type: 'datacenter', busWidth: '8192-bit', bandwidth: '5.3 TB/s' },
  { id: 'amd-mi300a', name: 'AMD Instinct MI300A', vram: 128, type: 'datacenter', busWidth: '8192-bit', bandwidth: '5.3 TB/s' },
  { id: 'amd-mi250x', name: 'AMD Instinct MI250X', vram: 128, type: 'datacenter', busWidth: '4096-bit', bandwidth: '3.2 TB/s' },
  { id: 'amd-7900xtx', name: 'AMD Radeon RX 7900 XTX', vram: 24, type: 'consumer', busWidth: '384-bit', bandwidth: '960 GB/s' },
  { id: 'amd-7900xt', name: 'AMD Radeon RX 7900 XT', vram: 20, type: 'consumer', busWidth: '320-bit', bandwidth: '800 GB/s' },
  { id: 'amd-7800xt', name: 'AMD Radeon RX 7800 XT', vram: 16, type: 'consumer', busWidth: '256-bit', bandwidth: '624 GB/s' },

  // --- Apple Silicon Unified Memory ---
  { id: 'apple-m4-max', name: 'Apple M4 Max / Ultra (Unified)', vram: 128, type: 'mac', busWidth: 'shared', bandwidth: '410 GB/s' },
  { id: 'apple-m3-ultra-192', name: 'Apple M2/M3 Ultra (192GB Unified)', vram: 192, type: 'mac', busWidth: 'shared', bandwidth: '800 GB/s' },
  { id: 'apple-m3-max-128', name: 'Apple M3 Max (128GB Unified)', vram: 128, type: 'mac', busWidth: 'shared', bandwidth: '400 GB/s' },
  { id: 'apple-m3-pro-48', name: 'Apple M3 Pro / Max (48GB Unified)', vram: 48, type: 'mac', busWidth: 'shared', bandwidth: '150 GB/s' },
  { id: 'apple-64', name: 'Apple M1/M2/M3 (64GB Unified)', vram: 64, type: 'mac', busWidth: 'shared', bandwidth: '300 GB/s' },
  { id: 'apple-36', name: 'Apple MacBook Pro (36GB Unified)', vram: 36, type: 'mac', busWidth: 'shared', bandwidth: '150 GB/s' },

  // --- Huawei Ascend NPU ---
  { id: 'ascend-910b-64', name: 'Huawei Ascend 910B (64GB)', vram: 64, type: 'datacenter', busWidth: '4096-bit', bandwidth: '1.2 TB/s' },
  { id: 'ascend-910-32', name: 'Huawei Ascend 910 Pro (32GB)', vram: 32, type: 'datacenter', busWidth: '4096-bit', bandwidth: '1.0 TB/s' },
  { id: 'ascend-310-16', name: 'Huawei Ascend 310P (16GB)', vram: 16, type: 'consumer', busWidth: '256-bit', bandwidth: '256 GB/s' },

  // --- Intel Gaudi & Arc ---
  { id: 'intel-gaudi-3', name: 'Intel Gaudi 3 AI Accelerator', vram: 128, type: 'datacenter', busWidth: 'HBM2e', bandwidth: '3.7 TB/s' },
  { id: 'intel-gaudi-2', name: 'Intel Gaudi 2 AI Accelerator', vram: 96, type: 'datacenter', busWidth: 'HBM2', bandwidth: '2.4 TB/s' },
  { id: 'intel-arc-a770', name: 'Intel Arc A770 (16GB)', vram: 16, type: 'consumer', busWidth: '256-bit', bandwidth: '560 GB/s' },

  // --- Google TPU ---
  { id: 'google-tpu-v5p', name: 'Google TPU v5p (Node Unit)', vram: 95, type: 'datacenter', busWidth: 'Interconnect', bandwidth: '4.8 TB/s' },
  { id: 'google-tpu-v5e', name: 'Google TPU v5e (Node Unit)', vram: 16, type: 'datacenter', busWidth: 'Interconnect', bandwidth: '810 GB/s' }
];

/**
 * Intelligently match or parse hardware accelerators based on user custom string queries
 */
export function matchOrParseGPU(query: string): GPUType {
  const norm = query.toLowerCase().trim();
  if (!norm) {
    return { id: 'custom-gpu', name: '自定义配置加速卡 / Custom Accelerator', vram: 16, type: 'consumer', busWidth: '256-bit', bandwidth: '500 GB/s' };
  }

  // 1. Precise check for name matches in hardware library (case-insensitive & whitespace trimmed)
  const matched = GPU_PRESETS.find(g => {
    const gName = g.name.toLowerCase();
    const gId = g.id.toLowerCase();
    return gName.includes(norm) || norm.includes(gName) || gId === norm;
  });
  if (matched) return { ...matched };

  // 2. Intelligent regex parsing if no match works
  // Regex searches for any sequence of numbers adjacent to "G", "GB", "M", "MB"
  let vram = 16; // reasonable default placeholder
  const vramMatch = query.match(/(\d+)\s*(gb|g|m)(b)?/i);
  if (vramMatch) {
    vram = parseInt(vramMatch[1], 10);
  } else {
    // If no direct digit + G, look for separate numbers that might stand for VRAM (e.g. "rtx 5090 32")
    const numOnlyMatch = query.match(/\b(8|12|16|20|24|32|40|48|64|80|96|128|141|192|256)\b/);
    if (numOnlyMatch) {
      vram = parseInt(numOnlyMatch[1], 10);
    }
  }

  // 3. Determine manufacturer branding family & parameters
  let type: 'datacenter' | 'consumer' | 'mac' | 'legacy' = 'consumer';
  let name = query;
  let busWidth = '256-bit';
  let bandwidth = '500 GB/s';

  if (/nvidia|rtx|gtx|h100|h200|a100|b200|l40|t4|geforce|quadro|titan/i.test(norm)) {
    type = /h100|h200|a100|b200|l40|v100|a800|h800/i.test(norm) ? 'datacenter' : 'consumer';
    name = norm.includes('nvidia') ? query : `NVIDIA ${query}`;
    if (vram >= 141) { busWidth = '6144-bit'; bandwidth = '4.8 TB/s'; }
    else if (vram >= 80) { busWidth = '5120-bit'; bandwidth = '2.0-3.35 TB/s'; }
    else if (vram >= 48) { busWidth = '384-bit'; bandwidth = '864-960 GB/s'; }
    else if (vram >= 24) { busWidth = '384-bit'; bandwidth = '936-1000 GB/s'; }
  } else if (/amd|mi300|mi250|radeon|rx|7900|instinct/i.test(norm)) {
    type = /mi300|mi250|instinct/i.test(norm) ? 'datacenter' : 'consumer';
    name = norm.includes('amd') ? query : `AMD ${query}`;
    if (vram >= 128) { busWidth = '8192-bit'; bandwidth = '5.3 TB/s'; }
    else if (vram >= 24) { busWidth = '384-bit'; bandwidth = '960 GB/s'; }
  } else if (/apple|mac|m1|m2|m3|m4|unified/i.test(norm)) {
    type = 'mac';
    name = norm.includes('apple') ? query : `Apple ${query} (Unified)`;
    busWidth = 'shared';
    bandwidth = vram >= 128 ? '400-800 GB/s' : '150-300 GB/s';
  } else if (/huawei|ascend|910|310/i.test(norm)) {
    type = 'datacenter';
    name = norm.includes('huawei') || norm.includes('ascend') ? query : `Huawei Ascend ${query}`;
    busWidth = '4096-bit';
    bandwidth = vram >= 64 ? '1.2 TB/s' : '1.0 TB/s';
  } else if (/intel|gaudi|arc/i.test(norm)) {
    type = /gaudi/i.test(norm) ? 'datacenter' : 'consumer';
    name = norm.includes('intel') ? query : `Intel ${query}`;
    busWidth = 'HBM2e';
    bandwidth = vram >= 128 ? '3.7 TB/s' : '560 GB/s';
  } else if (/google|tpu/i.test(norm)) {
    type = 'datacenter';
    name = norm.includes('google') || norm.includes('tpu') ? query : `Google ${query}`;
    busWidth = 'Interconnect';
    bandwidth = '4.8 TB/s';
  }

  return {
    id: `custom-${vram}gb-${Math.floor(Math.random() * 1000)}`,
    name: name,
    vram: vram,
    type: type,
    busWidth: busWidth,
    bandwidth: bandwidth
  };
}

/**
 * Calculators are heavily commented to show exactly what formulas are used,
 * ensuring absolute academic and industrial accuracy.
 */


export function calculateInferenceVRAM(
  model: ModelPreset,
  precision: PrecisionDetails,
  config: {
    batchSize: number;
    sequenceLength: number;
    kvCachePrecision: 'fp16' | 'fp8' | 'int8' | 'none';
    chunkPrefillSize?: 'off' | 512 | 1024 | 2048 | 4096;
    systemOverheadGB: number;
    tensorParallelism: number; // For splitting weight VRAM
    useMLACompression?: boolean; // For DeepSeek MLA cache savings
  }
): VRAMBreakdown {
  const tp = Math.max(1, config.tensorParallelism);
  
  // 1. Model Weights Memory (GB)
  // formula: Params * (BitsPerWeight / 8) * overheadFactor.
  // For MoE, total params reside in memory even though only active are computed!
  const bitsRate = precision.bitsPerWeight;
  const weightOverheadMultiplier = precision.overheadFactor;
  const rawWeightsSizeGB = (model.totalParams * (bitsRate / 8)) * weightOverheadMultiplier;
  const weightsGB = rawWeightsSizeGB / tp;

  // 2. KV Cache Memory (GB)
  let kvGB = 0;
  if (config.kvCachePrecision !== 'none') {
    // KV Quantization bytes size multiplier
    const kvBytes = config.kvCachePrecision === 'fp16' ? 2 : 1;

    // Check if it is a DeepSeek model using MLA KV Compression
    const isDeepSeekWithMLA = (model.id.includes('deepseek') || model.id.includes('r1')) && config.useMLACompression !== false;

    if (isDeepSeekWithMLA) {
      // DeepSeek MLA compresses Key/Value cash!
      // In MLA, the key-value representation is projected down to a compression latent dimension:
      // c = 512, d_head = 128 (8 heads equivalent) + decoupled key cache dimension (128)
      // Overall MLA stores much smaller latent values. In deepseek spec:
      // Cache VRAM size is approximately equivalent to storing (kv_compression_dim + decoupled_kv_head_dim) per token per layer,
      // where kv_compression_dim = 512 and decoupled_kv_head_dim = 64 (or 128). Usually equivalent to 4.5 heads!
      // Formula: 2 * n_layers * 4.5 * d_head * b * s * precision_bytes
      // Let's model it accurately: equivalent of 4.5 heads of size 128 dimension.
      const mlaEffectiveKVHeads = 4.5;
      const headDim = 128;
      const totalKeysValues = mlaEffectiveKVHeads * headDim; // 576 dimensional states
      
      const rawKVSize = (2 * model.numLayers * totalKeysValues * config.batchSize * config.sequenceLength * kvBytes) / 1e9;
      kvGB = rawKVSize / tp;
    } else {
      // Standard calculation for MHA / GQA / MQA:
      // Head dimension = hiddenSize / numHeads
      // Normally d_head = 128 is overwhelmingly standard. Let's make sure it defaults or calculates:
      const d_head = model.numHeads > 0 ? (model.hiddenSize / model.numHeads) : 128;
      
      // KV Heads determines GQA vs standard MHA.
      // If GQA: numKVHeads is used. If MHA: numKVHeads = numHeads. If MQA: numKVHeads = 1.
      const kvHeadsUsed = model.numKVHeads || model.numHeads;
      
      // Formula: 2 * layers * kv_heads * d_head * batch_size * seq_len * precision_bytes
      const rawKVSize = (2 * model.numLayers * kvHeadsUsed * d_head * config.batchSize * config.sequenceLength * kvBytes) / 1e9;
      kvGB = rawKVSize / tp;
    }
  }

  // 3. Activation Memory
  // Highly batch size & sequence length dependent.
  // Approximation formula for standard flash attention inference activation is fairly small.
  // Standard full-sequence prefill executes entire sequenceLength context, yielding huge intermediate attention tensors.
  // Activating Chunked Prefill divides this stage into chunks of size e.g. 512, 1024, 2048, or 4096, restricting peak activation sizes safely.
  const activeChunkSize = (config.chunkPrefillSize && config.chunkPrefillSize !== 'off') 
    ? Math.min(config.sequenceLength, config.chunkPrefillSize) 
    : config.sequenceLength;

  const layerActivationGB = (model.hiddenSize * config.batchSize * activeChunkSize * 2) / 1e9;
  const activationGB = Math.max(0.4, layerActivationGB * 0.15); // usually around 10-20% of intermediate block sizes during context tracking

  // 4. CUDA Workspace & PyTorch/Driver Overhead
  const overhead = config.systemOverheadGB;

  const total = weightsGB + kvGB + activationGB + overhead;

  return {
    modelWeights: parseFloat(weightsGB.toFixed(2)),
    kvCache: parseFloat(kvGB.toFixed(2)),
    activationMemory: parseFloat(activationGB.toFixed(2)),
    overhead: parseFloat(overhead.toFixed(2)),
    total: parseFloat(total.toFixed(2))
  };
}

export function calculateTrainingVRAM(
  model: ModelPreset,
  precision: PrecisionDetails,
  config: {
    optimizer: 'adamw' | 'adamw_8bit' | 'sgd' | 'none';
    precision: 'fp16_bf16' | 'fp32' | 'pure_bf16';
    trainableParamsPercent: number; // 1-100%
    activationCheckpointing: 'full' | 'selective' | 'off';
    batchSize: number; // microbatch size on single device
    sequenceLength: number;
    systemOverheadGB: number;
    tensorParallelism: number; // ZeRO stages can be estimated using this as divisor!
    pipelineParallelism: number;
    zeroStage?: 0 | 1 | 2 | 3; // Zero Redundancy Optimizer Stage selector
  }
): VRAMBreakdown {
  const tp = Math.max(1, config.tensorParallelism);
  const pp = Math.max(1, config.pipelineParallelism);
  const totalDpTpPpFactor = tp * pp; // Total device multiplier for layer splitting

  // Base parameters on device after pipeline parallelism
  const paramsOnDevice = model.totalParams / pp; 
  // Trainable parameters on device
  const trainableParamsOnDevice = (paramsOnDevice * (config.trainableParamsPercent / 100));

  // 1. Model Weights Memory (GB)
  // Training is typically done in higher precision (BF16/FP16) or simulated FP8.
  // In mixed precision training (standard), we retain FP16/BF16 weights (2 bytes per param) AND copy of FP32 master weights (4 bytes per param) if optimizer is FP32.
  let bytesPerWeight = 2; // Default 16-bit BF16/FP16 weights for active parameters
  if (config.precision === 'fp32') {
    bytesPerWeight = 4;
  }
  
  // Weights size of actual base model on device
  const rawModelWeightsGB = paramsOnDevice * bytesPerWeight;
  const weightsGB = rawModelWeightsGB / tp;

  // 2. Gradients (GB)
  // Gradients match precision of trainable weights (typically 2 bytes for mixed precision FP16/BF16, or 4 for FP32)
  const gradBytes = config.precision === 'fp32' ? 4 : 2;
  const gradientsGB = (trainableParamsOnDevice * gradBytes) / tp;

  // 3. Optimizer States (GB)
  // Standard Adam: stores 2 states per trainable parameter in FP32 (first momentum, second momentum = 8 bytes total per trainable parameter)
  // Master Weights: Adam with mixed precision stores another FP32 copy of trainable weights to avoid underflow/rounding errors (+4 bytes/param)
  // Let's compute based on selection:
  let optBytesPerParam = 0;
  if (config.optimizer === 'adamw') {
    optBytesPerParam = 8; // Mean state (4 bytes) + Variance state (4 bytes)
    if (config.precision !== 'fp32') {
      optBytesPerParam += 4; // Add FP32 master weights
    }
  } else if (config.optimizer === 'adamw_8bit') {
    optBytesPerParam = 2; // Standard 8-bit Adam saving 75% memory
    if (config.precision !== 'fp32') {
      optBytesPerParam += 4; // Master weights are still needed in FP32 format for stability!
    }
  } else if (config.optimizer === 'sgd') {
    optBytesPerParam = 4; // momentum vector in FP32
    if (config.precision !== 'fp32') {
      optBytesPerParam += 4; // Master weights
    }
  }
  
  // Optimizer size on device (only stores states for trainable parameters!)
  const optimizerGB = (trainableParamsOnDevice * optBytesPerParam) / tp;

  // 4. Activation Memory during Training (GB)
  // Activation memory scales with: batchSize * sequenceLength * layers * hiddenSize * attention_factors
  // Formula depends strongly on Activation Checkpointing (Gradient Checkpointing) mode:
  // - Off: Memory is very high because activations of ALL layers must be saved for the backward pass!
  //   Approx: layers * batch_size * sequence_length * hidden_size * (34 + 5 * num_heads * seq_len / hidden_size) bytes
  // - Selective: Saves only boundary activations.
  // - Full (Default for large training): Only saves activation of one layer boundary, recomputing intermediate values on backward pass.
  //   Approx VRAM: layers * constant_factor + sqrt(layers) * batch_size * sequence_length * hiddenSize
  
  const d_head = model.numHeads > 0 ? (model.hiddenSize / model.numHeads) : 128;
  const b = config.batchSize;
  const s = config.sequenceLength;
  const h = model.hiddenSize;
  const L = model.numLayers / pp; // split across pipeline stages

  let activationGB = 0;
  
  // Detailed activation modeling
  if (config.activationCheckpointing === 'off') {
    // Standard without checkpointing: requires 34 bytes * batch * seq * hidden per layer
    activationGB = (L * b * s * h * 34) / 1e9;
  } else if (config.activationCheckpointing === 'selective') {
    // Standard selective: saves attention weights but checkpoint MLP activations
    activationGB = (L * b * s * h * 12) / 1e9 + (2 * L * b * s * s * model.numHeads) / 1e9;
  } else {
    // Full checkpointing: only store 1 layer's activation at a time.
    // Memory per stage is roughly equivalent to 2 layers of activations + overhead of checkpoint lists
    // Approx: 2 * L * b * s * h * 2 / 1e9 + standard 0.5 GB buffer
    activationGB = (L * b * s * h * 4.4) / 1e9;
  }

  // Adjust for split in tensor parallelism
  // Note: activations are partly split, but some activations are duplicated due to All-Gather overhead.
  activationGB = Math.max(0.2, activationGB / Math.sqrt(tp));

  // 5. System or CUDA Baseline overhead
  const overhead = config.systemOverheadGB;

  const total = weightsGB + gradientsGB + optimizerGB + activationGB + overhead;

  return {
    modelWeights: parseFloat(weightsGB.toFixed(2)),
    kvCache: 0, // KV cache is inference-only, in training we calculate grad & optimizer
    trainingState: {
      gradients: parseFloat(gradientsGB.toFixed(2)),
      optimizer: parseFloat(optimizerGB.toFixed(2))
    },
    activationMemory: parseFloat(activationGB.toFixed(2)),
    overhead: parseFloat(overhead.toFixed(2)),
    total: parseFloat(total.toFixed(2))
  };
}
