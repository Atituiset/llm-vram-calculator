/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface ModelPreset {
  id: string;
  name: string;
  creator: string;
  totalParams: number; // in Billions (e.g., 8 for 8B)
  activeParams?: number; // for MoE, in Billions (if omitted, same as totalParams)
  numLayers: number;
  numHeads: number; // Query heads
  numKVHeads: number; // KV heads (for GQA/MQA representation)
  hiddenSize: number; // model dimension
  maxContext: number; // Native maximum designer context window (e.g., 8192, 32768, 131072)
  description?: string;
  defaultPrecisionId?: string; // Automatically load default precision format (e.g., 'fp16', 'fp8')
}

export enum PrecisionMode {
  FP16_BF16 = 'fp16',
  FP8 = 'fp8',
  INT8 = 'int8',
  INT4 = 'int4',
  GGUF_Q4 = 'gguf_q4',
  GGUF_Q8 = 'gguf_q8',
  FP4 = 'fp4',
  CUSTOM = 'custom'
}

export interface PrecisionDetails {
  id: PrecisionMode;
  name: string;
  bitsPerWeight: number;
  overheadFactor: number; // multiplier for memory overhead like GGUF metadata or quantization scale factors
  description: string;
}

export interface GPUType {
  id: string;
  name: string;
  vram: number; // in GB
  busWidth?: string; // e.g., "384-bit"
  bandwidth?: string; // e.g., "1008 GB/s"
  type: 'datacenter' | 'consumer' | 'mac' | 'legacy';
}

export type CalcMode = 'inference' | 'training';

export interface InferenceConfig {
  batchSize: number;
  sequenceLength: number;
  kvCachePrecision: 'fp16' | 'fp8' | 'int8' | 'none';
  chunkPrefillSize: 'off' | 512 | 1024 | 2048 | 4096; // Chunked prefill setting to manage peak activation memory
  systemOverheadGB: number; // standard workspace/CUDA context overhead (usually 1-2 GB)
  tensorParallelism: number; // TP degree (number of GPUs)
  memoryFraction: number; // SGLang --mem-fraction-static / vLLM --gpu-memory-utilization
  avgTokensPerRequest: number; // average input + output tokens per request for concurrency estimate
}

export interface TrainingConfig {
  optimizer: 'adamw' | 'adamw_8bit' | 'sgd' | 'none';
  precision: 'fp16_bf16' | 'fp32' | 'pure_bf16';
  trainableParamsPercent: number; // 100 for full tuning, e.g., 1% for LoRA
  loraRank: number; // for displaying LoRA-specific tuning config
  activationCheckpointing: 'full' | 'selective' | 'off';
  batchSize: number; // per-device batch size
  sequenceLength: number;
  systemOverheadGB: number;
  tensorParallelism: number; // TP degree
  pipelineParallelism: number; // PP degree
}

export interface VRAMBreakdown {
  modelWeights: number;       // GB
  kvCache: number;            // GB
  trainingState?: {
    gradients: number;       // GB
    optimizer: number;       // GB
  };
  activationMemory: number;   // GB (Estimated based on context, layers, standard parameters)
  overhead: number;           // GB (CUDA context, PyTorch structures, etc.)
  total: number;              // GB
}

export interface ConcurrencyEstimate {
  kvPoolPerGPU_GB: number;
  perTokenKV_GB: number;
  maxTokensTotal: number;
  maxConcurrentRequests: number;
  limitingFactor: 'weight' | 'kv' | 'fit';
  isFeasible: boolean;
  message?: string;
}
