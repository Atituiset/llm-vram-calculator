/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Interactive Deployment & Launcher Script Generator
 * Supports SGLang and vLLM production CLI arguments tailored for the selected GPU / Model topologies.
 */

import React, { useState, useMemo } from 'react';
import { ModelPreset, PrecisionDetails, GPUType, InferenceConfig, VRAMBreakdown, ConcurrencyEstimate } from '../types';
import { Terminal, Copy, Check, Info, ShieldAlert, Cpu } from 'lucide-react';

interface DeploymentScriptGeneratorProps {
  selectedModel: ModelPreset;
  selectedPrecision: PrecisionDetails;
  selectedGPU: GPUType;
  inferenceConfig: InferenceConfig;
  vramBreakdown: VRAMBreakdown;
  concurrencyEstimate: ConcurrencyEstimate | null;
}

export const DeploymentScriptGenerator: React.FC<DeploymentScriptGeneratorProps> = ({
  selectedModel,
  selectedPrecision,
  selectedGPU,
  inferenceConfig,
  vramBreakdown,
  concurrencyEstimate,
}) => {
  const [activeEngine, setActiveEngine] = useState<'sglang' | 'vllm'>('sglang');
  const [copied, setCopied] = useState(false);

  // Calculate cards needed
  const cardsNeeded = useMemo(() => {
    return Math.ceil(vramBreakdown.total / selectedGPU.vram);
  }, [vramBreakdown.total, selectedGPU.vram]);

  // Handle setting appropriate default Tensor Parallel size
  // Typically, users run with cardsNeeded, or a factor of 2 (1, 2, 4, 8)
  const recommendedTp = useMemo(() => {
    const tp = cardsNeeded;
    if (tp <= 1) return 1;
    if (tp <= 2) return 2;
    if (tp <= 4) return 4;
    if (tp <= 8) return 8;
    return tp;
  }, [cardsNeeded]);

  // Recommended Memory Utilization / Mem Fraction Dynamic calculation
  // (Weights + systemOverheadGB) / (RecommendedTP * GPU_VRAM) is the proportion of weights.
  // SGLang/vLLM use gpu-memory-utilization (e.g., 0.90) to define the max VRAM allocated for BOTH model weights and KV Cache.
  // If the weights themselves need more, the engine will OOM.
  // We can calculate a smart recommendation.
  const recommendedMemoryUtilization = useMemo(() => {
    const singleGPUVramBytes = selectedGPU.vram;
    const totalVRAMAvailable = recommendedTp * singleGPUVramBytes;
    const weightsAndOverhead = vramBreakdown.modelWeights + vramBreakdown.overhead;
    
    if (totalVRAMAvailable <= 0) return 0.90;
    
    const weightRatio = weightsAndOverhead / totalVRAMAvailable;
    // We want to leave a small buffer for activation memory
    const targetUtil = Math.max(0.85, Math.min(0.95, weightRatio + 0.10));
    return parseFloat(targetUtil.toFixed(2));
  }, [recommendedTp, selectedGPU.vram, vramBreakdown.modelWeights, vramBreakdown.overhead]);

  // Model Hub Identifier Resolution
  const modelHubPath = useMemo(() => {
    if (selectedModel.id === 'deepseek-v4-flash') {
      return 'deepseek-ai/DeepSeek-V3-Flash'; // Or representation
    }
    if (selectedModel.id === 'deepseek-r1-v3') {
      return 'deepseek-ai/DeepSeek-R1';
    }
    if (selectedModel.id.includes('llama-3.1-405b')) {
      return 'meta-llama/Llama-3.1-405B-Instruct';
    }
    if (selectedModel.id.includes('llama-3-8b')) {
      return 'meta-llama/Meta-Llama-3-8B-Instruct';
    }
    if (selectedModel.id.includes('llama-3-70b')) {
      return 'meta-llama/Meta-Llama-3-70B-Instruct';
    }
    if (selectedModel.id.includes('qwen-2.5-72b')) {
      return 'Qwen/Qwen2.5-72B-Instruct';
    }
    if (selectedModel.id.includes('gemma-2-27b')) {
      return 'google/gemma-2-27b-it';
    }
    
    // Default fallback
    return `${selectedModel.creator.toLowerCase()}/${selectedModel.id}-instruct`;
  }, [selectedModel.id, selectedModel.creator]);

  // Generate SGLang Command
  const sglangCommand = useMemo(() => {
    const args: string[] = [
      'python -m sglang.launch_server',
      `  --model-path ${modelHubPath}`,
    ];

    // Quantization
    if (selectedPrecision.id === 'int4') {
      args.push('  --quantization gptq'); // Standard GPTQ parser for INT4
    } else if (selectedPrecision.id === 'fp8') {
      args.push('  --quantization fp8');
    } else if (selectedPrecision.id === 'int8') {
      args.push('  --quantization bitsandbytes');
    }

    // Tensor Parallelism
    if (recommendedTp > 1) {
      args.push(`  --tensor-parallel-size ${recommendedTp}`);
    }

    // Max model length (Sequence length)
    args.push(`  --max-model-len ${inferenceConfig.sequenceLength}`);

    // Context / batch configurations
    args.push(`  --max-num-seqs ${inferenceConfig.batchSize * 2}`);

    // SGLang Memory Limit Config is --mem-fraction-static
    // For large MoE like DeepSeek, we need to balance this
    args.push(`  --mem-fraction-static ${recommendedMemoryUtilization}`);

    // Concurrency limit derived from KV cache pool
    if (concurrencyEstimate?.isFeasible) {
      args.push(`  --max-running-requests ${concurrencyEstimate.maxConcurrentRequests}`);
    }

    // Trust remote code for custom kernels (especially deepseek MLA/MoE implementations)
    if (selectedModel.id.includes('deepseek') || selectedModel.id.includes('r1')) {
      args.push('  --trust-remote-code');
      // MLA Optimized Kernel
      args.push('  --enable-flashinfer-mla');
    }

    // Port definition
    args.push('  --port 3000');
    // Host definition
    args.push('  --host 0.0.0.0');

    return args.join(' \\\n');
  }, [modelHubPath, selectedPrecision.id, recommendedTp, inferenceConfig.sequenceLength, inferenceConfig.batchSize, recommendedMemoryUtilization, selectedModel.id]);

  // Generate vLLM Command
  const vllmCommand = useMemo(() => {
    const args: string[] = [
      'python -m vllm.entrypoints.openai.api_server',
      `  --model ${modelHubPath}`,
    ];

    // Quantization
    if (selectedPrecision.id === 'int4') {
      args.push('  --quantization gptq');
    } else if (selectedPrecision.id === 'fp8') {
      args.push('  --quantization fp8');
    } else if (selectedPrecision.id === 'int8') {
      args.push('  --quantization bitsandbytes');
    }

    // Tensor Parallelism
    if (recommendedTp > 1) {
      args.push(`  --tensor-parallel-size ${recommendedTp}`);
    }

    // Max model length
    args.push(`  --max-model-len ${inferenceConfig.sequenceLength}`);

    // KV Cache Precision optimization
    if (inferenceConfig.kvCachePrecision === 'fp8') {
      args.push('  --kv-cache-dtype fp8');
    } else if (inferenceConfig.kvCachePrecision === 'int8') {
      args.push('  --kv-cache-dtype int8');
    }

    // vLLM Memory ratio
    args.push(`  --gpu-memory-utilization ${recommendedMemoryUtilization}`);

    // Concurrency limit derived from KV cache pool
    if (concurrencyEstimate?.isFeasible) {
      args.push(`  --max-num-seqs ${concurrencyEstimate.maxConcurrentRequests}`);
    }

    if (selectedModel.id.includes('deepseek') || selectedModel.id.includes('r1')) {
      args.push('  --trust-remote-code');
    }

    // Port definition
    args.push('  --port 3000');
    // Host definition
    args.push('  --host 0.0.0.0');

    return args.join(' \\\n');
  }, [modelHubPath, selectedPrecision.id, recommendedTp, inferenceConfig.sequenceLength, inferenceConfig.kvCachePrecision, recommendedMemoryUtilization, selectedModel.id]);

  const activeCommand = activeEngine === 'sglang' ? sglangCommand : vllmCommand;

  const handleCopy = () => {
    navigator.clipboard.writeText(activeCommand);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div id="deployment-script-generator" className="bg-slate-900 text-slate-100 rounded-2xl p-6 shadow-xl border border-slate-800 flex flex-col gap-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 bg-indigo-500/10 border border-indigo-500/20 rounded-xl flex items-center justify-center text-indigo-400">
            <Terminal className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-100 flex items-center gap-1.5 leading-snug">
              部署命令生成器 / Deployment Script Generator
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              根据当前估算的架构规模，自动生成高并发服务端部署指令。
            </p>
          </div>
        </div>

        {/* Engine Tabs */}
        <div className="bg-slate-800 p-0.5 rounded-lg flex self-start sm:self-center border border-slate-700/50">
          <button
            type="button"
            onClick={() => setActiveEngine('sglang')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
              activeEngine === 'sglang'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-450 hover:text-slate-200'
            }`}
          >
            SGLang (极速 MLA / MoE)
          </button>
          <button
            type="button"
            onClick={() => setActiveEngine('vllm')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
              activeEngine === 'vllm'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-450 hover:text-slate-200'
            }`}
          >
            vLLM Standard
          </button>
        </div>
      </div>

      {/* Code Display Area */}
      <div className="relative rounded-xl bg-black/40 border border-slate-800 p-4 font-mono text-[11px] text-slate-300 leading-relaxed overflow-x-auto group">
        <button
          type="button"
          onClick={handleCopy}
          className="absolute top-3 right-3 p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-100 transition-colors border border-slate-700"
          title="复制到剪贴板"
        >
          {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
        </button>
        <pre className="whitespace-pre">{activeCommand}</pre>
        
        {copied && (
          <span className="absolute bottom-3 right-3 text-[10px] bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded">
            已复制！/ Copied!
          </span>
        )}
      </div>

      {/* Configuration Specifications Alerts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-3.5 flex gap-2.5 items-start">
          <Cpu className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
          <div className="flex flex-col gap-1">
            <span className="font-bold text-slate-200">
              硬件适配建议 / GPU Matching Info
            </span>
            <p className="text-[11px] text-slate-400 leading-normal">
              使用 <strong>{selectedGPU.name}</strong> ({selectedGPU.vram}GB)。
              模型加载加上系统与 KV 缓冲共需 <strong>{vramBreakdown.total.toFixed(1)} GB</strong> 显存。
              {cardsNeeded > 1 ? (
                <>
                  推荐建立包含 <strong>{recommendedTp} 块显卡</strong> 的张量并行 (TP) 节点。
                </>
              ) : (
                <>
                  可以通过 <strong>1 块显卡</strong> 正常独立启动部署。
                </>
              )}
            </p>
            {concurrencyEstimate && !concurrencyEstimate.isFeasible && (
              <p className="text-[11px] text-amber-400 mt-2">
                {concurrencyEstimate.message}
              </p>
            )}
          </div>
        </div>

        {/* Dynamic Engine Optimizations advice */}
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-3.5 flex gap-2.5 items-start">
          <Info className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
          <div className="flex flex-col gap-1">
            <span className="font-bold text-slate-200">
              引擎关键参数释义 / Parameter Insights
            </span>
            <p className="text-[11px] text-slate-400 leading-normal">
              {activeEngine === 'sglang' ? (
                <>
                  <code>--mem-fraction-static {recommendedMemoryUtilization}</code> 会在加载静态模型与框架开销后，将剩余 GPU 空间的约 {(recommendedMemoryUtilization * 100).toFixed(0)}% 全量借调为高速 KV Caches，最大程度保障并发吞吐不溢出越界。
                </>
              ) : (
                <>
                  <code>--gpu-memory-utilization {recommendedMemoryUtilization}</code> 约束 vLLM 的静态显存预分配上限。为防止激活权重及 PyTorch 空间挤占崩溃，该值会在 GPU 卡组之间合理按比例留足富余安全边界。
                </>
              )}
            </p>
          </div>
        </div>
      </div>

      {/* DeepSeek and L40S Target Guidance if matches */}
      {((selectedModel.id.includes('v4-flash') || selectedModel.id.includes('r1')) && selectedGPU.id === 'l40s') && (
        <div className="bg-indigo-950/40 border border-indigo-900/50 rounded-xl p-4 flex gap-3 text-xs">
          <ShieldAlert className="w-5 h-5 text-indigo-400 shrink-0 mt-0.5" />
          <div className="flex flex-col gap-1">
            <span className="font-bold text-indigo-300">
              ⚡ 针对 NVIDIA L40S 的 DeepSeek MoE 优化部署专家路线 / Expert Tip
            </span>
            <p className="text-[11px] text-indigo-200 leading-relaxed">
              根据 DeepSeek 量化部署分析报告，在 <strong>NVIDIA L40S</strong> 上：
              <br />
              1. <strong>INT4 多卡并联</strong>: 4x L40S (共192GB VRAM) 部署 284B 的 INT4 极其合适，静态权重占用 149.1~157 GiB，留下充足裕量给 Flash Attention / MLA KV 缓存。
              <br />
              2. <strong>SGLang 提效参数</strong>: 在 H100 之外，L40S 性价比卓越！由于 L40S 不支持 NVLink 物理集群互联，TP4 张量切分会产生一定的 PCIe 跨卡带宽压力。使用 SGLang 自研的 <code>--enable-flashinfer-mla</code> (潜空间多头注意力算子) 可直接从底层算法角度节省高达 80%+ 的 KV VRAM，成倍弥补卡间带宽短板，将端到端延迟推向极致。
              <br />
              3. <strong>FP8 KV-Cache</strong>: 考虑搭配 <code>--kv-cache-dtype fp8</code> 启动，可获得极致吞吐表现。
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
