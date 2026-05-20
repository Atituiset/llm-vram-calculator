/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { InferenceConfig } from '../types';
import { COMPONENT_IDS } from '../data';
import { HelpCircle, Layers, Group, Sliders } from 'lucide-react';

interface InferenceParamsProps {
  config: InferenceConfig;
  onConfigChange: (config: InferenceConfig) => void;
  useMLACompression: boolean;
  onMLACompressionChange: (useMLA: boolean) => void;
  isDeepSeekModel: boolean;
}

export const InferenceParams: React.FC<InferenceParamsProps> = ({
  config,
  onConfigChange,
  useMLACompression,
  onMLACompressionChange,
  isDeepSeekModel,
}) => {
  const handleChange = (key: keyof InferenceConfig, value: number | string) => {
    onConfigChange({
      ...config,
      [key]: value,
    });
  };

  return (
    <div id={COMPONENT_IDS.INF_PARAMS} className="bg-white rounded-2xl border border-slate-100 p-6 shadow-sm flex flex-col gap-6">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <Sliders className="w-5 h-5 text-emerald-600" />
          <h2 className="text-lg font-semibold text-slate-900 tracking-tight">
            3. 推理阶段参数配置 / Inference Settings
          </h2>
        </div>
        <p className="text-sm text-slate-500">
          设定模型在部署及运行时(Inference)的批处理大小、上下文窗口以及 KV Cache 保留精度。
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Batch Size Slider */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center text-sm">
            <label className="font-semibold text-slate-700 flex items-center gap-1.5">
              并发用户数 / Batch Size (b)
              <HelpCircle className="w-3.5 h-3.5 text-slate-400 hover:text-emerald-600 cursor-help" title="同时进行模型推理的并发请求总数。由于KV-Cache需要针对每个请求存储独立的向量，因此显寸会随此项数值成正比例线性膨胀。" />
            </label>
            <span className="font-mono bg-slate-100 px-2 py-0.5 rounded text-sm text-slate-700 font-bold">
              {config.batchSize}
            </span>
          </div>
          <input
            id="inference-batch-slider"
            type="range"
            min="1"
            max="128"
            step="1"
            value={config.batchSize}
            onChange={(e) => handleChange('batchSize', parseInt(e.target.value))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-500"
          />
          <div className="flex justify-between text-[11px] text-slate-400 font-mono">
            <span>1 (Single)</span>
            <span>32 (Mid Load)</span>
            <span>64 (Heavy)</span>
            <span>128 (Enterprise)</span>
          </div>
        </div>

        {/* Sequence Length Slider */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center text-sm">
            <label className="font-semibold text-slate-700 flex items-center gap-1.5">
              最大上下文大小 / Context Length (s)
              <HelpCircle className="w-3.5 h-3.5 text-slate-400 hover:text-emerald-600 cursor-help" title="单次生成流程包含的历史Prompt和新增Token的上限。长上下文(如32k、128k)会带来极其沉重的KV-Cache存储开销。" />
            </label>
            <span className="font-mono bg-slate-100 px-2 py-0.5 rounded text-sm text-slate-700 font-bold">
              {(config.sequenceLength / 1024).toFixed(0)}k ({config.sequenceLength} tkn)
            </span>
          </div>
          <input
            id="inference-seq-length-slider"
            type="range"
            min="1024"
            max="131072"
            step="1024"
            value={config.sequenceLength}
            onChange={(e) => handleChange('sequenceLength', parseInt(e.target.value))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-500"
          />
          <div className="flex justify-between text-[11px] text-slate-400 font-mono">
            <span>1k (Default)</span>
            <span>32k (Long Context)</span>
            <span>64k (Extremely Long)</span>
            <span>128k (Max Window)</span>
          </div>
        </div>

        {/* KV Cache Precision Selection */}
        <div className="flex flex-col gap-2">
          <label className="font-semibold text-slate-700 text-sm flex items-center gap-1.5">
            KV Cache 精度 / KV Cache Precision
            <HelpCircle className="w-3.5 h-3.5 text-slate-400 hover:text-emerald-600 cursor-help" title="是否使用更简短的量化位宽来存储每一层KV向量。使用8位(FP8/INT8)通常能在极微精度损耗下节省高达50%的KV缓存显存！" />
          </label>
          <div className="grid grid-cols-4 gap-2">
            {(['fp16', 'fp8', 'int8', 'none'] as const).map((mode) => (
              <button
                key={mode}
                id={`kv-mode-${mode}`}
                type="button"
                onClick={() => handleChange('kvCachePrecision', mode)}
                className={`py-2 px-1 text-center font-mono text-xs font-semibold rounded-lg border transition-all ${
                  config.kvCachePrecision === mode
                    ? 'border-emerald-500 bg-emerald-50 text-emerald-800'
                    : 'border-slate-200 hover:border-slate-300 text-slate-600 bg-white'
                }`}
              >
                {mode.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* Tensor Parallelism slider */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center text-sm">
            <label className="font-semibold text-slate-700 flex items-center gap-1.5">
              张量并行度 / Tensor Parallelism (TP)
              <HelpCircle className="w-3.5 h-3.5 text-slate-400 hover:text-emerald-600 cursor-help" title="多卡联合推理。模型参数与KV缓存将被均匀分布切割并加载至多块显卡上并行执行(显存除以GPU数量)。" />
            </label>
            <span className="font-mono bg-slate-100 px-2 py-0.5 rounded text-sm text-slate-700 font-bold">
              {config.tensorParallelism} GPU(s)
            </span>
          </div>
          <input
            id="inference-tp-slider"
            type="range"
            min="1"
            max="16"
            step="1"
            value={config.tensorParallelism}
            onChange={(e) => handleChange('tensorParallelism', parseInt(e.target.value))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-500"
          />
          <div className="flex justify-between text-[11px] text-slate-400 font-mono">
            <span>1 (单卡)</span>
            <span>2</span>
            <span>4</span>
            <span>8 (标准服务器)</span>
            <span>16 (双节点等)</span>
          </div>
        </div>

        {/* System overhead context slider */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center text-sm">
            <label className="font-semibold text-slate-700 flex items-center gap-1.5">
              系统与 CUDA 驱动运行时开销 / Reserved CUDA Overhead
              <HelpCircle className="w-3.5 h-3.5 text-slate-400 hover:text-emerald-600 cursor-help" title="包括PyTorch上下文、CUDA引擎、通信缓存以及系统缓冲区占用的顽固空间。一般默认保留1.5GB至3.0GB可大幅规避OOM溢出。" />
            </label>
            <span className="font-mono bg-slate-100 px-2 py-0.5 rounded text-sm text-slate-700 font-bold">
              {config.systemOverheadGB.toFixed(1)} GB
            </span>
          </div>
          <input
            id="inference-overhead-slider"
            type="range"
            min="0.5"
            max="10.0"
            step="0.5"
            value={config.systemOverheadGB}
            onChange={(e) => handleChange('systemOverheadGB', parseFloat(e.target.value))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-500"
          />
          <div className="flex justify-between text-[11px] text-slate-400 font-mono">
            <span>0.5 GB (极精简)</span>
            <span>2.0 GB (默认推荐)</span>
            <span>5.0 GB</span>
            <span>10.0 GB (重载驱动)</span>
          </div>
        </div>

        {/* DeepSeek MLA Compression Highlight Toggle */}
        {isDeepSeekModel && (
          <div id={COMPONENT_IDS.COMPRESSED_MLA} className="bg-blue-50/50 rounded-xl p-4 border border-blue-100 flex flex-col gap-2.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-blue-800 uppercase tracking-wider flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-blue-600" />
                DeepSeek MLA 缓存高度压缩机制
              </span>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={useMLACompression}
                  onChange={(e) => onMLACompressionChange(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-9 h-5 bg-slate-200 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-blue-600"></div>
              </label>
            </div>
            <p className="text-[11px] text-blue-600 leading-relaxed">
              <strong>激活多头潜空间注意力 (MLA) 压缩:</strong> 默认为其保留高效的缓存算法。MLA 通过低秩向量压缩将极度沉重的键/值维度压缩为轻小的 512 潜空间，KV-Cache VRAM 仅相当于原本 128 Heads 架设的 <strong>约 1/14 (在8bit/16bit下表现为 4.5 虚拟头)</strong>! 关闭此项将降级为标准的 MHA/GQA 膨胀计算。
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
