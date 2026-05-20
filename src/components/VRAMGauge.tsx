/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { VRAMBreakdown, CalcMode, ModelPreset, PrecisionDetails, InferenceConfig, TrainingConfig, GPUType } from '../types';
import { COMPONENT_IDS } from '../data';
import { HeartCrack, ShieldAlert, Award, Activity, Database, Sparkles, Binary } from 'lucide-react';

interface VRAMGaugeProps {
  breakdown: VRAMBreakdown;
  selectedMode: CalcMode;
  gpuCapacity: number; // reference capacity (GB)
  selectedModel?: ModelPreset;
  selectedPrecision?: PrecisionDetails;
  inferenceConfig?: InferenceConfig;
  trainingConfig?: TrainingConfig;
  selectedGPU?: GPUType;
  gpuCount?: number;
}

export const VRAMGauge: React.FC<VRAMGaugeProps> = ({
  breakdown,
  selectedMode,
  gpuCapacity,
  selectedModel,
  selectedPrecision,
  inferenceConfig,
  trainingConfig,
  selectedGPU,
  gpuCount,
}) => {
  const { modelWeights, kvCache, trainingState, activationMemory, overhead, total } = breakdown;
  
  // Weights state: gradients and optimizer
  const gradientsGB = trainingState?.gradients ?? 0;
  const optimizerGB = trainingState?.optimizer ?? 0;

  // Percentage of typical selected capacity
  const percentOfGPU = parseFloat(((total / Math.max(8, gpuCapacity)) * 100).toFixed(1));
  
  // Status check
  const getStatusInfo = () => {
    if (total <= 8) {
      return {
        label: '标准家用级 / Highly Accessible',
        color: 'text-emerald-700 bg-emerald-50 border-emerald-150',
        text: '可在大多数主流消费级电脑、手机或轻量边缘设备上轻松运行推理/量化。',
        icon: Award,
        barColor: 'bg-emerald-500'
      };
    } else if (total <= 24) {
      return {
        label: '高端单卡级 / Single High-End GPU',
        color: 'text-blue-700 bg-blue-50 border-blue-150',
        text: '契合单块 RTX 3090/4090 (24GB) 或 Mac 设备。属于高性价比的顶配自部署方案。',
        icon: Activity,
        barColor: 'bg-blue-500'
      };
    } else if (total <= 80) {
      return {
        label: '专业企业单卡级 / Professional Server GPU',
        color: 'text-amber-700 bg-amber-50 border-amber-150',
        text: '需要企业级 A100 (80GB)、H100 或 Mac Studio。多用于中大规模的企业服务。',
        icon: ShieldAlert,
        barColor: 'bg-amber-500'
      };
    } else {
      return {
        label: '多卡分布式集群 / Multi-GPU Distributed Cluster',
        color: 'text-rose-700 bg-rose-50 border-rose-150',
        text: '显存需求极度庞大！必须依赖张量并行(TP)或流水线(PP)分布式切分到多台 GPU 节点分布式运行。',
        icon: HeartCrack,
        barColor: 'bg-rose-500'
      };
    }
  };

  const status = getStatusInfo();
  const StatusIcon = status.icon;

  // Compute percentage segments
  const activeWeightsPct = (modelWeights / Math.max(0.1, total)) * 100;
  const activeKVPct = (kvCache / Math.max(0.1, total)) * 100;
  const gradPct = (gradientsGB / Math.max(0.1, total)) * 100;
  const optPct = (optimizerGB / Math.max(0.1, total)) * 100;
  const activeActivatePct = (activationMemory / Math.max(0.1, total)) * 100;
  const activeOverheadPct = (overhead / Math.max(0.1, total)) * 100;

  return (
    <div id={COMPONENT_IDS.VRAM_GAUGE} className="bg-slate-900 text-white rounded-2xl p-6 shadow-md flex flex-col gap-6 border border-slate-800">
      
      {/* Header Banner */}
      <div className="flex justify-between items-start gap-4">
        <div>
          <span className="text-[10px] font-bold tracking-wider text-indigo-400 bg-indigo-950/80 border border-indigo-800 px-2 py-0.5 rounded uppercase">
            {selectedMode === 'inference' ? '⚡ 推理总显存估算 / Inference VRAM' : '🏋️ Training VRAM / 训练总显存估算'}
          </span>
          <div className="text-4xl font-extrabold font-mono text-slate-100 mt-2 flex items-baseline gap-1 animate-pulse">
            {total.toFixed(2)} <span className="text-lg font-semibold text-slate-400">GB</span>
          </div>
        </div>

        {/* Capacity compare indicator */}
        <div className="flex flex-col items-end shrink-0">
          <span className="text-xs text-slate-400">适配典型容量 ({gpuCapacity}GB)</span>
          <div className={`text-base font-bold font-mono mt-1 ${percentOfGPU > 100 ? 'text-rose-400' : percentOfGPU > 85 ? 'text-amber-400' : 'text-emerald-400'}`}>
            {percentOfGPU}% 负载
          </div>
        </div>
      </div>

      {/* Active Selection Details Summary Card */}
      {(selectedModel && selectedPrecision) && (
        <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800/80 grid grid-cols-2 gap-2.5 font-sans">
          <div className="flex flex-col col-span-2 pb-1.5 border-b border-slate-850">
            <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold flex items-center gap-1">
              <Sparkles className="w-3 h-3 text-indigo-400" />
              当前评估配置组合 / ACTIVE SIZING PARAMS
            </span>
          </div>
          
          <div className="flex flex-col">
            <span className="text-[9.5px] text-slate-500 uppercase">评估模型物种</span>
            <span className="text-xs font-semibold text-slate-100 truncate mt-0.5" title={selectedModel.name}>
              {selectedModel.name} <span className="text-[9.5px] font-mono text-slate-400">({selectedModel.totalParams}B)</span>
            </span>
          </div>
          
          <div className="flex flex-col">
            <span className="text-[9.5px] text-slate-400 uppercase">数据计算精度</span>
            <span className="text-xs font-semibold text-indigo-300 mt-0.5 font-mono">
              {selectedPrecision.name} ({selectedPrecision.bitsPerWeight}位)
            </span>
          </div>

          <div className="flex flex-col">
            <span className="text-[9.5px] text-slate-400 uppercase">运行算力宿主</span>
            <span className="text-xs font-semibold text-slate-100 mt-0.5 truncate" title={selectedGPU?.name}>
              {gpuCount ?? 1}卡 × {selectedGPU?.name || 'GPU'} <span className="font-mono text-[9px] text-slate-400">({gpuCapacity}GB)</span>
            </span>
          </div>

          {selectedMode === 'inference' ? (
            <div className="flex flex-col">
              <span className="text-[9.5px] text-slate-400 uppercase">推理并发与长度</span>
              <span className="text-xs font-semibold text-emerald-400 mt-0.5 font-mono">
                Bsz={inferenceConfig?.batchSize || 1} · SeqL={inferenceConfig?.sequenceLength || 4096}
              </span>
            </div>
          ) : (
            <div className="flex flex-col">
              <span className="text-[9.5px] text-slate-400 uppercase">训练并发与长度</span>
              <span className="text-xs font-semibold text-amber-400 mt-0.5 font-mono">
                Bsz={trainingConfig?.batchSize || 1} · SeqL={trainingConfig?.sequenceLength || 4096} · {trainingConfig?.trainableParamsPercent === 100 ? '全参' : 'LoRA'}
              </span>
            </div>
          )}

          {/* Inline configuration chips */}
          <div className="col-span-2 flex flex-wrap gap-1.5 mt-1 pt-1.5 border-t border-slate-800/40 text-[9px]">
            {selectedMode === 'inference' && inferenceConfig && (
              <>
                <span className="bg-slate-900/80 px-1.5 py-0.5 rounded border border-slate-800 text-slate-400 font-mono">
                  KV-Cache: {inferenceConfig.kvCachePrecision.toUpperCase()}
                </span>
                <span className={`px-1.5 py-0.5 rounded border font-mono ${
                  inferenceConfig.chunkPrefillSize && inferenceConfig.chunkPrefillSize !== 'off'
                    ? 'bg-emerald-950/45 border-emerald-900/60 text-emerald-400'
                    : 'bg-slate-900/85 border-slate-800 text-slate-500'
                }`}>
                  分块预填: {inferenceConfig.chunkPrefillSize && inferenceConfig.chunkPrefillSize !== 'off' ? `${inferenceConfig.chunkPrefillSize} tkn` : 'OFF'}
                </span>
                {inferenceConfig.tensorParallelism > 1 && (
                  <span className="bg-indigo-950/45 border-indigo-900/60 text-indigo-400 px-1.5 py-0.5 rounded border font-mono">
                    张量并行TP: {inferenceConfig.tensorParallelism}卡
                  </span>
                )}
              </>
            )}
            {selectedMode === 'training' && trainingConfig && (
              <>
                <span className="bg-slate-900/80 px-1.5 py-0.5 rounded border border-slate-800 text-slate-400 font-mono">
                  优化器: {trainingConfig.optimizer.toUpperCase()}
                </span>
                <span className="bg-indigo-950/45 border-indigo-900/60 text-indigo-400 px-1.5 py-0.5 rounded border font-mono animate-pulse">
                  激活重算 Checkpoint: On
                </span>
              </>
            )}
          </div>
        </div>
      )}

      {/* Segmented Stack Bar */}
      <div className="flex flex-col gap-1.5 mt-2">
        <div className="h-4 w-full bg-slate-800 rounded-full flex overflow-hidden ring-4 ring-slate-950">
          {modelWeights > 0 && (
            <div 
              id="bar-weights"
              className="bg-indigo-500 transition-all duration-300 hover:opacity-90" 
              style={{ width: `${activeWeightsPct}%` }}
              title={`模型参数: ${modelWeights} GB`}
            />
          )}
          {kvCache > 0 && (
            <div 
              id="bar-kv"
              className="bg-emerald-500 transition-all duration-300 hover:opacity-90" 
              style={{ width: `${activeKVPct}%` }}
              title={`KV 缓存: ${kvCache} GB`}
            />
          )}
          {gradientsGB > 0 && (
            <div 
              id="bar-grads"
              className="bg-amber-500 transition-all duration-300 hover:opacity-90" 
              style={{ width: `${gradPct}%` }}
              title={`权重梯度: ${gradientsGB} GB`}
            />
          )}
          {optimizerGB > 0 && (
            <div 
              id="bar-optimizer"
              className="bg-rose-500 transition-all duration-300 hover:opacity-90" 
              style={{ width: `${optPct}%` }}
              title={`优化器状态: ${optimizerGB} GB`}
            />
          )}
          {activationMemory > 0 && (
            <div 
              id="bar-activations"
              className="bg-sky-500 transition-all duration-300 hover:opacity-90" 
              style={{ width: `${activeActivatePct}%` }}
              title={`激活计算/重算: ${activationMemory} GB`}
            />
          )}
          {overhead > 0 && (
            <div 
              id="bar-overhead"
              className="bg-slate-500 transition-all duration-300 hover:opacity-90" 
              style={{ width: `${activeOverheadPct}%` }}
              title={`驱动开销: ${overhead} GB`}
            />
          )}
        </div>

        <div className="flex flex-wrap gap-x-4 gap-y-2 text-xs text-slate-400 font-mono pt-1">
          <div className="flex items-center gap-1.5 shrink-0">
            <span className="w-2.5 h-2.5 rounded bg-indigo-500 shrink-0" />
            <span>模型权重: {modelWeights} GB ({activeWeightsPct.toFixed(0)}%)</span>
          </div>
          {selectedMode === 'inference' && kvCache > 0 && (
            <div className="flex items-center gap-1.5 shrink-0">
              <span className="w-2.5 h-2.5 rounded bg-emerald-500 shrink-0" />
              <span>KV-Cache: {kvCache} GB ({activeKVPct.toFixed(0)}%)</span>
            </div>
          )}
          {selectedMode === 'training' && gradientsGB > 0 && (
            <div className="flex items-center gap-1.5 shrink-0">
              <span className="w-2.5 h-2.5 rounded bg-amber-500 shrink-0" />
              <span>权重梯度: {gradientsGB} GB ({gradPct.toFixed(0)}%)</span>
            </div>
          )}
          {selectedMode === 'training' && optimizerGB > 0 && (
            <div className="flex items-center gap-1.5 shrink-0">
              <span className="w-2.5 h-2.5 rounded bg-rose-500 shrink-0" />
              <span>优化器状态: {optimizerGB} GB ({optPct.toFixed(0)}%)</span>
            </div>
          )}
          {activationMemory > 0 && (
            <div className="flex items-center gap-1.5 shrink-0">
              <span className="w-2.5 h-2.5 rounded bg-sky-500 shrink-0" />
              <span>激活中间体: {activationMemory} GB ({activeActivatePct.toFixed(0)}%)</span>
            </div>
          )}
          {overhead > 0 && (
            <div className="flex items-center gap-1.5 shrink-0">
              <span className="w-2.5 h-2.5 rounded bg-slate-500 shrink-0" />
              <span>驱动开销: {overhead} GB ({activeOverheadPct.toFixed(0)}%)</span>
            </div>
          )}
        </div>
      </div>

      {/* Deployment tier advisory status card */}
      <div className={`p-4 rounded-xl border flex gap-3.5 items-start text-sm ${status.color}`}>
        <StatusIcon className="w-5 h-5 shrink-0 mt-0.5" />
        <div className="flex flex-col gap-1">
          <span className="font-bold tracking-tight">{status.label}</span>
          <p className="text-xs leading-relaxed opacity-90">{status.text}</p>
        </div>
      </div>

      {/* Numeric Breakdown lists */}
      <div id={COMPONENT_IDS.BREAKDOWN} className="border-t border-slate-800 pt-5 flex flex-col gap-3.5">
        <span className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1">
          <Binary className="w-4 h-4 text-slate-500" />
          显存开销解构 / Breakdown Details
        </span>

        {/* Weights details */}
        <div className="flex justify-between items-center text-sm text-slate-300">
          <div className="flex flex-col">
            <span className="font-medium text-slate-200">静态模型权重 / Model Weights (Frozen & Active States)</span>
            <span className="text-[11px] text-slate-500 leading-none mt-1">
              存放模型全部骨架层所必需的基本占用。
            </span>
          </div>
          <span className="font-mono font-bold text-slate-100 shrink-0 bg-slate-800/80 px-2.5 py-1 rounded">
            {modelWeights.toFixed(2)} GB
          </span>
        </div>

        {/* Inference specific KV Cache */}
        {selectedMode === 'inference' && (
          <div className="flex justify-between items-center text-sm text-slate-300 border-t border-slate-800/40 pt-3">
            <div className="flex flex-col">
              <span className="font-medium text-slate-200">运行时 KV Cache 缓存 / Context KV Cache</span>
              <span className="text-[11px] text-slate-500 leading-none mt-1">
                存储注意力机制中历史 Token 对应的 Key/Value 矩阵。随并发(Batch)及文本长度成线性剧增。
              </span>
            </div>
            <span className="font-mono font-bold text-slate-100 shrink-0 bg-slate-800/80 px-2.5 py-1 rounded">
              {kvCache.toFixed(2)} GB
            </span>
          </div>
        )}

        {/* Training specify items */}
        {selectedMode === 'training' && (
          <>
            {/* Gradients */}
            <div className="flex justify-between items-center text-sm text-slate-300 border-t border-slate-800/40 pt-3">
              <div className="flex flex-col">
                <span className="font-medium text-slate-200">梯度保存 / Gradient Space</span>
                <span className="text-[11px] text-slate-500 leading-none mt-1">
                  反向传播(BP)流程中存储偏导值。仅在可训练参数上产生开销。
                </span>
              </div>
              <span className="font-mono font-bold text-slate-100 shrink-0 bg-slate-800/80 px-2.5 py-1 rounded">
                {gradientsGB.toFixed(2)} GB
              </span>
            </div>

            {/* Optimizer */}
            <div className="flex justify-between items-center text-sm text-slate-300 border-t border-slate-800/40 pt-3">
              <div className="flex flex-col">
                <span className="font-medium text-slate-200">优化器状态 / Optimizer States</span>
                <span className="text-[11px] text-slate-500 leading-none mt-1">
                  跟踪一阶、二阶等梯度惯性。如 Adam 优化器需要为权重配置 FP32 的长效记录缓存。
                </span>
              </div>
              <span className="font-mono font-bold text-slate-100 shrink-0 bg-slate-800/80 px-2.5 py-1 rounded">
                {optimizerGB.toFixed(2)} GB
              </span>
            </div>
          </>
        )}

        {/* Activations */}
        <div className="flex justify-between items-center text-sm text-slate-300 border-t border-slate-800/40 pt-3">
          <div className="flex flex-col">
            <span className="font-medium text-slate-200">局部前向求导激活空间 / Activation Memory</span>
            <span className="text-[11px] text-slate-500 leading-none mt-1">
              {selectedMode === 'inference' 
                ? '存放进行 softmax 矩阵和多头融合计算的中间矢量缓转。'
                : '前向传递所计算的特征映射，用以反向回流。已融入梯度Checkpointing降低策略。'}
            </span>
          </div>
          <span className="font-mono font-bold text-slate-100 shrink-0 bg-slate-800/80 px-2.5 py-1 rounded">
            {activationMemory.toFixed(2)} GB
          </span>
        </div>

        {/* CUDA Context Overhead */}
        <div className="flex justify-between items-center text-sm text-slate-300 border-t border-slate-800/40 pt-3">
          <div className="flex flex-col">
            <span className="font-medium text-slate-200">CUDA / PyTorch 驻留开销 / System Reserved Buffer</span>
            <span className="text-[11px] text-slate-500 leading-none mt-1">
              驱动程序、各种计算核(Kernels)缓存和 CUDA Context 必备的刚性硬件底层缓存。
            </span>
          </div>
          <span className="font-mono font-bold text-slate-100 shrink-0 bg-slate-800/80 px-2.5 py-1 rounded">
            {overhead.toFixed(2)} GB
          </span>
        </div>
      </div>
    </div>
  );
};
