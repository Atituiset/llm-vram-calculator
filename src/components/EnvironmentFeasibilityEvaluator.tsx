/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Environment Deployment Feasibility Evaluator & Cross-Platform Adaptation Matrix
 * Tailored to directly resolve cross-platform querying pain points.
 */

import React, { useMemo, useState } from 'react';
import { ModelPreset, PrecisionDetails, GPUType, InferenceConfig, PrecisionMode } from '../types';
import { MODEL_PRESETS, PRECISION_OPTS, GPU_PRESETS, calculateInferenceVRAM } from '../data';
import { ShieldCheck, ShieldAlert, AlertTriangle, Cpu, Check, HelpCircle, Layers, ArrowDownUp, Sparkles, BookOpen } from 'lucide-react';

interface EnvironmentFeasibilityEvaluatorProps {
  // Current App States
  selectedModel: ModelPreset;
  selectedPrecision: PrecisionDetails;
  selectedGPU: GPUType;
  inferenceConfig: InferenceConfig;
  useMLACompression: boolean;
  
  // Setters to sync state on one-click cell load
  onModelChange: (model: ModelPreset) => void;
  onPrecisionChange: (precision: PrecisionDetails) => void;
  onGPUTypeSelect: (gpu: GPUType) => void;
  onConfigChange: (config: InferenceConfig) => void;
  
  // Dynamic Environment state controlled locally or bound
  envGPUCount: number;
  onEnvGPUCountChange: (count: number) => void;
}

export const EnvironmentFeasibilityEvaluator: React.FC<EnvironmentFeasibilityEvaluatorProps> = ({
  selectedModel,
  selectedPrecision,
  selectedGPU,
  inferenceConfig,
  useMLACompression,
  onModelChange,
  onPrecisionChange,
  onGPUTypeSelect,
  onConfigChange,
  envGPUCount,
  onEnvGPUCountChange,
}) => {
  // Config state for the matrix
  const [matrixSeqLen, setMatrixSeqLen] = useState<number>(4096);
  const [matrixBatchSize, setMatrixBatchSize] = useState<number>(1);
  const [matrixKVCachePrecision, setMatrixKVCachePrecision] = useState<'fp16' | 'fp8' | 'int8'>('fp16');

  // Sync matrix options to primary config helper
  const syncMatrixToConfig = () => {
    onConfigChange({
      ...inferenceConfig,
      sequenceLength: matrixSeqLen,
      batchSize: matrixBatchSize,
      kvCachePrecision: matrixKVCachePrecision,
      tensorParallelism: envGPUCount, // Optimal: align TP with host physical card count
    });
  };

  // Hardware capabilities calculation
  const totalPoolVRAM = useMemo(() => {
    return selectedGPU.vram * envGPUCount;
  }, [selectedGPU.vram, envGPUCount]);

  // VRAM Breakdown of Currently Selected configuration
  const currentBreakdown = useMemo(() => {
    return calculateInferenceVRAM(selectedModel, selectedPrecision, {
      batchSize: inferenceConfig.batchSize,
      sequenceLength: inferenceConfig.sequenceLength,
      kvCachePrecision: inferenceConfig.kvCachePrecision,
      systemOverheadGB: inferenceConfig.systemOverheadGB,
      tensorParallelism: envGPUCount,
      useMLACompression: useMLACompression,
    });
  }, [selectedModel, selectedPrecision, inferenceConfig, envGPUCount, useMLACompression]);

  const loadPercent = useMemo(() => {
    if (totalPoolVRAM <= 0) return 0;
    return (currentBreakdown.total / totalPoolVRAM) * 100;
  }, [currentBreakdown.total, totalPoolVRAM]);

  // Multi-card physical deployment recommendations
  const diagnosticVerdict = useMemo(() => {
    const totalNeeded = currentBreakdown.total;
    const available = totalPoolVRAM;

    if (totalNeeded <= available * 0.82) {
      return {
        status: 'success' as const,
        title: '完全可行 / Highly Feasible',
        bg: 'bg-emerald-50 border-emerald-200 text-emerald-800',
        iconBg: 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20',
        advice: `当前模型加缓存耗用约 ${totalNeeded.toFixed(1)} GB 显存，处于硬件资源池 (${available} GB) 负载安全红线以下。模型推理吞吐响应将达到峰值，可以完全放心在该节点独立承载高并发。`,
        action: '该拓扑性价比优异，可正常启动高速 vLLM / SGLang 微服务集群服务。'
      };
    } else if (totalNeeded <= available) {
      return {
        status: 'warn' as const,
        title: '容灾可行 (高负载) / Borderline Deployable',
        bg: 'bg-amber-50 border-amber-200 text-amber-800',
        iconBg: 'bg-amber-500/10 text-amber-600 border-amber-500/20',
        advice: `总显存需求为 ${totalNeeded.toFixed(1)} GB，紧贴物理上限 (${available} GB)。单卡负载高达 ${loadPercent.toFixed(0)}%。在并发增大 (Batch Size > 4) 或处理 8K 以上超级长文本 (Prefill 阶段) 时，极其容易因激活权重暴增诱发 Out-Of-Memory (OOM) 报错崩溃。`,
        action: '💡 专家优化策略：请在推理配置中强制开启 "FP8 KV-Cache 降轨"，或者使用本页面的一键部署工具并调低 mem-fraction-static 参数比例，以腾出安全的 PyTorch 物理堆空间。'
      };
    } else {
      // Out Of Memory Case
      const weightDeficit = currentBreakdown.modelWeights > available;
      let reason = `所选模型所需总显存 (${totalNeeded.toFixed(1)} GB) 超过了硬件资源池总和 (${available} GB)。`;
      let suggestion = '';

      if (weightDeficit) {
        reason += ` 仅静态模型权重 (${currentBreakdown.modelWeights.toFixed(1)} GB) 就已经把整组硬件塞满了！这是绝对物理硬件不可行的硬边界。`;
        suggestion = '🛠️ 专家破局路线：模型尺寸太大，请点击下方适配矩阵，选择 INT4、FP4 或 GGUF Q4 这种极致量化版本；或者增加本地 GPU 槽位卡数。';
      } else {
        reason += ` 模型权重可纳，但是 KV Cache 缓存及上下文中间变量对显存产生了极限挤占。`;
        suggestion = '🛠️ 专家破局路线：请将 Sequence Length 上下文或 Batch Size 设低，或者一键开启 FP8 量化缓存以强行压缩 KV 存储，从而让模型成功上线。';
      }

      return {
        status: 'error' as const,
        title: '物理溢出 / Definite OOM State',
        bg: 'bg-rose-50 border-rose-200 text-rose-800',
        iconBg: 'bg-rose-500/10 text-rose-600 border-rose-500/20',
        advice: reason,
        action: suggestion
      };
    }
  }, [currentBreakdown, totalPoolVRAM, loadPercent]);

  // Support for checking entire matrix dynamically
  const matrixData = useMemo(() => {
    // We want to test common open-source models with representative precisions
    // for the custom hardware pool configured
    const targetModels = MODEL_PRESETS;
    // We'll map columns to specific precision modes
    const targetPrecisions = [
      PrecisionMode.FP16_BF16,
      PrecisionMode.FP8,
      PrecisionMode.INT8,
      PrecisionMode.INT4,
      PrecisionMode.GGUF_Q4
    ];

    return targetModels.map((model) => {
      const rowPrecisions = targetPrecisions.map((pMode) => {
        const precisionDetails = PRECISION_OPTS.find(p => p.id === pMode) || PRECISION_OPTS[0];
        
        // Calculate dynamic cost under current active matrix constraints
        const breakdown = calculateInferenceVRAM(model, precisionDetails, {
          batchSize: matrixBatchSize,
          sequenceLength: matrixSeqLen,
          kvCachePrecision: matrixKVCachePrecision,
          systemOverheadGB: inferenceConfig.systemOverheadGB,
          tensorParallelism: envGPUCount,
          useMLACompression: useMLACompression,
        });

        const fits = breakdown.total <= totalPoolVRAM;
        const loadRatio = (breakdown.total / totalPoolVRAM) * 100;

        let status: 'safe' | 'tight' | 'oom' = 'safe';
        if (loadRatio > 98) {
          status = 'oom';
        } else if (loadRatio > 83) {
          status = 'tight';
        }

        return {
          precisionMode: pMode,
          precisionDetails,
          breakdown,
          loadRatio,
          status,
          fits
        };
      });

      return {
        model,
        rowPrecisions
      };
    });
  }, [matrixSeqLen, matrixBatchSize, matrixKVCachePrecision, totalPoolVRAM, envGPUCount, useMLACompression, inferenceConfig.systemOverheadGB]);

  return (
    <div id="environment-feasibility-evaluator" className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm flex flex-col gap-6">
      
      {/* Component Title */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-emerald-600 text-white rounded-xl flex items-center justify-center">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-900 tracking-tight flex items-center gap-2">
              环境部署可行性一键评估 / Environment Feasibility Sizing
              <span className="text-[10px] uppercase font-bold px-1.5 py-0.5 rounded bg-emerald-100 border border-emerald-200 text-emerald-800 tracking-wider">专属专家模块</span>
            </h2>
            <p className="text-xs text-slate-500">
              解决跨平台查询痛点：在此输入您所拥有的物理硬件卡组，一分钟测算开源大模型生态兼容谱系。
            </p>
          </div>
        </div>
      </div>

      {/* STEP A: Config My Environment Resources */}
      <div className="bg-slate-50 rounded-xl p-5 border border-slate-200/60 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
            <Layers className="w-4 h-4 text-emerald-600" />
            1. 定义我的物理卡组环境 / HOST HARDWARE INFRASTRUCTURE
          </span>
          <span className="text-xs text-slate-500 font-mono">
            总配额: <strong className="text-slate-800">{totalPoolVRAM.toFixed(0)} GB</strong> (单卡: {selectedGPU.vram}G × {envGPUCount}卡)
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
          
          {/* Card selection */}
          <div className="md:col-span-8 flex flex-col gap-1.5">
            <label className="text-xs text-slate-600 font-semibold flex items-center justify-between">
              <span>主板物理算力卡型号 / Selected Hardware Model:</span>
              <span className="text-[10px] text-indigo-650 bg-indigo-50 px-1.5 rounded">
                支持多卡商跨平台适配机架验证
              </span>
            </label>
            
            <div className="flex flex-col gap-2">
              {/* Comprehensive Grouped Selector Dropdown */}
              <select
                aria-label="选择算力卡型号"
                value={GPU_PRESETS.some(g => g.id === selectedGPU.id) ? selectedGPU.id : 'custom'}
                onChange={(e) => {
                  const val = e.target.value;
                  if (val === 'custom') return; // let them edit below
                  const found = GPU_PRESETS.find(g => g.id === val);
                  if (found) onGPUTypeSelect(found);
                }}
                className="w-full bg-white border border-slate-205 rounded-lg p-2 text-xs text-slate-800 focus:outline-none focus:ring-2 focus:ring-emerald-600/20 focus:border-emerald-600"
              >
                {!GPU_PRESETS.some(g => g.id === selectedGPU.id) && (
                  <option value="custom">★ 已载入自定义算力卡: {selectedGPU.name} ({selectedGPU.vram}GB)</option>
                )}
                
                <optgroup label="NVIDIA Data Center (AI 数据中心专用 SXM/PCIe)">
                  {GPU_PRESETS.filter(g => g.name.toLowerCase().includes('nvidia') && g.type === 'datacenter').map(g => (
                    <option key={g.id} value={g.id}>{g.name} ({g.vram} GB)</option>
                  ))}
                </optgroup>

                <optgroup label="NVIDIA GeForce / Workstation (轻量消费级显卡)">
                  {GPU_PRESETS.filter(g => g.name.toLowerCase().includes('nvidia') && g.type === 'consumer').map(g => (
                    <option key={g.id} value={g.id}>{g.name} ({g.vram} GB)</option>
                  ))}
                </optgroup>

                <optgroup label="AMD Instinct / Radeon 全系列 (ROCm平台)">
                  {GPU_PRESETS.filter(g => g.name.toLowerCase().includes('amd')).map(g => (
                    <option key={g.id} value={g.id}>{g.name} ({g.vram} GB)</option>
                  ))}
                </optgroup>

                <optgroup label="Apple Mac 统一内存节点 (M系列芯片)">
                  {GPU_PRESETS.filter(g => g.name.toLowerCase().includes('apple')).map(g => (
                    <option key={g.id} value={g.id}>{g.name} ({g.vram} GB)</option>
                  ))}
                </optgroup>

                <optgroup label="华为昇腾系列自主算力卡 (Huawei Ascend CANN/C++架构)">
                  {GPU_PRESETS.filter(g => g.name.toLowerCase().includes('huawei') || g.name.toLowerCase().includes('ascend')).map(g => (
                    <option key={g.id} value={g.id}>{g.name} ({g.vram} GB)</option>
                  ))}
                </optgroup>

                <optgroup label="Intel Gaudi / Google TPU 云原生阵列">
                  {GPU_PRESETS.filter(g => g.name.toLowerCase().includes('intel') || g.name.toLowerCase().includes('tpu') || g.id.includes('tpu')).map(g => (
                    <option key={g.id} value={g.id}>{g.name} ({g.vram} GB)</option>
                  ))}
                </optgroup>
              </select>

              {/* 4 Flagship Quick Select Buttons for Multi-Vendor Showcase */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
                {[
                  { label: '🟢 NVIDIA H200', id: 'h200' },
                  { label: '🔴 AMD MI300X', id: 'amd-mi300x' },
                  { label: '🔴 华为 910B', id: 'ascend-910b-64' },
                  { label: '🍏 Apple M3 Max', id: 'apple-m3-max-128' }
                ].map((item) => {
                  const preset = GPU_PRESETS.find(g => g.id === item.id);
                  const isSelected = selectedGPU.id === item.id;
                  return (
                    <button
                      key={item.id}
                      type="button"
                      disabled={!preset}
                      onClick={() => preset && onGPUTypeSelect(preset)}
                      className={`py-1.5 px-2 rounded-md text-[10.5px] font-sans transition-all text-center border truncate font-medium ${
                        isSelected
                          ? 'border-emerald-600 bg-emerald-50/50 text-emerald-950 font-semibold shadow-xs'
                          : 'border-slate-200 bg-white hover:border-slate-350 text-slate-700 hover:bg-slate-50'
                      }`}
                    >
                      {item.label} ({preset?.vram}G)
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Cards Count selection */}
          <div className="md:col-span-4 flex flex-col gap-1.5">
            <label className="text-xs text-slate-500 font-semibold">物理显卡插槽数(卡数) / Physical GPU Count:</label>
            <div className="grid grid-cols-5 gap-1.5 h-full">
              {[1, 2, 4, 8, 16].map((count) => (
                <button
                  key={count}
                  type="button"
                  onClick={() => onEnvGPUCountChange(count)}
                  className={`rounded-lg border flex flex-col items-center justify-center py-2 transition-all ${
                    envGPUCount === count
                      ? 'border-emerald-600 bg-emerald-600 text-white font-bold shadow-sm'
                      : 'border-slate-200 bg-white hover:border-emerald-200 hover:bg-emerald-50/20 text-slate-700'
                  }`}
                >
                  <span className="text-sm font-mono">{count}</span>
                  <span className="text-[9px] uppercase tracking-wider opacity-80">卡</span>
                </button>
              ))}
            </div>
          </div>

        </div>
      </div>

      {/* STEP B: Primary Verdict diagnostics */}
      <div className={`p-5 rounded-2xl border ${diagnosticVerdict.bg} flex flex-col sm:flex-row gap-4`}>
        <div className={`w-12 h-12 rounded-xl shrink-0 border flex items-center justify-center ${diagnosticVerdict.iconBg}`}>
          {diagnosticVerdict.status === 'success' && <ShieldCheck className="w-6 h-6" />}
          {diagnosticVerdict.status === 'warn' && <AlertTriangle className="w-6 h-6" />}
          {diagnosticVerdict.status === 'error' && <ShieldAlert className="w-6 h-6" />}
        </div>
        
        <div className="flex flex-col gap-1.5">
          <h4 className="text-sm font-bold text-slate-900 flex flex-wrap items-center gap-2">
            当前配置部署可行性诊断报告:
            <span className="font-mono text-indigo-600 bg-white/80 px-2 py-0.5 rounded border border-slate-250 text-xs">
              {selectedModel.name} @ {selectedPrecision.name}
            </span>
          </h4>
          <p className="text-xs text-slate-700 leading-normal">
            <strong>诊断结论:</strong> {diagnosticVerdict.advice}
          </p>
          <div className="text-[11px] text-slate-650 leading-relaxed border-t border-slate-300/30 pt-1.5 mt-0.5 font-medium/60">
            <strong>🎯 本地部署优化方案 / Deployment Action Item:</strong> {diagnosticVerdict.action}
          </div>
        </div>
      </div>

      {/* STEP C: Matrix Grid lookup */}
      <div className="border border-slate-200 rounded-xl overflow-hidden flex flex-col">
        
        {/* Sizing filters specifically for the look up matrix */}
        <div className="bg-slate-50 border-b border-slate-200 p-4 flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
          <div className="flex flex-col">
            <span className="text-xs font-bold text-slate-800 flex items-center gap-1">
              <Sparkles className="w-3.5 h-3.5 text-yellow-500 fill-yellow-500" />
              2. 开源生态一链检测适配矩阵 / Multi-Model Adaptation Lookup Matrix
            </span>
            <p className="text-[10px] text-slate-500 mt-0.5">
              模拟不同上下文与并发场景，全景探知。<strong>温馨提示：点击任意可用单元格，可直接一键反向加载加载该模型与量化组合至计算器！</strong>
            </p>
          </div>

          {/* Matrix Parameters Filters */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] text-slate-600 font-medium">矩阵上下文:</span>
              <select
                value={matrixSeqLen}
                onChange={(e) => setMatrixSeqLen(Number(e.target.value))}
                className="bg-white border border-slate-350 rounded px-1.5 py-1 text-xs font-mono text-slate-800"
              >
                <option value={2048}>2k (短文对话)</option>
                <option value={4096}>4k (标准对话)</option>
                <option value={8192}>8k (多轮会话)</option>
                <option value={16384}>16k (中等文档)</option>
                <option value={32768}>32k (研报书籍)</option>
                <option value={65536}>64k (代码库)</option>
                <option value={131072}>128k (超长输入)</option>
              </select>
            </div>

            <div className="flex items-center gap-1.5">
              <span className="text-[11px] text-slate-600 font-medium">并发 batch:</span>
              <select
                value={matrixBatchSize}
                onChange={(e) => setMatrixBatchSize(Number(e.target.value))}
                className="bg-white border border-slate-350 rounded px-1.5 py-1 text-xs font-mono text-slate-800"
              >
                <option value={1}>1 (单卡轻量)</option>
                <option value={4}>4 (小团队并发)</option>
                <option value={16}>16 (中高并发)</option>
                <option value={32}>32 (高频生产)</option>
                <option value={64}>64 (高负载极限)</option>
              </select>
            </div>

            <button
              type="button"
              onClick={syncMatrixToConfig}
              className="px-2.5 py-1 text-xs bg-indigo-650 hover:bg-indigo-750 text-white rounded font-semibold transition-colors flex items-center gap-1"
              title="将此矩阵参数一键同步至系统真实计算区"
            >
              <ArrowDownUp className="w-3.5 h-3.5" />
              同步全局设定
            </button>
          </div>
        </div>

        {/* The Matrix Table scrollable container */}
        <div className="overflow-x-auto w-full">
          <table className="w-full text-left text-xs text-slate-600 min-w-[700px]">
            
            <thead className="bg-slate-100 text-slate-700 uppercase tracking-tight text-[11px] border-b border-slate-200">
              <tr>
                <th className="px-4 py-3 font-bold w-[210px] border-r border-slate-200 bg-slate-100/50">
                  主流开源模型 / Model Name
                </th>
                <th className="px-3 py-3 text-center border-r border-slate-250">
                  FP16 / BF16
                </th>
                <th className="px-3 py-3 text-center border-r border-slate-250">
                  FP8 (Float)
                </th>
                <th className="px-3 py-3 text-center border-r border-slate-250">
                  INT8 (Integer)
                </th>
                <th className="px-3 py-3 text-center border-r border-slate-250">
                  INT4 (4-Bit)
                </th>
                <th className="px-3 py-3 text-center">
                  GGUF Q4 (M)
                </th>
              </tr>
            </thead>

            <tbody className="divide-y divide-slate-200 bg-white">
              {matrixData.map(({ model, rowPrecisions }) => {
                const isActiveModel = selectedModel.id === model.id;
                
                return (
                  <tr 
                    key={model.id} 
                    className={`transition-colors ${
                      isActiveModel ? 'bg-indigo-50/20 font-medium' : 'hover:bg-slate-50/40'
                    }`}
                  >
                    
                    {/* Model Metadata Name */}
                    <td className="px-4 py-3 border-r border-slate-200/80 font-bold bg-slate-50/20">
                      <div className="flex flex-col">
                        <span className="text-slate-900 text-xs flex items-center gap-1.5 flex-wrap">
                          {model.name}
                          {isActiveModel && (
                            <span className="text-[9px] bg-indigo-600 text-white font-extrabold px-1 py-0.2 rounded-full scale-90">
                              Active
                            </span>
                          )}
                        </span>
                        <span className="text-[10px] text-slate-400 font-normal mt-0.5 font-mono">
                          总参数: {model.totalParams}B {model.activeParams ? `(激活 ${model.activeParams}B MoE)` : ''} · {model.creator}
                        </span>
                      </div>
                    </td>

                    {/* Precisions rendering loop */}
                    {rowPrecisions.map(({ precisionMode, precisionDetails, breakdown, loadRatio, status }) => {
                      const isActiveCell = selectedModel.id === model.id && selectedPrecision.id === precisionMode;
                      
                      let badgeCls = '';
                      let statusText = '';
                      if (status === 'safe') {
                        badgeCls = 'bg-emerald-50 text-emerald-800 border-emerald-100/80 hover:bg-emerald-100/50 hover:border-emerald-300 cursor-pointer';
                        statusText = '良好 / Recommend';
                      } else if (status === 'tight') {
                        badgeCls = 'bg-amber-50 text-amber-800 border-amber-200 hover:bg-amber-100/50 hover:border-amber-400 cursor-pointer';
                        statusText = '吃紧 / Borderline';
                      } else {
                        badgeCls = 'bg-rose-50 text-rose-800 border-rose-200 hover:bg-rose-100/50 cursor-pointer';
                        statusText = 'OOM / Blocked';
                      }

                      return (
                        <td 
                          key={precisionMode}
                          onClick={() => {
                            // On-clicks immediately binds this row and col selection to parent app states!
                            onModelChange(model);
                            onPrecisionChange(precisionDetails);
                            // Also update configuration context limit TP to match environment count
                            onConfigChange({
                              ...inferenceConfig,
                              tensorParallelism: envGPUCount
                            });
                          }}
                          className={`p-2 text-center border-r last:border-r-0 border-slate-100 align-middle transition-all shrink-0 select-none ${
                            isActiveCell ? 'ring-2 ring-indigo-500 ring-inset bg-indigo-50/[0.15]' : ''
                          }`}
                          title={`点击一键模拟。计算规格: 权重+激活+缓存=${breakdown.total.toFixed(1)}GB / 硬件池容量=${totalPoolVRAM.toFixed(0)}GB`}
                        >
                          <div className={`mx-auto max-w-[130px] rounded-lg p-2 border transition-all text-left flex flex-col gap-1 ${badgeCls}`}>
                            <div className="flex justify-between items-center text-[10px] font-bold">
                              <span>{breakdown.total.toFixed(1)} G</span>
                              <span className="font-mono">{loadRatio.toFixed(0)}%</span>
                            </div>
                            
                            {/* Visual mini bar */}
                            <div className="w-full bg-black/5 rounded-full h-1 overflow-hidden">
                              <div 
                                className={`h-full rounded-full ${
                                  status === 'oom' ? 'bg-rose-500' : status === 'tight' ? 'bg-amber-500' : 'bg-emerald-500'
                                }`}
                                style={{ width: `${Math.min(100, loadRatio)}%` }}
                              />
                            </div>
                            
                            <div className="text-[9px] scale-95 origin-left tracking-tight opacity-90 truncate">
                              {statusText}
                            </div>
                          </div>
                        </td>
                      );
                    })}

                  </tr>
                );
              })}
            </tbody>

          </table>
        </div>

        {/* Bottom Legend details */}
        <div className="bg-slate-50 border-t border-slate-200 px-4 py-3 text-[11px] text-slate-500 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex flex-wrap gap-4">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
              <span>绿色等级 (≤83%): 显存宽绰，首选推荐部署。</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
              <span>黄色等级 (83% ~ 98%): 资源贴顶。需降低并发，或使用 FP8 KV-Cache。</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
              <span>红色等级 (＞98%): 一定发生 PyTorch OOM 溢出报错崩溃。</span>
            </span>
          </div>

          <span className="font-semibold text-slate-800 flex items-center gap-1">
            <BookOpen className="w-3.5 h-3.5 text-indigo-600" />
            快速部署指南：选择绿色，并复制下方的高性能启动命令
          </span>
        </div>

      </div>

    </div>
  );
};
