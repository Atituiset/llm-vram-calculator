/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { TrainingConfig } from '../types';
import { COMPONENT_IDS } from '../data';
import { HelpCircle, Sliders, ShieldCheck, Percent } from 'lucide-react';

interface TrainingParamsProps {
  config: TrainingConfig;
  onConfigChange: (config: TrainingConfig) => void;
}

export const TrainingParams: React.FC<TrainingParamsProps> = ({
  config,
  onConfigChange,
}) => {
  const handleChange = (key: keyof TrainingConfig, value: number | string) => {
    onConfigChange({
      ...config,
      [key]: value,
    });
  };

  const isLoRA = config.trainableParamsPercent < 100;

  return (
    <div id={COMPONENT_IDS.TRAIN_PARAMS} className="bg-white rounded-2xl border border-slate-100 p-6 shadow-sm flex flex-col gap-6">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <Sliders className="w-5 h-5 text-amber-600" />
          <h2 className="text-lg font-semibold text-slate-900 tracking-tight">
            3. 训练阶段参数配置 / Training Settings
          </h2>
        </div>
        <p className="text-sm text-slate-500">
          估算在开展全参数预训练(Pre-training)、微调(SFT)或低秩自适应(LoRA / QLoRA)时，由于梯度和优化器状态引发的指数级显存膨胀。
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Fine Tuning vs LoRA Config Toggle */}
        <div className="flex flex-col gap-3 md:col-span-2 bg-amber-50/50 rounded-xl p-4 border border-amber-100/50">
          <div className="flex justify-between items-center">
            <label className="font-semibold text-amber-900 text-sm flex items-center gap-1.5 animate-pulse">
              <Percent className="w-4 h-4 text-amber-700" />
              可训练参数占比 / Trainable Parameters (%)
            </label>
            <span className="font-mono bg-white px-2.5 py-1 text-xs border border-amber-200 text-amber-800 font-bold rounded-full shadow-sm">
              {config.trainableParamsPercent === 100 
                ? '全参数微调 / Full Fine-Tuning' 
                : `LoRA/PEFT 局部微调 (${config.trainableParamsPercent}%)`}
            </span>
          </div>
          
          <div className="flex gap-2">
            <button
              id="trainable-params-pct-100"
              type="button"
              onClick={() => handleChange('trainableParamsPercent', 100)}
              className={`flex-1 py-2 px-3 text-center text-xs font-semibold rounded-lg border transition-all ${
                config.trainableParamsPercent === 100
                  ? 'border-amber-600 bg-amber-600 text-white shadow-sm'
                  : 'border-slate-200 hover:border-slate-300 text-slate-600 bg-white'
              }`}
            >
              100% 全参数 / SFT
            </button>
            <button
              id="trainable-params-pct-lora"
              type="button"
              onClick={() => handleChange('trainableParamsPercent', 1.5)}
              className={`flex-1 py-2 px-3 text-center text-xs font-semibold rounded-lg border transition-all ${
                config.trainableParamsPercent < 100
                  ? 'border-amber-600 bg-amber-600 text-white shadow-sm'
                  : 'border-slate-200 hover:border-slate-300 text-slate-600 bg-white'
              }`}
            >
              微量 LoRA (默认 1.5%)
            </button>
          </div>

          {isLoRA && (
            <div className="flex flex-col gap-1.5 mt-2 transition-all duration-300">
              <div className="flex justify-between text-xs">
                <span className="text-slate-600">手动精确设置可训练参数百分比:</span>
                <span className="font-mono font-bold text-amber-700">{config.trainableParamsPercent.toFixed(2)}%</span>
              </div>
              <input
                id="trainable-params-pct-slider"
                type="range"
                min="0.05"
                max="10.0"
                step="0.05"
                value={config.trainableParamsPercent}
                onChange={(e) => handleChange('trainableParamsPercent', parseFloat(e.target.value))}
                className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
              />
              <span className="text-[10px] text-amber-600">
                提示: 在 QLoRA 微调中，基础权重被量化冻结在 4-bit 精度下，仅有外挂的 1.5% - 3% LoRA 权重在 FP32 状态下计算梯度与优化器参数。
              </span>
            </div>
          )}
        </div>

        {/* Optimizer Selection */}
        <div className="flex flex-col gap-2">
          <label className="font-semibold text-slate-700 text-sm flex items-center gap-1.5">
            优化器格式 / Optimizer State
            <HelpCircle className="w-3.5 h-3.5 text-slate-400 hover:text-amber-600 cursor-help" title="优化器用于更新模型参数。每个可训练的权重均需要在后台分配存储来记录梯度矩(momentum)。FP32 Adam (AdamW) 会吃掉不可思议的 8B 显存。8-bit 版本的 Adam 可以大幅降低75%的额外消耗。" />
          </label>
          <div className="grid grid-cols-2 gap-2">
            {[
              { id: 'adamw', name: 'FP32 AdamW (+8B/param)' },
              { id: 'adamw_8bit', name: '8-bit AdamW (+2B/param)' },
              { id: 'sgd', name: 'FP32 SGD (+4B/param)' },
              { id: 'none', name: '梯度更新/冻结 / None (Freeze)' }
            ].map((opt) => (
              <button
                key={opt.id}
                id={`optimizer-opt-${opt.id}`}
                type="button"
                onClick={() => handleChange('optimizer', opt.id)}
                className={`p-2.5 text-left text-xs font-semibold rounded-lg border transition-all flex flex-col justify-center leading-normal ${
                  config.optimizer === opt.id
                    ? 'border-amber-500 bg-amber-50 text-amber-800'
                    : 'border-slate-200 hover:border-slate-300 text-slate-600 bg-white'
                }`}
              >
                <span>{opt.name.split(' (')[0].split(' (+')[0]}</span>
                <span className="text-[10px] font-normal text-slate-400 mt-0.5">
                  {opt.name.includes('+') ? `+${opt.name.split('+')[1]}` : '无状态 / No State'}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Precision Selector */}
        <div className="flex flex-col gap-2">
          <label className="font-semibold text-slate-700 text-sm flex items-center gap-1.5">
            训练混合精度 / Mixed-Precision Training
            <HelpCircle className="w-3.5 h-3.5 text-slate-400 hover:text-amber-600 cursor-help" title="训练通常在16位(BF16/FP16)格式下做权重传递和梯度保留，配以32位(FP32)做优化器计算与主权重(master weights)备份，这称为混合精度训练模式。若不采用则直接进入耗能且低效的FP32纯计算体系。" />
          </label>
          <div className="grid grid-cols-2 gap-2">
            {[
              { id: 'fp16_bf16', name: 'Mixed BF16 / FP16', desc: '权重16b / 梯度16b' },
              { id: 'fp32', name: 'Pure FP32 (Full Scale)', desc: '权重32b / 梯度32b' }
            ].map((p) => (
              <button
                key={p.id}
                id={`train-precision-opt-${p.id}`}
                type="button"
                onChange={() => handleChange('precision', p.id)}
                onClick={() => handleChange('precision', p.id)}
                className={`p-2.5 text-left text-xs font-semibold rounded-lg border transition-all flex flex-col justify-center leading-normal ${
                  config.precision === p.id
                    ? 'border-amber-500 bg-amber-50 text-amber-800'
                    : 'border-slate-200 hover:border-slate-300 text-slate-600 bg-white'
                }`}
              >
                <span>{p.name}</span>
                <span className="text-[10px] font-normal text-slate-400 mt-0.5">
                  {p.desc}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Batch Size (micro-batch size per device) */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center text-sm">
            <label className="font-semibold text-slate-700 flex items-center gap-1.5">
              单卡微批大小 / Micro-Batch Size (b)
              <HelpCircle className="w-3.5 h-3.5 text-slate-400 hover:text-amber-600 cursor-help" title="单卡上运行的局部批量。每次微批处理会存储并跨层堆叠海量激活值(Activations)，是诱发训练阶段开销最大的首要原因。" />
            </label>
            <span className="font-mono bg-slate-100 px-2 py-0.5 rounded text-sm text-slate-700 font-bold">
              {config.batchSize}
            </span>
          </div>
          <input
            id="training-batch-slider"
            type="range"
            min="1"
            max="32"
            step="1"
            value={config.batchSize}
            onChange={(e) => handleChange('batchSize', parseInt(e.target.value))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
          />
          <div className="flex justify-between text-[11px] text-slate-400 font-mono">
            <span>1 (Single Sample)</span>
            <span>4 (Standard)</span>
            <span>16 (High Core)</span>
            <span>32 (Peak Batch)</span>
          </div>
        </div>

        {/* Sequence Length slider */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center text-sm">
            <label className="font-semibold text-slate-700 flex items-center gap-1.5">
              训练上下文跨度 / Train Seq Length (s)
              <HelpCircle className="w-3.5 h-3.5 text-slate-400 hover:text-amber-600 cursor-help" title="每个样本参与回传计算的Token密度。该数字越大，回传时需要暂存对齐的中间偏导数就越多，对激活显存有超凡级别的累加！" />
            </label>
            <span className="font-mono bg-slate-100 px-2 py-0.5 rounded text-sm text-slate-700 font-bold">
              {config.sequenceLength} token
            </span>
          </div>
          <input
            id="training-seq-length-slider"
            type="range"
            min="512"
            max="32768"
            step="512"
            value={config.sequenceLength}
            onChange={(e) => handleChange('sequenceLength', parseInt(e.target.value))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
          />
          <div className="flex justify-between text-[11px] text-slate-400 font-mono">
            <span>512</span>
            <span>2048 (Baseline)</span>
            <span>8192 (Intermediate)</span>
            <span>32768 (Heavy Context)</span>
          </div>
        </div>

        {/* Activation Checkpointing Mode */}
        <div className="flex flex-col gap-2">
          <label className="font-semibold text-slate-700 text-sm flex items-center gap-1.5">
            重卷激活检测 / Activation Checkpointing
            <HelpCircle className="w-3.5 h-3.5 text-slate-400 hover:text-amber-600 cursor-help" title="核心降维算法：训练时不存储所有的中间导回节点，而是只保存核心节点，遇到BP流程时再算一遍。开启全参数重卷(Full Recomputation)能在微小时间代价下，抹除多达70%+的训练激活显卡负荷！" />
          </label>
          <div className="grid grid-cols-3 gap-2">
            {[
              { id: 'full', name: '全参数重算 / Full CP', desc: '显存极大释放' },
              { id: 'selective', name: '高频重算 / Select', desc: '中等重构释放' },
              { id: 'off', name: '完全重计关 / Off', desc: '显存完全溢出' }
            ].map((ac) => (
              <button
                key={ac.id}
                id={`activation-checkpoint-${ac.id}`}
                type="button"
                onClick={() => handleChange('activationCheckpointing', ac.id)}
                className={`p-2.5 text-center text-xs font-semibold rounded-lg border transition-all flex flex-col justify-center leading-normal ${
                  config.activationCheckpointing === ac.id
                    ? 'border-amber-500 bg-amber-50 text-amber-800 shadow-sm'
                    : 'border-slate-200 hover:border-slate-300 text-slate-600 bg-white'
                }`}
              >
                <span>{ac.name.split(' / ')[0]}</span>
                <span className="text-[9px] font-normal text-slate-400 mt-0.5">
                  {ac.desc}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Tensor Parallelism slider in Training */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center text-sm">
            <label className="font-semibold text-slate-700 flex items-center gap-1.5">
              张量并行度 / Tensor Parallelism (TP)
              <HelpCircle className="w-3.5 h-3.5 text-slate-400 hover:text-amber-600 cursor-help" title="训练多卡切割TP。由于层内计算需要跨卡收集(All-Reduce)梯度与权重，模型对通信带宽依赖极高。" />
            </label>
            <span className="font-mono bg-slate-100 px-2 py-0.5 rounded text-sm text-slate-700 font-bold">
              {config.tensorParallelism} GPU(s)
            </span>
          </div>
          <input
            id="training-tp-slider"
            type="range"
            min="1"
            max="16"
            step="1"
            value={config.tensorParallelism}
            onChange={(e) => handleChange('tensorParallelism', parseInt(e.target.value))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
          />
          <div className="flex justify-between text-[11px] text-slate-400 font-mono">
            <span>1 (单卡)</span>
            <span>2</span>
            <span>4</span>
            <span>8 (标准底盘)</span>
            <span>16 (双机组)</span>
          </div>
        </div>

        {/* Pipeline Parallelism slider check */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center text-sm">
            <label className="font-semibold text-slate-700 flex items-center gap-1.5">
              流水线并行度 / Pipeline Parallelism (PP)
              <HelpCircle className="w-3.5 h-3.5 text-slate-400 hover:text-amber-600 cursor-help" title="训练流水线级并行。将模型不同的物理层(Layers)分配至串联的多个GPU组合上层进行计算(显存除以串机GPU数)。" />
            </label>
            <span className="font-mono bg-slate-100 px-2 py-0.5 rounded text-sm text-slate-700 font-bold">
              {config.pipelineParallelism} Stage(s)
            </span>
          </div>
          <input
            id="training-pp-slider"
            type="range"
            min="1"
            max="8"
            step="1"
            value={config.pipelineParallelism}
            onChange={(e) => handleChange('pipelineParallelism', parseInt(e.target.value))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
          />
          <div className="flex justify-between text-[11px] text-slate-400 font-mono">
            <span>1 (无流水线)</span>
            <span>2</span>
            <span>4</span>
            <span>8 (八机流水)</span>
          </div>
        </div>

        {/* System or CUDA Baseline overhead for training */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center text-sm">
            <label className="font-semibold text-slate-700 flex items-center gap-1.5">
              训练基座预留 / reserved CUDA Baseline
              <HelpCircle className="w-3.5 h-3.5 text-slate-400 hover:text-amber-600 cursor-help" title="包括训练底层引擎、通信张量对齐通道NCCL空间、DDP进程通信空间以及各种系统核心堆栈缓存。" />
            </label>
            <span className="font-mono bg-slate-100 px-2 py-0.5 rounded text-sm text-slate-700 font-bold">
              {config.systemOverheadGB.toFixed(1)} GB
            </span>
          </div>
          <input
            id="training-overhead-slider"
            type="range"
            min="0.5"
            max="12.0"
            step="0.5"
            value={config.systemOverheadGB}
            onChange={(e) => handleChange('systemOverheadGB', parseFloat(e.target.value))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
          />
          <div className="flex justify-between text-[11px] text-slate-400 font-mono">
            <span>0.5 GB</span>
            <span>3.0 GB (标准预留)</span>
            <span>6.0 GB (多卡对冲)</span>
            <span>12.0 GB (特大主架)</span>
          </div>
        </div>
      </div>
    </div>
  );
};
