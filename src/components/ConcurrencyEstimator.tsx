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
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-3 flex flex-col gap-1">
          <span className="text-[10px] text-slate-400 uppercase tracking-wide flex items-center gap-1">
            <Database className="w-3 h-3" /> KV Pool / GPU
          </span>
          <span className="text-sm font-bold text-slate-200">{estimate.kvPoolPerGPU_GB.toFixed(1)} GB</span>
        </div>

        <div className="bg-slate-800 border border-slate-700 rounded-xl p-3 flex flex-col gap-1">
          <span className="text-[10px] text-slate-400 uppercase tracking-wide flex items-center gap-1">
            <Layers className="w-3 h-3" /> Per-Token KV
          </span>
          <span className="text-sm font-bold text-slate-200">{(estimate.perTokenKV_GB * 1024).toFixed(2)} KB</span>
        </div>

        <div className="bg-slate-800 border border-slate-700 rounded-xl p-3 flex flex-col gap-1 col-span-2">
          <span className="text-[10px] text-slate-400 uppercase tracking-wide">最大容纳 Token 数 / Max Total Tokens</span>
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
