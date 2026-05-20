/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { PrecisionDetails, PrecisionMode } from '../types';
import { PRECISION_OPTS, COMPONENT_IDS } from '../data';
import { ShieldCheck, Sliders, Info } from 'lucide-react';

interface PrecisionSelectorProps {
  selectedPrecision: PrecisionDetails;
  onPrecisionChange: (precision: PrecisionDetails) => void;
  customBits: number;
  onCustomBitsChange: (bits: number) => void;
}

export const PrecisionSelector: React.FC<PrecisionSelectorProps> = ({
  selectedPrecision,
  onPrecisionChange,
  customBits,
  onCustomBitsChange,
}) => {
  const handlePrecisionSelect = (opt: PrecisionDetails) => {
    if (opt.id === PrecisionMode.CUSTOM) {
      onPrecisionChange({
        ...opt,
        bitsPerWeight: customBits,
      });
    } else {
      onPrecisionChange(opt);
    }
  };

  const handleCustomBitsSlider = (val: number) => {
    onCustomBitsChange(val);
    if (selectedPrecision.id === PrecisionMode.CUSTOM) {
      onPrecisionChange({
        ...selectedPrecision,
        bitsPerWeight: val,
      });
    }
  };

  return (
    <div id={COMPONENT_IDS.PRECISION} className="bg-white rounded-2xl border border-slate-100 p-6 shadow-sm flex flex-col gap-6">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <ShieldCheck className="w-5 h-5 text-indigo-600" />
          <h2 className="text-lg font-semibold text-slate-900 tracking-tight">
            2. 数据精度与运行量化 / Precision & Quantization
          </h2>
        </div>
        <p className="text-sm text-slate-500">
          设定模型权重的精度位数。更低的量化位数能显著减少显存体积，但会引入微小的困惑度(Loss)偏差。
        </p>
      </div>

      {/* Dynamic Quantization Status and Advisor */}
      <div className="bg-gradient-to-r from-slate-50 to-indigo-50/10 border border-indigo-100 rounded-xl p-4.5 flex flex-col gap-2.5">
        <span className="text-[11px] font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
          <Info className="w-4 h-4 text-indigo-600" />
          💡 物理权重基座与运行量化关系说明 (Quantization Design Concept)
        </span>
        <div className="text-xs text-slate-600 leading-relaxed flex flex-col gap-1.5 font-sans">
          <p>
            您的理解完全正确！模型的<strong>原始开发/出厂物理权重精度是固定的</strong>（例如 Meta LLaMA 3.1 默认权重精度为 <strong>FP16 (16位)</strong>，而 DeepSeek R1/V3 出厂精度默认为高效原生 <strong>FP8 (8位)</strong>）。
          </p>
          <p className="border-t border-indigo-100/30 pt-2 text-indigo-800 font-medium">
            👇 下方的精度菜单支持<strong>动态调优量化</strong>。改变此项等于对其进行 <strong>INT4 / FP4 / GGUF</strong> 等级的高阶量化压缩模拟。此机制可在极低精度损伤下，<strong>压缩 50% 到 75% 的显存占用</strong>，是本地低配显卡顺畅运行高参数量旗舰模型（如 70B 装入家用 24G 显卡）的唯一手段。
          </p>
        </div>
      </div>

      {/* Grid selector for Precision Mode */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {PRECISION_OPTS.map((opt) => {
          const isSelected = selectedPrecision.id === opt.id;
          return (
            <button
              key={opt.id}
              id={`precision-opt-${opt.id}`}
              type="button"
              onClick={() => handlePrecisionSelect(opt)}
              className={`flex flex-col text-left p-4 rounded-xl border transition-all duration-150 relative overflow-hidden ${
                isSelected
                  ? 'border-indigo-600 bg-indigo-50/40 ring-2 ring-indigo-600/10'
                  : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50/50'
              }`}
            >
              <div className="flex justify-between items-start gap-1 w-full">
                <span className="font-semibold text-slate-900 text-sm">{opt.name}</span>
                <span className="text-xs font-mono font-bold text-indigo-700 bg-indigo-50 shrink-0 px-2 py-0.5 rounded">
                  {opt.id === PrecisionMode.CUSTOM ? `${customBits.toFixed(1)} Bit` : `${opt.bitsPerWeight} Bit`}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-2 flex-grow">
                {opt.description}
              </p>
              {opt.overheadFactor > 1.0 && (
                <div className="mt-2 text-[10px] text-amber-700 font-mono bg-amber-50 rounded border border-amber-100/35 px-1.5 py-0.5 self-start">
                  量化开销 / Quant Overhead: +{Math.round((opt.overheadFactor - 1) * 100)}%
                </div>
              )}
            </button>
          );
        })}
      </div>

      {/* Custom Bits Options (Slider) */}
      {selectedPrecision.id === PrecisionMode.CUSTOM && (
        <div className="bg-slate-50/50 rounded-xl p-5 border border-slate-100 flex flex-col gap-4">
          <div className="flex justify-between items-center">
            <label className="text-xs font-semibold text-slate-600 flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-indigo-600" />
              自定义位宽 / Custom Bit Rate
            </label>
            <span className="font-mono text-sm font-bold text-indigo-700">
              {customBits.toFixed(2)} bits / parameter
            </span>
          </div>
          
          <input
            id="custom-bits-slider"
            type="range"
            min="1.5"
            max="16.0"
            step="0.05"
            value={customBits}
            onChange={(e) => handleCustomBitsSlider(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
          />

          <div className="flex justify-between text-[11px] text-slate-400 font-mono">
            <span>1.5b (Extreme AWQ)</span>
            <span>4.0b (INT4)</span>
            <span>6.0b (GGUF Q6)</span>
            <span>8.0b (INT8/FP8)</span>
            <span>16.0b (FP16/BF16)</span>
          </div>

          <p className="text-xs text-slate-500 bg-indigo-50/50 px-3 py-2 rounded-lg border border-indigo-100/50 flex items-start gap-2">
            <Info className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
            <span>
              <strong>自定义系数估算机制:</strong> 当您将位宽调至如 2.5 或 5.5 时，计算器会自动适配并预估带有相应标定参数(scale factors)与权重块分布(chunk weights)的对应 VRAM 数额。
            </span>
          </p>
        </div>
      )}
    </div>
  );
};
