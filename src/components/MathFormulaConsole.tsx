/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { ModelPreset, PrecisionDetails, CalcMode, InferenceConfig, TrainingConfig, VRAMBreakdown, ConcurrencyEstimate, GPUType } from '../types';
import { HelpCircle, ChevronRight, Calculator, X, Sparkles, BookOpen, Layers, Settings, ShieldCheck, Database, Zap, Users } from 'lucide-react';

interface MathFormulaConsoleProps {
  selectedModel: ModelPreset;
  selectedPrecision: PrecisionDetails;
  selectedMode: CalcMode;
  inferenceConfig: InferenceConfig;
  trainingConfig: TrainingConfig;
  useMLACompression: boolean;
  isDeepSeekModel: boolean;
  vramBreakdown: VRAMBreakdown;
  selectedGPU: GPUType;
  concurrencyEstimate: ConcurrencyEstimate | null;
}

export const MathFormulaConsole: React.FC<MathFormulaConsoleProps> = ({
  selectedModel,
  selectedPrecision,
  selectedMode,
  inferenceConfig,
  trainingConfig,
  useMLACompression,
  isDeepSeekModel,
  vramBreakdown,
  selectedGPU,
  concurrencyEstimate
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<'weights' | 'kv' | 'activation' | 'concurrency' | 'training'>('weights');

  // Lock body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      const originalOverflow = document.body.style.overflow;
      document.body.style.overflow = 'hidden';
      return () => {
        document.body.style.overflow = originalOverflow;
      };
    }
  }, [isOpen]);

  // Close modal on Escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsOpen(false);
    };
    if (isOpen) {
      window.addEventListener('keydown', handleEscape);
      return () => window.removeEventListener('keydown', handleEscape);
    }
  }, [isOpen]);

  // Math intermediate variables calculation
  const tp = selectedMode === 'inference' ? inferenceConfig.tensorParallelism : trainingConfig.tensorParallelism;
  const bpw = selectedPrecision.bitsPerWeight;
  const overheadFactor = selectedPrecision.overheadFactor;
  const params = selectedModel.totalParams;
  const layers = selectedModel.numLayers;
  const heads = selectedModel.numHeads;
  const kvHeads = selectedModel.numKVHeads || selectedModel.numHeads;
  const hiddenSize = selectedModel.hiddenSize;
  const d_head = heads > 0 ? (hiddenSize / heads) : 128;

  // Inference state helpers
  const batch = selectedMode === 'inference' ? inferenceConfig.batchSize : trainingConfig.batchSize;
  const seqLen = selectedMode === 'inference' ? inferenceConfig.sequenceLength : trainingConfig.sequenceLength;
  const kvPrecision = selectedMode === 'inference' ? inferenceConfig.kvCachePrecision : 'fp16';
  const kvBytes = kvPrecision === 'fp16' ? 2 : (kvPrecision === 'none' ? 0 : 1);

  // Training state helpers
  const trainPercent = trainingConfig.trainableParamsPercent;
  const trainableParams = params * (trainPercent / 100);
  const gradBytes = trainingConfig.precision === 'fp32' ? 4 : 2;
  const optType = trainingConfig.optimizer;
  
  let optBytes = 0;
  if (optType === 'adamw') optBytes = 12; // first moment + second moment + FP32 copy
  else if (optType === 'adamw_8bit') optBytes = 6;
  else if (optType === 'sgd') optBytes = 8;

  return (
    <div className="w-full">
      {/* Visual Trigger Button */}
      <button
        type="button"
        onClick={() => setIsOpen(true)}
        className="w-full bg-indigo-50 hover:bg-indigo-100 text-indigo-700 hover:text-indigo-800 border border-indigo-100 py-3 px-4 rounded-xl flex items-center justify-between transition-all font-semibold text-xs shadow-sm hover:cursor-pointer"
      >
        <span className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-indigo-600 animate-pulse" />
          <span>🔍 点击查看：显存占比数学推导公式与实时数值代入演算 / Math Derivations</span>
        </span>
        <span className="flex items-center gap-1 font-mono text-[10px] bg-indigo-600 text-white px-2 py-0.5 rounded-full">
          Live Math {`>`}
        </span>
      </button>

      {/* Overlay modal container — rendered via portal to escape parent stacking contexts */}
      {isOpen && createPortal(
        <div
          className="fixed inset-0 z-50 overflow-y-auto bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setIsOpen(false);
            }
          }}
        >
          <div className="bg-white rounded-2xl max-w-4xl w-full border border-slate-200 shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">

            {/* Modal Header */}
            <div className="bg-slate-50 border-b border-slate-150 p-5 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <Calculator className="w-5 h-5 text-indigo-600" />
                <div>
                  <h3 className="font-bold text-slate-900 text-base leading-tight">
                    大模型显存极客计算引擎 📐 数学推导与公式对照
                  </h3>
                  <p className="text-xs text-slate-500 font-serif mt-0.5">
                    Mathematical details representing LLM footprint algorithms dynamically parsed in real-time.
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setIsOpen(false)}
                className="p-1 px-2 text-slate-400 hover:text-slate-600 rounded bg-white border border-slate-200 text-xs font-semibold"
              >
                ✕ 关闭 (Close)
              </button>
            </div>

            {/* Modal Navigation Tabs */}
            <div className="flex border-b border-slate-150 overflow-x-auto bg-slate-50/50 p-1.5 gap-1.5 scrollbar-thin">
              <button
                type="button"
                onClick={() => setActiveTab('weights')}
                className={`px-4 py-2 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 shrink-0 ${
                  activeTab === 'weights' ? 'bg-white border text-indigo-700 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Database className="w-3.5 h-3.5" />
                1. 静态模型权重公式 (Model Weights)
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('kv')}
                className={`px-4 py-2 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 shrink-0 ${
                  activeTab === 'kv' ? 'bg-white border text-indigo-700 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                2. Key-Value 缓存公式 (KV Cache)
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('activation')}
                className={`px-4 py-2 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 shrink-0 ${
                  activeTab === 'activation' ? 'bg-white border text-indigo-700 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Zap className="w-3.5 h-3.5" />
                3. 动态激活值公式 (Activations)
              </button>
              {selectedMode === 'inference' && (
                <button
                  type="button"
                  onClick={() => setActiveTab('concurrency')}
                  className={`px-4 py-2 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 shrink-0 ${
                    activeTab === 'concurrency' ? 'bg-white border text-indigo-700 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <Users className="w-3.5 h-3.5" />
                  4. 反向并发估算 (Concurrency)
                </button>
              )}
              {selectedMode === 'training' && (
                <button
                  type="button"
                  onClick={() => setActiveTab('training')}
                  className={`px-4 py-2 text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 shrink-0 ${
                    activeTab === 'training' ? 'bg-white border text-indigo-700 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <Sparkles className="w-3.5 h-3.5 animate-pulse text-indigo-600" />
                  4. 微调训练梯度与优化器状态 (Optimizer & Grads)
                </button>
              )}
            </div>

            {/* Modal Core Area */}
            <div className="p-6 overflow-y-auto flex-1 flex flex-col gap-6 font-sans">
              
              {/* Active Tab Body render */}
              {activeTab === 'weights' && (
                <div className="flex flex-col gap-5 animate-in fade-in duration-200">
                  <div className="bg-slate-50 border-l-4 border-indigo-500 rounded-r-lg p-4">
                    <h4 className="text-xs uppercase tracking-wider font-bold text-indigo-950 mb-1">
                      公式：静态权重加载 VRAM 占用
                    </h4>
                    <p className="text-xs text-slate-600 leading-relaxed">
                      深度学习模型加载时，参数矩阵中的每个标量元素均需分配底层物理显存。权重显存取决于模型参数总量、采用的数据类型所对应的每个参数位宽 (Bits Per Weight, BPW)，以及为了保证吞吐速度和缩放对齐产生的开销溢出（如 GGUF 的元数据或是 INT8 / INT4 离散矩阵中的常数常项系数矩阵）。
                    </p>
                  </div>

                  {/* Math Render */}
                  <div className="bg-slate-900 rounded-xl p-5 text-center shadow-inner font-mono text-white select-all border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase tracking-widest mb-2 font-bold">Standard Sizing Equation</div>
                    <div className="text-lg md:text-xl font-semibold tracking-wide py-1 text-sky-300">
                      VRAM_weights = (Params_total × Bits_per_param / 8) × K_overhead / TP
                    </div>
                    <div className="text-[10px] text-slate-400 mt-2">
                      单位：GB | Total Params 以 B (10亿级) 为计算单位
                    </div>
                  </div>

                  {/* Variable breakdown */}
                  <div>
                    <h5 className="text-xs font-bold text-slate-800 mb-2">🏷️ 参数定义與取值参考 Legend:</h5>
                    <div className="border border-slate-150 rounded-lg overflow-hidden text-xs">
                      <table className="w-full text-left border-collapse">
                        <thead>
                          <tr className="bg-slate-50 border-b border-slate-150 text-slate-400 font-bold uppercase text-[10px]">
                            <th className="p-2.5">变量 / Variable</th>
                            <th className="p-2.5">核心含义 / Definition</th>
                            <th className="p-2.5">当前带入数值 / Current Live Value</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 text-slate-600">
                          <tr>
                            <td className="p-2.5 font-bold font-mono text-indigo-650">Params_total</td>
                            <td className="p-2.5">模型全部静态参数总量（不管是 Dense 模型还是 MoE 模型，全部参数都会先装载驻留至显存中）</td>
                            <td className="p-2.5 font-bold font-mono text-slate-800">{params} B 参数</td>
                          </tr>
                          <tr>
                            <td className="p-2.5 font-bold font-mono text-indigo-650">Bits_per_param</td>
                            <td className="p-2.5">精度选择位宽。例如 FP16/BF16 为 16；INT8 为 8；INT4 为 4B</td>
                            <td className="p-2.5 font-bold font-mono text-slate-800">{bpw} bits</td>
                          </tr>
                          <tr>
                            <td className="p-2.5 font-bold font-mono text-indigo-650">K_overhead</td>
                            <td className="p-2.5">量化溢出比率。低于 16 位精度下需记录 Scale 值或 zero_points，GGUF 有包封装信息（通常在 1.01 ~ 1.08 之间）</td>
                            <td className="p-2.5 font-bold font-mono text-slate-800">{overheadFactor} x</td>
                          </tr>
                          <tr>
                            <td className="p-2.5 font-bold font-mono text-indigo-650">TP (Tensor Parallelism)</td>
                            <td className="p-2.5">张量并行度（显卡总片数）。利用并行的 All-Reduce 总线在 GPU 层切分切片矩阵。</td>
                            <td className="p-2.5 font-bold font-mono text-slate-800">{tp} 片 GPU 分摊</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Step-by-step Execution Check */}
                  <div className="bg-sky-50/70 border border-sky-100 rounded-xl p-4.5">
                    <h5 className="text-xs font-bold text-sky-950 flex items-center gap-1 mb-2">
                      <Calculator className="w-4 h-4 text-sky-600" />
                      当前实际配置代入演算 (Live Math Run):
                    </h5>
                    <div className="font-mono text-xs text-slate-700 space-y-1.5">
                      <div>1. 首先将参数转换为 Bytes: <span className="bg-white px-2 py-0.5 rounded border border-sky-100 font-semibold">{params} B × ({bpw} bits / 8) = {((params * bpw) / 8).toFixed(2)} GB</span></div>
                      <div>2. 考虑量化常数包和数据对齐折损开销: <span className="bg-white px-2 py-0.5 rounded border border-sky-100 font-semibold">{((params * bpw) / 8).toFixed(2)} GB × {overheadFactor} = {((params * bpw * overheadFactor) / 8).toFixed(2)} GB</span></div>
                      <div>3. 在 TPU/GPU 张量集群分摊: <span className="bg-white px-2 py-0.5 rounded border border-sky-100 font-semibold">{((params * bpw * overheadFactor) / 8).toFixed(2)} GB / {tp} = <strong className="text-indigo-700">{vramBreakdown.modelWeights} GB</strong></span></div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'kv' && (
                <div className="flex flex-col gap-5 animate-in fade-in duration-200">
                  <div className="bg-slate-50 border-l-4 border-emerald-500 rounded-r-lg p-4">
                    <h4 className="text-xs uppercase tracking-wider font-bold text-emerald-950 mb-1">
                      公式：KV Cache 缓存显存开销
                    </h4>
                    <p className="text-xs text-slate-600 leading-relaxed">
                      模型自回归推理期间，为了避免多轮对话上下文中历史 Token 的 Key 和 Value 被反复进行高算力投影计算，系统会在物理显存中一直维持这些被计算过的 Attention 状态。KV 缓存随并发 Batch 以及上下文历史 Sequence 全景呈<strong>极其致命的线性飙升</strong>，是阻碍 LLM 突破十万级长文本的核心瓶颈。
                    </p>
                  </div>

                  {/* Standard GQA Formula */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-slate-900 rounded-xl p-4 text-center border border-slate-800">
                      <div className="text-[9px] text-slate-400 uppercase tracking-widest mb-1.5 font-bold font-mono">
                        标准 MHA / GQA / MQA 架构模型
                      </div>
                      <div className="font-mono text-xs md:text-sm text-emerald-400 py-1 font-semibold">
                        V_kv = (2 × L × H_kv × D_head × B × S × P_bytes) / 10^9 / TP
                      </div>
                      <div className="text-[10px] text-slate-500 mt-2 font-serif text-left p-1 border-t border-slate-800">
                        * GQA (如 Llama3) 仅包含 8 个 KV 对，比 MHA 节约高达 80% 到 85%!
                      </div>
                    </div>

                    <div className="bg-slate-900 rounded-xl p-4 text-center border border-slate-800">
                      <div className="text-[9px] text-sky-400 uppercase tracking-widest mb-1.5 font-bold font-mono">
                        DeepSeek 专有：MLA 潜向量压缩机制
                      </div>
                      <div className="font-mono text-xs md:text-sm text-sky-400 py-1 font-semibold">
                        V_kv_mla = (2 × L × 4.5 × 128 × B × S × P_bytes) / 10^9 / TP
                      </div>
                      <div className="text-[10px] text-slate-500 mt-2 font-serif text-left p-1 border-t border-slate-800">
                        * MLA（Multi-head Latent Attention）将 KV 映射投影至仅有 512 维的低秩空间中存储。等效为常规 4.5 个 KV 头！
                      </div>
                    </div>
                  </div>

                  {/* Variable Table */}
                  <div>
                    <h5 className="text-xs font-bold text-slate-800 mb-2">🏷️ 参数定义與取值参考 Legend:</h5>
                    <div className="border border-slate-150 rounded-lg overflow-hidden text-xs">
                      <table className="w-full text-left border-collapse">
                        <thead>
                          <tr className="bg-slate-50 border-b border-slate-150 text-slate-400 font-bold uppercase text-[10px]">
                            <th className="p-2.5">变量 / Variable</th>
                            <th className="p-2.5">核心含义 / Definition</th>
                            <th className="p-2.5">当前带入数值 / Current Live Value</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 text-slate-600">
                          <tr>
                            <td className="p-2.5 font-bold font-mono text-indigo-650">L (numLayers)</td>
                            <td className="p-2.5">模型的网络层数 (Transformer Layers)</td>
                            <td className="p-2.5 font-bold font-mono text-slate-800">{layers} 层</td>
                          </tr>
                          <tr>
                            <td className="p-2.5 font-bold font-mono text-indigo-650">H_kv (numKVHeads)</td>
                            <td className="p-2.5">KV头数。经典MHA等于注意力总头数（{heads}）；GQA 中缩减为总头数的分数（典型为 8）</td>
                            <td className="p-2.5 font-bold font-mono text-slate-800">{kvHeads} 头</td>
                          </tr>
                          <tr>
                            <td className="p-2.5 font-bold font-mono text-indigo-650">D_head (headDim)</td>
                            <td className="p-2.5">注意力的单头维度特征向量。通常 D_head = hiddenSize / numHeads</td>
                            <td className="p-2.5 font-bold font-mono text-slate-800">{d_head} 维度</td>
                          </tr>
                          <tr>
                            <td className="p-2.5 font-bold font-mono text-indigo-650">B × S</td>
                            <td className="p-2.5">并发批量 Batch Size × 最大上下文历史或单轮生成文本长度 Sequence Length</td>
                            <td className="p-2.5 font-bold font-mono text-slate-800">{batch} × {seqLen} tokens</td>
                          </tr>
                          <tr>
                            <td className="p-2.5 font-bold font-mono text-indigo-650">P_bytes</td>
                            <td className="p-2.5">KV 精度字节开销。默认 16 位下为 2 字节；FP8 或 INT8 缓存量化压缩后变 1 字节</td>
                            <td className="p-2.5 font-bold font-mono text-slate-800">{kvBytes} 字节</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Live KV Calculation Display */}
                  <div className="bg-emerald-50/70 border border-emerald-100 rounded-xl p-4.5">
                    <h5 className="text-xs font-bold text-emerald-950 flex items-center gap-1 mb-2">
                      <Calculator className="w-4 h-4 text-emerald-600" />
                      当前实际配置代入演算 (Live Math Run):
                    </h5>
                    <div className="font-mono text-xs text-slate-700 space-y-1.5">
                      {isDeepSeekModel && useMLACompression ? (
                        <>
                          <div className="text-[11px] text-sky-800 font-semibold mb-1">
                            ✨ 检测到正运行 DeepSeek MLA 并开启压缩优化。将采用 MLA 计算支线：
                          </div>
                          <div>1. 计算单个 Layer 保存压缩对齐后的 Token 位长: 
                            <span className="bg-white px-2 py-0.5 rounded border border-emerald-150 font-semibold">2 × 4.5 头 × 128 维 = 1152 维</span>
                          </div>
                          <div>2. 代入并发和历史文本矩阵序列大小: 
                            <span className="bg-white px-2 py-0.5 rounded border border-emerald-150 font-semibold">1152 维 × {batch} (B) × {seqLen} (S) × {kvBytes} 字节 = {1152 * batch * seqLen * kvBytes} 字节</span>
                          </div>
                          <div>3. 累乘模型 {layers} 层参数后除以多卡 TP {tp} 切片: 
                            <span className="bg-white px-2 py-0.5 rounded border border-emerald-150 font-semibold">({layers} 层 × {1152 * batch * seqLen * kvBytes} 字节) / 10^9 / {tp} = <strong className="text-indigo-700">{vramBreakdown.kvCache} GB</strong></span>
                          </div>
                        </>
                      ) : (
                        <>
                          <div>1. 计算单 Layer 内全部 KV 头在当前最大并发下的状态要素数量: 
                            <span className="bg-white px-2 py-0.5 rounded border border-emerald-150 font-semibold">2 (Key与Value) × {layers} 层 × {kvHeads} 头 × {d_head} 维度 = {2 * layers * kvHeads * d_head} 神经参数元</span>
                          </div>
                          <div>2. 考虑当前的推理总吞吐 Token 空间: 
                            <span className="bg-white px-2 py-0.5 rounded border border-emerald-150 font-semibold">{2 * layers * kvHeads * d_head} × {batch} 并发 × {seqLen} 长度 = {(2 * layers * kvHeads * d_head * batch * seqLen).toLocaleString()} floats</span>
                          </div>
                          <div>3. 乘以当前 KV 存储单点字节宽 {kvBytes} 字节并在 {tp} 片集群中分摊: 
                            <span className="bg-white px-2 py-0.5 rounded border border-emerald-150 font-semibold">({(2 * layers * kvHeads * d_head * batch * seqLen).toLocaleString()} × {kvBytes}) / 10^9 / {tp} = <strong className="text-indigo-700">{vramBreakdown.kvCache} GB</strong></span>
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'activation' && (
                <div className="flex flex-col gap-5 animate-in fade-in duration-200">
                  <div className="bg-slate-50 border-l-4 border-amber-500 rounded-r-lg p-4">
                    <h4 className="text-xs uppercase tracking-wider font-bold text-amber-950 mb-1">
                      公式：动态激活值显存 (Activation VRAM)
                    </h4>
                    <p className="text-xs text-slate-600 leading-relaxed">
                      激活值显存是指前向传播中用于存储各层中间神经元激活输出状态以便在反向传播（Backpropagation）中计算梯度的显存占用。它只在训练和推理的前向运算活动区间存在（生命周期较短），但开销大小高度依赖 <strong>批大小 (Batch)、文本长度 (Sequence Length)、网络通道宽度 (Hidden Size) </strong> 以及是否对残差、多层感知机（MLP）和 Attention 开辟了梯度投影。
                    </p>
                  </div>

                  {/* Formula Render (Inference vs Training selective) */}
                  <div className="bg-slate-900 rounded-xl p-5 text-center shadow-inner font-mono text-white border border-slate-800">
                    {selectedMode === 'inference' ? (
                      <>
                        <div className="text-[10px] text-slate-400 uppercase tracking-widest mb-2 font-bold">Inference Activations Approximation</div>
                        <div className="text-base md:text-lg font-semibold tracking-wide py-1 text-amber-300">
                          V_act ≈ (Hidden_size × Batch × S_active × 2 × 15%) / 10^9
                        </div>
                        <div className="text-[10px] text-slate-400 mt-2 text-left space-y-1.5 border-t border-slate-800 pt-3">
                          <div>• <strong>当前评估激活限制 active_chunk:</strong> {inferenceConfig.chunkPrefillSize !== 'off' ? `Math.min(SeqLen(${inferenceConfig.sequenceLength}), ChunkSize(${inferenceConfig.chunkPrefillSize}))` : `全序列(${inferenceConfig.sequenceLength})`} = {inferenceConfig.chunkPrefillSize !== 'off' ? Math.min(inferenceConfig.sequenceLength, Number(inferenceConfig.chunkPrefillSize)) : inferenceConfig.sequenceLength} tkn</div>
                          <div>• <strong>代入算式进行数值演算:</strong> ({hiddenSize} dim × {batch} B × {inferenceConfig.chunkPrefillSize !== 'off' ? Math.min(inferenceConfig.sequenceLength, Number(inferenceConfig.chunkPrefillSize)) : inferenceConfig.sequenceLength} active_tkn × 2) / 10^9 × 0.15 = <strong className="text-amber-400">{vramBreakdown.activationMemory} GB</strong></div>
                          <div className="text-slate-500 font-sans mt-1 text-[9px] leading-relaxed">
                            * 注意：未启用分块预填充且运行超长上下文时，前向注意力层计算矩阵尺寸产生的暂存中间激活 VRAM 会发生剧烈峰值暴增；分块预填充 (Chunked Prefill) 则能直接将峰值锁定在预填分块限制的大小！
                          </div>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="text-[10px] text-slate-400 uppercase tracking-widest mb-1 font-bold">Standard Activation Checkpointing (AC) Modeling</div>
                        <div className="grid grid-cols-1 gap-2.5 text-left text-xs text-slate-300 p-2 border border-slate-800 rounded bg-black/30 font-mono mt-1">
                          <div>• AC = OFF (全量保存中间激活梯度，显存压力极大):</div>
                          <div className="text-sky-300 pl-3">V_act ≈ (L × B × S × H × 34) / 10^9 / sqrt(TP)</div>
                          <div>• AC = SELECTIVE (注意力权重单独重算，其余保留):</div>
                          <div className="text-amber-300 pl-3">V_act ≈ ((L × B × S × H × 12) + (2 × L × B × S^2 × Heads)) / 10^9 / sqrt(TP)</div>
                          <div>• AC = FULL (完全不存中间结果，所有层反向时全部重算，牺牲 33% 算力保显存):</div>
                          <div className="text-emerald-300 pl-3">V_act ≈ (L × B × S × H × 4.4) / 10^9 / sqrt(TP)</div>
                        </div>
                      </>
                    )}
                  </div>

                  {/* Summary Callout explaining standard engineering parameters */}
                  <div className="border border-amber-100 bg-amber-50/40 rounded-xl p-4.5 text-xs text-amber-900 flex gap-2">
                    <HelpCircle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
                    <div>
                      <h5 className="font-bold mb-1">💡 工业界主流架构常识：</h5>
                      <ul className="list-disc pl-5 space-y-1.5 leading-relaxed text-slate-650">
                        <li><strong>不开启重算 (AC = OFF):</strong> 运行反向传播时，每层网络必须原地开辟约 34 个矩阵单位（17 矩阵的前向状态 + 17 矩阵的反向锁死缓存）保存前向结果。显存随层数和文本呈天文数字上升。</li>
                        <li><strong>开启全重算 (AC = FULL):</strong> 只保留当前微调中正活动的那一两层的前向激活值。等到反向求导时，模型会临时重新跑一遍前向流程把这些数据现场算出来。非常大幅降低显存占用，但系统算力开销将不可避免地平白产生 30%~33% 的阻折，这就是<strong>以计算时间换存储空间</strong>的精髓。</li>
                      </ul>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'concurrency' && selectedMode === 'inference' && concurrencyEstimate && (
                <div className="flex flex-col gap-5 animate-in fade-in duration-200">
                  <div className="bg-slate-50 border-l-4 border-purple-500 rounded-r-lg p-4">
                    <h4 className="text-xs uppercase tracking-wider font-bold text-purple-950 mb-1">
                      公式：反向并发估算 (Reverse Concurrency)
                    </h4>
                    <p className="text-xs text-slate-600 leading-relaxed">
                      与正向“给定 batch/seq 算显存”不同，反向并发从引擎实际行为出发：引擎先按 <code>--mem-fraction-static</code> / <code>--gpu-memory-utilization</code> 圈定可用显存池，加载静态权重并预留系统开销后，剩余空间全部作为 KV Cache 蓄水池。用单 Token KV 占用除以蓄水池容量，再按平均每请求 Token 数分摊，即可得到理论最大并发。
                    </p>
                  </div>

                  <div className="bg-slate-900 rounded-xl p-5 text-center shadow-inner font-mono text-white border border-slate-800">
                    <div className="text-[10px] text-slate-400 uppercase tracking-widest mb-2 font-bold">Reverse Concurrency Equation</div>
                    <div className="text-base md:text-lg font-semibold tracking-wide py-1 text-purple-300">
                      KV_Pool = GPU_VRAM × memoryFraction − weight − overhead
                    </div>
                    <div className="text-base md:text-lg font-semibold tracking-wide py-1 text-purple-300 mt-2">
                      Max_Tokens = floor(KV_Pool / perTokenKV)
                    </div>
                    <div className="text-base md:text-lg font-semibold tracking-wide py-1 text-purple-300 mt-2">
                      Max_Concurrent = floor(Max_Tokens / avgTokensPerRequest)
                    </div>
                  </div>

                  <div className="bg-purple-50/70 border border-purple-100 rounded-xl p-4.5">
                    <h5 className="text-xs font-bold text-purple-950 flex items-center gap-1 mb-2">
                      <Calculator className="w-4 h-4 text-purple-600" />
                      当前实际配置代入演算 (Live Math Run):
                    </h5>
                    <div className="font-mono text-xs text-slate-700 space-y-1.5">
                      <div>1. 单卡可用显存池: <span className="bg-white px-2 py-0.5 rounded border border-purple-100 font-semibold">{selectedGPU?.vram || inferenceConfig.tensorParallelism} GB × {inferenceConfig.memoryFraction} = {((selectedGPU?.vram || 0) * inferenceConfig.memoryFraction).toFixed(2)} GB</span></div>
                      <div>2. 扣除静态权重与开销后 KV 池: <span className="bg-white px-2 py-0.5 rounded border border-purple-100 font-semibold">{concurrencyEstimate.kvPoolPerGPU_GB.toFixed(2)} GB</span></div>
                      <div>3. 单 Token KV 占用 (已按 TP={tp} 切分): <span className="bg-white px-2 py-0.5 rounded border border-purple-100 font-semibold">{(concurrencyEstimate.perTokenKV_GB * 1024).toFixed(2)} KB</span></div>
                      <div>4. 最大容纳 Token 数: <span className="bg-white px-2 py-0.5 rounded border border-purple-100 font-semibold">{concurrencyEstimate.maxTokensTotal.toLocaleString()} tokens</span></div>
                      <div>5. 按每请求 {inferenceConfig.avgTokensPerRequest.toLocaleString()} Token 计算，最大并发: <span className="bg-white px-2 py-0.5 rounded border border-purple-100 font-semibold"><strong className="text-purple-700">{concurrencyEstimate.maxConcurrentRequests.toLocaleString()} 请求</strong></span></div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'training' && selectedMode === 'training' && (
                <div className="flex flex-col gap-5 animate-in fade-in duration-200">
                  <div className="bg-slate-50 border-l-4 border-indigo-500 rounded-r-lg p-4">
                    <h4 className="text-xs uppercase tracking-wider font-bold text-indigo-950 mb-1">
                      公式：微调训练状态显存 (Optimizer & Grads Math)
                    </h4>
                    <p className="text-xs text-slate-600 leading-relaxed">
                      模型在进行任何反向梯度更新训练时，底层的加速卡需要针对可训练的权重计算相应的<strong>梯度 (Gradients)</strong> 以及用来动态调整每步数值的<strong>优化器状态 (Optimizer States)</strong> 。
                    </p>
                  </div>

                  {/* Math Grid for Training */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-slate-900 rounded-xl p-4 text-center border border-slate-800 text-white font-mono">
                      <div className="text-[10px] text-sky-300 uppercase tracking-widest mb-1.5 font-bold">
                        1. 梯度显存 / Gradients VRAM
                      </div>
                      <div className="text-sm font-semibold tracking-wide py-1 text-white">
                        V_grad = (P_trainable × 2字节/4字节) / TP
                      </div>
                      <div className="text-[10px] text-slate-500 text-left mt-2 border-t border-slate-800 p-1 font-serif">
                        * Gradients 与训练位宽一致。混合精度下，使用 FP16 或 BF16 梯度（乘 2 字节系数）。
                      </div>
                    </div>

                    <div className="bg-slate-900 rounded-xl p-4 text-center border border-slate-800 text-white font-mono">
                      <div className="text-[10px] text-amber-300 uppercase tracking-widest mb-1.5 font-bold">
                        2. 优化器显存 / Optimizer VRAM (AdamW)
                      </div>
                      <div className="text-sm font-semibold tracking-wide py-1 text-white">
                        V_opt = (P_trainable × K_opt_scaler) / TP
                      </div>
                      <div className="text-[10px] text-slate-500 text-left mt-2 border-t border-slate-800 p-1 font-serif">
                        * FP32 经典 Adam 优化器存储 1 阶和 2 阶动量（共 8 字节）+ fp32 Master Weights（4 字节）= 12GB 每个十亿参数！使用 8-bit AdamW 可降至 6GB（省 50%）。
                      </div>
                    </div>
                  </div>

                  {/* Live Training calculation details */}
                  <div className="bg-indigo-50 border border-indigo-100 rounded-xl p-4.5">
                    <h5 className="text-xs font-bold text-indigo-950 flex items-center gap-1 mb-2 font-serif">
                      <Calculator className="w-4 h-4 text-indigo-650" />
                      当前实际配置代入演算 (Live Math Run)
                    </h5>
                    <div className="font-mono text-xs text-slate-705 space-y-1.5">
                      <div>1. 首先计算单卡上可训练的物理参数量: 
                        <span className="bg-white px-2 py-0.5 rounded border border-indigo-150 font-semibold">{params} B × {trainPercent}% ≈ {trainableParams.toFixed(3)} B 参数量</span>
                      </div>
                      <div>2. 梯度反算开销 (Gradients VRAM): 
                        <span className="bg-white px-2 py-0.5 rounded border border-indigo-150 font-semibold">{trainableParams} B × {gradBytes} 字节 / {tp} = <strong className="text-sky-700">{vramBreakdown.trainingState?.gradients || 0} GB</strong></span>
                      </div>
                      <div>3. 优化器状态开销 (Adam/SGD Optimizer VRAM): 
                        <span className="bg-white px-2 py-0.5 rounded border border-indigo-150 font-semibold">{trainableParams} B × {optBytes} 字节 / {tp} = <strong className="text-amber-700">{vramBreakdown.trainingState?.optimizer || 0} GB</strong></span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

            </div>

            {/* Modal Footer status banner */}
            <div className="bg-slate-50 border-t border-slate-150 p-4 px-6 flex flex-col sm:flex-row items-center justify-between text-[11px] text-slate-500">
              <span className="flex items-center gap-1 font-mono">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
                Live Calculators matched: {selectedModel.name} | Precision: {selectedPrecision.name}
              </span>
              <span className="font-sans text-slate-400 mt-1 sm:mt-0 font-light">
                该计算公式符合 Academic & Meta/DeepSeek 官方算力白皮书精度标准。
              </span>
            </div>

          </div>
        </div>,
        document.body
      )}
    </div>
  );
};
