/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useMemo } from 'react';
import { ModelPreset, PrecisionDetails, GPUType, CalcMode, InferenceConfig, TrainingConfig } from './types';
import { MODEL_PRESETS, PRECISION_OPTS, GPU_PRESETS, calculateInferenceVRAM, calculateTrainingVRAM, COMPONENT_IDS } from './data';
import { ModelSelector } from './components/ModelSelector';
import { PrecisionSelector } from './components/PrecisionSelector';
import { InferenceParams } from './components/InferenceParams';
import { TrainingParams } from './components/TrainingParams';
import { VRAMGauge } from './components/VRAMGauge';
import { GPUFitAdvisor } from './components/GPUFitAdvisor';
import { DeploymentScriptGenerator } from './components/DeploymentScriptGenerator';
import { EnvironmentFeasibilityEvaluator } from './components/EnvironmentFeasibilityEvaluator';
import { MathFormulaConsole } from './components/MathFormulaConsole';
import { Calculator, Flame, Shuffle, HelpCircle, ArrowRight } from 'lucide-react';

export default function App() {
  // Modes: 
  // 'inference' for API serving, local runs, or dynamic batch deployment scaling
  // 'training' for custom SFT, LoRA tuning, or pre-training calculations
  const [selectedMode, setSelectedMode] = useState<CalcMode>('inference');

  // Model States
  const [selectedModel, setSelectedModel] = useState<ModelPreset>(MODEL_PRESETS[0]);
  const [customModelSpecs, setCustomModelSpecs] = useState<ModelPreset>({
    id: 'custom-model',
    name: '我的自定义模型 / My Custom Model',
    creator: 'Local',
    totalParams: 8.0,
    activeParams: 8.0,
    numLayers: 32,
    numHeads: 32,
    numKVHeads: 8,
    hiddenSize: 4096,
    maxContext: 131072, // Default custom max context window
  });

  // Precision States
  const [selectedPrecision, setSelectedPrecision] = useState<PrecisionDetails>(PRECISION_OPTS[0]); // Default FP16
  const [customBits, setCustomBits] = useState<number>(6.0); // Default slider for custom bits

  // Custom handler for model changes to auto-select Native Precision
  const handleModelChange = (model: ModelPreset) => {
    setSelectedModel(model);
    if (model.defaultPrecisionId) {
      const matchPrecision = PRECISION_OPTS.find((p) => p.id === model.defaultPrecisionId);
      if (matchPrecision) {
        setSelectedPrecision(matchPrecision);
      }
    }
    
    // Auto-clamp sequenceLength to fit model physical limitations
    if (inferenceConfig.sequenceLength > model.maxContext) {
      setInferenceConfig(prev => ({
        ...prev,
        sequenceLength: model.maxContext
      }));
    }
    if (trainingConfig.sequenceLength > model.maxContext) {
      setTrainingConfig(prev => ({
        ...prev,
        sequenceLength: model.maxContext
      }));
    }
  };

  // DeepSeek MLA compression flag
  const [useMLACompression, setUseMLACompression] = useState<boolean>(true);

  // Inference Settings
  const [inferenceConfig, setInferenceConfig] = useState<InferenceConfig>({
    batchSize: 1,
    sequenceLength: 4096,
    kvCachePrecision: 'fp16',
    chunkPrefillSize: 'off',
    systemOverheadGB: 2.0,
    tensorParallelism: 1,
  });

  // Training Settings
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    optimizer: 'adamw',
    precision: 'fp16_bf16',
    trainableParamsPercent: 100, // 100% full training by default
    loraRank: 16,
    activationCheckpointing: 'full',
    batchSize: 1,
    sequenceLength: 2048,
    systemOverheadGB: 3.0,
    tensorParallelism: 1,
    pipelineParallelism: 1,
  });

  // Reference GPU choice for Gauge limit matching
  const [selectedGPU, setSelectedGPU] = useState<GPUType>(GPU_PRESETS[4]); // Default RTX 4095 or chosen GPU preset in list (e.g. L40S)

  // Dynamic Environment Node slot counts (how many GPUs are physically present on host)
  const [envGPUCount, setEnvGPUCount] = useState<number>(2); // Default to a 2-card node setup structure for optimal MoE testing

  const handleEnvGPUCountChange = (count: number) => {
    setEnvGPUCount(count);
    // Auto-update system tensor parallelism count to align computation with simulated physical slot count
    setInferenceConfig(prev => ({
      ...prev,
      tensorParallelism: count
    }));
    setTrainingConfig(prev => ({
      ...prev,
      tensorParallelism: count
    }));
  };

  // Quick helper to determine if current selected is a DeepSeek MoE
  const isDeepSeekModel = useMemo(() => {
    return selectedModel.id.includes('deepseek') || selectedModel.id.includes('r1');
  }, [selectedModel.id]);

  // Main compute memo
  const vramBreakdown = useMemo(() => {
    if (selectedMode === 'inference') {
      return calculateInferenceVRAM(selectedModel, selectedPrecision, {
        batchSize: inferenceConfig.batchSize,
        sequenceLength: inferenceConfig.sequenceLength,
        kvCachePrecision: inferenceConfig.kvCachePrecision,
        chunkPrefillSize: inferenceConfig.chunkPrefillSize,
        systemOverheadGB: inferenceConfig.systemOverheadGB,
        tensorParallelism: inferenceConfig.tensorParallelism,
        useMLACompression: useMLACompression,
      });
    } else {
      return calculateTrainingVRAM(selectedModel, selectedPrecision, {
        optimizer: trainingConfig.optimizer,
        precision: trainingConfig.precision,
        trainableParamsPercent: trainingConfig.trainableParamsPercent,
        activationCheckpointing: trainingConfig.activationCheckpointing,
        batchSize: trainingConfig.batchSize,
        sequenceLength: trainingConfig.sequenceLength,
        systemOverheadGB: trainingConfig.systemOverheadGB,
        tensorParallelism: trainingConfig.tensorParallelism,
        pipelineParallelism: trainingConfig.pipelineParallelism,
      });
    }
  }, [
    selectedMode,
    selectedModel,
    selectedPrecision,
    useMLACompression,
    inferenceConfig,
    trainingConfig,
  ]);

  const handleGPUTypeSelect = (gpu: GPUType) => {
    setSelectedGPU(gpu);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans antialiased selection:bg-indigo-600/10 selection:text-indigo-600">
      
      {/* Top Main Navigation Header */}
      <header className="border-b border-slate-200 bg-white sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center text-white shadow-md shadow-indigo-600/10">
              <Calculator className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-xl font-extrabold tracking-tight text-slate-900 flex items-center gap-2">
                大模型显存估算器 <span className="text-indigo-600 text-sm font-semibold bg-indigo-50 border border-indigo-100 px-2 py-0.5 rounded">v1.1</span>
              </h1>
              <p className="text-xs text-slate-500 font-mono mt-0.5">
                LLM GPU VRAM Capacity & Sizing Calculator
              </p>
            </div>
          </div>

          {/* Sizing Mode Tabs */}
          <div className="bg-slate-100 p-1 rounded-xl flex border border-slate-200/50">
            <button
              id={`${COMPONENT_IDS.MODE_TAB}-inference`}
              type="button"
              onClick={() => setSelectedMode('inference')}
              className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
                selectedMode === 'inference'
                  ? 'bg-white text-slate-900 shadow-sm border border-slate-200/30'
                  : 'text-slate-500 hover:text-slate-800'
              }`}
            >
              <Shuffle className="w-3.5 h-3.5" />
              模型推理部署 / Inference Sizing
            </button>
            <button
              id={`${COMPONENT_IDS.MODE_TAB}-training`}
              type="button"
              onClick={() => setSelectedMode('training')}
              className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
                selectedMode === 'training'
                  ? 'bg-white text-slate-900 shadow-sm border border-slate-200/30'
                  : 'text-slate-500 hover:text-slate-800'
              }`}
            >
              <Flame className="w-3.5 h-3.5" />
              模型微调训练 / Training Sizing
            </button>
          </div>
        </div>
      </header>

      {/* Main Workspace Frame */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Responsive dual panel layout: Parameters on the left, interactive VRAM breakdown on the right */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* LEFT SIDE: Inputs configurations (takes 7 cols) */}
          <div className="lg:col-span-7 flex flex-col gap-8">
            
            {/* 1. Model & Specifications */}
            <ModelSelector
              selectedModel={selectedModel}
              onModelChange={handleModelChange}
              customModelSpecs={customModelSpecs}
              onCustomSpecsChange={setCustomModelSpecs}
            />

            {/* 2. Precision Settings */}
            <PrecisionSelector
              selectedPrecision={selectedPrecision}
              onPrecisionChange={setSelectedPrecision}
              customBits={customBits}
              onCustomBitsChange={setCustomBits}
            />

            {/* 3. Stage Specific Parameters */}
            {selectedMode === 'inference' ? (
              <InferenceParams
                config={inferenceConfig}
                onConfigChange={setInferenceConfig}
                useMLACompression={useMLACompression}
                onMLACompressionChange={setUseMLACompression}
                isDeepSeekModel={isDeepSeekModel}
                selectedModelMaxContext={selectedModel.maxContext}
              />
            ) : (
              <TrainingParams
                config={trainingConfig}
                onConfigChange={setTrainingConfig}
              />
            )}

            {/* 4. Environment Feasibility Evaluator (Exclusive master advisor) */}
            {selectedMode === 'inference' && (
              <EnvironmentFeasibilityEvaluator
                selectedModel={selectedModel}
                selectedPrecision={selectedPrecision}
                selectedGPU={selectedGPU}
                inferenceConfig={inferenceConfig}
                useMLACompression={useMLACompression}
                onModelChange={setSelectedModel}
                onPrecisionChange={setSelectedPrecision}
                onGPUTypeSelect={handleGPUTypeSelect}
                onConfigChange={setInferenceConfig}
                envGPUCount={envGPUCount}
                onEnvGPUCountChange={handleEnvGPUCountChange}
              />
            )}

            {/* 5. Hardware Fit Advisor matching */}
            <GPUFitAdvisor
              totalVRAM={vramBreakdown.total}
              selectedGPUType={selectedGPU}
              onGPUTypeSelect={handleGPUTypeSelect}
              tensorParallelDegree={selectedMode === 'inference' ? inferenceConfig.tensorParallelism : trainingConfig.tensorParallelism}
            />

            {/* 6. Deploy Launcher scripts generation (Inference only) */}
            {selectedMode === 'inference' && (
              <DeploymentScriptGenerator
                selectedModel={selectedModel}
                selectedPrecision={selectedPrecision}
                selectedGPU={selectedGPU}
                inferenceConfig={inferenceConfig}
                vramBreakdown={vramBreakdown}
              />
            )}
          </div>

          {/* RIGHT SIDE: Real-time Output gauge dashboard (takes 5 cols, sticky) */}
          <div className="lg:col-span-5 lg:sticky lg:top-24 flex flex-col gap-6">
            
            <VRAMGauge
              breakdown={vramBreakdown}
              selectedMode={selectedMode}
              gpuCapacity={selectedGPU.vram}
              selectedModel={selectedModel}
              selectedPrecision={selectedPrecision}
              inferenceConfig={inferenceConfig}
              trainingConfig={trainingConfig}
              selectedGPU={selectedGPU}
              gpuCount={envGPUCount}
            />
            
            <MathFormulaConsole
              selectedModel={selectedModel}
              selectedPrecision={selectedPrecision}
              selectedMode={selectedMode}
              inferenceConfig={inferenceConfig}
              trainingConfig={trainingConfig}
              useMLACompression={useMLACompression}
              isDeepSeekModel={isDeepSeekModel}
              vramBreakdown={vramBreakdown}
            />
            
            {/* Quick Sizing Documentation / Mini Q&A panel */}
            <div className="bg-white rounded-2xl border border-slate-100 p-6 shadow-sm flex flex-col gap-4">
              <h3 className="font-bold text-sm text-slate-850 flex items-center gap-1.5 uppercase tracking-wide">
                <HelpCircle className="w-4 h-4 text-indigo-600" />
                科学计算逻辑 / Memory Math Cheatsheet
              </h3>
              
              <ul className="text-xs text-slate-500 flex flex-col gap-3 leading-relaxed">
                <li className="flex gap-2">
                  <ArrowRight className="w-3.5 h-3.5 text-indigo-600 shrink-0 mt-0.5" />
                  <span>
                    <strong>静态模型权重公式:</strong> <code>参数量 (Params) × 位宽 (Precision Bits) / 8 × 量化开销比率</code>。因此 70B 模型在 FP16 (2 Bytes) 下需要至少 140GB，在 INT4 (0.5 Bytes) 仅需约 35GB - 38GB。
                  </span>
                </li>
                {selectedMode === 'inference' ? (
                  <li className="flex gap-2">
                    <ArrowRight className="w-3.5 h-3.5 text-emerald-600 shrink-0 mt-0.5" />
                    <span>
                      <strong>KV Cache 缓存公式:</strong> <code>2 × 层数 × 缓存头数 × 头维度 × 批处理并发 × 上下文长度 × 单Token缓存精度(字节)</code>。MHA 注意力中显存随并发及文本长度双重呈恐怖线性乘阶扩张。
                    </span>
                  </li>
                ) : (
                  <li className="flex gap-2">
                    <ArrowRight className="w-3.5 h-3.5 text-amber-600 shrink-0 mt-0.5" />
                    <span>
                      <strong>训练状态显存构成:</strong>
                      <br />
                      • 梯度(Gradients): 2x/4x 的可训练参数显存。
                      <br />
                      • 优化器状态(Optimizer States): AdamW 优化器需要另外留存 8x 的显存记录(包含一阶和二阶梯度)；使用 8-bit Adam 能压缩此项 75% 开销。
                    </span>
                  </li>
                )}
                <li className="flex gap-2">
                  <ArrowRight className="w-3.5 h-3.5 text-indigo-600 shrink-0 mt-0.5" />
                  <span>
                    <strong>多卡 TP 硬件切分:</strong> 启动张量并行时，计算器默认将模型参数、梯度、乃至大部分 KV 缓存均分于各卡内以减少单卡负重。
                  </span>
                </li>
              </ul>
            </div>
            
          </div>
        </div>
      </main>

      {/* Humble Footer */}
      <footer className="border-t border-slate-200 mt-16 bg-white py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-xs text-slate-400 font-mono">
          <p>大模型显存估算器 (LLM GPU VRAM Sizing Calculator)</p>
          <p className="mt-1">基于主流大模型变种架构，结合 attention 重复计算、MLA / GQA机制以及 QLoRA 微调梯度拟合估算。</p>
        </div>
      </footer>
    </div>
  );
}
