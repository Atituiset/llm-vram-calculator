/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { ModelPreset } from '../types';
import { MODEL_PRESETS, COMPONENT_IDS } from '../data';
import { Cpu, Settings, Sparkles, HelpCircle, Search, Download, Loader2, CheckCircle2, AlertCircle, Globe } from 'lucide-react';

interface ModelSelectorProps {
  selectedModel: ModelPreset;
  onModelChange: (model: ModelPreset) => void;
  customModelSpecs: ModelPreset;
  onCustomSpecsChange: (specs: ModelPreset) => void;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedModel,
  onModelChange,
  customModelSpecs,
  onCustomSpecsChange,
}) => {
  const [isCustom, setIsCustom] = useState<boolean>(false);

  // Hugging Face Integrator states
  const [hfSearchQuery, setHfSearchQuery] = useState<string>('');
  const [hfLoading, setHfLoading] = useState<boolean>(false);
  const [hfImportingId, setHfImportingId] = useState<string | null>(null);
  const [hfResults, setHfResults] = useState<any[]>([]);
  const [hfError, setHfError] = useState<string | null>(null);
  const [hfImportedModelName, setHfImportedModelName] = useState<string | null>(null);

  const formatDownloads = (num?: number) => {
    if (!num) return '0';
    if (num >= 1e6) return `${(num / 1e6).toFixed(1)}M+`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(0)}k+`;
    return `${num}`;
  };

  const handleHuggingFaceSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!hfSearchQuery.trim()) return;
    setHfLoading(true);
    setHfError(null);
    setHfResults([]);
    
    try {
      const res = await fetch(`https://huggingface.co/api/models?search=${encodeURIComponent(hfSearchQuery.trim())}&limit=6&sort=downloads&direction=-1`);
      if (!res.ok) {
        throw new Error('无法连接至 Hugging Face Hub API (请检查网络状态或稍后再试)。');
      }
      const data = await res.json();
      setHfResults(data);
      if (data.length === 0) {
        setHfError('未在 Hugging Face 发现任何匹配的开源大模型。可尝试输入精确的机构/名称 (如 "Qwen/Qwen2.5-7B-Instruct")');
      }
    } catch (err: any) {
      setHfError(err.message || '网络连接异常，检索失败。');
    } finally {
      setHfLoading(false);
    }
  };

  const parseHFConfig = (config: any, modelId: string, totalParamsMetadata?: number) => {
    // Extract layers
    const numLayers = config.num_hidden_layers ?? config.n_layer ?? config.num_layers ?? config.n_layers ?? 32;
    
    // Extract hidden size
    const hiddenSize = config.hidden_size ?? config.d_model ?? config.dim ?? 4096;
    
    // Extract query heads
    const numHeads = config.num_attention_heads ?? config.n_head ?? config.num_heads ?? 32;
    
    // Extract KV heads
    const numKVHeads = config.num_key_value_heads ?? config.n_kv_head ?? config.num_kv_heads ?? numHeads;
    
    // Extract creator / Name
    const parts = modelId.split('/');
    const creator = parts.length > 1 ? parts[0] : 'HuggingFace';
    const shortName = parts.length > 1 ? parts[1] : modelId;
    
    // Fallback / Parse parameters
    let totalParams = 0;
    if (totalParamsMetadata) {
      totalParams = totalParamsMetadata / 1e9;
    } else {
      // Parse from modelId name (e.g. 8b, 7b)
      const bMatch = modelId.match(/(\d+(?:\.\d+)?)\s*[Bb]/);
      if (bMatch) {
        totalParams = parseFloat(bMatch[1]);
      } else if (config.intermediate_size) {
        const intermediateSize = config.intermediate_size;
        const numExperts = config.num_local_experts ?? config.moe_num_experts ?? config.n_experts ?? 1;
        
        const attParams = 4 * hiddenSize * hiddenSize;
        let mlpParams = 3 * hiddenSize * intermediateSize; // Standard SwiGLU 3-projection width
        
        if (numExperts > 1) {
          mlpParams = mlpParams * numExperts; // Factor in experts for MoE size
        }
        
        const layerParams = numLayers * (attParams + mlpParams);
        const embedParams = 2 * (config.vocab_size ?? 32000) * hiddenSize;
        totalParams = (layerParams + embedParams) / 1e9;
      } else {
        totalParams = 7.0; // Default baseline size Guess
      }
    }
    
    totalParams = parseFloat(totalParams.toFixed(2));
    
    // Active parameters
    let activeParams = totalParams;
    const numExperts = config.num_local_experts ?? config.moe_num_experts ?? config.n_experts ?? 1;
    const activeExpertsPerTok = config.num_experts_per_tok ?? config.moe_num_experts_activated ?? config.num_expert_per_tok ?? 1;
    
    if (numExperts > 1) {
      const baseRatio = 0.35; // Standard heuristic attention/routing ratio
      activeParams = totalParams * (baseRatio + (1 - baseRatio) * (activeExpertsPerTok / numExperts));
      activeParams = parseFloat(activeParams.toFixed(2));
    }

    return {
      id: `hf-${modelId.replace(/\//g, '-')}`,
      name: shortName,
      creator,
      totalParams,
      activeParams: numExperts > 1 ? activeParams : undefined,
      numLayers,
      numHeads,
      numKVHeads,
      hiddenSize,
      description: `从 Hugging Face Hub 极速拉取的实时配置结构。架构类型: ${config.model_type || 'transformer'}。`
    };
  };

  const handleImportHFModel = async (modelId: string, safetensorsTotal?: number) => {
    setHfImportingId(modelId);
    setHfError(null);
    
    try {
      const configRes = await fetch(`https://huggingface.co/${modelId}/raw/main/config.json`);
      if (!configRes.ok) {
        const fallbackRes = await fetch(`https://huggingface.co/${modelId}/resolve/main/config.json`);
        if (!fallbackRes.ok) {
          throw new Error('未能在该公开仓中找到正确的 config.json。此仓库可能为私有模型、或者是 GGUF 权重单文件版，需要填写完整的原生主干架构基座名 (如 Llama 3)。');
        }
        const configData = await fallbackRes.json();
        loadSpecs(configData, modelId, safetensorsTotal);
        return;
      }
      const configData = await configRes.json();
      loadSpecs(configData, modelId, safetensorsTotal);
    } catch (err: any) {
      setHfError(err.message || '获取结构参数失败，请确认模型是否为正常支持 config.json 的标准架构。');
    } finally {
      setHfImportingId(null);
    }
  };

  const loadSpecs = (configData: any, modelId: string, safetensorsTotal?: number) => {
    const parsedPreset = parseHFConfig(configData, modelId, safetensorsTotal);
    setIsCustom(true);
    onCustomSpecsChange(parsedPreset);
    onModelChange(parsedPreset);
    setHfImportedModelName(parsedPreset.name);
    setTimeout(() => setHfImportedModelName(null), 5000);
  };

  const handlePresetSelect = (presetId: string) => {
    if (presetId === 'custom') {
      setIsCustom(true);
      onModelChange(customModelSpecs);
    } else {
      setIsCustom(false);
      const preset = MODEL_PRESETS.find((m) => m.id === presetId);
      if (preset) {
        onModelChange(preset);
      }
    }
  };

  const updateCustomField = (key: keyof ModelPreset, value: string | number) => {
    const updated = {
      ...customModelSpecs,
      [key]: value,
    };
    onCustomSpecsChange(updated);
    if (isCustom) {
      onModelChange(updated);
    }
  };

  const isMoE = (selectedModel.activeParams !== undefined && selectedModel.activeParams < selectedModel.totalParams);

  return (
    <div id={COMPONENT_IDS.PRESETS} className="bg-white rounded-2xl border border-slate-100 p-6 shadow-sm flex flex-col gap-6">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <Cpu className="w-5 h-5 text-indigo-600" />
          <h2 className="text-lg font-semibold text-slate-900 tracking-tight">
            1. 选型与参数设定 / Model & Parameters
          </h2>
        </div>
        <p className="text-sm text-slate-500">
          选择一个大语言模型预设，或选择“自定义”以调整层数、注意力头和隐层维度。支持从 Hugging Face 一键拉取任意公有模型。
        </p>
      </div>

      {/* Dynamic Hugging Face Hub Import Panel */}
      <div id="hf-resolver-panel" className="bg-gradient-to-br from-slate-50 to-indigo-50/20 border border-slate-200/80 rounded-xl p-4.5 flex flex-col gap-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 text-xs font-bold text-slate-800 uppercase tracking-wide">
            <Globe className="w-4 h-4 text-indigo-600" />
            🔥 Hugging Face 开源模型一键检索导入 / LIVE HF HUB RESOLVER
          </div>
          <span className="text-[10px] text-slate-500 leading-normal font-sans">
            解决多平台查询痛点：省去科学查找时间，直接从 HF 公共 APIs 实时拉取模型层数、维度
          </span>
        </div>

        <form onSubmit={handleHuggingFaceSearch} className="flex gap-2">
          <div className="relative flex-1">
            <input
              type="text"
              placeholder="输入模型名称、关键字，或完整路径。如 meta-llama/Llama-3-8B 或 deepseek-ai/DeepSeek-V3"
              value={hfSearchQuery}
              onChange={(e) => setHfSearchQuery(e.target.value)}
              className="w-full bg-white border border-slate-250 rounded-lg pl-3 pr-8 py-2 text-xs font-mono text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-600/10 focus:border-indigo-600 placeholder:font-sans placeholder:text-slate-450"
            />
            {hfSearchQuery && (
              <button
                type="button"
                onClick={() => setHfSearchQuery('')}
                className="absolute right-2.5 top-2.5 text-slate-400 hover:text-slate-600 text-xs"
              >
                ✕
              </button>
            )}
          </div>
          <button
            type="submit"
            disabled={hfLoading || !hfSearchQuery.trim()}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-200 text-white disabled:text-slate-400 text-xs font-semibold rounded-lg transition-all flex items-center gap-1.5 shrink-0 hover:cursor-pointer"
          >
            {hfLoading ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Search className="w-3.5 h-3.5" />
            )}
            检索 / Search
          </button>
        </form>

        {/* Success toast banner */}
        {hfImportedModelName && (
          <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-3.5 py-2.5 text-xs text-emerald-800 flex items-center gap-2.5 animate-in fade-in slide-in-from-top-1 duration-200">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
            <div>
              解析成功！已成功导入 <strong>{hfImportedModelName}</strong> 规格参数至“自定义配置”并联动全局计算。
            </div>
          </div>
        )}

        {/* Error message */}
        {hfError && (
          <div className="bg-rose-50 border border-rose-200 rounded-lg px-3.5 py-2.5 text-xs text-rose-800 flex items-center gap-2.5">
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
            <div className="leading-relaxed font-medium">{hfError}</div>
          </div>
        )}

        {/* HF API Results Container */}
        {hfResults.length > 0 && (
          <div className="border border-slate-200/80 rounded-lg bg-white overflow-hidden divide-y divide-slate-150 animate-in fade-in slide-in-from-top-1">
            <div className="bg-slate-50 px-3 py-2 text-[10px] uppercase font-bold text-slate-500 flex justify-between">
              <span>检索到的社区匹配结果 / Found in HF Hub Registry</span>
              <span>下载热度排序 (Downloads)</span>
            </div>
            {hfResults.map((result: any) => {
              const totalParamsEst = result.safetensors?.parameters?.total;
              const isImporting = hfImportingId === result.id;
              
              return (
                <div key={result.id} className="p-3 flex items-center justify-between hover:bg-slate-50/20 gap-4">
                  <div className="flex flex-col gap-0.5 min-w-0">
                    <span className="text-xs font-mono font-bold text-slate-800 truncate" title={result.id}>
                      {result.id}
                    </span>
                    <div className="flex items-center gap-2.5 text-[10px] text-slate-500 font-mono">
                      <span className="flex items-center gap-0.5">
                        ⬇️ {formatDownloads(result.downloads)} downloads
                      </span>
                      <span>·</span>
                      <span className="flex items-center gap-0.5">
                        ⭐ {result.likes || 0} stars
                      </span>
                      {totalParamsEst && (
                        <>
                          <span>·</span>
                          <span className="text-indigo-600 font-semibold">
                            📁 {(totalParamsEst / 1e9).toFixed(1)}B params
                          </span>
                        </>
                      )}
                    </div>
                  </div>

                  <button
                    type="button"
                    disabled={isImporting}
                    onClick={() => handleImportHFModel(result.id, totalParamsEst)}
                    className="px-3 py-1.5 bg-indigo-50 hover:bg-indigo-100 disabled:bg-slate-100 border border-indigo-100/50 hover:border-indigo-200 text-[11px] font-bold text-indigo-700 disabled:text-slate-400 rounded-md transition-colors flex items-center gap-1 shrink-0 hover:cursor-pointer"
                  >
                    {isImporting ? (
                      <>
                        <Loader2 className="w-3 h-3 animate-spin text-indigo-600" />
                        读取中...
                      </>
                    ) : (
                      <>
                        <Download className="w-3 h-3" />
                        拉取解析 / Load Size
                      </>
                    )}
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Preset Selector Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {MODEL_PRESETS.map((model) => {
          const isSelected = !isCustom && selectedModel.id === model.id;
          return (
            <button
              key={model.id}
              id={`preset-${model.id}`}
              type="button"
              onClick={() => handlePresetSelect(model.id)}
              className={`flex flex-col text-left p-4 rounded-xl border transition-all duration-200 ${
                isSelected
                  ? 'border-indigo-600 bg-indigo-50/40 ring-2 ring-indigo-600/10 shadow-sm'
                  : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50/50'
              }`}
            >
              <div className="flex justify-between items-start gap-2 w-full">
                <span className="font-semibold text-slate-900 truncate">
                  {model.name}
                </span>
                <span className="text-[10px] font-mono font-medium text-slate-500 bg-slate-100 hover:bg-slate-200/60 px-2 py-0.5 rounded shrink-0">
                  {model.creator}
                </span>
              </div>
              <div className="text-xs font-mono text-indigo-700 font-bold mt-1.5 flex items-center gap-1.5">
                <span>{model.totalParams}B params</span>
                {model.activeParams && model.activeParams < model.totalParams && (
                  <span className="text-amber-700 bg-amber-50 px-1.5 rounded border border-amber-100/50 text-[10px]">
                    MoE (Active: {model.activeParams}B)
                  </span>
                )}
              </div>
              {model.description && !isSelected && (
                <p className="text-xs text-slate-400 mt-2 line-clamp-2">
                  {model.description}
                </p>
              )}

              {/* Collapsible Architecture Details inside the Card (Responsive default expansion) */}
              {isSelected && (
                <div className="mt-3.5 pt-3.5 border-t border-indigo-150/60 w-full animate-in fade-in slide-in-from-top-1 duration-200">
                  <span className="text-[10px] font-bold text-indigo-800 uppercase tracking-wider block mb-2">
                    🔬 激活模型层数与维度规格:
                  </span>
                  <div className="grid grid-cols-2 gap-1.5 text-[10px] font-mono">
                    <div className="p-1 px-1.5 bg-white/70 rounded border border-indigo-100/40">
                      <span className="text-slate-400 text-[8px] uppercase block">Transformer层数</span>
                      <strong className="text-slate-800">{model.numLayers} Layers</strong>
                    </div>
                    <div className="p-1 px-1.5 bg-white/70 rounded border border-indigo-100/40">
                      <span className="text-slate-400 text-[8px] uppercase block">隐层维度 size</span>
                      <strong className="text-slate-800">{model.hiddenSize} Dim</strong>
                    </div>
                    <div className="p-1 px-1.5 bg-white/70 rounded border border-indigo-100/40">
                      <span className="text-slate-400 text-[8px] uppercase block">关注头数 (Q)</span>
                      <strong className="text-slate-800">{model.numHeads} Q-Heads</strong>
                    </div>
                    <div className="p-1 px-1.5 bg-white/70 rounded border border-indigo-100/40">
                      <span className="text-slate-400 text-[8px] uppercase block">缓存头数 (KV)</span>
                      <strong className="text-slate-800">{model.numKVHeads} KV-Heads</strong>
                    </div>
                  </div>
                  <div className="mt-2.5 bg-indigo-500/10 text-[9.5px] leading-relaxed text-indigo-800 p-1.5 px-2.5 rounded-md border border-indigo-100 flex justify-between items-center">
                    <span>
                      GQA 比例: <strong>{model.numKVHeads === model.numHeads ? 'MHA 1:1 无衰减' : `GQA 1:${model.numHeads / model.numKVHeads}`}</strong>
                    </span>
                    <span className="font-bold underline">首发精度: {model.defaultPrecisionId?.toUpperCase() || 'FP16'}</span>
                  </div>
                </div>
              )}
            </button>
          );
        })}

        {/* Custom Option Button */}
        <button
          id="preset-custom"
          type="button"
          onClick={() => handlePresetSelect('custom')}
          className={`flex flex-col text-left p-4 rounded-xl border transition-all duration-200 sm:col-span-2 ${
            isCustom
              ? 'border-indigo-600 bg-indigo-50/40 ring-2 ring-indigo-600/10 shadow-sm'
              : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50/50'
          }`}
        >
          <div className="flex items-center gap-1.5 justify-between w-full">
            <span className="flex items-center gap-1.5 font-semibold text-slate-900">
              <Settings className="w-4 h-4 text-indigo-600 animate-spin" style={{ animationDuration: '12s' }} />
              自定义配置 / Custom Architecture Spec
            </span>
            {isCustom && (
              <span className="text-[9px] font-mono text-indigo-700 bg-indigo-100 border border-indigo-200 font-bold px-1.5 py-0.5 rounded uppercase tracking-wider animate-pulse">
                已激活编辑
              </span>
            )}
          </div>
          <p className="text-xs text-slate-450 mt-1">
            编辑专属层数、首批注意力宽度和维度大小，可用于评估未公开模型或进行特定架构微调实验。
          </p>
        </button>
      </div>

      {/* Inputs Section - ONLY shown when isCustom is active to make relations perfectly transparent! */}
      {isCustom && (
        <div 
          id={COMPONENT_IDS.SPECS} 
          className="bg-indigo-50/5 border border-indigo-150 rounded-xl p-5 border-dashed transition-all duration-200 animate-in fade-in slide-in-from-top-2"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 mb-4 pb-2 border-b border-indigo-100/40">
            <div className="flex flex-col">
              <span className="text-xs font-bold uppercase tracking-wider text-indigo-850 flex items-center gap-1.5">
                ✏️ 架构规格实时调节面板 / ARCHITECTURE CONFIGURATION EDITOR
              </span>
              <p className="text-[10px] text-slate-500 leading-normal font-sans">
                以下调节项目将直接作用于大模型显存估算算式和 KV 密集缓存系数中。
              </p>
            </div>
            <span className="text-[10px] text-indigo-600 bg-indigo-150/40 border border-indigo-200 px-2 py-0.5 rounded font-mono font-bold shrink-0">
              [ ✏️ EDIT MODE SELECTIVE ]
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-x-4 gap-y-4 font-sans text-sm">
            {/* Total Parameter */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-slate-600 flex items-center gap-1">
                总参数量 / Total Params
                <HelpCircle className="w-3 h-3 text-slate-400 hover:text-indigo-600 cursor-help" title="计算模型权重显存时占主导地位的总文件参数体积(以十亿B为单位)" />
              </label>
              <div className="relative">
                <input
                  id={`${COMPONENT_IDS.CUSTOM_SPEC}-totalParams`}
                  type="number"
                  disabled={!isCustom}
                  value={isCustom ? customModelSpecs.totalParams : selectedModel.totalParams}
                  onChange={(e) => updateCustomField('totalParams', parseFloat(e.target.value) || 0)}
                  className="w-full bg-white border border-slate-200 rounded-lg px-3 py-1.5 font-mono text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-600/20 focus:border-indigo-600 disabled:bg-slate-100 disabled:text-slate-500"
                  min="0.1"
                  step="0.1"
                />
                <span className="absolute right-3 top-2 text-xs text-slate-400 font-mono">B</span>
              </div>
            </div>

            {/* Active Parameter */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-slate-600 flex items-center gap-1">
                单次激活 / Active Params
                <HelpCircle className="w-3 h-3 text-slate-400 hover:text-indigo-600 cursor-help" title="主要针对MoE(混合专家模型)。通常只有小部分参数真正用于单次Token计算，但全部参数依然需要完全驻留在显存中。" />
              </label>
              <div className="relative">
                <input
                  id={`${COMPONENT_IDS.CUSTOM_SPEC}-activeParams`}
                  type="number"
                  disabled={!isCustom}
                  value={
                    isCustom 
                      ? (customModelSpecs.activeParams ?? customModelSpecs.totalParams) 
                      : (selectedModel.activeParams ?? selectedModel.totalParams)
                  }
                  onChange={(e) => updateCustomField('activeParams', parseFloat(e.target.value) || 0)}
                  className="w-full bg-white border border-slate-200 rounded-lg px-3 py-1.5 font-mono text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-600/20 focus:border-indigo-600 disabled:bg-slate-100 disabled:text-slate-500"
                  min="0.1"
                  step="0.1"
                />
                <span className="absolute right-3 top-2 text-xs text-slate-400 font-mono">B</span>
              </div>
            </div>

            {/* Num Layers */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-slate-600 flex items-center gap-1">
                模型层数 / Num Layers
                <HelpCircle className="w-3 h-3 text-slate-400 hover:text-indigo-600 cursor-help" title="Transformer 模块总层数。直接影响KV缓存和训练梯度的深度累加。" />
              </label>
              <input
                id={`${COMPONENT_IDS.CUSTOM_SPEC}-numLayers`}
                type="number"
                disabled={!isCustom}
                value={isCustom ? customModelSpecs.numLayers : selectedModel.numLayers}
                onChange={(e) => updateCustomField('numLayers', parseInt(e.target.value) || 0)}
                className="w-full bg-white border border-slate-200 rounded-lg px-3 py-1.5 font-mono text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-600/20 focus:border-indigo-600 disabled:bg-slate-100 disabled:text-slate-500"
                min="1"
              />
            </div>

            {/* Num Q Heads */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-slate-600 flex items-center gap-1">
                查询头数 (Q Heads)
                <HelpCircle className="w-3 h-3 text-slate-400 hover:text-indigo-600 cursor-help" title="Self-Attention 的 Query 多头注意层总数。" />
              </label>
              <input
                id={`${COMPONENT_IDS.CUSTOM_SPEC}-numHeads`}
                type="number"
                disabled={!isCustom}
                value={isCustom ? customModelSpecs.numHeads : selectedModel.numHeads}
                onChange={(e) => updateCustomField('numHeads', parseInt(e.target.value) || 0)}
                className="w-full bg-white border border-slate-200 rounded-lg px-3 py-1.5 font-mono text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-600/20 focus:border-indigo-600 disabled:bg-slate-100 disabled:text-slate-500"
                min="1"
              />
            </div>

            {/* Num KV Heads */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-slate-600 flex items-center gap-1">
                缓存头数 (KV Heads)
                <HelpCircle className="w-3 h-3 text-slate-400 hover:text-indigo-600 cursor-help" title="若数值小于Query头，则是GQA(组注意力)机制或MQA(单头缓存)。极大缩减 inference KV-cache 深度显存占用。如Llama3-8B中此项数值为8。" />
              </label>
              <input
                id={`${COMPONENT_IDS.CUSTOM_SPEC}-numKVHeads`}
                type="number"
                disabled={!isCustom}
                value={isCustom ? customModelSpecs.numKVHeads : selectedModel.numKVHeads}
                onChange={(e) => updateCustomField('numKVHeads', parseInt(e.target.value) || 0)}
                className="w-full bg-white border border-slate-200 rounded-lg px-3 py-1.5 font-mono text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-600/20 focus:border-indigo-600 disabled:bg-slate-100 disabled:text-slate-500"
                min="1"
              />
            </div>

            {/* Hidden Size */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-slate-600 flex items-center gap-1">
                隐层维度 / Hidden Size
                <HelpCircle className="w-3 h-3 text-slate-400 hover:text-indigo-600 cursor-help" title="模型内部每个 Token 对应的密集向量大小 (例如 LLaMA-8B 通常为 4096)。" />
              </label>
              <input
                id={`${COMPONENT_IDS.CUSTOM_SPEC}-hiddenSize`}
                type="number"
                disabled={!isCustom}
                value={isCustom ? customModelSpecs.hiddenSize : selectedModel.hiddenSize}
                onChange={(e) => updateCustomField('hiddenSize', parseInt(e.target.value) || 0)}
                className="w-full bg-white border border-slate-200 rounded-lg px-3 py-1.5 font-mono text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-600/20 focus:border-indigo-600 disabled:bg-slate-100 disabled:text-slate-500"
                min="128"
                step="128"
              />
            </div>
          </div>

          {/* Dynamic GQA Note indicator */}
          <div className="mt-4 pt-3 border-t border-slate-200/50 flex flex-wrap gap-2 text-xs text-slate-500 items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="font-medium text-slate-755">当前自定义注意力机制:</span>
              <span className="px-2 py-0.5 font-bold font-mono text-indigo-700 bg-indigo-50 border border-indigo-100 rounded-full text-[10.5px]">
                {(() => {
                  const q = isCustom ? customModelSpecs.numHeads : selectedModel.numHeads;
                  const kv = isCustom ? customModelSpecs.numKVHeads : selectedModel.numKVHeads;
                  if (!kv || kv === q) return 'MHA (Multi-Head Attention)';
                  if (kv === 1) return 'MQA (Multi-Query Attention)';
                  return `GQA (Grouped-Query GQA 1:${q / kv})`;
                })()}
              </span>
            </div>
            <div className="flex items-center gap-1 font-mono text-[11px]">
              <span>每个注意力头维度 (Head Dim):</span>
              <span className="font-bold text-slate-800">
                {(() => {
                  const hidden = isCustom ? customModelSpecs.hiddenSize : selectedModel.hiddenSize;
                  const heads = isCustom ? customModelSpecs.numHeads : selectedModel.numHeads;
                  return heads > 0 ? Math.round(hidden / heads) : 128;
                })()}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
