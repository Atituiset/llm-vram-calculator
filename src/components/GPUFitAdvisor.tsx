/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo } from 'react';
import { GPUType } from '../types';
import { GPU_PRESETS, COMPONENT_IDS, matchOrParseGPU } from '../data';
import { 
  HardDrive, 
  HelpCircle, 
  CheckCircle2, 
  AlertTriangle, 
  AlertOctagon, 
  Search, 
  Sliders, 
  Cpu, 
  Sparkles, 
  BadgeHelp,
  CornerDownRight,
  RefreshCw,
  Plus
} from 'lucide-react';

interface GPUFitAdvisorProps {
  totalVRAM: number;
  selectedGPUType: GPUType;
  onGPUTypeSelect: (gpu: GPUType) => void;
  tensorParallelDegree: number;
}

export const GPUFitAdvisor: React.FC<GPUFitAdvisorProps> = ({
  totalVRAM,
  selectedGPUType,
  onGPUTypeSelect,
  tensorParallelDegree,
}) => {
  // Local interface states
  const [searchQuery, setSearchQuery] = useState('');
  const [activeBrandFilter, setActiveBrandFilter] = useState<'all' | 'nvidia' | 'amd' | 'apple' | 'huawei' | 'intel_google'>('all');
  const [isManualEditing, setIsManualEditing] = useState(false);
  const [customInputText, setCustomInputText] = useState('');

  // Brand categorization catalog
  const filterByBrand = (gpu: GPUType) => {
    const name = gpu.name.toLowerCase();
    const id = gpu.id.toLowerCase();
    if (activeBrandFilter === 'all') return true;
    if (activeBrandFilter === 'nvidia') return name.includes('nvidia') || name.includes('rtx');
    if (activeBrandFilter === 'amd') return name.includes('amd') || name.includes('radeon') || name.includes('instinct');
    if (activeBrandFilter === 'apple') return name.includes('apple') || name.includes('mac');
    if (activeBrandFilter === 'huawei') return name.includes('huawei') || name.includes('ascend');
    if (activeBrandFilter === 'intel_google') return name.includes('intel') || name.includes('gaudi') || name.includes('google') || name.includes('tpu') || id.includes('tpu');
    return true;
  };

  // Autocomplete matching list based on searchQuery and brand filter
  const suggestedGPUTemplates = useMemo(() => {
    let list = GPU_PRESETS;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter(g => 
        g.name.toLowerCase().includes(q) || 
        g.id.toLowerCase().includes(q) ||
        (g.bandwidth && g.bandwidth.toLowerCase().includes(q))
      );
    }
    return list.filter(filterByBrand);
  }, [searchQuery, activeBrandFilter]);

  // Live intelligent parser status
  const parsedPreviewGPUType = useMemo(() => {
    if (!searchQuery.trim()) return null;
    // Check if searchQuery perfectly matches any preset first, if so no need to show raw parse warnings
    const exactMatch = GPU_PRESETS.find(g => g.name.toLowerCase() === searchQuery.toLowerCase().trim());
    if (exactMatch) return null;
    return matchOrParseGPU(searchQuery);
  }, [searchQuery]);

  // Handle preset card click
  const handleSelectPreset = (gpu: GPUType) => {
    onGPUTypeSelect(gpu);
    setSearchQuery(''); // clear search to highlight selection
  };

  // Trigger custom typing auto-parse and save
  const handleTriggerAutoParse = () => {
    if (!searchQuery.trim()) return;
    const parsed = matchOrParseGPU(searchQuery);
    onGPUTypeSelect(parsed);
    setSearchQuery('');
    setCustomInputText('');
  };

  // Modify currently selected GPU fields directly
  const handleUpdateSelectedGPUField = (field: keyof GPUType, value: any) => {
    onGPUTypeSelect({
      ...selectedGPUType,
      [field]: value,
    });
  };

  // Sizing cluster calculation
  const cardsNeeded = Math.ceil(totalVRAM / selectedGPUType.vram);
  const percentPerCard = parseFloat(((totalVRAM / (cardsNeeded * selectedGPUType.vram)) * 100).toFixed(0));

  // Determine load warning badges
  let loadBadgeColor = 'bg-emerald-50 text-emerald-800 border-emerald-100';
  let loadLabel = '宽裕 / Safe Layout';
  let ImpactIcon = CheckCircle2;

  if (percentPerCard > 92) {
    loadBadgeColor = 'bg-rose-50 text-rose-700 border-rose-100';
    loadLabel = '极度吃紧 (OOM风险极高)';
    ImpactIcon = AlertOctagon;
  } else if (percentPerCard > 80) {
    loadBadgeColor = 'bg-amber-50 text-amber-700 border-amber-100';
    loadLabel = '中高负荷 / Warm Run';
    ImpactIcon = AlertTriangle;
  }

  return (
    <div id={COMPONENT_IDS.GPU_FIT} className="bg-white rounded-2xl border border-slate-100 p-6 shadow-sm flex flex-col gap-6">
      
      {/* Header Block with Dual Languages */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <HardDrive className="w-5 h-5 text-indigo-600" />
          <h2 className="text-lg font-semibold text-slate-900 tracking-tight">
            4. 硬件配置适配推荐与多厂商验证 / Accelerator Matcher
          </h2>
        </div>
        <p className="text-sm text-slate-500">
          支持自主录入或检索 NVIDIA、AMD Instinct、华为昇腾 (Ascend)、Apple M芯片及 Intel 算力底座，智能评估所需集群节点及显存安全阈值。
        </p>
      </div>

      {/* Dynamic Accelerator Entry Section */}
      <div className="bg-slate-50/50 rounded-xl p-4.5 border border-slate-200/60 flex flex-col gap-4">
        
        {/* Search Input and AI Engine Parser Row */}
        <div className="flex flex-col gap-2">
          <label className="text-xs text-slate-700 font-bold flex items-center justify-between">
            <span className="flex items-center gap-1">
              🔎 算力芯片检索或自主输入型号 / Search & Smart Parser:
            </span>
            <span className="text-[10px] text-indigo-600 font-mono">
              输入任意文本(如 "MI300 192G", "910B 64G", "rtx5090")智能提取显存与厂商
            </span>
          </label>
          <div className="relative flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-2.5 w-4 h-4 text-slate-400" />
              <input
                type="text"
                placeholder="请输入型号, 如: AMD Arctic MI300X, 华为 910B, Mac Studio M4 128G, RTX 5090..."
                value={searchQuery}
                aria-label="请输入型号"
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-white border border-slate-200 rounded-lg pl-9 pr-3 py-2 font-sans text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-600/20 focus:border-indigo-600"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  type="button"
                  className="absolute right-3 top-2.5 text-xs text-slate-400 hover:text-slate-600"
                >
                  ✕
                </button>
              )}
            </div>
            {searchQuery.trim() && (
              <button
                type="button"
                onClick={handleTriggerAutoParse}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-xs text-white font-bold rounded-lg transition-colors shadow-sm flex items-center gap-1.5 shrink-0"
              >
                <Sparkles className="w-3.5 h-3.5" />
                智能解析加载
              </button>
            )}
          </div>
        </div>

        {/* Brand quick-filters tabs */}
        <div className="flex flex-wrap gap-1.5 items-center">
          <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider mr-1">
            厂商分类:
          </span>
          <button
            type="button"
            onClick={() => setActiveBrandFilter('all')}
            className={`px-2.5 py-1 text-[11px] font-medium rounded-full border transition-all ${
              activeBrandFilter === 'all'
                ? 'bg-indigo-600 border-indigo-600 text-white font-semibold'
                : 'bg-white border-slate-200 text-slate-600 hover:border-slate-350'
            }`}
          >
            全部 ({GPU_PRESETS.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveBrandFilter('nvidia')}
            className={`px-2.5 py-1 text-[11px] font-medium rounded-full border transition-all ${
              activeBrandFilter === 'nvidia'
                ? 'bg-green-600 border-green-600 text-white font-semibold'
                : 'bg-white border-slate-200 text-slate-600 hover:border-green-200 hover:text-green-700'
            }`}
          >
            NVIDIA
          </button>
          <button
            type="button"
            onClick={() => setActiveBrandFilter('amd')}
            className={`px-2.5 py-1 text-[11px] font-medium rounded-full border transition-all ${
              activeBrandFilter === 'amd'
                ? 'bg-rose-600 border-rose-600 text-white font-semibold'
                : 'bg-white border-slate-200 text-slate-600 hover:border-rose-200 hover:text-rose-700'
            }`}
          >
            AMD Instinct
          </button>
          <button
            type="button"
            onClick={() => setActiveBrandFilter('apple')}
            className={`px-2.5 py-1 text-[11px] font-medium rounded-full border transition-all ${
              activeBrandFilter === 'apple'
                ? 'bg-slate-800 border-slate-800 text-white font-semibold'
                : 'bg-white border-slate-200 text-slate-600 hover:border-slate-300'
            }`}
          >
            Apple Unified
          </button>
          <button
            type="button"
            onClick={() => setActiveBrandFilter('huawei')}
            className={`px-2.5 py-1 text-[11px] font-medium rounded-full border transition-all ${
              activeBrandFilter === 'huawei'
                ? 'bg-red-600 border-red-600 text-white font-semibold'
                : 'bg-white border-slate-200 text-slate-600 hover:border-red-200 hover:text-red-700'
            }`}
          >
            华为昇腾
          </button>
          <button
            type="button"
            onClick={() => setActiveBrandFilter('intel_google')}
            className={`px-2.5 py-1 text-[11px] font-medium rounded-full border transition-all ${
              activeBrandFilter === 'intel_google'
                ? 'bg-cyan-600 border-cyan-600 text-white font-semibold'
                : 'bg-white border-slate-200 text-slate-600 hover:border-cyan-200 hover:text-cyan-700'
            }`}
          >
            Intel / TPU / Other
          </button>
        </div>

        {/* AI Auto-extract smart feedback alert */}
        {parsedPreviewGPUType && (
          <div className="bg-indigo-50/70 border border-indigo-100 rounded-lg p-3 text-xs leading-relaxed text-indigo-900 flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 animate-in fade-in duration-200">
            <div className="flex gap-2 items-start">
              <Sparkles className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
              <div>
                <span>
                  💡 <strong>动态识别引擎反馈：</strong>您输入的文本将被解析为全新算力载体 ——
                </span>
                <div className="mt-1 font-mono text-[11px] bg-white/80 p-1 px-2 rounded inline-block border border-indigo-100/50">
                  名称：<strong className="text-slate-800">{parsedPreviewGPUType.name}</strong> | 
                  解析显存：<strong className="text-indigo-700">{parsedPreviewGPUType.vram} GB</strong> | 
                  网络架构：<strong className="text-amber-800">{parsedPreviewGPUType.type.toUpperCase()}</strong>
                </div>
              </div>
            </div>
            <button
              onClick={handleTriggerAutoParse}
              type="button"
              className="bg-indigo-600 text-white font-bold text-[10px] hover:bg-indigo-700 px-3 py-1.5 rounded transition-colors uppercase shrink-0"
            >
              应用以上解析参数
            </button>
          </div>
        )}

        {/* Filtered suggestions list - Shows horizontally scrollable/grid chips to save precious vertical space */}
        <div className="flex flex-col gap-1.5">
          <div className="flex justify-between items-center text-[10.5px] font-serif font-bold text-slate-400">
            <span>预设参考库匹配列表 ({suggestedGPUTemplates.length}):</span>
            {searchQuery && <span>点击直接选择内置项</span>}
          </div>
          <div className="max-h-36 overflow-y-auto grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 pr-1 custom-scrollbar">
            {suggestedGPUTemplates.map((gpu) => {
              const isSelected = selectedGPUType.id === gpu.id;
              return (
                <button
                  key={gpu.id}
                  type="button"
                  onClick={() => handleSelectPreset(gpu)}
                  className={`px-3 py-2 rounded-lg border text-left transition-all flex justify-between items-center ${
                    isSelected
                      ? 'border-indigo-650 bg-indigo-50/50 text-indigo-950 font-semibold ring-1 ring-indigo-500/10'
                      : 'border-slate-200 bg-white hover:border-slate-350 hover:bg-slate-50/60 text-slate-700'
                  }`}
                >
                  <div className="truncate flex-1">
                    <span className="text-xs truncate block font-medium">{gpu.name}</span>
                    <span className="text-[9.5px] text-slate-400 font-mono">
                      带宽: {gpu.bandwidth || '未知'}
                    </span>
                  </div>
                  <span className="text-[10px] font-bold font-mono text-indigo-700 bg-indigo-50 border border-indigo-100/40 px-1.5 py-0.5 rounded shrink-0">
                    {gpu.vram}G
                  </span>
                </button>
              );
            })}
            {suggestedGPUTemplates.length === 0 && (
              <div className="col-span-full py-4 text-center text-xs text-slate-400 bg-white rounded border border-dashed flex flex-col items-center justify-center gap-1">
                <BadgeHelp className="w-5 h-5 text-slate-300" />
                <span>没有找到完全符合的内置型号。</span>
                <span>请直接点击右上方 "智能解析加载" 以生成自定义加速卡。</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Manual Fine-tuning Panel Toggle */}
      <div className="border border-slate-100 rounded-xl overflow-hidden">
        <button
          onClick={() => setIsManualEditing(!isManualEditing)}
          type="button"
          className="w-full bg-slate-50 hover:bg-slate-100/70 py-2 px-4 flex items-center justify-between text-xs font-semibold text-slate-700 transition-colors"
        >
          <span className="flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-indigo-600" />
            🔧 硬件物理底层规格底层微调 (Manual Refinement Panel)  
            {selectedGPUType.id.startsWith('custom') && (
              <span className="ml-2 font-mono bg-indigo-100 border border-indigo-200 text-indigo-700 font-bold px-1.5 rounded text-[9px] uppercase tracking-wider">
                自定义配置生效中
              </span>
            )}
          </span>
          <span className="font-mono text-[10px] text-slate-400">
            {isManualEditing ? '收起配置 [-]' : '展开精确微调参数 [+]'}
          </span>
        </button>

        {isManualEditing && (
          <div className="p-4 bg-white border-t border-slate-100 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 animate-in fade-in slide-in-from-top-1 duration-150">
            {/* Custom VRAM */}
            <div className="flex flex-col gap-1.5">
              <label className="text-[11px] font-bold text-slate-600">物理显存容量 (GB):</label>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  aria-label="物理显存容量 (GB)"
                  value={selectedGPUType.vram}
                  min={1}
                  max={1024}
                  onChange={(e) => handleUpdateSelectedGPUField('vram', Math.max(1, parseInt(e.target.value) || 1))}
                  className="w-20 bg-white border border-slate-200 rounded px-2 py-1 font-mono text-xs text-slate-800 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                />
                <input
                  type="range"
                  aria-label="显存容量拖拽"
                  min={4}
                  max={192}
                  step={4}
                  value={selectedGPUType.vram}
                  onChange={(e) => handleUpdateSelectedGPUField('vram', parseInt(e.target.value))}
                  className="flex-1 accent-indigo-600 h-1 bg-slate-100 rounded"
                />
              </div>
            </div>

            {/* Custom Name */}
            <div className="flex flex-col gap-1.5">
              <label className="text-[11px] font-bold text-slate-600">自定义物理型号名称 (Name):</label>
              <input
                type="text"
                aria-label="物理型号名称"
                value={selectedGPUType.name}
                onChange={(e) => handleUpdateSelectedGPUField('name', e.target.value)}
                className="w-full bg-white border border-slate-200 rounded px-2 py-1 font-sans text-xs text-slate-800 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </div>

            {/* Custom Bandwidth */}
            <div className="flex flex-col gap-1.5">
              <label className="text-[11px] font-bold text-slate-600">吞吐带宽 (Bandwidth):</label>
              <input
                type="text"
                aria-label="吞吐带宽"
                value={selectedGPUType.bandwidth || ''}
                placeholder="例如: 1.0 TB/s, shared"
                onChange={(e) => handleUpdateSelectedGPUField('bandwidth', e.target.value)}
                className="w-full bg-white border border-slate-200 rounded px-2 py-1 font-mono text-xs text-slate-800 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </div>

            {/* Custom Vendor Type */}
            <div className="flex flex-col gap-1.5">
              <label className="text-[11px] font-bold text-slate-600">总线运行级别 (Device Mode):</label>
              <select
                value={selectedGPUType.type}
                aria-label="总线运行级别"
                onChange={(e) => handleUpdateSelectedGPUField('type', e.target.value)}
                className="w-full bg-white border border-slate-200 rounded px-2 py-1 font-sans text-xs text-slate-800 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              >
                <option value="datacenter">Datacenter (云端大容量加速阵列)</option>
                <option value="consumer">Consumer (消费级显卡 / 移动工作站)</option>
                <option value="mac">Mac Studio (苹果统一内存统一吞吐)</option>
                <option value="legacy">Legacy (老式小频带设备)</option>
              </select>
            </div>
          </div>
        )}
      </div>

      {/* ACTIVE SELECTED ACCELERATOR METRICS PANEL */}
      <div className="bg-gradient-to-br from-indigo-550 to-indigo-700 rounded-xl p-5 text-white shadow-md flex flex-col md:flex-row md:items-center justify-between gap-6 relative overflow-hidden">
        {/* Abstract background vector accent */}
        <div className="absolute right-0 bottom-0 top-0 w-1/3 opacity-15 flex items-center justify-center pointer-events-none">
          <Cpu className="w-48 h-48 text-indigo-300" />
        </div>

        {/* Spec Overview */}
        <div className="z-10 flex flex-col gap-1.5">
          <span className="text-[10px] uppercase font-bold tracking-widest text-indigo-200 block mb-0.5">
            Active Platform Accelerator / 正在运行的计算卡规格 
          </span>
          <h3 className="text-xl font-bold tracking-tight">
            {selectedGPUType.name}
          </h3>
          <div className="flex flex-wrap gap-2.5 mt-1 text-[11px] text-indigo-100 font-mono">
            <span className="bg-white/10 px-2 py-0.5 rounded border border-white/15">
              物理单卡: <strong>{selectedGPUType.vram} GB VRAM</strong>
            </span>
            <span className="bg-white/10 px-2 py-0.5 rounded border border-white/15">
              显存接口位宽: {selectedGPUType.busWidth || 'HBM2/Unified'}
            </span>
            <span className="bg-white/10 px-2 py-0.5 rounded border border-white/15">
              总线总带宽: {selectedGPUType.bandwidth || '未知'}
            </span>
            <span className="bg-white/15 text-indigo-300 font-bold px-1.5 rounded uppercase text-[10px]">
              {selectedGPUType.type.toUpperCase()}
            </span>
          </div>
        </div>

        {/* Cluster Evaluation Sizing Output */}
        <div className="z-10 bg-white/10 border border-white/15 backdrop-blur-sm rounded-lg p-3.5 min-w-[240px] flex flex-col gap-2 shrink-0">
          <div className="flex justify-between items-center text-xs">
            <span className="text-indigo-200">最小部署所需硬件节点:</span>
            <strong className="text-white text-sm font-mono">
              {cardsNeeded} 卡 ({selectedGPUType.type === 'mac' ? 'Mac芯片单元' : '加速卡'})
            </strong>
          </div>

          <div className="flex justify-between items-center text-xs">
            <span className="text-indigo-200">满载平均负载 occupancy:</span>
            <strong className="text-white font-mono">
              {percentPerCard}%
            </strong>
          </div>

          {/* Micro Slider graphic output */}
          <div className="w-full bg-indigo-900/40 rounded-full h-1 my-1 overflow-hidden">
            <div 
              className={`h-full rounded-full transition-all duration-300 ${
                percentPerCard > 92 
                  ? 'bg-rose-400' 
                  : percentPerCard > 80 
                    ? 'bg-amber-300' 
                    : 'bg-emerald-400'
              }`}
              style={{ width: `${Math.min(100, percentPerCard)}%` }}
            />
          </div>

          {/* Load health tag */}
          <div className={`mt-1 inline-flex items-center gap-1.5 self-start text-[10px] font-bold px-2 py-0.5 rounded border ${loadBadgeColor}`}>
            <ImpactIcon className="w-3.5 h-3.5" />
            <span>【{loadLabel}】</span>
          </div>
        </div>
      </div>

      {/* Hardware Fit Callout Info */}
      <div className="bg-slate-50 border border-slate-150 rounded-xl p-4 text-xs text-slate-600 leading-relaxed flex flex-col gap-2">
        <div className="font-bold text-slate-800 flex items-center gap-1.5">
          <CheckCircle2 className="w-4 h-4 text-indigo-600" />
          💡 全球多端硬件分布式技术架构常识建议
        </div>
        <p>
          1. <strong>NVIDIA (SXM / NVLink):</strong> 支持 NVLink 并行组网，在机架级 TP 划分下显存几乎是无损共享。
        </p>
        <p>
          2. <strong>AMD Instinct (Infinity Fabric):</strong> MI300X 自带高达 192GB 巨量 HBM3 显存，单卡即可吞下如 Llama3-70B 的低量化版本。在 ROCm 驱动套件支持下可完成弹性浮点切换。
        </p>
        <p>
          3. <strong>Apple Silicon (Unified Memory):</strong> 运行于 Mac 机器（macOS 统一内存阵列）。CPU/GPU 共享物理大内存。可通过 Metal / Llama.cpp 将本地 90% 物理内存分配为 VRAM 使用，免遭 OOM 问题，对极大型模型（如 70B/405B）极具性价比。
        </p>
        <p>
          4. <strong>华为昇腾 (Huawei Ascend - MindSpore/CANN):</strong> 主要围绕 Ascend 910B 算力卡进行全量参数加载及训练优化，是国产自主可控智算中心的首选硬件平台。
        </p>
      </div>

    </div>
  );
};
