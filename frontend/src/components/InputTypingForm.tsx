import React, { useState } from 'react'
import {
  Search,
  RotateCcw,
  Sparkles,
  ChevronDown,
  ChevronRight,
  Info,
  Sliders,
  User,
  Heart,
  Utensils,
  Sun,
  Activity,
  Layers,
} from 'lucide-react'
import { FEATURE_CATEGORIES, FEATURE_DEFINITIONS } from '../data/featureDefinitions'
import { PatientMetadata } from '../types/api'

interface InputTypingFormProps {
  features: number[]
  metadata: PatientMetadata
  onFeatureChange: (index: number, value: number) => void
  onMetadataChange: (field: keyof PatientMetadata, value: any) => void
  onLoadSample: () => void
  onReset: () => void
}

const CATEGORY_ICONS: Record<string, React.ReactNode> = {
  'Demographics & Vitals': <User className="w-4 h-4 text-blue-500" />,
  'Serum Biomarkers': <Heart className="w-4 h-4 text-rose-500" />,
  'Dietary Macronutrients': <Utensils className="w-4 h-4 text-amber-500" />,
  'Vitamins & Micronutrients': <Sun className="w-4 h-4 text-yellow-500" />,
  'Minerals & Bioactives': <Sparkles className="w-4 h-4 text-emerald-500" />,
  'Dietary Ratios & AMDR': <Sliders className="w-4 h-4 text-indigo-500" />,
  'Lifestyle & Physical Activity': <Activity className="w-4 h-4 text-cyan-500" />,
  'Data Quality & Missingness': <Layers className="w-4 h-4 text-slate-500" />,
}

export const InputTypingForm: React.FC<InputTypingFormProps> = ({
  features,
  metadata,
  onFeatureChange,
  onMetadataChange,
  onLoadSample,
  onReset,
}) => {
  const [searchQuery, setSearchQuery] = useState('')
  const [openCategories, setOpenCategories] = useState<Record<string, boolean>>({
    'Demographics & Vitals': true,
    'Serum Biomarkers': true,
    'Dietary Macronutrients': true,
    'Vitamins & Micronutrients': false,
    'Minerals & Bioactives': false,
    'Dietary Ratios & AMDR': false,
    'Lifestyle & Physical Activity': false,
    'Data Quality & Missingness': false,
  })

  const toggleCategory = (cat: string) => {
    setOpenCategories((prev) => ({ ...prev, [cat]: !prev[cat] }))
  }

  const expandAll = () => {
    const all: Record<string, boolean> = {}
    FEATURE_CATEGORIES.forEach((c) => (all[c] = true))
    setOpenCategories(all)
  }

  const collapseAll = () => {
    const all: Record<string, boolean> = {}
    FEATURE_CATEGORIES.forEach((c) => (all[c] = false))
    setOpenCategories(all)
  }

  // Filter features based on search query
  const filteredFeatures = FEATURE_DEFINITIONS.filter(
    (f) =>
      f.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
      f.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      f.category.toLowerCase().includes(searchQuery.toLowerCase())
  )

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
      {/* Header & Quick Action Bar */}
      <div className="p-5 border-b border-slate-200 bg-slate-50/70 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
            <h2 className="text-lg font-bold text-slate-900 tracking-tight">
              Enter patient information
            </h2>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button
            type="button"
            onClick={onLoadSample}
            className="px-3.5 py-1.5 bg-emerald-50 text-emerald-700 hover:bg-emerald-100 border border-emerald-300 rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition shadow-sm"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Pre-fill Sample Patient</span>
          </button>
          <button
            type="button"
            onClick={onReset}
            className="px-3 py-1.5 bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-300 rounded-lg text-xs font-medium flex items-center space-x-1 transition"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Defaults</span>
          </button>
        </div>
      </div>

      {/* Patient Clinical Metadata Card (Age, Sex, Weight, Height, Activity) */}
      <div className="p-5 bg-gradient-to-r from-blue-50/60 to-indigo-50/40 border-b border-slate-200">
        <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-3 flex items-center space-x-1.5">
          <User className="w-4 h-4 text-blue-600" />
          <span>Patient Baseline Vitals & Physical Anthropometrics</span>
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3.5">
          {/* Age */}
          <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-sm">
            <label className="block text-[11px] font-semibold text-slate-600 mb-1">
              Age (Years)
            </label>
            <input
              type="number"
              min="18"
              max="105"
              step="1"
              value={metadata.age_years}
              onChange={(e) => onMetadataChange('age_years', parseFloat(e.target.value) || 0)}
              className="w-full font-mono text-sm font-semibold text-slate-900 border border-slate-200 rounded-md px-2 py-1 focus:ring-2 focus:ring-blue-500 focus:outline-none"
            />
          </div>

          {/* Biological Sex */}
          <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-sm">
            <label className="block text-[11px] font-semibold text-slate-600 mb-1">
              Biological Sex
            </label>
            <select
              value={metadata.is_female ? 'female' : 'male'}
              onChange={(e) => onMetadataChange('is_female', e.target.value === 'female')}
              className="w-full text-sm font-semibold text-slate-900 border border-slate-200 rounded-md px-2 py-1 bg-white focus:ring-2 focus:ring-blue-500 focus:outline-none"
            >
              <option value="male">Male</option>
              <option value="female">Female</option>
            </select>
          </div>

          {/* Weight */}
          <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-sm">
            <label className="block text-[11px] font-semibold text-slate-600 mb-1">
              Weight (kg)
            </label>
            <input
              type="number"
              min="30"
              max="250"
              step="0.5"
              value={metadata.weight_kg}
              onChange={(e) => onMetadataChange('weight_kg', parseFloat(e.target.value) || 0)}
              className="w-full font-mono text-sm font-semibold text-slate-900 border border-slate-200 rounded-md px-2 py-1 focus:ring-2 focus:ring-blue-500 focus:outline-none"
            />
          </div>

          {/* Height */}
          <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-sm">
            <label className="block text-[11px] font-semibold text-slate-600 mb-1">
              Height (cm)
            </label>
            <input
              type="number"
              min="100"
              max="230"
              step="0.5"
              value={metadata.height_cm}
              onChange={(e) => onMetadataChange('height_cm', parseFloat(e.target.value) || 0)}
              className="w-full font-mono text-sm font-semibold text-slate-900 border border-slate-200 rounded-md px-2 py-1 focus:ring-2 focus:ring-blue-500 focus:outline-none"
            />
          </div>

          {/* Activity Level */}
          <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-sm col-span-2 sm:col-span-1">
            <label className="block text-[11px] font-semibold text-slate-600 mb-1">
              Activity Level
            </label>
            <select
              value={metadata.activity_level}
              onChange={(e) => onMetadataChange('activity_level', e.target.value)}
              className="w-full text-xs font-semibold text-slate-900 border border-slate-200 rounded-md px-2 py-1 bg-white focus:ring-2 focus:ring-blue-500 focus:outline-none capitalize"
            >
              <option value="sedentary">Sedentary (1.2×)</option>
              <option value="light">Light (1.375×)</option>
              <option value="moderate">Moderate (1.55×)</option>
              <option value="active">Active (1.725×)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Search Bar & Expansion Controls */}
      <div className="p-4 bg-white border-b border-slate-200 flex flex-wrap items-center justify-between gap-3">
        <div className="relative flex-1 min-w-[240px]">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search features by name, label, or category (e.g. cholesterol, vitamin c, sleep)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-1.5 text-xs border border-slate-200 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
          />
        </div>

        <div className="flex items-center space-x-2 text-xs">
          <button
            type="button"
            onClick={expandAll}
            className="text-slate-600 hover:text-slate-900 px-2 py-1 rounded hover:bg-slate-100"
          >
            Expand All
          </button>
          <span className="text-slate-300">|</span>
          <button
            type="button"
            onClick={collapseAll}
            className="text-slate-600 hover:text-slate-900 px-2 py-1 rounded hover:bg-slate-100"
          >
            Collapse All
          </button>
        </div>
      </div>

      {/* Categorized Features Grid */}
      <div className="p-5 space-y-4 max-h-[600px] overflow-y-auto">
        {FEATURE_CATEGORIES.map((category) => {
          const categoryFeatures = filteredFeatures.filter((f) => f.category === category)
          if (categoryFeatures.length === 0) return null

          const isOpen = searchQuery ? true : !!openCategories[category]

          return (
            <div key={category} className="border border-slate-200 rounded-xl overflow-hidden shadow-sm">
              {/* Category Accordion Header */}
              <button
                type="button"
                onClick={() => toggleCategory(category)}
                className="w-full px-4 py-3 bg-slate-50 hover:bg-slate-100 flex items-center justify-between text-left transition border-b border-slate-200/60"
              >
                <div className="flex items-center space-x-2.5">
                  <span className="p-1 rounded-md bg-white border border-slate-200 shadow-2xs">
                    {CATEGORY_ICONS[category] || <Info className="w-4 h-4 text-slate-500" />}
                  </span>
                  <div>
                    <h3 className="text-xs font-bold text-slate-800 tracking-tight">{category}</h3>
                    <p className="text-[10px] text-slate-400 font-normal">
                      {categoryFeatures.length} {categoryFeatures.length === 1 ? 'parameter' : 'parameters'}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-200/70 text-slate-600 font-semibold font-mono">
                    {categoryFeatures.length}
                  </span>
                  {isOpen ? (
                    <ChevronDown className="w-4 h-4 text-slate-400" />
                  ) : (
                    <ChevronRight className="w-4 h-4 text-slate-400" />
                  )}
                </div>
              </button>

              {/* Category Feature Inputs Grid */}
              {isOpen && (
                <div className="p-4 bg-white grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {categoryFeatures.map((field) => {
                    // Global index in the 87 features array
                    const featureIndex = FEATURE_DEFINITIONS.findIndex((d) => d.name === field.name)
                    const val = features[featureIndex] !== undefined ? features[featureIndex] : field.default_val

                    return (
                      <div
                        key={field.name}
                        className="p-2.5 rounded-lg border border-slate-100 hover:border-slate-300 bg-slate-50/40 hover:bg-white transition flex flex-col justify-between"
                      >
                        <div className="flex items-start justify-between mb-1.5">
                          <label
                            htmlFor={`field-${field.name}`}
                            className="text-xs font-semibold text-slate-800 leading-tight pr-2"
                            title={field.description}
                          >
                            {field.label}
                          </label>
                          <span className="text-[10px] text-slate-400 font-mono shrink-0">
                            {field.unit}
                          </span>
                        </div>

                        <div className="flex items-center space-x-2">
                          <input
                            id={`field-${field.name}`}
                            type="number"
                            step={field.step}
                            min={field.min_val}
                            max={field.max_val}
                            value={val}
                            onChange={(e) => {
                              const parsed = parseFloat(e.target.value)
                              onFeatureChange(featureIndex, isNaN(parsed) ? 0 : parsed)
                            }}
                            className="w-full text-xs font-mono font-medium text-slate-900 bg-white border border-slate-200 rounded px-2 py-1 focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                          />
                        </div>

                        <div className="flex items-center justify-between text-[10px] text-slate-400 mt-1">
                          <span className="font-mono text-[9px] text-slate-400">{field.name}</span>
                          <span className="text-[9px] text-slate-400">def: {field.default_val}</span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
