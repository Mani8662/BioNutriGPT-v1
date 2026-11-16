import React from 'react'
import { Activity, ShieldCheck, Zap, HeartPulse } from 'lucide-react'
import { HealthStatus } from '../types/api'

interface NavbarProps {
  health: HealthStatus | null
  activeTab: 'input' | 'results'
  hasResults: boolean
  onTabChange: (tab: 'input' | 'results') => void
  onReset: () => void
}

export const Navbar: React.FC<NavbarProps> = ({
  health,
  activeTab,
  hasResults,
  onTabChange,
  onReset,
}) => {
  return (
    <header className="bg-slate-900 border-b border-slate-800 text-white sticky top-0 z-50 shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Clinical Brand */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => onTabChange('input')}>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <HeartPulse className="w-6 h-6 text-slate-950 font-bold" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-100 to-emerald-300 bg-clip-text text-transparent">
                  BioNutriGPT
                </span>
                <span className="px-2 py-0.5 text-[10px] font-semibold tracking-wider uppercase bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 rounded-full">
                  Clinical 4-Stage
                </span>
              </div>
              <p className="text-xs text-slate-400 font-medium hidden sm:block">
                Cardiometabolic Diagnostic & Indian Meal Synthesis Platform
              </p>
            </div>
          </div>

          {/* Navigation Controls */}
          <div className="flex items-center space-x-3">
            <button
              onClick={() => onTabChange('input')}
              className={`px-3.5 py-1.5 rounded-lg text-sm font-semibold transition-all ${
                activeTab === 'input'
                  ? 'bg-slate-800 text-emerald-400 border border-slate-700 shadow-inner'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              Patient Intake
            </button>
            <button
              disabled={!hasResults}
              onClick={() => onTabChange('results')}
              className={`px-3.5 py-1.5 rounded-lg text-sm font-semibold transition-all flex items-center space-x-1.5 ${
                !hasResults
                  ? 'text-slate-600 cursor-not-allowed'
                  : activeTab === 'results'
                  ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-600/30'
                  : 'text-emerald-400 hover:bg-slate-800/80 border border-emerald-500/30'
              }`}
            >
              <Activity className="w-4 h-4" />
              <span>Clinical Results</span>
            </button>

            {/* Quick Reset */}
            <button
              onClick={onReset}
              title="Reset intake form"
              className="text-xs text-slate-400 hover:text-slate-200 px-2 py-1 rounded hover:bg-slate-800/80 transition"
            >
              Reset
            </button>
          </div>

          {/* Hardware & Safety Status */}
          <div className="hidden lg:flex items-center space-x-4 pl-4 border-l border-slate-800 text-xs">
            <div className="flex items-center space-x-1.5">
              <span className={`w-2 h-2 rounded-full ${health?.cuda_available ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
              <span className="text-slate-300 font-medium">
                {health?.cuda_available ? health.cuda_device || 'CUDA GPU Active' : 'CPU Inference'}
              </span>
            </div>
            <div className="flex items-center space-x-1 text-slate-400">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              <span>AST Guarded</span>
            </div>
            <div className="flex items-center space-x-1 text-slate-400">
              <Zap className="w-3.5 h-3.5 text-blue-400" />
              <span>HiGHS MILP</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
