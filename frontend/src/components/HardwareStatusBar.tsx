import React from 'react'
import { Cpu, Database, Server, ShieldCheck } from 'lucide-react'
import { HealthStatus } from '../types/api'

interface HardwareStatusBarProps {
  health: HealthStatus | null
  stageCount?: number
}

export const HardwareStatusBar: React.FC<HardwareStatusBarProps> = ({
  health,
  stageCount = 4,
}) => {
  return (
    <aside aria-label="System status" className="bg-slate-900 border-t border-slate-800 text-slate-400 text-xs py-2 px-4 shadow-inner">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1.5">
            <Server className="w-3.5 h-3.5 text-slate-500" />
            <span className="text-slate-300 font-medium">Backend:</span>
            <span className="text-emerald-400 font-mono">FastAPI :8000</span>
          </div>

          <div className="flex items-center space-x-1.5">
            <Cpu className="w-3.5 h-3.5 text-slate-500" />
            <span className="text-slate-300 font-medium">Device:</span>
            <span className="text-slate-200">
              {health?.cuda_available ? (
                <span className="text-emerald-400 font-mono">{health.cuda_device} (NF4 4-bit)</span>
              ) : (
                <span className="text-amber-400">Host CPU (PyTorch)</span>
              )}
            </span>
          </div>

          {health?.vram_total_gb && (
            <div className="hidden sm:flex items-center space-x-1.5">
              <Database className="w-3.5 h-3.5 text-slate-500" />
              <span className="text-slate-300 font-medium">VRAM:</span>
              <span className="font-mono text-slate-300">
                {health.vram_free_gb} GB free / {health.vram_total_gb} GB
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1 text-slate-400">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-slate-300">AST Guarded</span>
          </div>

          <div className="hidden md:flex items-center space-x-1 text-slate-500">
            <span>Stages Active:</span>
            <span className="text-emerald-400 font-semibold">{stageCount}/4</span>
          </div>
        </div>
      </div>
    </aside>
  )
}
