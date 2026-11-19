import React from 'react'
import { Activity, BookOpen, HeartPulse, History, Info, LogOut, RotateCcw, ShieldCheck, UserRound } from 'lucide-react'
import { AccountUser, HealthStatus } from '../types/api'

export type AppSection = 'input' | 'results' | 'history' | 'account' | 'about' | 'admin'

interface NavbarProps {
  health: HealthStatus | null
  activeTab: AppSection
  hasResults: boolean
  account: AccountUser | null
  onTabChange: (tab: AppSection) => void
  onAccountClick: () => void
  onLogout: () => void
  onReset: () => void
}

export const Navbar: React.FC<NavbarProps> = ({ health, activeTab, hasResults, account, onTabChange, onAccountClick, onLogout, onReset }) => {
  const itemClass = (section: AppSection, disabled = false) => `flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm font-semibold transition ${disabled ? 'cursor-not-allowed text-slate-600' : activeTab === section ? 'bg-emerald-500/15 text-emerald-300 ring-1 ring-inset ring-emerald-400/20' : 'text-slate-300 hover:bg-white/5 hover:text-white'}`

  return (
    <aside className="sticky top-0 z-40 w-full shrink-0 border-b border-slate-800 bg-slate-950 text-white shadow-lg md:h-screen md:w-64 md:border-b-0 md:border-r">
      <div className="flex h-full flex-col">
        <button type="button" onClick={() => onTabChange('input')} className="flex items-center gap-3 px-4 py-4 text-left sm:px-5 md:py-6">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400"><HeartPulse className="h-6 w-6 text-slate-950" /></span>
          <span><span className="block text-lg font-bold tracking-tight">BioNutriGPT</span><span className="block text-[10px] font-semibold uppercase tracking-wider text-slate-400">Nutrition insights</span></span>
        </button>

        <nav aria-label="Main navigation" className="grid grid-cols-2 gap-1 px-3 pb-3 sm:grid-cols-3 md:flex md:flex-1 md:flex-col md:px-3">
          <p className="hidden px-3 pb-1 pt-3 text-[10px] font-bold uppercase tracking-[0.16em] text-slate-500 md:block">Workspace</p>
          <button type="button" onClick={() => onTabChange('input')} className={itemClass('input')}><BookOpen className="h-4 w-4 shrink-0" /><span>Patient intake</span></button>
          <button type="button" disabled={!hasResults} onClick={() => onTabChange('results')} aria-disabled={!hasResults} className={itemClass('results', !hasResults)}><Activity className="h-4 w-4 shrink-0" /><span>Clinical results</span></button>
          {account && <button type="button" onClick={() => onTabChange('history')} className={itemClass('history')}><History className="h-4 w-4 shrink-0" /><span>Risk history</span></button>}
          {account ? <button type="button" onClick={() => onTabChange('account')} className={itemClass('account')}><UserRound className="h-4 w-4 shrink-0" /><span>Account</span></button> : <button type="button" onClick={onAccountClick} className={itemClass('account')}><UserRound className="h-4 w-4 shrink-0" /><span>Sign in / create account</span></button>}
          <p className="hidden px-3 pb-1 pt-5 text-[10px] font-bold uppercase tracking-[0.16em] text-slate-500 md:block">Information</p>
          <button type="button" onClick={() => onTabChange('about')} className={itemClass('about')}><Info className="h-4 w-4 shrink-0" /><span>About</span></button>
          {account?.role === 'admin' && <button type="button" onClick={() => onTabChange('admin')} className={itemClass('admin')}><ShieldCheck className="h-4 w-4 shrink-0" /><span>Admin</span></button>}
        </nav>

        <div className="hidden border-t border-slate-800 p-4 md:block">
          <div className="flex items-center gap-2 text-xs"><span className={`h-2 w-2 rounded-full ${health ? 'bg-emerald-400' : 'bg-amber-400'}`} /><span className="font-semibold text-slate-300">{health ? 'API connected' : 'Checking API'}</span></div>
          <p className="mt-1 pl-4 text-[11px] text-slate-500">{health?.cuda_available ? health.cuda_device || 'CUDA inference' : 'CPU inference'}</p>
          <div className="mt-4 flex items-center justify-between gap-2">
            {account ? <span className="min-w-0 truncate text-[11px] text-slate-400" title={account.email}>{account.email}</span> : <span className="text-[11px] text-slate-500">Research prototype</span>}
            {account && <button type="button" onClick={onLogout} title="Sign out" aria-label="Sign out" className="rounded-lg p-2 text-slate-400 hover:bg-white/5 hover:text-white"><LogOut className="h-4 w-4" /></button>}
          </div>
          {activeTab === 'input' && <button type="button" onClick={onReset} className="mt-3 flex w-full items-center gap-2 rounded-lg px-2 py-2 text-xs font-medium text-slate-400 hover:bg-white/5 hover:text-white"><RotateCcw className="h-3.5 w-3.5" />Reset intake</button>}
        </div>
      </div>
    </aside>
  )
}
