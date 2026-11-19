import React, { useEffect, useMemo, useState } from 'react'
import { Activity, RefreshCw, Server, ShieldCheck, Users } from 'lucide-react'
import { getAdminSummary } from '../services/api'
import { AdminSummary, HealthStatus } from '../types/api'

interface Props { health: HealthStatus | null }

export const AdminDashboard: React.FC<Props> = ({ health }) => {
  const [summary, setSummary] = useState<AdminSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = async () => {
    setLoading(true)
    setError(null)
    try { setSummary(await getAdminSummary()) }
    catch (err) { setError(err instanceof Error ? err.message : 'Could not load administrator metrics.') }
    finally { setLoading(false) }
  }

  useEffect(() => { void refresh() }, [])

  const chartDays = useMemo(() => {
    const counts = new Map((summary?.daily_assessments ?? []).map((item) => [item.date, item.count]))
    return Array.from({ length: 7 }, (_, index) => {
      const date = new Date()
      date.setUTCHours(0, 0, 0, 0)
      date.setUTCDate(date.getUTCDate() - (6 - index))
      const key = date.toISOString().slice(0, 10)
      return { key, date, count: counts.get(key) ?? 0 }
    })
  }, [summary])
  const maxCount = Math.max(1, ...chartDays.map((day) => day.count))

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <section className="flex flex-wrap items-end justify-between gap-4 rounded-3xl bg-gradient-to-br from-slate-950 to-indigo-950 p-8 text-white">
        <div><p className="text-xs font-bold uppercase tracking-[0.18em] text-indigo-300">Administration</p><h1 className="mt-2 text-3xl font-bold">System overview</h1><p className="mt-2 text-sm text-slate-300">Aggregate usage and service status. Assessment values and account emails are kept out of this page.</p></div>
        <button type="button" onClick={() => void refresh()} disabled={loading} className="inline-flex items-center gap-2 rounded-xl border border-white/20 px-4 py-2.5 text-sm font-semibold hover:bg-white/10 disabled:opacity-50"><RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />Refresh</button>
      </section>

      {error && <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">{error}</div>}

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {[
          { label: 'Accounts', value: summary?.total_accounts, icon: Users, color: 'text-blue-700 bg-blue-50' },
          { label: 'Assessments saved', value: summary?.total_assessments, icon: Activity, color: 'text-emerald-700 bg-emerald-50' },
          { label: 'Assessments · 7 days', value: summary?.assessments_last_7_days, icon: RefreshCw, color: 'text-violet-700 bg-violet-50' },
          { label: 'Administrator accounts', value: summary?.admin_accounts, icon: ShieldCheck, color: 'text-amber-700 bg-amber-50' },
        ].map(({ label, value, icon: Icon, color }) => <article key={label} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><div className="flex items-center justify-between"><span className="text-sm font-semibold text-slate-500">{label}</span><span className={`flex h-9 w-9 items-center justify-center rounded-xl ${color}`}><Icon className="h-4 w-4" /></span></div><p className="mt-4 text-3xl font-bold text-slate-900">{loading && value === undefined ? '…' : value ?? '—'}</p></article>)}
      </section>

      <section className="grid gap-6 lg:grid-cols-[1.4fr_0.8fr]">
        <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
          <h2 className="font-bold text-slate-900">Daily assessment volume</h2><p className="mt-1 text-xs text-slate-500">Last seven days, UTC</p>
          <div className="mt-6 grid h-48 grid-cols-7 items-end gap-3 border-b border-slate-200 pb-2">
            {chartDays.map((day) => <div key={day.key} className="flex h-full flex-col items-center justify-end gap-2">
              <span className="text-xs font-semibold text-slate-600">{day.count}</span>
              <div title={`${day.count} assessments`} className="w-full max-w-12 rounded-t-lg bg-gradient-to-t from-emerald-700 to-teal-400" style={{ height: `${Math.max(day.count ? 10 : 2, day.count / maxCount * 72)}%` }} />
              <span className="text-[10px] text-slate-500">{day.date.toLocaleDateString(undefined, { weekday: 'short', timeZone: 'UTC' })}</span>
            </div>)}
          </div>
        </article>

        <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
          <h2 className="font-bold text-slate-900">Service status</h2>
          <div className="mt-4 flex items-center justify-between rounded-xl bg-slate-50 p-4"><div className="flex items-center gap-3"><span className={`h-2.5 w-2.5 rounded-full ${health ? 'bg-emerald-500' : 'bg-rose-500'}`} /><div><p className="text-sm font-semibold text-slate-800">API</p><p className="text-xs text-slate-500">{health ? 'Responding to health checks' : 'No health response'}</p></div></div><Server className="h-5 w-5 text-slate-400" /></div>
          <div className="mt-3 flex items-center justify-between rounded-xl bg-slate-50 p-4"><div><p className="text-sm font-semibold text-slate-800">Inference device</p><p className="mt-0.5 text-xs text-slate-500">{health?.cuda_available ? health.cuda_device || 'CUDA GPU' : 'CPU'}</p></div><span className="rounded-full bg-white px-2.5 py-1 text-[11px] font-semibold text-slate-600">{health?.cuda_available ? `${health.vram_free_gb ?? '—'} GB VRAM free` : 'Active'}</span></div>
          <p className="mt-4 text-xs leading-5 text-slate-500">LLM availability is configured separately through the Stage 4 model host.</p>
        </article>
      </section>

      <section className="grid gap-6 lg:grid-cols-2">
        <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6"><h2 className="font-bold text-slate-900">Recent activity</h2><p className="mt-1 text-xs text-slate-500">Timestamps and source only; no user identifiers or risk scores.</p>
          <ul className="mt-4 divide-y divide-slate-100">{summary?.recent_activity.length ? summary.recent_activity.map((item, index) => <li key={`${item.created_at}-${index}`} className="flex items-center justify-between gap-3 py-3 text-sm"><span className="capitalize text-slate-700">{item.source} assessment</span><time className="shrink-0 text-xs text-slate-500">{new Date(item.created_at).toLocaleString()}</time></li>) : <li className="py-6 text-center text-sm text-slate-500">No assessments saved yet.</li>}</ul>
        </article>
        <article className="rounded-2xl border border-indigo-200 bg-indigo-50 p-5 sm:p-6"><h2 className="font-bold text-indigo-950">Manage administrators</h2><p className="mt-2 text-sm leading-6 text-indigo-900">Administrator access is assigned by the backend environment setting <code className="rounded bg-white/70 px-1.5 py-0.5">BIONUTRIGPT_ADMIN_EMAILS</code>. Add trusted email addresses there and restart the API; users cannot grant themselves administrator access.</p><p className="mt-3 text-xs leading-5 text-indigo-800">This dashboard shows aggregate counts only. Keep the SQLite database and signing secret on persistent, access-controlled storage.</p></article>
      </section>
    </div>
  )
}
