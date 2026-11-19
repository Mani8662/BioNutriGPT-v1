import React, { useMemo, useState } from 'react'
import { Download, RefreshCw, Trash2, TrendingDown, TrendingUp } from 'lucide-react'
import { AccountUser, RiskHistoryEntry } from '../types/api'

interface Props {
  account: AccountUser
  records: RiskHistoryEntry[]
  loading: boolean
  onRefresh: () => void
  onDelete: (id: string) => void
}

const RISK_SERIES = [
  { key: 'diabetes', label: 'Diabetes', color: '#0f766e' },
  { key: 'obesity', label: 'Obesity', color: '#2563eb' },
  { key: 'cvd', label: 'Cardiovascular', color: '#c2415d' },
] as const

type RiskKey = typeof RISK_SERIES[number]['key']
type RangeKey = '30' | '90' | 'all'

const formatDate = (value: string, options: Intl.DateTimeFormatOptions = { month: 'short', day: 'numeric', year: 'numeric' }) =>
  new Date(value).toLocaleDateString(undefined, options)

const score = (value: number) => `${Math.round(value * 100)}%`

export const RiskHistoryDashboard: React.FC<Props> = ({ account, records, loading, onRefresh, onDelete }) => {
  const [range, setRange] = useState<RangeKey>('all')
  const [visible, setVisible] = useState<Record<RiskKey, boolean>>({ diabetes: true, obesity: true, cvd: true })
  const filteredRecords = useMemo(() => {
    if (range === 'all' || records.length === 0) return records
    const minTime = Date.now() - Number(range) * 24 * 60 * 60 * 1000
    return records.filter((record) => new Date(record.created_at).getTime() >= minTime)
  }, [range, records])

  const exportCsv = () => {
    const rows = [
      ['date', 'source', 'diabetes_probability', 'obesity_probability', 'cardiovascular_probability'],
      ...records.map((record) => [record.created_at, record.source, record.probabilities.diabetes, record.probabilities.obesity, record.probabilities.cvd]),
    ]
    const csv = rows.map((row) => row.map((value) => `"${String(value).replace(/"/g, '""')}"`).join(',')).join('\n')
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'bionutrigpt-risk-history.csv'
    anchor.click()
    URL.revokeObjectURL(url)
  }

  const trendDelta = (key: RiskKey) => {
    if (filteredRecords.length < 2) return null
    return (filteredRecords[filteredRecords.length - 1].probabilities[key] - filteredRecords[0].probabilities[key]) * 100
  }

  const width = 820
  const height = 300
  const left = 54
  const right = 20
  const top = 18
  const bottom = 38
  const plotWidth = width - left - right
  const plotHeight = height - top - bottom
  const x = (index: number) => filteredRecords.length <= 1 ? left + plotWidth / 2 : left + index * plotWidth / (filteredRecords.length - 1)
  const y = (probability: number) => top + (1 - probability) * plotHeight

  return (
    <div className="space-y-6">
      <section className="flex flex-col justify-between gap-4 rounded-2xl bg-gradient-to-br from-slate-900 to-emerald-950 p-6 text-white sm:flex-row sm:items-end">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.16em] text-emerald-300">Personal health timeline</p>
          <h1 className="mt-2 text-2xl font-bold">Risk history</h1>
          <p className="mt-1 text-sm text-slate-300">Assessments for {account.email}</p>
        </div>
        <div className="flex gap-2">
          <button type="button" onClick={onRefresh} disabled={loading} className="inline-flex items-center gap-2 rounded-xl border border-white/20 px-3.5 py-2.5 text-sm font-semibold hover:bg-white/10 disabled:opacity-50"><RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />Refresh</button>
          <button type="button" onClick={exportCsv} disabled={!records.length} className="inline-flex items-center gap-2 rounded-xl bg-white px-3.5 py-2.5 text-sm font-semibold text-slate-900 hover:bg-emerald-50 disabled:opacity-50"><Download className="h-4 w-4" />Export CSV</button>
        </div>
      </section>

      {records.length === 0 ? (
        <section className="rounded-2xl border border-dashed border-slate-300 bg-white p-10 text-center">
          <h2 className="font-semibold text-slate-900">No saved assessments yet</h2>
          <p className="mx-auto mt-2 max-w-md text-sm text-slate-600">Run an assessment while signed in and its risk estimates will appear here over time.</p>
        </section>
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-3">
            {RISK_SERIES.map((series) => {
              const latestRecord = filteredRecords[filteredRecords.length - 1]
              const latest = latestRecord?.probabilities[series.key]
              const delta = trendDelta(series.key)
              const DeltaIcon = delta === null || delta === 0 ? null : delta > 0 ? TrendingUp : TrendingDown
              return <article key={series.key} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-600"><span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: series.color }} />{series.label} estimate</div>
                <div className="mt-3 flex items-end justify-between">
                  <p className="text-3xl font-bold text-slate-900">{latest === undefined ? '—' : score(latest)}</p>
                  {delta !== null && <p className={`flex items-center gap-1 text-sm font-semibold ${delta > 0 ? 'text-rose-700' : delta < 0 ? 'text-emerald-700' : 'text-slate-500'}`}>{DeltaIcon && <DeltaIcon className="h-4 w-4" />}{delta > 0 ? '+' : ''}{delta.toFixed(1)} pp</p>}
                </div>
                <p className="mt-1 text-xs text-slate-500">{latestRecord ? `Latest in range · ${formatDate(latestRecord.created_at)}` : 'No assessments in this range'}</p>
              </article>
            })}
          </div>

          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
            <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
              <div><h2 className="text-lg font-bold text-slate-900">Risk estimates over time</h2><p className="mt-1 text-xs text-slate-500">Model probabilities are estimates and are not diagnoses.</p></div>
              <div className="flex rounded-xl bg-slate-100 p-1">
                {(['30', '90', 'all'] as const).map((item) => <button key={item} type="button" onClick={() => setRange(item)} className={`rounded-lg px-3 py-1.5 text-xs font-semibold ${range === item ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}>{item === 'all' ? 'All time' : `${item} days`}</button>)}
              </div>
            </div>

            <div className="mt-5 flex flex-wrap gap-2">
              {RISK_SERIES.map((series) => <button key={series.key} type="button" aria-pressed={visible[series.key]} onClick={() => setVisible((prev) => ({ ...prev, [series.key]: !prev[series.key] }))} className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold ${visible[series.key] ? 'border-slate-200 bg-white text-slate-700' : 'border-transparent bg-slate-100 text-slate-400'}`}><span className="h-2 w-2 rounded-full" style={{ backgroundColor: series.color }} />{series.label}</button>)}
            </div>

            <div className="mt-4 overflow-x-auto">
              {filteredRecords.length === 0 ? <p className="py-12 text-center text-sm text-slate-500">No assessments in this time range.</p> : <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Line chart of diabetes, obesity, and cardiovascular risk estimates over time" className="min-w-[620px] w-full">
                {[0, 25, 50, 75, 100].map((tick) => <g key={tick}><line x1={left} x2={width - right} y1={y(tick / 100)} y2={y(tick / 100)} stroke="#e2e8f0" strokeDasharray={tick === 0 ? undefined : '4 5'} /><text x={left - 10} y={y(tick / 100) + 4} textAnchor="end" fontSize="11" fill="#64748b">{tick}%</text></g>)}
                {RISK_SERIES.filter((series) => visible[series.key]).map((series) => <g key={series.key}>
                  {filteredRecords.length > 1 && <polyline fill="none" stroke={series.color} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" points={filteredRecords.map((record, index) => `${x(index)},${y(record.probabilities[series.key])}`).join(' ')} />}
                  {filteredRecords.map((record, index) => <circle key={record.id} cx={x(index)} cy={y(record.probabilities[series.key])} r="4" fill="white" stroke={series.color} strokeWidth="2"><title>{`${series.label}: ${score(record.probabilities[series.key])} · ${formatDate(record.created_at)}`}</title></circle>)}
                </g>)}
                {filteredRecords.length > 0 && <><text x={left} y={height - 9} textAnchor="start" fontSize="11" fill="#64748b">{formatDate(filteredRecords[0].created_at)}</text><text x={width - right} y={height - 9} textAnchor="end" fontSize="11" fill="#64748b">{formatDate(filteredRecords[filteredRecords.length - 1].created_at)}</text></>}
              </svg>}
            </div>
            {filteredRecords.length === 1 && <p className="text-center text-xs text-slate-500">Add another assessment to see a trend line.</p>}
          </section>

          <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
            <div className="border-b border-slate-100 px-5 py-4"><h2 className="font-bold text-slate-900">Assessment log</h2><p className="mt-1 text-xs text-slate-500">{records.length} saved {records.length === 1 ? 'assessment' : 'assessments'} · newest up to 250 shown</p></div>
            <div className="overflow-x-auto"><table className="w-full min-w-[650px] text-left text-sm">
              <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500"><tr><th className="px-5 py-3">Date</th><th className="px-4 py-3">Source</th><th className="px-4 py-3">Diabetes</th><th className="px-4 py-3">Obesity</th><th className="px-4 py-3">Cardiovascular</th><th className="px-4 py-3"><span className="sr-only">Actions</span></th></tr></thead>
              <tbody>{[...records].reverse().map((record) => <tr key={record.id} className="border-t border-slate-100"><td className="whitespace-nowrap px-5 py-3.5 font-medium text-slate-800">{formatDate(record.created_at, { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit' })}</td><td className="px-4 py-3.5 capitalize text-slate-600">{record.source}</td><td className="px-4 py-3.5 font-semibold text-slate-800">{score(record.probabilities.diabetes)}</td><td className="px-4 py-3.5 font-semibold text-slate-800">{score(record.probabilities.obesity)}</td><td className="px-4 py-3.5 font-semibold text-slate-800">{score(record.probabilities.cvd)}</td><td className="px-4 py-3.5 text-right"><button type="button" onClick={() => onDelete(record.id)} aria-label="Delete assessment" title="Delete assessment" className="rounded-lg p-2 text-slate-400 hover:bg-rose-50 hover:text-rose-700"><Trash2 className="h-4 w-4" /></button></td></tr>)}</tbody>
            </table></div>
          </section>
        </>
      )}
    </div>
  )
}
