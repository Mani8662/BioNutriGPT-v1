import React, { useState } from 'react'

type Turn = { role: 'user' | 'assistant'; content: string }
type Source = { food_id?: string; name: string; serving?: string; serving_weight_g?: number; nutrients?: Record<string, number>; source?: string }

export const NutritionChat: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const [turns, setTurns] = useState<Turn[]>([{ role: 'assistant', content: 'Ask me about nutrient values for foods in the local food database. I can share database facts, but I can’t diagnose or prescribe.' }])
  const [sources, setSources] = useState<Source[]>([])
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function send(event: React.FormEvent) {
    event.preventDefault()
    const question = message.trim()
    if (!question || loading) return
    const next = [...turns, { role: 'user' as const, content: question }]
    setTurns(next)
    setMessage('')
    setLoading(true)
    setError('')
    try {
      const response = await fetch('/api/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: question, history: turns.slice(-6) }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Chat is temporarily unavailable.')
      setTurns([...next, { role: 'assistant', content: data.answer }])
      setSources(data.sources || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Chat is temporarily unavailable.')
    } finally { setLoading(false) }
  }

  return <section className="fixed bottom-24 right-5 z-50 flex h-[min(640px,75vh)] w-[min(420px,calc(100vw-2rem))] flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl" aria-label="Nutrition chat">
    <header className="flex items-center justify-between bg-emerald-800 px-4 py-3 text-white"><div><h2 className="font-bold">Nutrition chat</h2><p className="text-xs text-emerald-100">Answers grounded in food data</p></div><button onClick={onClose} aria-label="Close chat" className="rounded px-2 py-1 hover:bg-emerald-700">✕</button></header>
    <div className="flex-1 space-y-3 overflow-y-auto p-4" aria-live="polite">
      {turns.map((turn, i) => <div key={i} className={`max-w-[90%] whitespace-pre-wrap rounded-xl px-3 py-2 text-sm ${turn.role === 'user' ? 'ml-auto bg-emerald-100 text-slate-900' : 'bg-slate-100 text-slate-800'}`}>{turn.content}</div>)}
      {loading && <p className="text-sm text-slate-500">Looking up food data and composing an answer…</p>}
      {!!sources.length && <details className="rounded-lg border p-3 text-xs text-slate-600"><summary className="cursor-pointer font-semibold">Sources and nutrient data ({sources.length})</summary>{sources.map(s => <div className="mt-2 border-t pt-2" key={s.food_id}><b>{s.name}</b> · {s.serving || 'listed serving'}{s.serving_weight_g ? ` (${s.serving_weight_g} g)` : ''}<p>{Object.entries(s.nutrients || {}).map(([k,v])=>`${k}: ${v}`).join(' · ')}</p><p>{s.source}</p></div>)}</details>}
      {error && <p role="alert" className="text-sm text-rose-700">{error}</p>}
    </div>
    <form onSubmit={send} className="flex gap-2 border-t p-3"><input value={message} onChange={e=>setMessage(e.target.value)} maxLength={2000} aria-label="Your nutrition question" placeholder="e.g. How much protein is in amaranth?" className="min-w-0 flex-1 rounded-lg border px-3 py-2 text-sm"/><button disabled={loading || !message.trim()} className="rounded-lg bg-emerald-800 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">Send</button></form>
    <p className="px-3 pb-2 text-[10px] text-slate-500">Food-composition values only; not medical advice. Availability and permissions for this dataset must be confirmed before public/commercial use.</p>
  </section>
}
