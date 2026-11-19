import React, { useState } from 'react'
import { MessageSquare, Send, Sparkles, RotateCcw, X } from 'lucide-react'
import { sendIntakeChat } from '../services/api'
import type { IntakeChatState, MealPlanResult } from '../types/api'

interface Props {
  dossier: Record<string, any>
  onPlanGenerated: (plan: MealPlanResult) => void
  onClose?: () => void
}

type DietMode = 'pure_vegetarian' | 'veg_nonveg_combo' | 'pure_non_vegetarian'

const DIET_SWITCH_OPTIONS: Array<{ id: DietMode; label: string; sub: string }> = [
  { id: 'pure_vegetarian', label: 'Vegetarian', sub: 'Pure vegetarian only' },
  { id: 'veg_nonveg_combo', label: 'Veg + Non-Veg', sub: 'Vegetarian & non-veg combo' },
  { id: 'pure_non_vegetarian', label: 'Non-Vegetarian', sub: 'Pure non-vegetarian mains' },
]

const INITIAL_ASSISTANT_MESSAGE =
  'Choose your diet switch above (Vegetarian, Veg + Non-Veg, or Non-Vegetarian), then type below in your own words:\n' +
  '1. Any food allergies or intolerances (e.g., allergic to nuts, lactose intolerant — or type "no allergies")\n' +
  '2. Any dishes or ingredients you want removed (e.g., remove lady finger, no cabbage — or type "none")'

export const MealPlanIntakeChat: React.FC<Props> = ({ dossier, onPlanGenerated, onClose }) => {
  const [dietMode, setDietMode] = useState<DietMode>('veg_nonveg_combo')
  const [messages, setMessages] = useState<Array<{ role: 'assistant' | 'user'; text: string; llmUsed?: boolean }>>([
    { role: 'assistant', text: INITIAL_ASSISTANT_MESSAGE },
  ])
  const [intakeState, setIntakeState] = useState<IntakeChatState>({ dietary_mode: 'veg_nonveg_combo' })
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleDietSwitch = (mode: DietMode) => {
    setDietMode(mode)
    setIntakeState(prev => ({ ...prev, dietary_mode: mode }))
  }

  const handleRestart = () => {
    setMessages([{ role: 'assistant', text: INITIAL_ASSISTANT_MESSAGE }])
    setIntakeState({ dietary_mode: dietMode })
    setInput('')
    setError(null)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const clean = input.trim()
    if (!clean || busy) return
    setBusy(true)
    setError(null)
    setMessages(prev => [...prev, { role: 'user', text: clean }])
    setInput('')

    try {
      const stateWithSwitch: IntakeChatState = { ...intakeState, dietary_mode: dietMode }
      const res = await sendIntakeChat(clean, dossier, stateWithSwitch, true)
      setIntakeState(res.state)
      if (res.state.dietary_mode) {
        setDietMode(res.state.dietary_mode)
      }
      setMessages(prev => [...prev, { role: 'assistant', text: res.assistant_reply, llmUsed: res.llm_used }])
      if (res.ready_to_generate && res.meal_plan) {
        onPlanGenerated(res.meal_plan)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not understand your input. Please try again.')
    } finally {
      setBusy(false)
    }
  }

  const modeLabels: Record<string, string> = {
    pure_vegetarian: 'Vegetarian (Pure)',
    veg_nonveg_combo: 'Vegetarian + Non-Vegetarian',
    pure_non_vegetarian: 'Non-Vegetarian (Pure)',
  }

  return (
    <section className="bg-white rounded-2xl border-2 border-emerald-200 p-5 shadow-md space-y-4">
      <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-3">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-xl bg-emerald-50 text-emerald-700 border border-emerald-200">
            <MessageSquare className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-900 flex items-center space-x-1.5">
              <span>Tell the Assistant Your Meal Preferences</span>
              <Sparkles className="w-4 h-4 text-emerald-600" />
            </h3>
            <p className="text-xs text-slate-500">
              Select your diet mode with the 3-way switch, then type your allergies and disliked dishes freely.
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-1.5">
          <button
            type="button"
            onClick={handleRestart}
            disabled={busy}
            className="px-2.5 py-1 rounded-lg border border-slate-200 text-xs text-slate-600 hover:bg-slate-50 flex items-center space-x-1"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Start Over</span>
          </button>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600"
              aria-label="Close preference chat"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* 3-Way Diet Mode Switch: Vegetarian | Veg + Non-Veg | Non-Vegetarian */}
      <div className="space-y-1.5">
        <span className="text-[11px] font-bold uppercase tracking-wider text-slate-600 block">
          Diet Mode Switch
        </span>
        <div
          role="radiogroup"
          aria-label="Diet mode"
          className="grid grid-cols-1 sm:grid-cols-3 gap-2 p-1.5 rounded-xl bg-slate-100 border border-slate-200"
        >
          {DIET_SWITCH_OPTIONS.map(opt => {
            const active = dietMode === opt.id
            return (
              <button
                key={opt.id}
                type="button"
                role="radio"
                aria-checked={active}
                disabled={busy}
                onClick={() => handleDietSwitch(opt.id)}
                className={`rounded-lg px-3 py-2 text-left transition ${
                  active
                    ? 'bg-emerald-700 text-white shadow-sm'
                    : 'bg-transparent text-slate-700 hover:bg-white/80'
                }`}
              >
                <div className="text-xs font-bold">{opt.label}</div>
                <div className={`text-[10px] ${active ? 'text-emerald-100' : 'text-slate-500'}`}>
                  {opt.sub}
                </div>
              </button>
            )
          })}
        </div>
      </div>

      {/* Live extracted summary strip */}
      <div className="flex flex-wrap items-center gap-2 rounded-xl bg-emerald-50/60 border border-emerald-200 px-3.5 py-2 text-xs text-slate-700">
        <span className="font-bold text-emerald-900">Active settings:</span>
        <span className="rounded-md bg-white border border-emerald-200 px-2 py-0.5 font-medium">
          Diet: {modeLabels[dietMode]}
        </span>
        {intakeState.allergies_answered && (
          <span className="rounded-md bg-white border border-emerald-200 px-2 py-0.5 font-medium">
            Allergies: {intakeState.allergies?.length ? intakeState.allergies.join(', ') : 'None'}
          </span>
        )}
        {intakeState.dislikes_answered && (
          <span className="rounded-md bg-white border border-emerald-200 px-2 py-0.5 font-medium">
            Exclude: {intakeState.disliked_terms?.length ? intakeState.disliked_terms.join(', ') : 'None'}
          </span>
        )}
      </div>

      {/* Conversation Messages */}
      <div aria-live="polite" className="space-y-3 max-h-80 overflow-y-auto pr-1">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`rounded-xl px-4 py-3 text-sm whitespace-pre-line leading-relaxed ${
              msg.role === 'assistant'
                ? 'bg-slate-50 border border-slate-200 text-slate-800'
                : 'bg-emerald-700 text-white ml-8'
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <span className={`text-[11px] font-bold uppercase tracking-wider ${msg.role === 'assistant' ? 'text-emerald-700' : 'text-emerald-100'}`}>
                {msg.role === 'assistant' ? 'BioNutriGPT Assistant' : 'You'}
              </span>
              {msg.llmUsed && (
                <span className="text-[10px] font-mono text-slate-400">Qwen 2.5 3B NLU</span>
              )}
            </div>
            <p>{msg.text}</p>
          </div>
        ))}
        {busy && (
          <div className="rounded-xl bg-slate-50 border border-slate-200 px-4 py-3 text-xs text-slate-600 flex items-center space-x-2">
            <div className="w-3.5 h-3.5 border-2 border-emerald-600/30 border-t-emerald-600 rounded-full animate-spin" />
            <span>Understanding your preferences and generating checked meal plan…</span>
          </div>
        )}
      </div>

      {/* Free-Text Typed Input Box */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <label htmlFor="intake-chat-input" className="sr-only">
          Type your allergies and dishes to exclude
        </label>
        <input
          id="intake-chat-input"
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          disabled={busy}
          maxLength={1000}
          placeholder="Type allergies & dishes to remove (e.g., 'remove lady finger, allergic to nuts, lactose intolerant' or 'no allergies, none')"
          className="flex-1 min-w-0 rounded-xl border border-slate-300 px-4 py-2.5 text-sm focus:ring-2 focus:ring-emerald-500 focus:outline-none disabled:bg-slate-50"
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className="rounded-xl bg-emerald-700 hover:bg-emerald-600 px-5 py-2.5 text-white text-sm font-semibold flex items-center space-x-1.5 disabled:opacity-50 transition"
        >
          <Send className="w-4 h-4" />
          <span>Send</span>
        </button>
      </form>

      {error && <p role="alert" className="text-xs text-rose-700">{error}</p>}
    </section>
  )
}
