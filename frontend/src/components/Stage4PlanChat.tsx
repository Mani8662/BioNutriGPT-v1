import React, { useEffect, useState } from 'react'
import { History, RotateCcw, ShieldAlert, Wrench, Sparkles, CheckCircle2 } from 'lucide-react'
import {
  askPlanChat,
  acceptPlanProposal,
  createPlanSession,
  getPlanSession,
  rollbackPlanVersion,
} from '../services/api'
import type { ChatAcceptResponse, ChatResponse, PlanSessionRecord, RevisionDiff } from '../types/api'

interface Props {
  dossier: Record<string, any>
  onAccept: (accepted: ChatAcceptResponse) => void
}

const RevisionDiffCard: React.FC<{ diff: RevisionDiff }> = ({ diff }) => (
  <div className="rounded-lg border border-slate-200 bg-white p-3 text-xs space-y-1.5">
    <div className="flex items-center justify-between font-semibold text-slate-700">
      <span>Controlled Revision Diff</span>
      <span className="font-mono text-[11px] text-slate-500">
        Total serving delta: {diff.total_serving_delta} · Unchanged: {diff.unchanged.length} dish(es)
      </span>
    </div>
    <div className="flex flex-wrap gap-1.5">
      {diff.removed.map(item => (
        <span key={`rm-${item.food_id}`} className="rounded-md bg-rose-50 border border-rose-200 px-2 py-0.5 text-rose-800 font-medium">
          − {item.quantity} × {item.name}
        </span>
      ))}
      {diff.added.map(item => (
        <span key={`add-${item.food_id}`} className="rounded-md bg-emerald-50 border border-emerald-200 px-2 py-0.5 text-emerald-800 font-medium">
          + {item.quantity} × {item.name}
        </span>
      ))}
      {diff.quantity_changed.map(item => (
        <span key={`chg-${item.food_id}`} className="rounded-md bg-amber-50 border border-amber-200 px-2 py-0.5 text-amber-800 font-medium">
          ~ {item.name}: {item.previous_quantity} → {item.new_quantity}
        </span>
      ))}
    </div>
  </div>
)

export const Stage4PlanChat: React.FC<Props> = ({ dossier, onAccept }) => {
  const hasAllergyWarning = (value: Record<string, any>) =>
    (value?.stage3_discrete_meal_synthesis?.flagged_unsuitable_for_allergies?.length ?? 0) > 0
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(dossier?.session_id ?? null)
  const [planVersion, setPlanVersion] = useState<number>(
    dossier?.stage3_discrete_meal_synthesis?.plan_version ?? 1,
  )
  const [flaggedUnsuitable, setFlaggedUnsuitable] = useState<boolean>(
    (dossier?.stage3_discrete_meal_synthesis?.flagged_unsuitable_for_allergies?.length ?? 0) > 0,
  )
  const [sessionRecord, setSessionRecord] = useState<PlanSessionRecord | null>(null)
  const [showHistory, setShowHistory] = useState(false)
  const [autoApplyVerified, setAutoApplyVerified] = useState(true)
  const [history, setHistory] = useState<Array<{ question: string; response: ChatResponse; autoAppliedVersion?: number }>>([])
  const [acceptedTurns, setAcceptedTurns] = useState<number[]>([])

  const refreshSession = async (sid: string) => {
    try {
      const rec = await getPlanSession(sid)
      setSessionRecord(rec)
      setPlanVersion(rec.current_version)
      setFlaggedUnsuitable(
        rec.preferences.reported_allergies.length > 0 ||
        (rec.versions.find(v => v.version === rec.current_version)?.flagged_unsuitable_for_allergies?.length ?? 0) > 0,
      )
    } catch {
      // Non-fatal if session polling fails
    }
  }

  useEffect(() => {
    let active = true
    if (!sessionId && dossier?.stage3_discrete_meal_synthesis?.accepted) {
      createPlanSession(dossier)
        .then(res => {
          if (active) {
            setSessionId(res.session_id)
            setPlanVersion(res.current_version)
            refreshSession(res.session_id)
          }
        })
        .catch(() => {
          // Fallback cleanly to stateless chat
        })
    } else if (sessionId) {
      refreshSession(sessionId)
    }
    return () => { active = false }
  }, [dossier, sessionId])

  const accept = async (index: number, proposal: NonNullable<ChatResponse['proposed_meal_plan']>) => {
    setBusy(true)
    setError(null)
    try {
      const accepted = await acceptPlanProposal(dossier, proposal, sessionId)
      if (accepted.plan_version) setPlanVersion(accepted.plan_version)
      setFlaggedUnsuitable(hasAllergyWarning(accepted.dossier))
      onAccept(accepted)
      setAcceptedTurns(prev => [...prev, index])
      if (sessionId) await refreshSession(sessionId)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not accept this proposal.')
    } finally {
      setBusy(false)
    }
  }

  const handleRollbackTo = async (targetVersion: number) => {
    if (!sessionId || busy || targetVersion === planVersion) return
    setBusy(true)
    setError(null)
    try {
      const restored = await rollbackPlanVersion(sessionId, targetVersion)
      if (restored.plan_version) setPlanVersion(restored.plan_version)
      setFlaggedUnsuitable(hasAllergyWarning(restored.dossier))
      onAccept(restored)
      await refreshSession(sessionId)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not roll back plan version.')
    } finally {
      setBusy(false)
    }
  }

  const sendMessage = async (text: string) => {
    const clean = text.trim()
    if (!clean || busy) return
    setBusy(true)
    setError(null)
    try {
      const response = await askPlanChat(clean, dossier, sessionId)
      if (response.session_id && !sessionId) setSessionId(response.session_id)
      if (response.plan_version) setPlanVersion(response.plan_version)
      if (response.plan_flagged_unsuitable) setFlaggedUnsuitable(true)

      let autoAppliedVersion: number | undefined
      if (autoApplyVerified && response.proposed_meal_plan) {
        const accepted = await acceptPlanProposal(dossier, response.proposed_meal_plan, sessionId)
        if (accepted.plan_version) {
          setPlanVersion(accepted.plan_version)
          autoAppliedVersion = accepted.plan_version
        }
        setFlaggedUnsuitable(hasAllergyWarning(accepted.dossier))
        onAccept(accepted)
      }

      setHistory(prev => [...prev, { question: clean, response, autoAppliedVersion }])
      setQuestion('')
      if (sessionId) {
        await refreshSession(sessionId)
        if (response.intent === 'rollback' && response.plan_version) {
          const restored = await rollbackPlanVersion(sessionId, response.plan_version)
          setFlaggedUnsuitable(hasAllergyWarning(restored.dossier))
          onAccept(restored)
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not check this question.')
    } finally {
      setBusy(false)
    }
  }

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    await sendMessage(question)
  }

  return (
    <section className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm space-y-4">
      {/* Header & Session Controls */}
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-100 pb-3">
        <div>
          <div className="flex items-center space-x-2">
            <Sparkles className="w-4 h-4 text-emerald-600" />
            <h2 className="text-lg font-bold text-slate-900">Stage 4: Controlled Plan Assistant</h2>
          </div>
          <p className="text-xs text-slate-600 mt-0.5">
            Every number is recalculated from the food catalog and every swap is re-solved and independently validated by Stage 3 before acceptance.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <label className="inline-flex items-center space-x-1.5 rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-700 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={autoApplyVerified}
              onChange={e => setAutoApplyVerified(e.target.checked)}
              className="rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
            />
            <span>Auto-update plan on verified swaps</span>
          </label>
          <span className="rounded-full bg-emerald-50 border border-emerald-200 px-3 py-1 text-xs font-mono font-bold text-emerald-800">
            Plan v{planVersion}
          </span>
          {sessionId && (
            <button
              type="button"
              onClick={() => setShowHistory(prev => !prev)}
              className="rounded-lg border border-slate-300 px-2.5 py-1 text-xs font-semibold text-slate-700 hover:bg-slate-50 flex items-center space-x-1"
            >
              <History className="w-3.5 h-3.5" />
              <span>History ({sessionRecord?.versions.length ?? 1})</span>
            </button>
          )}
          {sessionId && planVersion > 1 && (
            <button
              type="button"
              disabled={busy}
              onClick={() => handleRollbackTo(planVersion - 1)}
              className="rounded-lg border border-slate-300 px-2.5 py-1 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50 flex items-center space-x-1"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Rollback to v{planVersion - 1}</span>
            </button>
          )}
        </div>
      </div>

      {/* Allergy Suitability Banner */}
      {flaggedUnsuitable && (
        <div role="alert" className="rounded-xl border border-amber-300 bg-amber-50 p-3 text-xs text-amber-900 flex items-start space-x-2">
          <ShieldAlert className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
          <div>
            <strong>Allergy safety warning:</strong> This plan version has been flagged unsuitable due to a reported allergy/intolerance. Unknown allergen entries are <strong>not allergen-free</strong>; consult a clinician or dietitian.
          </div>
        </div>
      )}

      {/* Expandable Version History Drawer */}
      {showHistory && sessionRecord && (
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-3.5 space-y-2 text-xs">
          <div className="flex items-center justify-between">
            <span className="font-bold text-slate-800">Persistent Version History ({sessionRecord.session_id})</span>
            <span className="text-[11px] text-slate-500">Observed Stage 1/2 profile remains immutable</span>
          </div>
          <div className="space-y-2">
            {sessionRecord.versions.map(v => {
              const isCurrent = v.version === sessionRecord.current_version
              return (
                <div key={v.version} className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-white border border-slate-200 px-3 py-2">
                  <div>
                    <span className="font-mono font-bold text-slate-900 mr-2">v{v.version}</span>
                    <span className="text-slate-700">{v.reason}</span>
                    {v.diff_from_previous && (
                      <span className="ml-2 text-[11px] text-slate-500 font-mono">
                        (Δ {v.diff_from_previous.total_serving_delta} servings)
                      </span>
                    )}
                    {(v.flagged_unsuitable_for_allergies?.length ?? 0) > 0 && (
                      <span className="ml-2 rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-semibold text-amber-800">
                        flagged: {v.flagged_unsuitable_for_allergies!.join(', ')}
                      </span>
                    )}
                  </div>
                  {isCurrent ? (
                    <span className="inline-flex items-center space-x-1 text-[11px] font-semibold text-emerald-700">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      <span>Active</span>
                    </span>
                  ) : (
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => handleRollbackTo(v.version)}
                      className="rounded-md border border-slate-300 px-2 py-0.5 text-[11px] font-semibold text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                    >
                      Restore v{v.version}
                    </button>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Conversation Transcript */}
      <div aria-live="polite" className="space-y-4">
        {history.map((turn, index) => (
          <div key={index} className="rounded-xl bg-slate-50 border border-slate-200/70 p-4 text-sm space-y-2.5">
            <p className="text-slate-900"><strong>You:</strong> {turn.question}</p>
            <p className="text-slate-800 leading-relaxed"><strong>Plan assistant:</strong> {turn.response.answer}</p>

            {turn.response.revision_diff && (
              <RevisionDiffCard diff={turn.response.revision_diff} />
            )}

            {turn.response.proposed_meal_plan && (
              <div className="border-t border-slate-200 pt-2.5 space-y-2">
                {turn.autoAppliedVersion ? (
                  <p className="inline-flex items-center space-x-1.5 text-xs font-semibold text-emerald-700">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Meal plan above reconfigured and saved as Plan v{turn.autoAppliedVersion}. Use Rollback if you want the earlier version.</span>
                  </p>
                ) : (
                  <>
                    <p className="font-semibold text-xs uppercase tracking-wider text-slate-600">
                      Checked Replacement Proposal (Not Yet Accepted)
                    </p>
                    <ul className="grid grid-cols-1 sm:grid-cols-2 gap-1 text-xs text-slate-700 list-disc pl-5">
                      {turn.response.proposed_meal_plan.solved_food_basket.map(food => (
                        <li key={food.food_id}>{food.quantity} × {food.name}</li>
                      ))}
                    </ul>
                    <button
                      type="button"
                      disabled={busy || acceptedTurns.includes(index)}
                      onClick={() => accept(index, turn.response.proposed_meal_plan!)}
                      className="mt-1 rounded-lg bg-emerald-700 hover:bg-emerald-600 px-3.5 py-2 text-white text-xs font-semibold disabled:opacity-50 transition"
                    >
                      {acceptedTurns.includes(index) ? 'Accepted & Saved as New Version' : 'Accept Checked Proposal'}
                    </button>
                  </>
                )}
              </div>
            )}

            <div className="flex flex-wrap items-center gap-2 pt-1 text-[11px] text-slate-500">
              <span className="rounded bg-slate-200/70 px-1.5 py-0.5 font-mono text-[10px] text-slate-700">
                intent: {turn.response.intent}
              </span>
              {turn.response.tool_calls_used && turn.response.tool_calls_used.length > 0 && (
                <span className="inline-flex items-center space-x-1 rounded bg-indigo-50 border border-indigo-200 px-1.5 py-0.5 font-mono text-[10px] text-indigo-700">
                  <Wrench className="w-3 h-3" />
                  <span>{turn.response.tool_calls_used.join(', ')}</span>
                </span>
              )}
              {turn.response.sources.length > 0 && (
                <span className="truncate">Sources: {turn.response.sources.join(', ')}</span>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Input Form */}
      <form onSubmit={submit} className="flex gap-2">
        <label htmlFor="plan-question" className="sr-only">Question about your plan</label>
        <input
          id="plan-question"
          value={question}
          onChange={event => setQuestion(event.target.value)}
          maxLength={1000}
          placeholder="Ask about nutrients, Stage 1 risk, or request a swap (e.g., 'Remove Baingan Bharta, keep my lunch')"
          className="flex-1 min-w-0 rounded-xl border border-slate-300 px-3.5 py-2.5 text-sm focus:ring-2 focus:ring-emerald-500 focus:outline-none"
        />
        <button
          type="submit"
          disabled={busy || !question.trim()}
          className="rounded-xl bg-emerald-700 hover:bg-emerald-600 px-5 py-2.5 text-white text-sm font-semibold disabled:opacity-50 transition"
        >
          {busy ? 'Checking…' : 'Ask'}
        </button>
      </form>
      {error && <p role="alert" className="text-sm text-rose-700">{error}</p>}
    </section>
  )
}
