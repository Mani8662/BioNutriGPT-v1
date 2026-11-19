import React from 'react'
import { ShieldCheck, Flame, Scale, Check, SlidersHorizontal, AlertCircle } from 'lucide-react'
import { ScreeningResult } from '../types/api'

interface Stage2PrescriptionProps {
  screening: ScreeningResult
}

export const Stage2Prescription: React.FC<Stage2PrescriptionProps> = ({ screening }) => {
  const { energy_prescription, stage2_prescription, certified_for_stage3 } = screening
  const envelope = screening.dossier?.stage2_counterfactual_prescription || screening.dossier?.stage2_prescription || {}
  const records: Array<Record<string, any>> = envelope.nutrient_constraint_envelope || []
  const energy: Record<string, any> = energy_prescription || {}
  const formatNumber = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? Math.round(value).toLocaleString() : 'Unavailable'

  // Extract key targets
  const targetKcal = energy.target_calories
  const bmrKcal = energy.BMR ?? energy.bmr_kcal
  const tdeeKcal = energy.TDEE ?? energy.tdee_kcal
  const deficit = typeof tdeeKcal === 'number' && typeof targetKcal === 'number' ? tdeeKcal - targetKcal : undefined

  const keyNutrients = [
    { code: 'DR1TPROT', name: 'Protein', unit: 'g' },
    { code: 'DR1TCARB', name: 'Carbohydrates', unit: 'g' },
    { code: 'DR1TFIBE', name: 'Dietary Fiber', unit: 'g' },
    { code: 'DR1TSODI', name: 'Sodium', unit: 'mg' },
    { code: 'DR1TPOTA', name: 'Potassium', unit: 'mg' },
    { code: 'DR1TSFAT', name: 'Saturated Fat', unit: 'g' },
  ].map(n => ({ ...n, target: records.find(r => r.nutrient === n.code)?.target ?? stage2_prescription?.[n.code], guideline: records.find(r => r.nutrient === n.code)?.rule_provenance?.planning_role || 'Legacy rule; review needed' }))

  const checkStatus = (key: string) => {
    const check = envelope.audit_filters?.[key]
    const passed = check?.passed_after_all_later_filters ?? check?.passed
    return passed === true ? 'PASS' : passed === false ? 'WARNING' : 'UNAVAILABLE'
  }

  const safetyGates = [
    { title: 'Gate 1: Energy Arithmetic', desc: 'Checks the implemented energy consistency rule.', status: checkStatus('filter_1_thermodynamic_balance') },
    { title: 'Gate 2: Project Clamps', desc: 'Checks configured bounds; clinical applicability still needs review.', status: checkStatus('filter_2_iom_safety_clamps') },
    { title: 'Gate 3: Ratio Checks', desc: 'Checks the configured dietary ratios.', status: checkStatus('filter_3_clinical_ratio_coherence') },
    { title: 'Gate 4: Model Rescoring', desc: 'Measures model-score changes; does not establish a treatment effect.', status: checkStatus('filter_4_closed_loop_rescoring') },
  ]

  return (
    <div className="space-y-6">
      {/* Energy Envelope Header */}
      <div className="bg-gradient-to-r from-emerald-900 to-teal-950 text-white rounded-2xl p-6 shadow-md border border-emerald-800">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="p-1 rounded-md bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                <Flame className="w-5 h-5 text-emerald-400" />
              </span>
              <h2 className="text-xl font-bold tracking-tight">Stage 2: Physiological Counterfactual Envelope</h2>
            </div>
            <p className="text-xs text-emerald-300/80 mt-1">
              Nutrient search using reference intervals and project assumptions. Clinical benefit and prescribing safety are not established.
            </p>
          </div>

          <div className="flex items-center space-x-3">
            <div className="bg-white/10 px-4 py-2 rounded-xl border border-white/10 text-center">
              <span className="text-[10px] text-emerald-200 uppercase tracking-wider block font-semibold">Prescribed Energy</span>
              <span className="text-2xl font-black font-mono text-white">{formatNumber(targetKcal)}</span>
              <span className="text-[10px] text-emerald-300 ml-1">kcal/day</span>
            </div>
          </div>
        </div>

        {/* Energy breakdown tiles */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-5 pt-4 border-t border-emerald-800/80 text-xs">
          <div>
            <span className="text-emerald-400 block font-medium">BMR (Mifflin-St Jeor):</span>
            <span className="text-sm font-bold font-mono text-white">{formatNumber(bmrKcal)} kcal</span>
          </div>
          <div>
            <span className="text-emerald-400 block font-medium">TDEE (Activity-adjusted):</span>
            <span className="text-sm font-bold font-mono text-white">{formatNumber(tdeeKcal)} kcal</span>
          </div>
          <div>
            <span className="text-emerald-400 block font-medium">Caloric Delta:</span>
            <span className="text-sm font-bold font-mono text-white">
              {typeof deficit !== 'number' ? 'Unavailable' : `${formatNumber(deficit)} kcal below estimated TDEE`}
            </span>
          </div>
          <div>
            <span className="text-emerald-400 block font-medium">Programmed checks:</span>
            <span className="inline-flex items-center space-x-1 text-xs font-bold text-emerald-300 bg-emerald-500/20 px-2 py-0.5 rounded border border-emerald-500/30">
              <Check className="w-3 h-3" />
              <span>{certified_for_stage3 ? 'PASSED' : 'UNVERIFIED'}</span>
            </span>
          </div>
        </div>
      </div>

      {/* 4 Clinical Safety Filters Badges */}
      <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
        <h3 className="text-sm font-bold text-slate-900 tracking-tight flex items-center space-x-2 mb-3">
          <ShieldCheck className="w-4 h-4 text-emerald-600" />
          <span>Automated arithmetic and project-rule checks</span>
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {safetyGates.map((gate, i) => (
            <div key={i} className="p-3.5 bg-slate-50 rounded-xl border border-slate-200">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-bold text-slate-800">{gate.title}</span>
                <span className={`px-2 py-0.5 text-[10px] font-bold rounded-full border ${
                  gate.status === 'PASS'
                    ? 'bg-emerald-100 text-emerald-800 border-emerald-200'
                    : gate.status === 'WARNING'
                    ? 'bg-amber-100 text-amber-800 border-amber-200'
                    : 'bg-slate-100 text-slate-600 border-slate-200'
                }`}>
                  {gate.status}
                </span>
              </div>
              <p className="text-[11px] text-slate-500 leading-relaxed">{gate.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Key Prescribed Nutrients Grid */}
      <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
        <h3 className="text-sm font-bold text-slate-900 tracking-tight flex items-center space-x-2 mb-4">
          <SlidersHorizontal className="w-4 h-4 text-indigo-600" />
          <span>Core Counterfactual Micronutrient & Electrolyte Targets</span>
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {keyNutrients.map((n) => (
            <div key={n.code} className="p-3 bg-slate-50 rounded-xl border border-slate-200 text-center">
              <span className="text-[10px] text-slate-400 font-mono block">{n.code}</span>
              <span className="text-xs font-bold text-slate-800 block mt-0.5">{n.name}</span>
              <div className="my-1.5">
                <span className="text-xl font-black font-mono text-emerald-700">
                  {formatNumber(n.target)}
                </span>
                <span className="text-xs text-slate-500 ml-1">{n.unit}</span>
              </div>
              <span className="text-[10px] text-slate-500 block leading-tight">{n.guideline}</span>
            </div>
          ))}
        </div>
      </div>
      <details className="bg-white rounded-2xl border border-slate-200 p-5 text-sm shadow-sm">
        <summary className="cursor-pointer font-semibold flex flex-wrap items-center justify-between gap-2">
          <span>Explicit Stage 2 Constraint Specification ({records.length} fields)</span>
          <span className="font-mono text-xs text-slate-500">
            {envelope.constraint_spec_version || 'adult-food-rule-review-v4-exposure-separated'}
          </span>
        </summary>
        <p className="mt-3 text-xs text-slate-600">
          {envelope.counterfactual_scope_note ||
            'Reference intervals are advisory planning inputs. Active bounds are protected against chat relaxation; caffeine and alcohol are tracked separately as beverage exposure.'}
        </p>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500">
                <th className="py-2 pr-3">Nutrient</th>
                <th className="py-2 pr-3">Bounds</th>
                <th className="py-2 pr-3">Enforcement</th>
                <th className="py-2 pr-3">Relaxation Policy</th>
                <th className="py-2">Note &amp; Source</th>
              </tr>
            </thead>
            <tbody>
              {records.map(row => {
                const prov = row.rule_provenance || {}
                const role = prov.hard_or_soft || prov.planning_role || 'hard'
                const unit = prov.unit || ''
                const hiDisplay = prov.upper_kind === 'numerical_search_cap_not_UL'
                  ? 'no food UL cap'
                  : (typeof row.max_bound === 'number' ? `${Number(row.max_bound).toFixed(1)} ${unit}` : '—')
                return (
                  <tr key={row.nutrient} className="border-b border-slate-100 align-top">
                    <td className="py-2 pr-3 font-mono font-semibold text-slate-800">{row.nutrient}</td>
                    <td className="py-2 pr-3 font-mono text-slate-700 whitespace-nowrap">
                      {typeof row.min_bound === 'number' ? Number(row.min_bound).toFixed(1) : '—'} – {hiDisplay}
                    </td>
                    <td className="py-2 pr-3">
                      <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                        role === 'hard'
                          ? 'bg-emerald-100 text-emerald-800'
                          : role === 'exposure_reporting'
                          ? 'bg-blue-100 text-blue-800'
                          : 'bg-slate-100 text-slate-600'
                      }`}>
                        {role}
                      </span>
                    </td>
                    <td className="py-2 pr-3 font-mono text-[10px] text-slate-600">
                      {prov.relaxation_policy || 'protected_no_chat_relaxation'}
                    </td>
                    <td className="py-2 text-slate-600">
                      <span>{prov.review_note || 'Source and applicability not recorded.'}</span>
                      {prov.source_url && (
                        <>
                          {' '}
                          <a href={prov.source_url} target="_blank" rel="noreferrer" className="text-emerald-700 underline">
                            Source
                          </a>
                        </>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  )
}
