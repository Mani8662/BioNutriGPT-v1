import React from 'react'
import { AlertTriangle, CheckCircle, ShieldAlert, TrendingUp } from 'lucide-react'
import { ScreeningResult } from '../types/api'

interface Stage1RiskCardsProps {
  screening: ScreeningResult
}

const DISEASE_CONFIG = {
  diabetes: {
    title: 'Type 2 Diabetes',
    code: 'T2D',
    threshold: 0.617,
    description: 'Estimated cross-sectional probability based on fasting glucose, HbA1c criteria & metabolic features.',
    gradient: 'from-amber-500 to-orange-600',
  },
  obesity: {
    title: 'Clinical Obesity',
    code: 'OBES',
    threshold: 0.445,
    description: 'Estimated probability of BMI ≥ 30 kg/m² and adiposity-associated metabolic load.',
    gradient: 'from-rose-500 to-red-600',
  },
  cvd: {
    title: 'Cardiovascular Disease',
    code: 'CVD',
    threshold: 0.603,
    description: 'Estimated probability of atherosclerotic cardiovascular disease or ischemic events.',
    gradient: 'from-purple-500 to-indigo-600',
  },
}

export const Stage1RiskCards: React.FC<Stage1RiskCardsProps> = ({ screening }) => {
  const { probabilities, classifications, top_feature_drivers } = screening

  return (
    <div className="space-y-6">
      {/* Risk Gauges Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {(['diabetes', 'obesity', 'cvd'] as const).map((key) => {
          const cfg = DISEASE_CONFIG[key]
          const prob = probabilities[key] || 0.0
          const cls = classifications[key]
          const isHigh = cls ? cls.positive : prob >= cfg.threshold
          const percent = Math.round(prob * 100)
          const thresholdPercent = Math.round(cfg.threshold * 100)

          return (
            <div
              key={key}
              className={`rounded-2xl border p-5 bg-white shadow-sm transition-all hover:shadow-md relative overflow-hidden ${
                isHigh ? 'border-rose-200' : 'border-slate-200'
              }`}
            >
              {/* Top Accent Strip */}
              <div
                className={`absolute top-0 left-0 right-0 h-1.5 ${
                  isHigh ? 'bg-rose-500' : 'bg-emerald-500'
                }`}
              />

              <div className="flex items-start justify-between mb-3">
                <div>
                  <span className="text-[10px] font-bold tracking-wider uppercase text-slate-400 font-mono">
                    Endpoint: {cfg.code}
                  </span>
                  <h3 className="text-base font-bold text-slate-900 tracking-tight">{cfg.title}</h3>
                </div>
                <span
                  className={`px-2.5 py-1 rounded-full text-xs font-bold flex items-center space-x-1 ${
                    isHigh
                      ? 'bg-rose-100 text-rose-800 border border-rose-200'
                      : 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                  }`}
                >
                  {isHigh ? (
                    <>
                      <AlertTriangle className="w-3.5 h-3.5 text-rose-600" />
                      <span>HIGH RISK</span>
                    </>
                  ) : (
                    <>
                      <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                      <span>LOW RISK</span>
                    </>
                  )}
                </span>
              </div>

              {/* Big Percentage & Circular Metric */}
              <div className="my-4 flex items-baseline justify-between">
                <div>
                  <div className="flex items-baseline space-x-1">
                    <span className="text-4xl font-extrabold text-slate-900 tracking-tight font-mono">
                      {percent}%
                    </span>
                    <span className="text-xs text-slate-400">calibrated</span>
                  </div>
                  <p className="text-xs text-slate-500 mt-1">
                    Threshold: <span className="font-semibold text-slate-700">{thresholdPercent}%</span> (Platt ECE calibrated)
                  </p>
                </div>

                {/* Visual Mini Progress Dial */}
                <div className="w-14 h-14 rounded-full border-4 border-slate-100 flex items-center justify-center relative">
                  <div
                    className={`text-xs font-bold font-mono ${
                      isHigh ? 'text-rose-600' : 'text-emerald-600'
                    }`}
                  >
                    {isHigh ? 'ALERT' : 'SAFE'}
                  </div>
                </div>
              </div>

              {/* Progress Bar with Threshold Marker */}
              <div className="relative pt-1 mb-3">
                <div className="overflow-hidden h-2.5 text-xs flex rounded-full bg-slate-100">
                  <div
                    style={{ width: `${Math.min(percent, 100)}%` }}
                    className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center transition-all duration-700 ${
                      isHigh ? 'bg-gradient-to-r from-amber-500 to-rose-600' : 'bg-gradient-to-r from-teal-400 to-emerald-500'
                    }`}
                  />
                </div>
                {/* Threshold tick line */}
                <div
                  style={{ left: `${thresholdPercent}%` }}
                  className="absolute top-0 bottom-0 w-0.5 bg-slate-400 z-10"
                  title={`Clinical decision threshold: ${thresholdPercent}%`}
                />
              </div>

              <p className="text-[11px] text-slate-500 leading-normal">{cfg.description}</p>
            </div>
          )
        })}
      </div>

      {/* Model Attention Drivers Breakdown (Feature Attributions) */}
      <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
        <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-100">
          <div>
            <h3 className="text-sm font-bold text-slate-900 tracking-tight flex items-center space-x-2">
              <TrendingUp className="w-4 h-4 text-indigo-600" />
              <span>Cross-Attention Modality Driver Attribution (Stage 1 Transformer)</span>
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Learned query-pooling weights identifying the principal biometric drivers influencing risk estimations.
            </p>
          </div>
          <span className="text-[11px] font-mono px-2 py-1 bg-indigo-50 text-indigo-700 rounded-md font-semibold border border-indigo-200">
            Top-2 MoE Router
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            { key: 'bio', label: 'Serum Biomarkers', color: 'rose' },
            { key: 'diet', label: 'Dietary Intake', color: 'amber' },
            { key: 'life', label: 'Lifestyle & Sleep', color: 'blue' },
            { key: 'demo', label: 'Demographics', color: 'slate' },
          ].map(({ key, label, color }) => {
            const drivers = top_feature_drivers[key as keyof typeof top_feature_drivers] || []

            return (
              <div key={key} className="bg-slate-50/70 p-3.5 rounded-xl border border-slate-200">
                <span className="text-xs font-bold text-slate-800 uppercase tracking-wider block mb-2">
                  {label}
                </span>
                <div className="space-y-2">
                  {drivers.map((d, i) => (
                    <div key={i} className="flex items-center justify-between text-xs">
                      <span className="text-slate-600 font-mono text-[11px] truncate max-w-[140px]" title={d.feature}>
                        {d.feature}
                      </span>
                      <span className={`font-mono font-semibold text-[11px] text-${color}-600 bg-white px-1.5 py-0.5 rounded border border-slate-200`}>
                        {d.weight_pct}%
                      </span>
                    </div>
                  ))}
                  {drivers.length === 0 && (
                    <p className="text-[11px] text-slate-400 italic">No salient drivers</p>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
