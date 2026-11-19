import React from 'react'
import { Activity, FileText, ShieldCheck, Utensils } from 'lucide-react'

const stages = [
  { icon: Activity, label: 'Assess', detail: 'Review model-estimated cardiometabolic risks from the supplied patient inputs.' },
  { icon: ShieldCheck, label: 'Set nutrition targets', detail: 'Apply the pipeline’s nutrient-envelope and clinical checks.' },
  { icon: Utensils, label: 'Build a meal plan', detail: 'Generate a meal schedule from the bundled food catalog and stated preferences.' },
  { icon: FileText, label: 'Prepare a report', detail: 'Generate a patient-facing report after the meal plan passes its checks.' },
]

export const AboutPage: React.FC = () => (
  <div className="mx-auto max-w-5xl space-y-6">
    <section className="rounded-3xl bg-gradient-to-br from-slate-950 via-slate-900 to-emerald-950 p-8 text-white sm:p-10">
      <p className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-300">About BioNutriGPT</p>
      <h1 className="mt-3 text-3xl font-bold sm:text-4xl">Nutrition insights with a clear audit trail</h1>
      <p className="mt-4 max-w-3xl text-sm leading-7 text-slate-300">BioNutriGPT is a research prototype that connects cardiometabolic risk estimation with nutrient-aware Indian meal planning. Each stage is shown separately so people can review what the system produced.</p>
    </section>

    <section className="grid gap-4 sm:grid-cols-2">
      {stages.map(({ icon: Icon, label, detail }, index) => <article key={label} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-center gap-3"><span className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-50 text-emerald-800"><Icon className="h-5 w-5" /></span><div><p className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Stage {index + 1}</p><h2 className="font-bold text-slate-900">{label}</h2></div></div>
        <p className="mt-3 text-sm leading-6 text-slate-600">{detail}</p>
      </article>)}
    </section>

    <section className="rounded-2xl border border-amber-200 bg-amber-50 p-6">
      <h2 className="font-bold text-amber-950">Research prototype, not medical care</h2>
      <p className="mt-2 text-sm leading-6 text-amber-900">Risk values are model estimates, not diagnoses. Food records may be incomplete, and meal plans need review by a qualified clinician or dietitian before being used for care. Do not use this app for emergencies or to change medication.</p>
      <p className="mt-3 text-xs leading-5 text-amber-800">A typical lab report may not contain enough information for every model input. Missing demographic answers are requested; other unavailable model inputs use the saved training-set imputation values.</p>
    </section>
  </div>
)
