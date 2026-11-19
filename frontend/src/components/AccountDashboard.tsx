import React from 'react'
import { ArrowRight, CalendarDays, Clock3, ShieldCheck, UserRound } from 'lucide-react'
import { AccountUser } from '../types/api'

interface Props {
  account: AccountUser
  assessmentCount: number
  onHistory: () => void
  onSignOut: () => void
}

export const AccountDashboard: React.FC<Props> = ({ account, assessmentCount, onHistory, onSignOut }) => (
  <div className="mx-auto max-w-4xl space-y-6">
    <section className="rounded-3xl bg-gradient-to-br from-emerald-950 to-slate-900 p-8 text-white sm:p-10">
      <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-white/10"><UserRound className="h-6 w-6 text-emerald-300" /></div>
      <p className="mt-5 text-xs font-bold uppercase tracking-[0.18em] text-emerald-300">Your BioNutriGPT account</p>
      <h1 className="mt-2 break-all text-2xl font-bold sm:text-3xl">{account.email}</h1>
      <p className="mt-2 text-sm text-slate-300">{account.role === 'admin' ? 'Administrator access' : 'Personal account'}</p>
    </section>

    <section className="grid gap-4 sm:grid-cols-2">
      <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-center gap-2 text-sm font-semibold text-slate-500"><CalendarDays className="h-4 w-4 text-emerald-700" />Account created</div>
        <p className="mt-3 text-lg font-bold text-slate-900">{new Date(account.created_at).toLocaleDateString(undefined, { month: 'long', day: 'numeric', year: 'numeric' })}</p>
      </article>
      <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-center gap-2 text-sm font-semibold text-slate-500"><Clock3 className="h-4 w-4 text-blue-700" />Saved assessments</div>
        <p className="mt-3 text-lg font-bold text-slate-900">{assessmentCount}</p>
      </article>
    </section>

    <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex items-start gap-3"><ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-emerald-700" /><div><h2 className="font-bold text-slate-900">What this account saves</h2><p className="mt-1 text-sm leading-6 text-slate-600">Your risk history stores assessment dates, source type, and the three model probability scores. Uploaded files and raw feature vectors are not added to the history log. You can export your history or remove individual entries from the History page.</p></div></div>
      <div className="mt-5 flex flex-wrap gap-3">
        <button type="button" onClick={onHistory} className="inline-flex items-center gap-2 rounded-xl bg-emerald-800 px-4 py-2.5 text-sm font-semibold text-white hover:bg-emerald-700">Open risk history <ArrowRight className="h-4 w-4" /></button>
        <button type="button" onClick={onSignOut} className="rounded-xl px-4 py-2.5 text-sm font-semibold text-slate-500 hover:bg-slate-100">Sign out</button>
      </div>
      <p className="mt-4 text-[11px] text-slate-500">CSV export is available from Risk History.</p>
    </section>
  </div>
)
