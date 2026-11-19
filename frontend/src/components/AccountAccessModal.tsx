import React, { useEffect, useState } from 'react'
import { X } from 'lucide-react'
import { AccountUser } from '../types/api'
import { loginAccount, registerAccount } from '../services/api'

interface Props {
  onClose: () => void
  onAuthenticated: (account: AccountUser) => void
}

export const AccountAccessModal: React.FC<Props> = ({ onClose, onAuthenticated }) => {
  const [mode, setMode] = useState<'login' | 'register'>('register')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [onClose])

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const account = mode === 'register'
        ? await registerAccount(email, password)
        : await loginAccount(email, password)
      onAuthenticated(account)
      onClose()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Account request failed.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-slate-950/60 p-4" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
      <section role="dialog" aria-modal="true" aria-labelledby="account-title" aria-describedby="account-description" className="w-full max-w-md rounded-2xl bg-white p-6 shadow-2xl">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-wider text-emerald-700">Your private workspace</p>
            <h2 id="account-title" className="mt-1 text-xl font-bold text-slate-900">{mode === 'register' ? 'Create your account' : 'Welcome back'}</h2>
          </div>
          <button type="button" onClick={onClose} aria-label="Close" className="rounded-lg p-2 text-slate-500 hover:bg-slate-100"><X className="h-5 w-5" /></button>
        </div>
        <p id="account-description" className="mt-2 text-sm text-slate-600">Save your risk assessment timeline on this BioNutriGPT server.</p>
        <form onSubmit={submit} className="mt-5 space-y-4">
          <label className="block text-sm font-medium text-slate-700">Email
            <input autoFocus autoComplete="email" required type="email" maxLength={254} value={email} onChange={(event) => setEmail(event.target.value)} className="mt-1.5 w-full rounded-xl border border-slate-300 px-3 py-2.5 outline-none focus:border-emerald-600 focus:ring-2 focus:ring-emerald-100" />
          </label>
          <label className="block text-sm font-medium text-slate-700">Password
            <input autoComplete={mode === 'register' ? 'new-password' : 'current-password'} required type="password" minLength={mode === 'register' ? 10 : 1} maxLength={128} value={password} onChange={(event) => setPassword(event.target.value)} className="mt-1.5 w-full rounded-xl border border-slate-300 px-3 py-2.5 outline-none focus:border-emerald-600 focus:ring-2 focus:ring-emerald-100" />
            {mode === 'register' && <span className="mt-1 block text-xs text-slate-500">Use at least 10 characters.</span>}
          </label>
          {error && <p role="alert" className="rounded-lg bg-rose-50 p-3 text-sm text-rose-700">{error}</p>}
          <button disabled={busy} type="submit" className="w-full rounded-xl bg-emerald-800 px-4 py-3 font-semibold text-white hover:bg-emerald-700 disabled:opacity-60">
            {busy ? 'Please wait…' : mode === 'register' ? 'Create account' : 'Sign in'}
          </button>
        </form>
        <p className="mt-4 text-center text-sm text-slate-600">
          {mode === 'register' ? 'Already have an account?' : 'New to BioNutriGPT?'}{' '}
          <button type="button" onClick={() => { setMode(mode === 'register' ? 'login' : 'register'); setError(null) }} className="font-semibold text-emerald-800 hover:underline">
            {mode === 'register' ? 'Sign in' : 'Create an account'}
          </button>
        </p>
        <p className="mt-4 text-[11px] leading-relaxed text-slate-500">Your timeline stores risk probabilities and assessment dates. Do not use this research prototype for medical diagnosis or treatment.</p>
      </section>
    </div>
  )
}
