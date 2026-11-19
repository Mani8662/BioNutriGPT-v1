import React from "react";
import { ChevronDown, LoaderCircle } from "lucide-react";
export function Button({
  children,
  variant = "primary",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost";
}) {
  return (
    <button {...props} className={`btn ${variant} ${props.className || ""}`}>
      {children}
    </button>
  );
}
export function Select({
  label,
  children,
  ...props
}: React.SelectHTMLAttributes<HTMLSelectElement> & { label: string }) {
  return (
    <label className="field">
      <span>{label}</span>
      <div className="select-wrap">
        <select {...props}>{children}</select>
        <ChevronDown size={16} />
      </div>
    </label>
  );
}
export function Field({
  label,
  hint,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement> & {
  label: string;
  hint?: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input {...props} />
      {hint && <small>{hint}</small>}
    </label>
  );
}
export function Empty({
  icon: Icon,
  title,
  children,
  action,
}: {
  icon: React.ElementType;
  title: string;
  children: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <section className="empty card">
      <span className="empty-icon">
        <Icon size={30} />
      </span>
      <h2>{title}</h2>
      <p>{children}</p>
      {action}
    </section>
  );
}
export function Busy({ text }: { text: string }) {
  return (
    <div className="busy" role="status">
      <LoaderCircle className="spin" size={19} />
      {text}
      <span>Please keep this window open.</span>
    </div>
  );
}
