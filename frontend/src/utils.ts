import { FEATURE_DEFINITIONS } from "./data/featureDefinitions";
import type { PatientMetadata } from "./types/api";
export const words = (value: string) =>
  value.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
export const splitList = (text: string) => [
  ...new Set(
    text
      .split(/[,\n]/)
      .map((s) => s.trim())
      .filter(Boolean),
  ),
];
export function validateProfile(values: string[], meta: PatientMetadata) {
  if (meta.renal_disease === true || meta.pregnant === true || meta.lactating === true)
    return "These meal-planning rules do not cover kidney disease, pregnancy or breastfeeding. Individual review is required.";
  if (
    !Number.isFinite(meta.age_years) ||
    meta.age_years < 18 ||
    meta.age_years > 100
  )
    return "Enter an age between 18 and 100.";
  if (
    !Number.isFinite(meta.weight_kg) ||
    meta.weight_kg < 20 ||
    meta.weight_kg > 400
  )
    return "Enter a weight between 20 and 400 kg.";
  if (
    !Number.isFinite(meta.height_cm) ||
    meta.height_cm < 80 ||
    meta.height_cm > 250
  )
    return "Enter a height between 80 and 250 cm.";
  for (let i = 0; i < FEATURE_DEFINITIONS.length; i++) {
    const f = FEATURE_DEFINITIONS[i],
      v = Number(values[i]);
    if (values[i] === "" || !Number.isFinite(v))
      return `Please complete ${f.label}. You can find it under ${f.category}.`;
    if (v < f.min_val || v > f.max_val)
      return `${f.label} must be between ${f.min_val} and ${f.max_val} ${f.unit}.`;
    if (f.unit === "0 or 1" && !Number.isInteger(v))
      return `Please choose Yes or No for ${f.label}.`;
  }
  return null;
}
export function downloadFile(
  name: string,
  content: string,
  type = "text/plain",
) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
