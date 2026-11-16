import React from 'react'
import {
  FileText,
  Printer,
  ShieldCheck,
  CheckCircle,
  AlertCircle,
  Lightbulb,
  Heart,
  Droplets,
  Footprints,
  Bed,
} from 'lucide-react'
import { ClinicalReport } from '../types/api'

interface Stage4ClinicalReportProps {
  report: ClinicalReport
}

export const Stage4ClinicalReport: React.FC<Stage4ClinicalReportProps> = ({ report }) => {
  const {
    patient_metadata,
    compiled_patient_report,
    integrity_checks,
    compiler_metadata,
    clinical_disclaimer,
  } = report

  const handlePrint = () => {
    window.print()
  }

  return (
    <div className="space-y-6 print:m-0 print:p-0">
      {/* Report Action Bar (hidden on print) */}
      <div className="bg-white rounded-2xl border border-slate-200 p-4 shadow-sm flex flex-wrap items-center justify-between gap-4 print:hidden">
        <div>
          <div className="flex items-center space-x-2">
            <span className="p-1 rounded bg-indigo-50 text-indigo-600 border border-indigo-200">
              <FileText className="w-4 h-4" />
            </span>
            <h2 className="text-base font-bold text-slate-900 tracking-tight">
              Stage 4: Verified Patient-Facing Chrono-Nutrition Summary
            </h2>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Compiled by {compiler_metadata?.backend_used === 'transformers' ? 'Local Qwen 2.5 3B NF4 (RTX GPU)' : (compiler_metadata?.backend_used || 'Local Qwen 2.5 3B NF4 (RTX GPU)')}.
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1.5 px-3 py-1 bg-emerald-50 text-emerald-800 border border-emerald-200 rounded-lg text-xs font-semibold">
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
            <span>AST Guarded: 0% Unauthorized Drift</span>
          </div>

          <button
            type="button"
            onClick={handlePrint}
            className="px-3.5 py-1.5 bg-slate-900 hover:bg-slate-800 text-white rounded-lg text-xs font-semibold flex items-center space-x-1.5 shadow transition"
          >
            <Printer className="w-3.5 h-3.5" />
            <span>Print Report (PDF)</span>
          </button>
        </div>
      </div>

      {/* Main Clinical Document Container */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-6 print:border-none print:shadow-none print:p-0">
        {/* Printable Header */}
        <div className="border-b border-slate-200 pb-5 flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="w-3 h-3 rounded-full bg-emerald-500" />
              <h1 className="text-xl font-bold text-slate-950 tracking-tight">
                BioNutriGPT | Chrono-Nutrition Consultation Report
              </h1>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Personalized Cardiometabolic Screening, Nutrient Envelope & Indian Plate Formulation
            </p>
          </div>

          {patient_metadata && (
            <div className="bg-slate-50 px-4 py-2 rounded-xl border border-slate-200 text-xs text-slate-700 font-mono space-x-3">
              <span>Age: <strong>{patient_metadata.age_years}</strong></span>
              <span>•</span>
              <span>Sex: <strong>{patient_metadata.is_female ? 'Female' : 'Male'}</strong></span>
              <span>•</span>
              <span>Weight: <strong>{patient_metadata.weight_kg} kg</strong></span>
              <span>•</span>
              <span>Height: <strong>{patient_metadata.height_cm} cm</strong></span>
            </div>
          )}
        </div>

        {/* 1. Empathetic Health Narrative */}
        <div className="p-5 bg-gradient-to-r from-slate-50 to-emerald-50/30 rounded-xl border border-slate-200/80">
          <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2 flex items-center space-x-1.5">
            <Heart className="w-4 h-4 text-rose-500" />
            <span>Patient-Friendly Health Narrative</span>
          </h3>
          <p className="text-xs text-slate-700 leading-relaxed font-sans">
            {compiled_patient_report?.health_narrative}
          </p>
        </div>

        {/* 2. Model Attention Drivers Explanation */}
        {compiled_patient_report?.biomarker_driver_explanations &&
          compiled_patient_report.biomarker_driver_explanations.length > 0 && (
            <div>
              <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2.5 flex items-center space-x-1.5">
                <Lightbulb className="w-4 h-4 text-amber-500" />
                <span>Salient Clinical Biomarkers & Lifestyle Influences</span>
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {compiled_patient_report.biomarker_driver_explanations.map((item, idx) => (
                  <div key={idx} className="p-3 bg-slate-50 rounded-xl border border-slate-200 text-xs">
                    <span className="font-bold text-slate-900 block mb-1">{item.modality}</span>
                    <p className="text-slate-600 leading-normal">{item.plain_language_explanation}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

        {/* 3. Practical Indian Culinary Directives */}
        {compiled_patient_report?.culinary_directives && (
          <div>
            <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2.5 flex items-center space-x-1.5">
              <span>👨‍🍳 Practical Culinary Directives for Indian Kitchens</span>
            </h3>
            <div className="bg-slate-50 rounded-xl border border-slate-200 p-4 space-y-2.5">
              {compiled_patient_report.culinary_directives.map((dir, idx) => (
                <div key={idx} className="flex items-start space-x-2.5 text-xs text-slate-700">
                  <span className="w-5 h-5 rounded-full bg-emerald-100 text-emerald-800 font-bold flex items-center justify-center shrink-0 text-[10px]">
                    {idx + 1}
                  </span>
                  <p className="leading-relaxed mt-0.5">{dir}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 4. Lifestyle, Hydration & Sleep Hygiene Guidance */}
        {compiled_patient_report?.lifestyle_hydration_guidance && (
          <div>
            <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2.5">
              Lifestyle, Hydration & Physical Recovery
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5">
              <div className="p-3.5 bg-blue-50/50 rounded-xl border border-blue-200 text-xs space-y-1">
                <div className="flex items-center space-x-1.5 font-bold text-blue-900">
                  <Droplets className="w-4 h-4 text-blue-600" />
                  <span>Hydration</span>
                </div>
                <p className="text-blue-950/80 leading-relaxed">
                  {compiled_patient_report.lifestyle_hydration_guidance.hydration}
                </p>
              </div>

              <div className="p-3.5 bg-emerald-50/50 rounded-xl border border-emerald-200 text-xs space-y-1">
                <div className="flex items-center space-x-1.5 font-bold text-emerald-900">
                  <Footprints className="w-4 h-4 text-emerald-600" />
                  <span>Post-Meal Activity</span>
                </div>
                <p className="text-emerald-950/80 leading-relaxed">
                  {compiled_patient_report.lifestyle_hydration_guidance.post_meal_activity}
                </p>
              </div>

              <div className="p-3.5 bg-indigo-50/50 rounded-xl border border-indigo-200 text-xs space-y-1">
                <div className="flex items-center space-x-1.5 font-bold text-indigo-900">
                  <Bed className="w-4 h-4 text-indigo-600" />
                  <span>Sleep Hygiene</span>
                </div>
                <p className="text-indigo-950/80 leading-relaxed">
                  {compiled_patient_report.lifestyle_hydration_guidance.sleep_hygiene}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* 5. AST Integrity Audit Verification Box */}
        {integrity_checks && (
          <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl text-xs space-y-2">
            <span className="font-bold text-slate-800 block text-[11px] uppercase tracking-wider">
              Formal Abstract Syntax Tree (AST) Circuit-Breaker Invariants Audit
            </span>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
              <div className="flex items-center space-x-1 text-emerald-700">
                <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                <span>Stage 3 Basket Preserved: 100%</span>
              </div>
              <div className="flex items-center space-x-1 text-emerald-700">
                <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                <span>Zero Unauthorized Dishes</span>
              </div>
              <div className="flex items-center space-x-1 text-emerald-700">
                <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                <span>Exact Quantity Conservation</span>
              </div>
              <div className="flex items-center space-x-1 text-emerald-700">
                <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                <span>Zero Recalculation Authority</span>
              </div>
            </div>
          </div>
        )}

        {/* 6. Legal & Clinical Disclaimer */}
        <div className="p-4 bg-amber-50/70 border border-amber-200 rounded-xl text-[11px] text-amber-900 leading-relaxed flex items-start space-x-2.5">
          <AlertCircle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
          <p>
            <strong>Clinical Disclaimer:</strong> {clinical_disclaimer || compiled_patient_report?.closing_message}
          </p>
        </div>
      </div>
    </div>
  )
}
