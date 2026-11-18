import React, { useState, useRef } from 'react'
import {
  UploadCloud,
  FileText,
  CheckCircle2,
  AlertCircle,
  Copy,
  Check,
  ArrowRight,
  Info,
} from 'lucide-react'
import { uploadDocumentForPipeline, uploadTxtFile, parseRawText } from '../services/api'
import { DocumentPipelineResponse, PatientMetadata, TxtParseResponse } from '../types/api'

interface TxtFileUploadProps {
  onParsedSuccess: (features: number[], metadata: Partial<PatientMetadata>) => void
  onRunFileDirectly: (file: File) => void
  onPipelineComplete: (response: DocumentPipelineResponse) => void
}

export const TxtFileUpload: React.FC<TxtFileUploadProps> = ({
  onParsedSuccess,
  onRunFileDirectly,
  onPipelineComplete,
}) => {
  const [dragActive, setDragActive] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [parseResult, setParseResult] = useState<TxtParseResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [rawTextPaste, setRawTextPaste] = useState('')
  const [copied, setCopied] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const documentInputRef = useRef<HTMLInputElement>(null)
  const [selectedDocument, setSelectedDocument] = useState<File | null>(null)
  const [documentResult, setDocumentResult] = useState<DocumentPipelineResponse | null>(null)
  const [documentLoading, setDocumentLoading] = useState(false)
  const [documentError, setDocumentError] = useState<string | null>(null)

  const processDocument = async () => {
    if (!selectedDocument || documentLoading) return
    setDocumentLoading(true)
    setDocumentError(null)
    setDocumentResult(null)
    try {
      const result = await uploadDocumentForPipeline(selectedDocument)
      setDocumentResult(result)
      if (result.model_output) onPipelineComplete(result)
    } catch (err) {
      setDocumentError(err instanceof Error ? err.message : 'Could not read this file')
    } finally {
      setDocumentLoading(false)
    }
  }

  const sampleFormatGuide = `56.0 178.0 122.0 3.178571 2.178571 1990.0 90.0 250.0 42.0 30.0 ... (87 space-separated floats)
# CLINICAL_METADATA: weight_kg=74.0 height_cm=176.0 activity_level=sedentary age=42 is_female=0`

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const processFile = async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.txt') && !file.name.toLowerCase().endsWith('.csv')) {
      setError('Please select a valid .txt or .csv patient vector file.')
      return
    }

    setSelectedFile(file)
    setError(null)
    setLoading(true)

    try {
      const result = await uploadTxtFile(file)
      setParseResult(result)
      onParsedSuccess(result.features, result.metadata)
    } catch (err: any) {
      setError(err.message || 'Failed to parse uploaded file')
      setParseResult(null)
    } finally {
      setLoading(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0])
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0])
    }
  }

  const handleParsePaste = async () => {
    if (!rawTextPaste.trim()) return
    setError(null)
    setLoading(true)
    try {
      const result = await parseRawText(rawTextPaste)
      setParseResult(result)
      setSelectedFile(null)
      onParsedSuccess(result.features, result.metadata)
    } catch (err: any) {
      setError(err.message || 'Failed to parse text')
      setParseResult(null)
    } finally {
      setLoading(false)
    }
  }

  const copySampleToClipboard = () => {
    const sample = `56.0 178.0 122.0 3.178571 2.178571 1990.0 90.0 250.0 42.0 30.0 70.0 18.0 28.0 17.0 220.0 10.5 350.0 900.0 500.0 3200.0 6000.0 1800.0 1.4 1.6 18.0 1.8 500.0 120.0 380.0 584.0 4.2 100.0 120.0 950.0 1400.0 380.0 14.0 11.0 1.2 1900.0 3500.0 90.0 80.0 0.0 18.09 50.25 31.65 15.07 1.35 0.54 0.39 7.5 0.0 300.0 0.0 120.0 60.0 60.0 60.0 1440.0 0 0 1 1 1 1 1 0 0 0 42.0 3.2 5 0 0 0 1 0 0 1 0 0 0 0 0 0 0
# CLINICAL_METADATA: weight_kg=74.0 height_cm=176.0 activity_level=sedentary age=42 is_female=0`
    navigator.clipboard.writeText(sample)
    setCopied(true)
    setRawTextPaste(sample)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
      {/* Header */}
      <div className="p-5 border-b border-slate-200 bg-slate-50/70 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
            <h2 className="text-lg font-bold text-slate-900 tracking-tight">
              TXT / CSV Patient Vector Upload or Paste
            </h2>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Upload or paste an 87-feature whitespace- or comma-separated TXT/CSV file matching the canonical patient record.
          </p>
        </div>

        <button
          type="button"
          onClick={copySampleToClipboard}
          className="px-3 py-1.5 bg-blue-50 text-blue-700 hover:bg-blue-100 border border-blue-200 rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
          <span>{copied ? 'Sample Copied & Pasted!' : 'Load Sample TXT Format'}</span>
        </button>
      </div>

      <div className="p-6 space-y-6">
        <section className="rounded-xl border border-emerald-200 bg-emerald-50/50 p-4 space-y-3">
          <div>
            <h3 className="text-sm font-bold text-slate-900">Analyze a health or nutrition document</h3>
            <p className="mt-1 text-xs text-slate-600">The LLM extracts only values shown in the file. Complete inputs are sent to the project predictor; its output then goes to the report LLM for analysis and a meal plan.</p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <input
              ref={documentInputRef}
              type="file"
              accept=".pdf,.png,.jpg,.jpeg,.txt,.csv,application/pdf,image/png,image/jpeg,text/plain,text/csv"
              className="sr-only"
              onChange={(event) => {
                const file = event.target.files?.[0] ?? null
                setSelectedDocument(file)
                setDocumentResult(null)
                setDocumentError(null)
              }}
            />
            <button type="button" onClick={() => documentInputRef.current?.click()} className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50">
              Choose a document
            </button>
            <span className="max-w-full truncate text-xs text-slate-600">{selectedDocument?.name ?? 'No file selected'}</span>
            <button type="button" onClick={processDocument} disabled={!selectedDocument || documentLoading} className="rounded-lg bg-emerald-700 px-4 py-2 text-xs font-semibold text-white disabled:opacity-50">
              {documentLoading ? 'Extracting values…' : 'Extract values and run analysis'}
            </button>
          </div>
          <p className="text-[11px] text-slate-500">Files up to 8 MB; PDFs read up to 6 pages. Image reading uses English OCR. Avoid files with personal identifiers.</p>
          {documentError && <p role="alert" className="rounded-lg bg-rose-50 p-3 text-xs text-rose-700">{documentError}</p>}
          {documentResult && (
            <div className="space-y-3" aria-live="polite">
              {documentResult.status === 'needs_more_input' && (
                <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-xs text-amber-950">
                  <p className="font-semibold">The model was not run because the document is incomplete.</p>
                  <p className="mt-1">This predictor requires all 87 trained values. Missing fields are never filled with demo defaults.</p>
                  <details className="mt-2">
                    <summary className="cursor-pointer font-semibold">View {documentResult.missing_features.length} missing model inputs</summary>
                    <ul className="mt-2 grid grid-cols-1 gap-1 sm:grid-cols-2">
                      {documentResult.missing_features.map((field) => <li key={field.name}>{field.label} ({field.unit})</li>)}
                    </ul>
                  </details>
                </div>
              )}
              {documentResult.extracted_values.length > 0 && (
                <details className="rounded-lg border border-slate-200 bg-white p-3" open={documentResult.status === 'needs_more_input'}>
                  <summary className="cursor-pointer text-xs font-semibold text-slate-700">Review {documentResult.extracted_values.length} values extracted by the LLM</summary>
                  <div className="mt-2 overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead className="text-slate-500"><tr><th className="py-1 pr-3">Measurement</th><th className="py-1 pr-3">Value</th><th className="py-1">Unit</th></tr></thead>
                      <tbody>{documentResult.extracted_values.map((value) => <tr key={value.name} className="border-t border-slate-100"><td className="py-1.5 pr-3">{value.label}</td><td className="py-1.5 pr-3 font-mono">{value.value}</td><td className="py-1.5">{value.unit}</td></tr>)}</tbody>
                    </table>
                  </div>
                </details>
              )}
              {documentResult.rejected_values.length > 0 && <p className="text-xs text-amber-800">Some extracted values were outside the model’s supported ranges and were excluded.</p>}
              {documentResult.truncated && <p className="text-xs text-amber-800">The file exceeded processing limits; only its supported portion was read.</p>}
              {documentResult.notices.map((notice) => <p key={notice} className="text-xs text-slate-600">{notice}</p>)}
            </div>
          )}
        </section>

        {/* Drag and Drop Zone */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all ${
            dragActive
              ? 'border-emerald-500 bg-emerald-50/50 scale-[0.99]'
              : selectedFile
              ? 'border-blue-400 bg-blue-50/30'
              : 'border-slate-300 hover:border-slate-400 bg-slate-50/50 hover:bg-slate-50'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.csv"
            onChange={handleFileChange}
            className="hidden"
          />

          <div className="flex flex-col items-center justify-center space-y-3">
            <div
              className={`w-14 h-14 rounded-2xl flex items-center justify-center shadow-md transition-transform ${
                selectedFile
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-slate-600 border border-slate-200'
              }`}
            >
              {selectedFile ? <FileText className="w-7 h-7" /> : <UploadCloud className="w-7 h-7 text-blue-600" />}
            </div>

            <div>
              <p className="text-sm font-semibold text-slate-800">
                {selectedFile ? (
                  <span>Selected: <span className="font-mono text-blue-600">{selectedFile.name}</span></span>
                ) : (
                  <span>Drop your <span className="text-blue-600 font-mono">.txt</span> patient vector here, or <span className="text-blue-600 underline">browse</span></span>
                )}
              </p>
              <p className="text-xs text-slate-500 mt-1">
                Supports single-line or multi-line files with 87 space-delimited floats and optional metadata comment lines.
              </p>
            </div>
          </div>
        </div>

        {/* Alternative: Raw Text Paste Box */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center space-x-1.5">
              <span>Or Paste Raw Text Directly</span>
            </label>
            {rawTextPaste && (
              <button
                type="button"
                onClick={() => setRawTextPaste('')}
                className="text-[11px] text-slate-400 hover:text-slate-600"
              >
                Clear
              </button>
            )}
          </div>
          <div className="relative">
            <textarea
              rows={4}
              value={rawTextPaste}
              onChange={(e) => setRawTextPaste(e.target.value)}
              placeholder={`Paste space-separated numbers here, e.g.:\n${sampleFormatGuide}`}
              className="w-full font-mono text-xs text-slate-800 bg-slate-50/50 border border-slate-200 rounded-xl p-3.5 focus:ring-2 focus:ring-blue-500 focus:bg-white focus:outline-none transition leading-relaxed"
            />
          </div>
          <div className="mt-2 flex justify-end">
            <button
              type="button"
              disabled={!rawTextPaste.trim() || loading}
              onClick={handleParsePaste}
              className="px-4 py-2 bg-slate-900 text-white hover:bg-slate-800 disabled:opacity-50 rounded-lg text-xs font-semibold flex items-center space-x-1.5 shadow transition"
            >
              <span>Parse Pasted Numbers</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Parsing Error Alert */}
        {error && (
          <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl flex items-start space-x-3 text-rose-700 text-xs">
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
            <div>
              <p className="font-semibold">Input Validation Error</p>
              <p className="mt-0.5 text-rose-600">{error}</p>
            </div>
          </div>
        )}

        {/* Successful Parse Preview Card */}
        {parseResult && (
          <div className="p-4 bg-emerald-50/70 border border-emerald-200 rounded-xl space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-emerald-800 font-semibold text-xs">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                <span>{parseResult.summary}</span>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 bg-emerald-100 text-emerald-800 rounded font-bold">
                {parseResult.feature_count} Features Verified
              </span>
            </div>

            {/* Extracted Metadata Badges */}
            {Object.keys(parseResult.metadata).length > 0 && (
              <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-emerald-200/60 text-xs">
                <span className="text-slate-600 font-medium text-[11px]">Extracted Metadata:</span>
                {parseResult.metadata.age_years !== undefined && (
                  <span className="px-2 py-0.5 bg-white border border-emerald-200 rounded text-slate-800 font-mono text-[11px]">
                    Age: {parseResult.metadata.age_years} yrs
                  </span>
                )}
                {parseResult.metadata.is_female !== undefined && (
                  <span className="px-2 py-0.5 bg-white border border-emerald-200 rounded text-slate-800 font-mono text-[11px]">
                    Sex: {parseResult.metadata.is_female ? 'Female' : 'Male'}
                  </span>
                )}
                {parseResult.metadata.weight_kg !== undefined && (
                  <span className="px-2 py-0.5 bg-white border border-emerald-200 rounded text-slate-800 font-mono text-[11px]">
                    Weight: {parseResult.metadata.weight_kg} kg
                  </span>
                )}
                {parseResult.metadata.height_cm !== undefined && (
                  <span className="px-2 py-0.5 bg-white border border-emerald-200 rounded text-slate-800 font-mono text-[11px]">
                    Height: {parseResult.metadata.height_cm} cm
                  </span>
                )}
                {parseResult.metadata.activity_level && (
                  <span className="px-2 py-0.5 bg-white border border-emerald-200 rounded text-slate-800 font-mono text-[11px] capitalize">
                    Activity: {parseResult.metadata.activity_level}
                  </span>
                )}
              </div>
            )}

            <p className="text-[11px] text-emerald-700">
              ✓ All values have been synced into the individual typing options above. You can switch to Option 1 to inspect or modify any specific biomarker before running.
            </p>
          </div>
        )}

        {/* Format Explanation Card */}
        <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl text-xs text-slate-600 space-y-1.5">
          <div className="flex items-center space-x-1.5 font-semibold text-slate-800">
            <Info className="w-4 h-4 text-slate-500" />
            <span>TXT File Format Guide</span>
          </div>
          <p className="text-[11px] leading-relaxed">
            • Line 1 contains exactly 87 space- or comma-delimited numeric floats representing the standardized NHANES cross-sectional feature set (HDL, Total Cholesterol, Calories, Protein, Carbs, Vitamins, Minerals, Activity METs, and Demographics).
          </p>
          <p className="text-[11px] leading-relaxed">
            • Optional comment lines starting with <code className="bg-slate-200 px-1 py-0.5 rounded font-mono text-[10px]"># CLINICAL_METADATA: weight_kg=... height_cm=... activity_level=...</code> can be included to pass vital parameters automatically.
          </p>
        </div>
      </div>
    </div>
  )
}
