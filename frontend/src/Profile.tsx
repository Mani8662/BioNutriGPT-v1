import React, { useState } from "react";
import {
  Upload,
  FileText,
  Search,
  Check,
  FlaskConical,
  UserRound,
  Camera,
  Sparkles,
  Utensils,
  CheckCircle2,
} from "lucide-react";
import { Button, Field, Select } from "./ui";
import {
  FEATURE_CATEGORIES,
  FEATURE_DEFINITIONS,
} from "./data/featureDefinitions";
import { api } from "./api";
import type { PatientMetadata, FoodPhotoAnalysisResponse } from "./types/api";

interface Props {
  values: string[];
  meta: PatientMetadata;
  setMeta: (m: PatientMetadata) => void;
  setValue: (i: number, v: string) => void;
  loadSample: () => void;
  importText: (t: string) => void;
  importDocument?: (file: File) => void;
  importFoodPhoto?: (p: FoodPhotoAnalysisResponse) => void;
  run: () => void;
  busy: boolean;
  sample: boolean;
  reviewed: boolean;
  setReviewed: (v: boolean) => void;
}
export default function Profile(p: Props) {
  const [section, setSection] = useState("Basics"),
    [tab, setTab] = useState("form"),
    [text, setText] = useState(""),
    [fileName, setFileName] = useState(""),
    [search, setSearch] = useState(""),
    [fileError, setFileError] = useState(""),
    [foodFileName, setFoodFileName] = useState(""),
    [foodAnalyzing, setFoodAnalyzing] = useState(false),
    [foodError, setFoodError] = useState(""),
    [foodAnalysis, setFoodAnalysis] = useState<FoodPhotoAnalysisResponse | null>(null),
    [hdlInput, setHdlInput] = useState(p.values[0] || "56"),
    [tcInput, setTcInput] = useState(p.values[1] || "178"),
    [nonHdlInput, setNonHdlInput] = useState(p.values[2] || "122"),
    [cholHdlRatioInput, setCholHdlRatioInput] = useState(p.values[3] || "3.18"),
    [nonHdlToHdlRatioInput, setNonHdlToHdlRatioInput] = useState(p.values[4] || "2.18");

  function updateHdl(val: string) {
    setHdlInput(val);
    const h = parseFloat(val);
    const tc = parseFloat(tcInput);
    if (!isNaN(h) && !isNaN(tc) && h > 0) {
      const nh = Math.max(0, tc - h);
      setNonHdlInput(nh.toFixed(1));
      setCholHdlRatioInput((tc / h).toFixed(2));
      setNonHdlToHdlRatioInput((nh / h).toFixed(2));
    }
  }

  function updateTc(val: string) {
    setTcInput(val);
    const tc = parseFloat(val);
    const h = parseFloat(hdlInput);
    if (!isNaN(h) && !isNaN(tc) && h > 0) {
      const nh = Math.max(0, tc - h);
      setNonHdlInput(nh.toFixed(1));
      setCholHdlRatioInput((tc / h).toFixed(2));
      setNonHdlToHdlRatioInput((nh / h).toFixed(2));
    }
  }

  async function handleFoodPhotoSelect(file?: File) {
    if (!file) return;
    if (file.size > 15 * 1024 * 1024) {
      setFoodError("Please choose an image smaller than 15 MB.");
      return;
    }
    setFoodError("");
    setFoodFileName(file.name);
    setFoodAnalyzing(true);
    try {
      const h = parseFloat(hdlInput) || 56.0;
      const tc = parseFloat(tcInput) || 178.0;
      const result = await api.uploadFoodPhoto(file, h, tc);
      setFoodAnalysis(result);
    } catch (err: any) {
      setFoodError(err?.message || "Could not analyze the food photo. Please try a clearer picture.");
    } finally {
      setFoodAnalyzing(false);
    }
  }

  function handleApplyFoodAndBiomarkers() {
    const h = parseFloat(hdlInput) || 56.0;
    const tc = parseFloat(tcInput) || 178.0;
    const nh = parseFloat(nonHdlInput) || Math.max(0, tc - h);
    const chr = parseFloat(cholHdlRatioInput) || (tc / Math.max(h, 1));
    const nhr = parseFloat(nonHdlToHdlRatioInput) || (nh / Math.max(h, 1));

    let baseFeatures: number[];
    let baseMetadata = p.meta;

    if (foodAnalysis && foodAnalysis.features && foodAnalysis.features.length === FEATURE_DEFINITIONS.length) {
      baseFeatures = [...foodAnalysis.features];
      if (foodAnalysis.metadata) {
        baseMetadata = { ...baseMetadata, ...foodAnalysis.metadata };
      }
    } else {
      baseFeatures = p.values.map((v) => parseFloat(v) || 0);
    }

    baseFeatures[0] = h;
    baseFeatures[1] = tc;
    baseFeatures[2] = nh;
    baseFeatures[3] = chr;
    baseFeatures[4] = nhr;

    if (p.importFoodPhoto) {
      p.importFoodPhoto({
        ...(foodAnalysis || {
          success: true,
          meal_type: "Manual",
          is_nutrition_label: false,
          detected_items: [],
          nutrition_summary: {
            calories: baseFeatures[5] || 0,
            protein_g: baseFeatures[6] || 0,
            carbs_g: baseFeatures[7] || 0,
            fat_g: baseFeatures[10] || 0,
            fiber_g: baseFeatures[9] || 0,
            sugars_g: baseFeatures[8] || 0,
            sodium_mg: baseFeatures[39] || 0,
            potassium_mg: baseFeatures[40] || 0,
            caffeine_mg: baseFeatures[42] || 0,
          },
          extracted_nutrients: {},
          derived_ratios: {},
          biomarkers: {
            LBDHDD: h,
            LBXTC: tc,
            NON_HDL_CHOL: nh,
            CHOL_HDL_RATIO: chr,
            NON_HDL_TO_HDL_RATIO: nhr,
          },
          features: baseFeatures,
          metadata: baseMetadata,
          summary: "Food nutrients and 5 serum biomarkers applied successfully to your health profile.",
        }),
        features: baseFeatures,
        biomarkers: {
          LBDHDD: h,
          LBXTC: tc,
          NON_HDL_CHOL: nh,
          CHOL_HDL_RATIO: chr,
          NON_HDL_TO_HDL_RATIO: nhr,
        },
      });
    } else {
      baseFeatures.forEach((val, idx) => p.setValue(idx, String(val)));
    }
    p.setReviewed(true);
    setTab("form");
  }

  const count = p.values.filter((v) => v !== "").length;
  async function file(file?: File) {
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) {
      setFileError("Choose a file smaller than 10 MB.");
      return;
    }
    const name = file.name.toLowerCase();
    const isDoc = /\.(pdf|png|jpe?g|webp|bmp|tiff)$/i.test(name);
    const isTxt = /\.(txt|csv)$/i.test(name);
    if (!isDoc && !isTxt) {
      setFileError("Please choose a supported file (.pdf, photo/image, .txt, or .csv).");
      return;
    }
    try {
      setFileError("");
      setFileName(file.name);
      if (isDoc) {
        if (p.importDocument) {
          p.importDocument(file);
        }
      } else {
        const textContent = await file.text();
        setText(textContent);
      }
    } catch {
      setFileError(
        "This file could not be read. Choose it again or paste the text below.",
      );
    }
  }
  const labels: Record<string, string> = {
    "Demographics & Vitals": "Background & measurements",
    "Serum Biomarkers": "Blood test results",
    "Dietary Macronutrients": "Daily nutrition",
    "Vitamins & Micronutrients": "Vitamins",
    "Minerals & Bioactives": "Minerals",
    "Dietary Ratios & AMDR": "Nutrition balance",
    "Lifestyle & Physical Activity": "Lifestyle",
    "Data Quality & Missingness": "Missing information",
  };
  return (
    <div className="profile-layout">
      <section className="card profile-main">
        <div className="card-title">
          <div>
            <span className="eyebrow">YOUR STARTING POINT</span>
            <h2>Let’s get to know you.</h2>
          </div>
          <span className="icon-tile">
            <UserRound size={22} />
          </span>
        </div>
        <p className="muted">
          Enter your details in manageable sections, or import a prepared health
          profile.
        </p>
        <div className="tabs" role="tablist" aria-label="Profile entry method">
          <button
            role="tab"
            aria-selected={tab === "form"}
            onClick={() => setTab("form")}
          >
            Enter details
          </button>
          <button
            role="tab"
            aria-selected={tab === "photo"}
            onClick={() => setTab("photo")}
            style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}
          >
            <Camera size={16} />
            Food Photo & Biomarkers
          </button>
          <button
            role="tab"
            aria-selected={tab === "import"}
            onClick={() => setTab("import")}
          >
            <Upload size={16} />
            Import profile
          </button>
        </div>
        <fieldset disabled={p.busy}>
          <div className="form-grid">
            {([
              ["renal_disease", "Known kidney disease"],
              ["pregnant", "Currently pregnant"],
              ["lactating", "Currently breastfeeding"],
            ] as const).map(([key, label]) => (
              <Select key={key} label={label}
                value={p.meta[key] == null ? "unknown" : String(p.meta[key])}
                onChange={(e) => p.setMeta({ ...p.meta, [key]: e.target.value === "unknown" ? null : e.target.value === "true" })}>
                <option value="unknown">Unknown / not provided</option>
                <option value="false">No</option>
                <option value="true">Yes</option>
              </Select>
            ))}
          </div>
          <p className="help">These meal-planning rules are for adults. A Yes answer requires individual review and stops automatic planning. Unknown answers remain unverified.</p>

          {tab === "photo" ? (
            <div className="import-panel food-photo-panel" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
              <div style={{ background: "#f8fcfb", border: "1.5px solid #d4e8e5", borderRadius: "12px", padding: "20px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "8px" }}>
                  <span style={{ background: "#0b5c56", color: "white", borderRadius: "50%", width: "26px", height: "26px", display: "inline-flex", alignItems: "center", justifyContent: "center", fontWeight: "bold", fontSize: "13px" }}>1</span>
                  <h3 style={{ margin: 0, fontSize: "17px", fontWeight: "600", color: "#1a3936" }}>Upload Food Plate Photo or Nutrition Facts Label</h3>
                </div>
                <p style={{ margin: "0 0 16px 0", fontSize: "14px", color: "#54716e" }}>
                  Upload a photo of your meal (e.g. roti, dal, rice, sabzi, chai) or packaged food nutrition label. The multimodal vision model identifies ingredients and calculates all 39 clinical nutrients (macros, fiber, vitamins, minerals, caffeine).
                </p>

                <label className="upload-box" style={{ padding: "28px 16px", cursor: foodAnalyzing ? "wait" : "pointer" }}>
                  <Camera size={32} />
                  <strong>{foodFileName || "Choose or drop a food photo / nutrition label"}</strong>
                  <span>PNG, JPG, JPEG, WEBP · maximum 15 MB</span>
                  <input
                    aria-label="Choose food photo"
                    type="file"
                    accept=".png,.jpg,.jpeg,.webp,image/*"
                    disabled={foodAnalyzing}
                    onChange={(e) => handleFoodPhotoSelect(e.target.files?.[0])}
                  />
                </label>

                {foodAnalyzing && (
                  <div style={{ display: "flex", alignItems: "center", gap: "10px", padding: "14px", background: "#e8f4f3", borderRadius: "8px", marginTop: "14px", color: "#0b5c56" }}>
                    <Sparkles size={20} className="animate-spin" />
                    <span><strong>Analyzing food photo...</strong> Estimating portion weights and extracting all 39 dietary nutrients with ICMR-NIN IFCT 2017 grounding...</span>
                  </div>
                )}

                {foodError && (
                  <p role="alert" className="inline-error" style={{ marginTop: "10px" }}>
                    {foodError}
                  </p>
                )}
              </div>

              {foodAnalysis && (
                <div style={{ background: "#ffffff", border: "1.5px solid #b7ded9", borderRadius: "12px", padding: "20px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px", flexWrap: "wrap", gap: "10px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <CheckCircle2 size={20} style={{ color: "#107c41" }} />
                      <h4 style={{ margin: 0, fontSize: "16px", fontWeight: "600", color: "#1a3936" }}>
                        Extracted Food Intake ({foodAnalysis.meal_type})
                      </h4>
                    </div>
                    <span style={{ fontSize: "12px", background: "#e2f2ef", color: "#0d5c56", padding: "4px 10px", borderRadius: "20px", fontWeight: "500" }}>
                      {foodAnalysis.is_nutrition_label ? "Packaged Nutrition Facts Label" : "Visual Meal Recognition"}
                    </span>
                  </div>

                  {foodAnalysis.detected_items && foodAnalysis.detected_items.length > 0 && (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", marginBottom: "16px" }}>
                      {foodAnalysis.detected_items.map((item, idx) => (
                        <div key={idx} style={{ display: "flex", alignItems: "center", gap: "6px", background: "#f0f8f7", border: "1px solid #c8e3df", borderRadius: "8px", padding: "6px 12px", fontSize: "13px" }}>
                          <Utensils size={14} style={{ color: "#0b5c56" }} />
                          <strong>{item.name}</strong>
                          <span style={{ color: "#5d7875" }}>({item.portion} · {item.weight_g}g)</span>
                        </div>
                      ))}
                    </div>
                  )}

                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(110px, 1fr))", gap: "10px", textAlign: "center" }}>
                    <div style={{ background: "#f7faf9", padding: "10px", borderRadius: "8px", border: "1px solid #e1ecea" }}>
                      <div style={{ fontSize: "11px", color: "#617d7a", textTransform: "uppercase", fontWeight: "600" }}>Calories</div>
                      <div style={{ fontSize: "18px", fontWeight: "700", color: "#1a3936" }}>{Math.round(foodAnalysis.nutrition_summary.calories)} kcal</div>
                    </div>
                    <div style={{ background: "#f7faf9", padding: "10px", borderRadius: "8px", border: "1px solid #e1ecea" }}>
                      <div style={{ fontSize: "11px", color: "#617d7a", textTransform: "uppercase", fontWeight: "600" }}>Protein</div>
                      <div style={{ fontSize: "18px", fontWeight: "700", color: "#1a3936" }}>{foodAnalysis.nutrition_summary.protein_g.toFixed(1)}g</div>
                    </div>
                    <div style={{ background: "#f7faf9", padding: "10px", borderRadius: "8px", border: "1px solid #e1ecea" }}>
                      <div style={{ fontSize: "11px", color: "#617d7a", textTransform: "uppercase", fontWeight: "600" }}>Carbs</div>
                      <div style={{ fontSize: "18px", fontWeight: "700", color: "#1a3936" }}>{foodAnalysis.nutrition_summary.carbs_g.toFixed(1)}g</div>
                    </div>
                    <div style={{ background: "#f7faf9", padding: "10px", borderRadius: "8px", border: "1px solid #e1ecea" }}>
                      <div style={{ fontSize: "11px", color: "#617d7a", textTransform: "uppercase", fontWeight: "600" }}>Total Fat</div>
                      <div style={{ fontSize: "18px", fontWeight: "700", color: "#1a3936" }}>{foodAnalysis.nutrition_summary.fat_g.toFixed(1)}g</div>
                    </div>
                    <div style={{ background: "#f7faf9", padding: "10px", borderRadius: "8px", border: "1px solid #e1ecea" }}>
                      <div style={{ fontSize: "11px", color: "#617d7a", textTransform: "uppercase", fontWeight: "600" }}>Fiber</div>
                      <div style={{ fontSize: "18px", fontWeight: "700", color: "#1a3936" }}>{foodAnalysis.nutrition_summary.fiber_g.toFixed(1)}g</div>
                    </div>
                    <div style={{ background: "#f7faf9", padding: "10px", borderRadius: "8px", border: "1px solid #e1ecea" }}>
                      <div style={{ fontSize: "11px", color: "#617d7a", textTransform: "uppercase", fontWeight: "600" }}>Sodium</div>
                      <div style={{ fontSize: "18px", fontWeight: "700", color: "#1a3936" }}>{Math.round(foodAnalysis.nutrition_summary.sodium_mg)} mg</div>
                    </div>
                  </div>
                  <p style={{ margin: "12px 0 0 0", fontSize: "12px", color: "#618480" }}>
                    ✓ 39 dietary nutrients (Vitamins A, B1-B12, C, E, K, Iron, Zinc, Calcium, Potassium, Magnesium, Caffeine) mapped into project format.
                  </p>
                </div>
              )}

              {/* Step 2: Confirm First 5 Serum Biomarkers */}
              <div style={{ background: "#ffffff", border: "2px solid #0b5c56", borderRadius: "12px", padding: "22px", boxShadow: "0 4px 16px rgba(11, 92, 86, 0.08)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "8px" }}>
                  <span style={{ background: "#0b5c56", color: "white", borderRadius: "50%", width: "26px", height: "26px", display: "inline-flex", alignItems: "center", justifyContent: "center", fontWeight: "bold", fontSize: "13px" }}>2</span>
                  <h3 style={{ margin: 0, fontSize: "18px", fontWeight: "700", color: "#0b5c56" }}>Confirm Your 5 Serum Biomarkers</h3>
                </div>
                <p style={{ margin: "0 0 18px 0", fontSize: "14px", color: "#446461" }}>
                  Except for the 24h food nutrients calculated above, these 5 serum biomarkers complete the clinical input vector. Enter your HDL and Total Cholesterol; Non-HDL and the atherogenic ratios calculate automatically!
                </p>

                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "16px", marginBottom: "20px" }}>
                  <Field
                    label="1. HDL Cholesterol (LBDHDD)"
                    type="number"
                    min={10}
                    max={150}
                    step={1}
                    value={hdlInput}
                    onChange={(e) => updateHdl(e.target.value)}
                    hint="mg/dL · High-density lipoprotein ('good' cholesterol)"
                  />
                  <Field
                    label="2. Total Cholesterol (LBXTC)"
                    type="number"
                    min={50}
                    max={500}
                    step={1}
                    value={tcInput}
                    onChange={(e) => updateTc(e.target.value)}
                    hint="mg/dL · Total serum cholesterol"
                  />
                  <Field
                    label="3. Non-HDL Cholesterol (Derived)"
                    type="number"
                    min={30}
                    max={450}
                    step={1}
                    value={nonHdlInput}
                    onChange={(e) => setNonHdlInput(e.target.value)}
                    hint="mg/dL · Total - HDL (atherogenic marker)"
                  />
                  <Field
                    label="4. Total / HDL Ratio (Derived)"
                    type="number"
                    min={1.0}
                    max={15.0}
                    step={0.05}
                    value={cholHdlRatioInput}
                    onChange={(e) => setCholHdlRatioInput(e.target.value)}
                    hint="Ratio of Total Cholesterol to HDL"
                  />
                  <Field
                    label="5. Non-HDL / HDL Ratio (Derived)"
                    type="number"
                    min={0.5}
                    max={12.0}
                    step={0.05}
                    value={nonHdlToHdlRatioInput}
                    onChange={(e) => setNonHdlToHdlRatioInput(e.target.value)}
                    hint="Atherogenic plasma index"
                  />
                </div>

                <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
                  <Button
                    disabled={p.busy || !hdlInput || !tcInput}
                    onClick={handleApplyFoodAndBiomarkers}
                  >
                    <Check size={18} />
                    Apply All 87 Inputs & Continue to Assessment
                  </Button>
                  <span style={{ fontSize: "13px", color: "#618480" }}>
                    Applies the 5 biomarkers + 39 food nutrients + 7 ratios to your active profile.
                  </span>
                </div>
              </div>
            </div>
          ) : tab === "import" ? (
            <div className="import-panel">
              <label className="upload-box">
                <Upload size={28} />
                <strong>{fileName || "Choose a medical report or health profile"}</strong>
                <span>PDF, photo (PNG/JPG/WEBP), or TXT/CSV · maximum 10 MB</span>
                <input
                  aria-label="Choose medical report or TXT file"
                  type="file"
                  accept=".txt,.csv,.pdf,.png,.jpg,.jpeg,.webp,.bmp,.tiff,application/pdf,image/*"
                  onChange={(e) => file(e.target.files?.[0])}
                />
              </label>
              {fileError && (
                <p role="alert" className="inline-error">
                  {fileError}
                </p>
              )}
              <label className="field">
                <span>Or paste your profile text</span>
                <textarea
                  rows={7}
                  value={text}
                  onChange={(e) => {
                    setText(e.target.value);
                    setFileName("");
                  }}
                  placeholder="Paste the contents of your prepared TXT profile here."
                />
              </label>
              <p className="help">
                Use your application's exported profile format. Imported values
                remain editable before assessment.
              </p>
              <Button
                disabled={!text.trim()}
                onClick={() => p.importText(text)}
              >
                Import & review
              </Button>
            </div>
          ) : (
            <>
              <div className="form-toolbar">
                <Select
                  label="Profile section"
                  value={section}
                  onChange={(e) => {
                    setSection(e.target.value);
                    setSearch("");
                  }}
                >
                  <option>Basics</option>
                  {FEATURE_CATEGORIES.map((c) => (
                    <option key={c} value={c}>
                      {labels[c]}
                    </option>
                  ))}
                </Select>
                <label className="search-field">
                  <Search size={17} />
                  <input
                    aria-label="Search profile fields"
                    placeholder="Find a measurement"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                  />
                </label>
              </div>
              {section === "Basics" && !search ? (
                <div className="fields-grid">
                  <Field
                    label="Age"
                    type="number"
                    min={18}
                    max={100}
                    value={
                      Number.isNaN(p.meta.age_years) ? "" : p.meta.age_years
                    }
                    onChange={(e) =>
                      p.setMeta({
                        ...p.meta,
                        age_years:
                          e.target.value === "" ? NaN : Number(e.target.value),
                      })
                    }
                    hint="Years"
                  />
                  <Select
                    label="Sex recorded at birth"
                    value={p.meta.is_female ? "female" : "male"}
                    onChange={(e) =>
                      p.setMeta({
                        ...p.meta,
                        is_female: e.target.value === "female",
                      })
                    }
                  >
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                  </Select>
                  <Field
                    label="Weight"
                    type="number"
                    min={20}
                    max={400}
                    step="any"
                    value={
                      Number.isNaN(p.meta.weight_kg) ? "" : p.meta.weight_kg
                    }
                    onChange={(e) =>
                      p.setMeta({
                        ...p.meta,
                        weight_kg:
                          e.target.value === "" ? NaN : Number(e.target.value),
                      })
                    }
                    hint="Kilograms (kg)"
                  />
                  <Field
                    label="Height"
                    type="number"
                    min={80}
                    max={250}
                    step="any"
                    value={
                      Number.isNaN(p.meta.height_cm) ? "" : p.meta.height_cm
                    }
                    onChange={(e) =>
                      p.setMeta({
                        ...p.meta,
                        height_cm:
                          e.target.value === "" ? NaN : Number(e.target.value),
                      })
                    }
                    hint="Centimetres (cm)"
                  />
                  <Select
                    label="Usual activity level"
                    value={p.meta.activity_level}
                    onChange={(e) =>
                      p.setMeta({
                        ...p.meta,
                        activity_level: e.target
                          .value as PatientMetadata["activity_level"],
                      })
                    }
                  >
                    <option value="sedentary">Mostly sitting</option>
                    <option value="light">Lightly active</option>
                    <option value="moderate">Moderately active</option>
                    <option value="active">Very active</option>
                  </Select>
                </div>
              ) : (
                <div className="fields-grid">
                  {FEATURE_DEFINITIONS.map((f, i) => ({ f, i }))
                    .filter(({ f }) =>
                      search
                        ? `${f.label} ${f.description}`
                            .toLowerCase()
                            .includes(search.toLowerCase())
                        : f.category === section,
                    )
                    .map(({ f, i }) => (
                      <div key={f.name}>
                        {f.unit === "0 or 1" ? (
                          <Select
                            label={f.label.replace(" (Binary)", "")}
                            value={p.values[i]}
                            onChange={(e) => p.setValue(i, e.target.value)}
                          >
                            <option value="">Choose an answer</option>
                            <option value="0">No</option>
                            <option value="1">Yes</option>
                          </Select>
                        ) : (
                          <Field
                            label={f.label}
                            type="number"
                            value={p.values[i]}
                            min={f.min_val}
                            max={f.max_val}
                            step="any"
                            onChange={(e) => p.setValue(i, e.target.value)}
                            hint={`${f.unit} · ${f.min_val}–${f.max_val}`}
                          />
                        )}
                        <details className="field-help">
                          <summary>About this measurement</summary>
                          <p>{f.description}</p>
                        </details>
                      </div>
                    ))}
                  {search &&
                    !FEATURE_DEFINITIONS.some((f) =>
                      `${f.label} ${f.description}`
                        .toLowerCase()
                        .includes(search.toLowerCase()),
                    ) && (
                      <p className="muted">
                        No matching measurements. Try “cholesterol” or “sleep”.
                      </p>
                    )}
                </div>
              )}
              <div className="form-bottom">
                <span>
                  {section === "Basics"
                    ? "Continue with your health measurements."
                    : `${count} of ${p.values.length} measurements completed`}
                </span>
                <Button
                  variant="secondary"
                  onClick={() => {
                    const all = ["Basics", ...FEATURE_CATEGORIES];
                    setSearch("");
                    setSection(all[(all.indexOf(section) + 1) % all.length]);
                  }}
                >
                  Next section
                </Button>
              </div>
            </>
          )}
        </fieldset>
      </section>
      <aside className="profile-side">
        <section className="card">
          <span className="eyebrow">PROFILE CHECKLIST</span>
          <h3>A few details. A clearer picture.</h3>
          <div className="completion">
            <div style={{ width: `${(count / p.values.length) * 100}%` }} />
          </div>
          <p className="muted">
            {count} of {p.values.length} measurements entered
          </p>
          <ul className="checklist">
            <li>
              <Check size={16} />
              Basics and measurements
            </li>
            <li>
              <Check size={16} />
              Nutrition and lifestyle
            </li>
            <li>
              <Check size={16} />
              Review before continuing
            </li>
          </ul>
          {p.sample && (
            <div className="note">
              Example profile loaded. Results will describe this example, not
              your own health.
            </div>
          )}
          <label className="check-label">
            <input
              type="checkbox"
              checked={p.reviewed}
              disabled={p.busy}
              onChange={(e) => p.setReviewed(e.target.checked)}
            />
            <span>I have reviewed these values and their units.</span>
          </label>
          <Button disabled={p.busy || !p.reviewed} onClick={p.run}>
            Get my health insights
          </Button>
          <p className="help">
            Results are screening estimates, not a diagnosis.
          </p>
        </section>
        <section className="card subtle">
          <FlaskConical size={22} />
          <h3>Just exploring?</h3>
          <p className="muted">
            Use an explicitly labeled example to explore the complete journey.
          </p>
          <Button variant="secondary" disabled={p.busy} onClick={p.loadSample}>
            Load example profile
          </Button>
        </section>
        <details className="card help-card">
          <summary>
            <FileText size={17} />
            Why all these details?
          </summary>
          <p>
            Your current service requires a complete health and nutrition
            profile. Ask your care provider to prepare an import file if you do
            not have these measurements. Do not guess missing health values.
          </p>
        </details>
      </aside>
    </div>
  );
}
