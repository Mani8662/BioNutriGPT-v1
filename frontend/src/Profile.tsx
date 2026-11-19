import React, { useState } from "react";
import {
  Upload,
  FileText,
  Search,
  Check,
  FlaskConical,
  UserRound,
} from "lucide-react";
import { Button, Field, Select } from "./ui";
import {
  FEATURE_CATEGORIES,
  FEATURE_DEFINITIONS,
} from "./data/featureDefinitions";
import type { PatientMetadata } from "./types/api";

interface Props {
  values: string[];
  meta: PatientMetadata;
  setMeta: (m: PatientMetadata) => void;
  setValue: (i: number, v: string) => void;
  loadSample: () => void;
  importText: (t: string) => void;
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
    [fileError, setFileError] = useState("");
  const count = p.values.filter((v) => v !== "").length;
  async function file(file?: File) {
    if (!file) return;
    if (file.size > 1024 * 1024) {
      setFileError("Choose a TXT file smaller than 1 MB.");
      return;
    }
    if (!file.name.toLowerCase().endsWith(".txt")) {
      setFileError("Please choose a .txt file.");
      return;
    }
    try {
      setFileError("");
      setText(await file.text());
      setFileName(file.name);
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
            aria-selected={tab === "import"}
            onClick={() => setTab("import")}
          >
            <Upload size={16} />
            Import profile
          </button>
        </div>
        <fieldset disabled={p.busy}>
          {tab === "import" ? (
            <div className="import-panel">
              <label className="upload-box">
                <Upload size={28} />
                <strong>{fileName || "Choose a health profile"}</strong>
                <span>TXT file · maximum 1 MB</span>
                <input
                  aria-label="Choose TXT file"
                  type="file"
                  accept=".txt,text/plain"
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
