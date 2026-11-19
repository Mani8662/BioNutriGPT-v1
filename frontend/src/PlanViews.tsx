import React, { useState } from "react";
import {
  Clock,
  Utensils,
  Download,
  ShoppingBasket,
  Check,
  Printer,
} from "lucide-react";
import { Button, Select } from "./ui";
import { downloadFile, words } from "./utils";
import type { MealPlanResult, ClinicalReport } from "./types/api";
export function MealPlan({
  plan,
  onEdit,
  onChat,
}: {
  plan: MealPlanResult;
  onEdit: () => void;
  onChat: () => void;
}) {
  const [view, setView] = useState("schedule"),
    [slot, setSlot] = useState("all"),
    [checked, setChecked] = useState<string[]>([]);
  const basket = plan.solved_food_basket || [],
    schedule = plan.meal_schedule || [];
  const list = basket
    .map((f) => `${f.name}: ${f.quantity} ${f.serving_unit}`)
    .join("\n");
  const allergies =
    plan.dossier?.stage3_discrete_meal_synthesis
      ?.flagged_unsuitable_for_allergies || [];
  return (
    <>
      <div className="page-actions">
        <div className="tabs" role="tablist" aria-label="Meal plan view">
          {[
            ["schedule", "Daily schedule"],
            ["shopping", "Shopping list"],
            ["nutrition", "Nutrition"],
          ].map(([v, l]) => (
            <button
              key={v}
              role="tab"
              aria-selected={view === v}
              onClick={() => setView(v)}
            >
              {l}
            </button>
          ))}
        </div>
        <Button variant="secondary" onClick={onEdit}>
          Edit preferences
        </Button>
      </div>
      {allergies.length > 0 && (
        <div className="alert" role="alert">
          This plan needs review for reported allergies: {allergies.join(", ")}.
          Please consult your care provider before using it.
        </div>
      )}
      {view === "schedule" && (
        <>
          <div className="plan-banner">
            <div>
              <span className="eyebrow">MADE AROUND YOUR PREFERENCES</span>
              <h2>Your day, thoughtfully planned.</h2>
              <p>Food you know. Portions picked for your profile.</p>
            </div>
            <span className="plan-banner-icon">
              <Utensils size={43} />
            </span>
          </div>
          <div className="filter-row">
            <Select
              label="Show meals"
              value={slot}
              onChange={(e) => setSlot(e.target.value)}
            >
              <option value="all">All meals</option>
              {schedule.map((s, i) => (
                <option key={i} value={String(i)}>
                  {s.meal}
                </option>
              ))}
            </Select>
            <span className="muted">
              {basket.length} foods in your daily plan
            </span>
          </div>
          <div className="meal-list">
            {schedule
              .filter((_, i) => slot === "all" || String(i) === slot)
              .map((s, i) => (
                <article className="card meal-row" key={`${s.meal}-${i}`}>
                  <div className="meal-time">
                    <span className="icon-tile">
                      <Utensils size={21} />
                    </span>
                    <h3>{s.meal}</h3>
                    <span>
                      <Clock size={14} />
                      {s.time}
                    </span>
                  </div>
                  <div className="meal-foods">
                    {s.items.map((f, j) => (
                      <div className="food-line" key={`${f.food_id}-${j}`}>
                        <strong>{f.name}</strong>
                        <span>
                          {f.quantity} {f.serving_unit}
                        </span>
                      </div>
                    ))}
                    {s.guidance && (
                      <details>
                        <summary>Meal notes</summary>
                        <p>{s.guidance}</p>
                      </details>
                    )}
                  </div>
                </article>
              ))}
          </div>
          {!schedule.length && (
            <section className="card">
              <h3>Your food portions</h3>
              <p className="muted">
                A timed schedule was not returned. Your selected foods are
                available in the Shopping list tab.
              </p>
            </section>
          )}
          <div className="callout">
            <div>
              <h3>Want to change something?</h3>
              <p>Ask for a food swap or an explanation of your portions.</p>
            </div>
            <Button onClick={onChat}>Talk to my assistant</Button>
          </div>
        </>
      )}
      {view === "shopping" && (
        <section className="card">
          <div className="card-title">
            <div>
              <h2>Your shopping list</h2>
              <p className="muted">
                Plan quantities for one day. Tick off what you already have.
              </p>
            </div>
            <ShoppingBasket size={27} />
          </div>
          <div className="shopping-list">
            {basket.map((f, i) => (
              <label
                key={`${f.food_id}-${i}`}
                className={
                  checked.includes(f.food_id)
                    ? "shopping-item done"
                    : "shopping-item"
                }
              >
                <input
                  type="checkbox"
                  checked={checked.includes(f.food_id)}
                  onChange={() =>
                    setChecked(
                      checked.includes(f.food_id)
                        ? checked.filter((x) => x !== f.food_id)
                        : [...checked, f.food_id],
                    )
                  }
                />
                <span>
                  {f.name}
                  <small>
                    {f.quantity} {f.serving_unit}
                  </small>
                </span>
                {checked.includes(f.food_id) && <Check size={17} />}
              </label>
            ))}
          </div>
          <div className="button-row">
            <Button onClick={() => downloadFile("my-shopping-list.txt", list)}>
              <Download size={16} />
              Download list
            </Button>
            <Button
              variant="ghost"
              disabled={!checked.length}
              onClick={() => setChecked([])}
            >
              Clear checkmarks
            </Button>
          </div>
        </section>
      )}
      {view === "nutrition" && (
        <section className="card">
          <h2>What’s in your plan</h2>
          <p className="muted">
            Compare planned nutrition with the targets returned for your
            profile.
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Nutrient</th>
                  <th>Planned</th>
                  <th>Target</th>
                </tr>
              </thead>
              <tbody>
                {plan.delivered_nutrition.map((n, i) => (
                  <tr key={i}>
                    <td>{words(n.nutrient)}</td>
                    <td>
                      {n.value_status === "unknown"
                        ? "Unavailable"
                        : format(n.delivered_value ?? n.delivered)}{" "}
                      {n.unit}
                    </td>
                    <td>
                      {format(n.prescribed_target ?? n.target)} {n.unit}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </>
  );
}
const format = (n: number | undefined) =>
  n === undefined
    ? "—"
    : Number(n).toLocaleString(undefined, { maximumFractionDigits: 1 });
export function Report({ report }: { report: ClinicalReport }) {
  const r = report.compiled_patient_report;
  const content = [
    r.health_narrative,
    ...(r.biomarker_driver_explanations || []).map(
      (x) => x.plain_language_explanation,
    ),
    ...(r.meal_schedule || []).flatMap((m) => [
      `${m.meal} · ${m.time}`,
      ...m.items.map((i) => `${i.name}: ${i.quantity} ${i.serving_unit}`),
      m.guidance,
    ]),
    ...(r.culinary_directives || []),
    ...Object.values(r.lifestyle_hydration_guidance || {}),
    r.closing_message,
    report.clinical_disclaimer,
  ]
    .filter(Boolean)
    .join("\n\n");
  return (
    <>
      <div className="page-actions no-print">
        <span className="pill">Your latest report</span>
        <div className="button-row">
          <Button
            variant="secondary"
            onClick={() => downloadFile("my-nutrition-report.txt", content)}
          >
            <Download size={16} />
            Download report
          </Button>
          <Button onClick={() => window.print()}>
            <Printer size={16} />
            Print / save PDF
          </Button>
        </div>
      </div>
      <article className="card report-document">
        <div className="report-brand">
          BioNutri<span>GPT</span>
          <span className="eyebrow">PERSONAL NUTRITION REPORT</span>
        </div>
        <h2>Your nutrition, explained.</h2>
        <p className="report-intro">{r.health_narrative}</p>
        {r.biomarker_driver_explanations?.length > 0 && (
          <section>
            <h3>Understanding your health</h3>
            {r.biomarker_driver_explanations.map((e, i) => (
              <p key={i}>{e.plain_language_explanation}</p>
            ))}
          </section>
        )}
        <section>
          <h3>Your meals</h3>
          {r.meal_schedule?.map((m, i) => (
            <div className="report-meal" key={i}>
              <h4>
                {m.meal} <span>{m.time}</span>
              </h4>
              <ul>
                {m.items.map((f, j) => (
                  <li key={j}>
                    {f.name} · {f.quantity} {f.serving_unit}
                  </li>
                ))}
              </ul>
              {m.guidance && <p>{m.guidance}</p>}
            </div>
          ))}
        </section>
        {r.culinary_directives?.length > 0 && (
          <section>
            <h3>In the kitchen</h3>
            <ul>
              {r.culinary_directives.map((d, i) => (
                <li key={i}>{d}</li>
              ))}
            </ul>
          </section>
        )}
        <section>
          <h3>Beyond your plate</h3>
          {Object.entries(r.lifestyle_hydration_guidance || {}).map(
            ([k, v]) => (
              <div key={k}>
                <h4>{words(k)}</h4>
                <p>{v}</p>
              </div>
            ),
          )}
        </section>
        <p>{r.closing_message}</p>
        <div className="report-disclaimer">
          {report.clinical_disclaimer ||
            "Review this plan with your care provider. Screening results are not a diagnosis."}
        </div>
      </article>
    </>
  );
}
