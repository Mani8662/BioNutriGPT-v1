import React from 'react'
import {
  Clock,
  Utensils,
  CheckCircle2,
  Coffee,
  Sun,
  Sunset,
  Moon,
  Apple,
  ShieldAlert,
  Info,
} from 'lucide-react'
import { MealPlanResult } from '../types/api'

interface Stage3MealScheduleProps {
  mealPlan: MealPlanResult
}

const SLOT_ICONS: Record<string, React.ReactNode> = {
  'Breakfast': <Coffee className="w-4 h-4 text-amber-500" />,
  'Mid-Morning Snack': <Apple className="w-4 h-4 text-emerald-500" />,
  'Lunch': <Sun className="w-4 h-4 text-orange-500" />,
  'Evening Snack': <Sunset className="w-4 h-4 text-indigo-500" />,
  'Dinner': <Moon className="w-4 h-4 text-purple-500" />,
}

const CATEGORY_TAGS: Record<string, { bg: string; text: string; label: string }> = {
  dal_pulse: { bg: 'bg-amber-100', text: 'text-amber-800', label: 'Dal / Pulse' },
  non_veg_protein: { bg: 'bg-rose-100', text: 'text-rose-800', label: 'Non-Veg Protein' },
  cereal_grain: { bg: 'bg-orange-100', text: 'text-orange-800', label: 'Staple Grain' },
  vegetable_sabzi: { bg: 'bg-emerald-100', text: 'text-emerald-800', label: 'Cooked Sabzi' },
  dairy: { bg: 'bg-blue-100', text: 'text-blue-800', label: 'Dairy' },
  fruit: { bg: 'bg-rose-100', text: 'text-rose-800', label: 'Whole Fruit / Nut' },
  oil_fat: { bg: 'bg-yellow-100', text: 'text-yellow-800', label: 'Healthy Lipid' },
}

export const Stage3MealSchedule: React.FC<Stage3MealScheduleProps> = ({ mealPlan }) => {
  const { meal_schedule, solved_food_basket, delivered_nutrition } = mealPlan
  const stage3Raw = mealPlan.dossier?.stage3_discrete_meal_synthesis || {}
  const verification = stage3Raw.clinical_verification || mealPlan.clinical_verification
  const acceptanceVal = verification?.acceptance_validation
  const planVersion: number = stage3Raw.plan_version || 1
  const flaggedAllergies: string[] = stage3Raw.flagged_unsuitable_for_allergies || []
  const prefsApplied = stage3Raw.preferences_applied
  const missingActive: string[] = mealPlan.failure?.details?.unknown_active_nutrients || []

  if (!mealPlan.accepted) {
    return (
      <div role="status" className="rounded-2xl border border-amber-300 bg-amber-50 p-5 text-amber-950 space-y-2">
        <h2 className="text-lg font-bold">No meal plan accepted</h2>
        <p>{mealPlan.failure?.message || 'The candidate did not pass the required checks.'}</p>
        <p className="text-sm">Reason: {mealPlan.failure?.code || 'unverified_plan'}</p>
        {missingActive.length > 0 && (
          <p className="text-sm">Missing active nutrient data: {missingActive.join(', ')}</p>
        )}
        <p className="text-sm">No rejected basket or meal schedule has been accepted. Stage 4 compilation is blocked until a plan passes.</p>
      </div>
    )
  }

  const totalWeightG = solved_food_basket.reduce(
    (acc, item) => acc + (Number(item.quantity || 0) * Number(item.serving_weight_g || 0)),
    0,
  )
  const activeCount = acceptanceVal?.active_nutrient_count ?? 30
  const unknownReportingFields = (delivered_nutrition || [])
    .filter(row => !row.constraint_applied && (row.unknown_food_count ?? 0) > 0)
    .map(row => row.nutrient)

  return (
    <div className="space-y-6">
      {/* Allergy suitability flag if reported */}
      {flaggedAllergies.length > 0 && (
        <div role="alert" className="rounded-2xl border border-rose-300 bg-rose-50 p-4 text-sm text-rose-900 flex items-start space-x-3">
          <ShieldAlert className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
          <div>
            <p className="font-bold">Plan v{planVersion} Flagged Unsuitable Pending Review</p>
            <p className="text-xs mt-1">
              Reported allergy/intolerance: <strong>{flaggedAllergies.join(', ')}</strong>. Unknown allergen records in the catalog are NOT allergen-free; review with a clinician or registered dietitian before use.
            </p>
          </div>
        </div>
      )}

      {/* Solver Status Header */}
      <div className="bg-slate-900 text-white rounded-2xl p-5 shadow-sm border border-slate-800">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="p-1 rounded-md bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                <Utensils className="w-5 h-5 text-emerald-400" />
              </span>
              <h2 className="text-xl font-bold tracking-tight">Stage 3: Discrete Indian Meal Synthesis (HiGHS MILP)</h2>
              <span className="px-2.5 py-0.5 rounded-full bg-indigo-500/20 text-indigo-200 border border-indigo-400/30 text-xs font-mono font-bold">
                Plan v{planVersion}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Sourced 49-entry catalog (20 INDB recipe variants + 29 USDA SR Legacy single-food entries covering vegetarian, veg+non-veg combo, pure non-vegetarian, and dairy-free calcium options). Acceptance verifies active arithmetic and portion rules — clinical validation has not been established.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="px-3 py-1 bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 rounded-lg text-xs font-bold font-mono flex items-center space-x-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>{activeCount}/{activeCount} Active Rules Passed</span>
            </span>
            <span className="px-3 py-1 bg-slate-800 text-slate-300 border border-slate-700 rounded-lg text-xs font-mono">
              {solved_food_basket.length} Dishes · ~{Math.round(totalWeightG)} g/day
            </span>
          </div>
        </div>
      </div>

      {/* Applied preferences summary if present */}
      {prefsApplied && (
        <div className="rounded-xl border border-emerald-200 bg-emerald-50/50 px-4 py-2.5 text-xs text-slate-700 flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-bold text-emerald-900">Preferences applied:</span>
            <span className="rounded-md bg-white border border-emerald-200 px-2 py-0.5 capitalize">
              {String(prefsApplied.dietary_preference || 'vegetarian').replace('_', '-')}
            </span>
            {prefsApplied.excluded_food_ids?.length > 0 && (
              <span className="rounded-md bg-white border border-emerald-200 px-2 py-0.5">
                {prefsApplied.excluded_food_ids.length} excluded dish(es)
              </span>
            )}
            {prefsApplied.max_active_dishes && (
              <span className="rounded-md bg-white border border-emerald-200 px-2 py-0.5">
                Max {prefsApplied.max_active_dishes} dishes
              </span>
            )}
          </div>
          {prefsApplied.catalog_note && (
            <span className="text-[11px] text-slate-600">{prefsApplied.catalog_note}</span>
          )}
        </div>
      )}

      {/* Calm informational scope bar (exposure reporting + conservative allergen caveat) */}
      <div className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-xs text-slate-600 flex flex-wrap items-center justify-between gap-2 shadow-2xs">
        <div className="flex items-center space-x-2">
          <Info className="w-4 h-4 text-blue-600 shrink-0" />
          <span>
            <strong>Allergen note:</strong> Catalog allergen tracking is partial — dishes without a listed allergen have unknown records and are <strong>not certified allergen-free</strong>.
          </span>
        </div>
        {unknownReportingFields.length > 0 && (
          <span className="font-mono text-[11px] text-slate-500">
            Separate exposure/reporting fields with partial data: {unknownReportingFields.join(', ')}
          </span>
        )}
      </div>

      {/* 5-Slot Circadian Feeding Timeline */}
      <div className="space-y-4">
        {meal_schedule.map((slot, index) => {
          const hasItems = slot.items && slot.items.length > 0

          return (
            <div
              key={index}
              className="bg-white rounded-2xl border border-slate-200 overflow-hidden shadow-xs hover:border-slate-300 transition"
            >
              {/* Slot Header */}
              <div className="p-4 bg-slate-50 border-b border-slate-200/80 flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <span className="p-2 rounded-xl bg-white border border-slate-200 shadow-2xs">
                    {SLOT_ICONS[slot.meal] || <Clock className="w-4 h-4 text-slate-500" />}
                  </span>
                  <div>
                    <h3 className="text-sm font-bold text-slate-900 tracking-tight">{slot.meal}</h3>
                    <p className="text-xs text-slate-500">{slot.guidance}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="px-2.5 py-1 rounded-lg bg-slate-200/70 text-slate-700 font-mono text-xs font-bold flex items-center space-x-1">
                    <Clock className="w-3 h-3 text-slate-500" />
                    <span>{slot.time}</span>
                  </span>
                </div>
              </div>

              {/* Dish Items */}
              <div className="p-4">
                {hasItems ? (
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                    {slot.items.map((item, itemIdx) => {
                      const tag = CATEGORY_TAGS[item.category] || {
                        bg: 'bg-slate-100',
                        text: 'text-slate-700',
                        label: item.category || 'Dish',
                      }

                      return (
                        <div
                          key={itemIdx}
                          className="p-3.5 bg-slate-50/60 rounded-xl border border-slate-200 hover:bg-white hover:border-emerald-300 transition flex flex-col justify-between"
                        >
                          <div>
                            <div className="flex items-start justify-between gap-2 mb-1.5">
                              <span className="text-xs font-bold text-slate-900 leading-tight">
                                {item.name}
                              </span>
                              <span
                                className={`text-[9px] font-semibold px-2 py-0.5 rounded-full shrink-0 ${tag.bg} ${tag.text}`}
                              >
                                {tag.label}
                              </span>
                            </div>

                            {/* Serving size badge */}
                            <div className="my-1.5 flex items-center space-x-1.5">
                              <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-900 font-mono font-bold text-xs">
                                {item.quantity} × {item.serving_unit}
                              </span>
                            </div>
                          </div>

                          {item.clinical_rationale && (
                            <div className="mt-2 pt-2 border-t border-slate-200/60 text-[10px] text-slate-500 italic leading-snug">
                              {item.clinical_rationale}
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                ) : (
                  <p className="text-xs text-slate-400 italic py-2">
                    No discrete basket item assigned to this feeding window.
                  </p>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Expandable Delivered Nutrition Verification Table */}
      {delivered_nutrition && delivered_nutrition.length > 0 && (
        <details className="bg-white rounded-2xl border border-slate-200 p-5 text-sm shadow-sm">
          <summary className="cursor-pointer font-semibold text-slate-900">
            Verify Delivered Nutrition Totals ({delivered_nutrition.length} tracked fields)
          </summary>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2 pr-3">Nutrient</th>
                  <th className="py-2 pr-3">Delivered</th>
                  <th className="py-2 pr-3">Allowed Interval</th>
                  <th className="py-2 pr-3">Role</th>
                  <th className="py-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {delivered_nutrition.map(row => {
                  const prov = row.rule_provenance || {}
                  const unit = prov.unit || row.unit || ''
                  const delivered = row.delivered ?? row.delivered_value ?? 0
                  const lo = row.min_bound ?? row.lower_bound
                  const hi = row.max_bound ?? row.upper_bound
                  const hiText = prov.upper_kind === 'numerical_search_cap_not_UL'
                    ? 'no food UL cap'
                    : (typeof hi === 'number' ? `${hi.toFixed(1)} ${unit}` : '—')
                  const role = prov.hard_or_soft || prov.planning_role || (row.constraint_applied ? 'hard' : 'report_only')
                  return (
                    <tr key={row.nutrient} className="border-b border-slate-100">
                      <td className="py-1.5 pr-3 font-mono font-semibold text-slate-800">{row.nutrient}</td>
                      <td className="py-1.5 pr-3 font-mono text-slate-800">
                        {Number(delivered).toFixed(1)} {unit}
                        {(row.unknown_food_count ?? 0) > 0 && (
                          <span className="ml-1 text-[10px] text-amber-700">(partial)</span>
                        )}
                      </td>
                      <td className="py-1.5 pr-3 font-mono text-slate-600">
                        {typeof lo === 'number' ? lo.toFixed(1) : '—'} – {hiText}
                      </td>
                      <td className="py-1.5 pr-3">
                        <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                          row.constraint_applied ? 'bg-emerald-50 text-emerald-700' : 'bg-slate-100 text-slate-600'
                        }`}>
                          {role}
                        </span>
                      </td>
                      <td className="py-1.5 font-semibold">
                        {!row.constraint_applied ? (
                          <span className="text-slate-500">Tracked only</span>
                        ) : row.within_bounds ? (
                          <span className="text-emerald-700">PASS</span>
                        ) : (
                          <span className="text-rose-700">FAIL</span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  )
}
