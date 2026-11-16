import React from 'react'
import {
  Clock,
  Utensils,
  CheckCircle2,
  Cpu,
  Coffee,
  Sun,
  Sunset,
  Moon,
  Apple,
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
  cereal_grain: { bg: 'bg-orange-100', text: 'text-orange-800', label: 'Staple Grain' },
  vegetable_sabzi: { bg: 'bg-emerald-100', text: 'text-emerald-800', label: 'Cooked Sabzi' },
  dairy: { bg: 'bg-blue-100', text: 'text-blue-800', label: 'Dairy' },
  fruit: { bg: 'bg-rose-100', text: 'text-rose-800', label: 'Whole Fruit' },
  oil_fat: { bg: 'bg-yellow-100', text: 'text-yellow-800', label: 'Healthy Lipid' },
}

export const Stage3MealSchedule: React.FC<Stage3MealScheduleProps> = ({ mealPlan }) => {
  const { meal_schedule, solved_food_basket } = mealPlan

  return (
    <div className="space-y-6">
      {/* Solver Status Header */}
      <div className="bg-slate-900 text-white rounded-2xl p-5 shadow-sm border border-slate-800">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="p-1 rounded-md bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                <Utensils className="w-5 h-5 text-emerald-400" />
              </span>
              <h2 className="text-xl font-bold tracking-tight">Stage 3: Discrete Indian Meal Synthesis (HiGHS MILP)</h2>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Translates continuous nutrient bounds into authentic cooked dishes from ICMR-NIN IFCT 2017 & RecipeDB.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="px-3 py-1 bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 rounded-lg text-xs font-bold font-mono flex items-center space-x-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>100% Integer Servings (No Fractions)</span>
            </span>
            <span className="px-3 py-1 bg-slate-800 text-slate-300 border border-slate-700 rounded-lg text-xs font-mono">
              {solved_food_basket.length} Active Dishes
            </span>
          </div>
        </div>
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

                          {/* Physiological Clinical Rationale */}
                          {item.clinical_rationale && (
                            <div className="mt-2 pt-2 border-t border-slate-200/60 text-[10px] text-slate-500 italic leading-snug">
                              💡 {item.clinical_rationale}
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
    </div>
  )
}
