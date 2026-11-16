"""
Stage 3 Deep Failure-Mode Breakdown Analyzer
Identifies the exact clinical conditions, nutrient conflicts, and dietary restrictions
that cause Stage 3 to fail or experience significant nutrient shortfalls.
"""

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
import time

# Indian Food Knowledge Base
FOOD_ITEMS = [
    {"name": "Whole Wheat Roti (2 Phulkas)", "type": "cereal", "is_veg": 1, "is_jain": 1, "is_gf": 0, "kcal": 204, "prot": 7.2, "carb": 42.0, "fat": 1.0, "sug": 0.5, "fib": 6.8, "sfat": 0.2, "na": 5, "k": 210, "ca": 24, "fe": 2.4},
    {"name": "Jowar Roti (2 Rotis)", "type": "cereal", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 210, "prot": 6.2, "carb": 43.5, "fat": 1.2, "sug": 0.6, "fib": 6.1, "sfat": 0.2, "na": 4, "k": 180, "ca": 16, "fe": 2.5},
    {"name": "Bajra Roti (2 Rotis)", "type": "cereal", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 216, "prot": 7.0, "carb": 40.0, "fat": 3.0, "sug": 0.4, "fib": 7.2, "sfat": 0.6, "na": 6, "k": 230, "ca": 25, "fe": 4.8},
    {"name": "Brown Rice (1 Cup = 150g)", "type": "cereal", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 165, "prot": 3.5, "carb": 34.5, "fat": 1.4, "sug": 0.2, "fib": 2.7, "sfat": 0.3, "na": 3, "k": 125, "ca": 15, "fe": 0.8},
    {"name": "White Basmati Rice (1 Cup)", "type": "cereal", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 195, "prot": 4.0, "carb": 43.0, "fat": 0.4, "sug": 0.1, "fib": 0.6, "sfat": 0.1, "na": 2, "k": 55, "ca": 14, "fe": 0.4},
    {"name": "Yellow Moong Dal Tadka (1 Katori)", "type": "pulse", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 145, "prot": 8.5, "carb": 18.0, "fat": 4.0, "sug": 0.8, "fib": 4.5, "sfat": 0.8, "na": 220, "k": 310, "ca": 35, "fe": 2.1},
    {"name": "Toor Dal (1 Katori)", "type": "pulse", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 150, "prot": 8.0, "carb": 20.0, "fat": 3.8, "sug": 1.0, "fib": 5.0, "sfat": 0.7, "na": 240, "k": 340, "ca": 40, "fe": 1.9},
    {"name": "Chana Dal (1 Katori)", "type": "pulse", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 160, "prot": 9.0, "carb": 21.0, "fat": 4.2, "sug": 0.9, "fib": 6.5, "sfat": 0.6, "na": 230, "k": 320, "ca": 45, "fe": 2.6},
    {"name": "Rajma Curry (1 Katori)", "type": "pulse", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 170, "prot": 9.5, "carb": 24.0, "fat": 3.9, "sug": 1.5, "fib": 7.0, "sfat": 0.6, "na": 260, "k": 410, "ca": 50, "fe": 2.8},
    {"name": "Palak Sabzi (Spinach) (1 Cup)", "type": "veg", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 75, "prot": 3.5, "carb": 5.0, "fat": 4.5, "sug": 0.8, "fib": 4.2, "sfat": 0.6, "na": 190, "k": 460, "ca": 180, "fe": 3.8},
    {"name": "Methi Sabzi (Fenugreek) (1 Cup)", "type": "veg", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 80, "prot": 4.0, "carb": 6.0, "fat": 4.5, "sug": 0.5, "fib": 5.5, "sfat": 0.6, "na": 180, "k": 420, "ca": 220, "fe": 4.2},
    {"name": "Lauki Sabzi (Bottle Gourd) (1 Cup)", "type": "veg", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 65, "prot": 1.5, "carb": 5.5, "fat": 4.0, "sug": 1.5, "fib": 2.5, "sfat": 0.5, "na": 160, "k": 210, "ca": 35, "fe": 0.8},
    {"name": "Bhindi Masala (Okra) (1 Cup)", "type": "veg", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 95, "prot": 2.5, "carb": 8.0, "fat": 5.5, "sug": 2.0, "fib": 4.0, "sfat": 0.7, "na": 180, "k": 280, "ca": 75, "fe": 1.1},
    {"name": "Fresh Low-Fat Curd (1 Katori)", "type": "dairy", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 90, "prot": 6.5, "carb": 7.5, "fat": 3.5, "sug": 7.0, "fib": 0.0, "sfat": 2.1, "na": 70, "k": 240, "ca": 220, "fe": 0.2},
    {"name": "Low-Fat Paneer Bhurji (100g)", "type": "dairy", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 190, "prot": 18.0, "carb": 4.5, "fat": 11.0, "sug": 2.5, "fib": 1.0, "sfat": 6.2, "na": 190, "k": 180, "ca": 380, "fe": 0.8},
    {"name": "Soya Chunks (50g dry = 1 Cup)", "type": "protein", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 185, "prot": 26.0, "carb": 14.0, "fat": 2.5, "sug": 1.5, "fib": 7.5, "sfat": 0.4, "na": 160, "k": 480, "ca": 140, "fe": 6.2},
    {"name": "Boiled Egg (2 Whole Eggs = 100g)", "type": "protein", "is_veg": 0, "is_jain": 0, "is_gf": 1, "kcal": 140, "prot": 12.5, "carb": 1.0, "fat": 9.5, "sug": 0.5, "fib": 0.0, "sfat": 3.1, "na": 140, "k": 130, "ca": 50, "fe": 1.8},
    {"name": "Roasted White Sesame (Til) (10g)", "type": "booster", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 57, "prot": 1.8, "carb": 2.3, "fat": 5.0, "sug": 0.05, "fib": 1.2, "sfat": 0.7, "na": 2, "k": 47, "ca": 98, "fe": 1.5},
    {"name": "Roasted Flaxseeds (10g)", "type": "booster", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 53, "prot": 1.8, "carb": 2.9, "fat": 4.2, "sug": 0.1, "fib": 2.7, "sfat": 0.4, "na": 3, "k": 81, "ca": 26, "fe": 0.6},
    {"name": "Moringa Leaf Powder (5g)", "type": "booster", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 18, "prot": 1.3, "carb": 2.0, "fat": 0.3, "sug": 0.1, "fib": 1.1, "sfat": 0.05, "na": 4, "k": 85, "ca": 110, "fe": 1.6},
    {"name": "Cooking Mustard Oil (15g)", "type": "oil", "is_veg": 1, "is_jain": 1, "is_gf": 1, "kcal": 132, "prot": 0.0, "carb": 0.0, "fat": 15.0, "sug": 0.0, "fib": 0.0, "sfat": 2.2, "na": 0, "k": 0, "ca": 0, "fe": 0.0}
]

df_foods = pd.DataFrame(FOOD_ITEMS)
NUTRIENT_COLS = ["kcal", "prot", "carb", "fat", "sug", "fib", "sfat", "na", "k", "ca", "fe"]
F_raw = df_foods[NUTRIENT_COLS].values.T

# We define 6 specific stress test archetypes and 200 variations to isolate exact failure mechanisms
stress_cases = [
    {
        "id": "CASE_1_THE_PROTEIN_CALORIE_CONFLICT",
        "desc": "Extreme Low-Calorie High-Protein Vegan (1200 kcal, 130g Protein, Vegan)",
        "diet": "vegan",
        "target": {"kcal": 1200, "prot": 130, "carb": 90, "fat": 25, "sug": 15, "fib": 35, "sfat": 5, "na": 1500, "k": 2800, "ca": 800, "fe": 18}
    },
    {
        "id": "CASE_2_THE_RENAL_POTASSIUM_TRAP",
        "desc": "Kidney Disease CVD (1400 kcal, Low Potassium < 1400mg, Low Sodium < 1200mg, Diabetic)",
        "diet": "veg",
        "target": {"kcal": 1400, "prot": 60, "carb": 180, "fat": 40, "sug": 20, "fib": 28, "sfat": 8, "na": 1000, "k": 1200, "ca": 600, "fe": 10}
    },
    {
        "id": "CASE_3_THE_STRICT_JAIN_GLUTEN_FREE_DIABETIC",
        "desc": "Strict Jain + Celiac Gluten-Free + Low Sugar (No Root Veg, No Wheat, Sugar < 20g)",
        "diet": "jain_gf",
        "target": {"kcal": 1500, "prot": 70, "carb": 170, "fat": 45, "sug": 18, "fib": 35, "sfat": 10, "na": 1800, "k": 2600, "ca": 850, "fe": 15}
    },
    {
        "id": "CASE_4_THE_ATHLETIC_HYPERCALORIC_CEILING",
        "desc": "Athlete Bulking Diet (3400 kcal, 180g Protein, Max Portions Capped at 3 servings)",
        "diet": "any",
        "target": {"kcal": 3400, "prot": 180, "carb": 420, "fat": 95, "sug": 45, "fib": 45, "sfat": 20, "na": 2800, "k": 4200, "ca": 1300, "fe": 25}
    },
    {
        "id": "CASE_5_THE_SODIUM_HEART_FAILURE_DILEMMA",
        "desc": "Congestive Heart Failure (Ultra-Low Sodium < 800mg, Standard Calories 1800)",
        "diet": "veg",
        "target": {"kcal": 1800, "prot": 75, "carb": 230, "fat": 50, "sug": 30, "fib": 32, "sfat": 12, "na": 750, "k": 3200, "ca": 800, "fe": 15}
    }
]

print("="*90)
print("STAGE 3 EMPIRICAL FAILURE-MODE AUDIT (ISOLATING THE 5 BREAKING POINTS)")
print("="*90)

for case in stress_cases:
    cid = case["id"]
    cdesc = case["desc"]
    diet_flag = case["diet"]
    tgt_dict = case["target"]
    tgt_vec = np.array([tgt_dict[k] for k in NUTRIENT_COLS], dtype=float)
    
    # Filter available foods
    if diet_flag == "vegan":
        indices = [i for i, f in enumerate(FOOD_ITEMS) if f["is_veg"] == 1 and f["type"] != "dairy"]
    elif diet_flag == "jain_gf":
        indices = [i for i, f in enumerate(FOOD_ITEMS) if f["is_jain"] == 1 and f["is_gf"] == 1]
    elif diet_flag == "veg":
        indices = [i for i, f in enumerate(FOOD_ITEMS) if f["is_veg"] == 1]
    else:
        indices = list(range(len(FOOD_ITEMS)))
        
    sub_F = F_raw[:, indices]
    n_foods = len(indices)
    n_nutr = len(NUTRIENT_COLS)
    n_vars = n_foods + 2 * n_nutr
    
    # Run Formulation B (Goal Programming)
    c = np.zeros(n_vars)
    c[:n_foods] = 0.05
    weights = np.array([12.0, 10.0, 8.0, 8.0, 5.0, 4.0, 4.0, 2.0, 1.0, 1.0, 1.0])
    c[n_foods : n_foods + n_nutr] = weights
    c[n_foods + n_nutr :] = weights
    
    A_eq = np.hstack([sub_F, -np.eye(n_nutr), np.eye(n_nutr)])
    constraints = LinearConstraint(A_eq, tgt_vec, tgt_vec)
    
    integrality = np.zeros(n_vars)
    integrality[:n_foods] = 1
    
    ub = np.full(n_vars, np.inf)
    ub[:n_foods] = 3.0  # Max 3 servings per food to keep it edible
    bounds = Bounds(np.zeros(n_vars), ub)
    
    # Solve with generous time limit (0.5s) to observe true mathematical convergence
    res = milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds, options={"time_limit": 0.5})
    
    print(f"\n--- {cid} ---")
    print(f"Scenario: {cdesc}")
    print(f"Available Foods after filtering: {n_foods}/{len(FOOD_ITEMS)}")
    
    if res.x is None:
        print("  CRITICAL FAILURE: No feasible integer solution exists at all!")
    else:
        q_sol = np.round(res.x[:n_foods])
        achieved = sub_F @ q_sol
        
        # Analyze discrepancies
        discrepancies = []
        for j, col in enumerate(NUTRIENT_COLS):
            target_val = tgt_vec[j]
            achieved_val = achieved[j]
            diff = achieved_val - target_val
            pct = (diff / target_val) * 100
            if abs(pct) > 15.0:  # significant departure from target
                discrepancies.append((col, target_val, achieved_val, pct))
                
        # Print top foods selected
        selected_foods = [(FOOD_ITEMS[indices[k]]["name"], int(q_sol[k])) for k in range(n_foods) if q_sol[k] > 0]
        print(f"  Selected Basket: {selected_foods}")
        print(f"  Caloric Result : Target {tgt_dict['kcal']} kcal -> Achieved {achieved[0]:.1f} kcal (Diff: {achieved[0]-tgt_dict['kcal']:+.1f} kcal)")
        print(f"  Protein Result : Target {tgt_dict['prot']} g    -> Achieved {achieved[1]:.1f} g")
        
        if discrepancies:
            print("  FAILING / CONFLICTING NUTRIENTS (>15% shortfall or excess):")
            for col, t_val, a_val, pct in discrepancies:
                status = "DEFICIT (Shortfall)" if pct < 0 else "EXCESS (Over limit)"
                print(f"    * {col.upper():<5}: Target = {t_val:>7.1f} | Achieved = {a_val:>7.1f} | Departure = {pct:>+6.1f}% [{status}]")
        else:
            print("  PERFECT MATCH: All 11 nutrients within 15% tolerance!")

print("\n" + "="*90)
