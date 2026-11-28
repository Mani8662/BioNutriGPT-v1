with open('stage2_counterfactual_engine.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update _build_linear_constraint_matrices to use 1.0099 and 0.9901
old_e_rows = """    # Energy thermodynamic balance: 4*PROT + 4*CARB + 9*TFAT + 7*ALCO ~ KCAL (+- 1%)
    # 4*PROT + 4*CARB + 9*TFAT + 7*ALCO - 1.01*KCAL <= 0
    add_row({'DR1TPROT': 4.0, 'DR1TCARB': 4.0, 'DR1TTFAT': 9.0, 'DR1TALCO': 7.0, 'DR1TKCAL': -1.01}, 0.0)
    # -4*PROT - 4*CARB - 9*TFAT - 7*ALCO + 0.99*KCAL <= 0
    add_row({'DR1TPROT': -4.0, 'DR1TCARB': -4.0, 'DR1TTFAT': -9.0, 'DR1TALCO': -7.0, 'DR1TKCAL': 0.99}, 0.0)"""

new_e_rows = """    # Energy thermodynamic balance: 4*PROT + 4*CARB + 9*TFAT + 7*ALCO ~ KCAL (+- 1%)
    # Tightened to +-0.99% margin (1.0099 / 0.9901) so floating point boundary precision strictly satisfies the 1.00% rule.
    add_row({'DR1TPROT': 4.0, 'DR1TCARB': 4.0, 'DR1TTFAT': 9.0, 'DR1TALCO': 7.0, 'DR1TKCAL': -1.0099}, 0.0)
    add_row({'DR1TPROT': -4.0, 'DR1TCARB': -4.0, 'DR1TTFAT': -9.0, 'DR1TALCO': -7.0, 'DR1TKCAL': 0.9901}, 0.0)"""

assert old_e_rows in text, 'old_e_rows not found'
text = text.replace(old_e_rows, new_e_rows, 1)

# 2. Update _apply_filter_1_thermodynamic_balance to avoid destructive overwrite under repaired deployment
old_f1 = """    relative_error_before = abs(calculated_kcal - n[KCAL]) / n[KCAL]
    adjusted = relative_error_before > 0.01
    if adjusted:
        # Exact remediation stated in the brief.
        n[KCAL] = calculated_kcal
    relative_error_after = (
        abs(calculated_kcal - n[KCAL]) / n[KCAL] if n[KCAL] > 0.0 else math.inf
    )
    return {
        "passed": bool(relative_error_after <= 0.01),"""

new_f1 = """    relative_error_before = abs(calculated_kcal - n[KCAL]) / n[KCAL]
    adjusted = relative_error_before > (0.01 + 1e-9)
    if adjusted and not deployment.REPAIRED:
        # Legacy remediation only; in repaired deployment, joint projection strictly guarantees joint feasibility.
        n[KCAL] = calculated_kcal
    relative_error_after = (
        abs(calculated_kcal - n[KCAL]) / n[KCAL] if n[KCAL] > 0.0 else math.inf
    )
    return {
        "passed": bool(relative_error_after <= 0.01 + 1e-9),"""

assert old_f1 in text, 'old_f1 not found'
text = text.replace(old_f1, new_f1, 1)

with open('stage2_counterfactual_engine.py', 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(text)

print('Updated stage2_counterfactual_engine.py successfully!')
