import pandas as pd
df_w = pd.read_csv('Saved Model/stage1_robustness_uncertainty_v1/within_model_deterioration.csv')
df_b = pd.read_csv('Saved Model/stage1_robustness_uncertainty_v1/between_model_contrasts.csv')

print('=== CALIBRATED WITHIN-MODEL DETERIORATION (BioNutriGPT) ===')
sub = df_w[(df_w['model']=='bionutrigpt') & (df_w['variant']=='calibrated')]
for out in ['TARGET_DIABETES', 'TARGET_CVD', 'TARGET_OBESITY']:
    print(f'-- {out} --')
    for met in ['auroc', 'brier']:
        s_df = sub[(sub['outcome']==out) & (sub['metric']==met)]
        for _, r in s_df.iterrows():
            sc = r['scenario']
            dp = r['deterioration_point']
            dlo = r['deterioration_ci_lower']
            dhi = r['deterioration_ci_upper']
            sp = r['stressed_point']
            print(f"{sc:25s} | {met:5s} | det: {dp:+.4f} [{dlo:+.4f}, {dhi:+.4f}] | stressed: {sp:.4f}")

print('\n=== CALIBRATED BETWEEN-MODEL CONTRASTS (BioNutriGPT vs CatBoost & FT-Transformer) ===')
sub_b = df_b[(df_b['variant']=='calibrated') & (df_b['comparator'].isin(['catboost', 'ft_transformer']))]
for out in ['TARGET_DIABETES', 'TARGET_CVD']:
    print(f'-- {out} --')
    for comp in ['catboost', 'ft_transformer']:
        for met in ['auroc', 'brier']:
            s_df = sub_b[(sub_b['outcome']==out) & (sub_b['comparator']==comp) & (sub_b['metric']==met)]
            for _, r in s_df.iterrows():
                excl = '*' if r['excludes_zero'] else ' '
                sc = r['scenario']
                ddp = r['diff_deterioration_point']
                ddlo = r['diff_deterioration_ci_lower']
                ddhi = r['diff_deterioration_ci_upper']
                dsp = r['diff_stressed_point']
                dslo = r['diff_stressed_ci_lower']
                dshi = r['diff_stressed_ci_upper']
                print(f"{comp:15s} | {sc:25s} | {met:5s} | diff_det: {ddp:+.4f} [{ddlo:+.4f}, {ddhi:+.4f}] {excl} | diff_stress: {dsp:+.4f} [{dslo:+.4f}, {dshi:+.4f}]")

print('\n=== HOW MANY EXCLUDES ZERO IN ALL 480 CONTRASTS? ===')
print(df_b['excludes_zero'].value_counts())
print('\nBreakdown of excludes_zero by comparator and variant:')
print(df_b[df_b['excludes_zero']].groupby(['comparator', 'variant', 'outcome', 'metric']).size())
