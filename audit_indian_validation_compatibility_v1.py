"""Inspect local LASI metadata; no training, inference or participant export."""
from pathlib import Path
import ast, hashlib, json, shutil, tempfile, zipfile
import pyreadstat
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'Saved Model/indian_validation_compatibility_v1'

def main():
    OUT.mkdir(exist_ok=True)
    archive = ROOT / 'Data/65dc53fced0231708938236.zip'
    metadata = {}
    frames = {}
    selected = {
        '4_LASI_W1_Biomarker.sav': ['prim_key','bm067','bm071','indiaindividualweight','ssuid'],
        '3_LASI_W1_Individual_v4.sav': ['prim_key','dm005','dm003','dm008','dm021','ht003','ht006','ht007','ht006a','ht006fs4','ht006fs5','tu002','hb001','hb003','hb003_a','hb211','hb212','hb213','hb214'],
    }
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory() as tmp:
        members = [{'name': x.filename, 'bytes': x.file_size} for x in z.infolist()]
        for name in z.namelist():
            if not name.endswith('.sav'):
                continue
            if name.startswith('LASI_W1_Community'):
                metadata[name] = {'inspection_status': 'Not parsed: pyreadstat 1.3.8 on Python 3.14 crashed with access violation. Community-level data cannot replace individual biomarker assays.'}
                continue
            p = Path(tmp) / Path(name).name
            with z.open(name) as src, p.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            _, m = pyreadstat.read_sav(str(p), metadataonly=True)
            metadata[name] = {'rows': m.number_rows, 'columns': m.number_columns,
                'variables': [{'name': n, 'label': l, 'values': m.variable_value_labels.get(n, {})}
                              for n, l in zip(m.column_names, m.column_labels)]}
            print(name, m.number_rows, m.number_columns, flush=True)
            if name in selected:
                frames[name], _ = pyreadstat.read_sav(str(p), usecols=selected[name])
            p.unlink()
    tree = ast.parse((ROOT / 'train_bionutrigpt_stage1.py').read_text(encoding='utf-8-sig'))
    groups = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ['BIO_COLS', 'DIET_COLS', 'LIFE_COLS', 'DEMO_COLS']:
                    groups[t.id] = ast.literal_eval(node.value)
    assert sum(map(len, groups.values())) == 87
    individual = frames['3_LASI_W1_Individual_v4.sav']
    biomarker = frames['4_LASI_W1_Biomarker.sav']
    assert individual.prim_key.notna().all() and individual.prim_key.is_unique
    assert biomarker.prim_key.notna().all() and biomarker.prim_key.is_unique
    merged = individual.merge(biomarker, on='prim_key', validate='one_to_one')
    adult = merged[merged.dm005.between(20, 100) & merged.dm003.isin([1, 2])].copy()
    adult['bmi'] = adult.bm071 / (adult.bm067/100)**2
    measured = adult.bm067.between(120,210) & adult.bm071.between(30,180) & adult.bmi.between(14,55)
    # Engineering plausibility filters for this availability audit, not a locked study cohort.
    summary = {'merged_one_to_one':len(merged), 'age20_100_valid_sex':len(adult),
       'age45_plus_within_audit_adults':int((adult.dm005>=45).sum()),
       'valid_anthropometry':int(measured.sum()),
       'bmi30_positive':int((measured & (adult.bmi>=30)).sum()),
       'bmi30_negative':int((measured & (adult.bmi<30)).sum()),
       'diabetes_selfreport_yes':int((adult.ht003==1).sum()),
       'diabetes_selfreport_no':int((adult.ht003==2).sum()),
       'diabetes_selfreport_unknown':int((~adult.ht003.isin([1,2])).sum()),
       'broad_heart_or_stroke_yes':int(((adult.ht006==1)|(adult.ht007==1)).sum()),
       'broad_heart_or_stroke_no':int(((adult.ht006==2)&(adult.ht007==2)).sum()),
       'positive_india_weight':int((adult.indiaindividualweight>0).sum()),
       'nonmissing_selected_fields':{c:int(adult[c].notna().sum()) for c in selected['3_LASI_W1_Individual_v4.sav'] if c!='prim_key'},
       'caution':'Nonmissing does not imply valid or harmonized. Counts are unweighted availability counts, not prevalence or validation metrics.'}
    result = {'date': '2026-11-22', 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'archive_members': members, 'model_feature_groups': groups, 'metadata': metadata,
              'availability_counts':summary}
    (OUT / 'lasi_metadata.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
