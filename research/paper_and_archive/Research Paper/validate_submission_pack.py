"""Rasterize Word-exported PDFs, inspect structural bounds, and bundle drafts."""
import ast
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pymupdf
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
PACK = ROOT / "Submission_Pack"


def main():
    qa=PACK/"_QA";qa.mkdir(exist_ok=True)
    checks=[];thumbs=[]
    for pdf in sorted(PACK.rglob("*.pdf")):
        if "Figures" in pdf.parts or "_QA" in pdf.parts:continue
        doc=pymupdf.open(pdf);pages=[];out=qa/pdf.stem;out.mkdir(exist_ok=True)
        for i,page in enumerate(doc):
            png=out/f"page-{i+1}.png";page.get_pixmap(matrix=pymupdf.Matrix(1.3,1.3),alpha=False).save(png)
            outside=sum(b[0]<0 or b[1]<0 or b[2]>page.rect.width+1 or b[3]>page.rect.height+1 for b in page.get_text("blocks"))
            pages.append({"page":i+1,"characters":len(page.get_text().strip()),"outside_page_text_blocks":outside})
            im=Image.open(png).convert("RGB");im.thumbnail((306,396));tile=Image.new("RGB",(330,430),"#e9edf0")
            tile.paste(im,((330-im.width)//2,20));ImageDraw.Draw(tile).text((8,410),pdf.stem[:32]+f" p{i+1}",fill="black");thumbs.append(tile)
        checks.append({"file":str(pdf.relative_to(PACK)),"pages":len(doc),"page_checks":pages})
    for start in range(0,len(thumbs),12):
        chunk=thumbs[start:start+12];sheet=Image.new("RGB",(990,430*((len(chunk)+2)//3)),"#cbd4db")
        for j,tile in enumerate(chunk):sheet.paste(tile,((j%3)*330,(j//3)*430))
        sheet.save(qa/f"contact_sheet_{start//12+1}.png")
    (qa/"layout_checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    assert len(checks)==8, "Expected PDF review copies for eight Word drafts"
    assert not any(p["outside_page_text_blocks"] for c in checks for p in c["page_checks"])
    assert not any(p["characters"]<120 for c in checks for p in c["page_checks"])
    ns={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for docx in PACK.rglob("*.docx"):
        with ZipFile(docx) as z:
            styles=ET.fromstring(z.read("word/styles.xml"))
            assert not styles.findall(".//w:pBdr",ns), f"Unexpected decorative border in {docx.name}"
    source=PACK/"Reproducibility/source_code"
    missing=set()
    for py in source.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8-sig"))):
            names=[a.name.split(".")[0] for a in node.names] if isinstance(node,ast.Import) else ([node.module.split(".")[0]] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                if (ROOT.parent/(name+".py")).exists() and not (source/(name+".py")).exists():missing.add(name)
    assert not missing, f"Missing local imports: {missing}"
    manifest=json.loads((PACK/"Reproducibility/artifact_manifest.json").read_text(encoding="utf-8"))
    for name,digest in manifest["source_files"].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    data=json.loads((PACK/"Sources/submission_data.json").read_text(encoding="utf-8"))
    assert len(data["features"])==87 and len(data["rules"])==39 and len(data["catalog"])==49
    status={"submission_ready":False,"journal_selected":False,"author_details_confirmed":False,
            "document_drafts":8,"review_pdf_pages":len(thumbs),"current_data_counts":{"features":87,"nutrient_fields":39,"food_entries":49},
            "automated_checks":"PDF page bounds, nonblank pages, Word border removal, copied source hashes and local import closure passed.",
            "visual_review":"PDFs rasterized for contact-sheet and representative full-page review.",
            "packaged_synthetic_chat_tests":"Current LASI audit: 10/11 chatbot checks passed; allergy declaration/question expectation fails. Earlier 11/11 package pass is historical.",
            "benchmark_correction_tests":"Eight benchmark checks passed before the verified checkpointed v4 run.",
            "completed_v4":{"distinct_full_participants":36,"full_protocol_passes":"76/78","matched_participants":12,"matched_guarded_passes":"24/24","matched_qwen_passes":"2/24","same_state_pairs":25,"same_state_reduction_pct":53.38},
            "benchmark_execution_during_preparation":False,
            "pending":["author and institutional confirmations","journal-specific formatting","matched current LP/MILP study and broader independent dialogue comparisons","prediction reporting repairs","application regression contract reconciliation","final official checklist subitems and pages"]}
    (PACK/"PREPARATION_STATUS.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    files=[p for p in PACK.rglob("*") if p.is_file() and not any(part in {"_QA","__pycache__"} for part in p.relative_to(PACK).parts) and p.name!="PACK_MANIFEST.json"]
    excluded={'lasi_w1_indian_cohort_100.json','stage4_lasi_v4_frozen_scenarios.json','stage4_lasi_conversational_benchmark_v4_report.json','lasi_stage3_indian_cohort_report.json','65dc53fced0231708938236.zip'}
    assert not any(p.name in excluded or p.suffix=='.jsonl' or 'plan_sessions' in p.parts for p in files), 'Participant-level evidence must not enter the aggregate submission pack.'
    bundle={"status":"Working draft; not submission-ready", "files":{str(p.relative_to(PACK)).replace("\\","/"):{"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size} for p in sorted(files)}}
    (PACK/"PACK_MANIFEST.json").write_text(json.dumps(bundle,indent=2),encoding="utf-8")
    zip_path=ROOT/"BioNutriGPT_Submission_Pack_DRAFT.zip"
    with ZipFile(zip_path,"w",compression=ZIP_DEFLATED) as archive:
        for path in files+[PACK/"PACK_MANIFEST.json"]:archive.write(path,"BioNutriGPT_Submission_Pack/"+str(path.relative_to(PACK)).replace("\\","/"))
    print(json.dumps({"documents":8,"pages":len(thumbs),"files":len(files)+1,"zip":str(zip_path),"zip_bytes":zip_path.stat().st_size,"checks":"passed"},indent=2))


if __name__ == "__main__":main()
