"""Build journal-neutral Word drafts using the bundled python-docx runtime."""
import json
from pathlib import Path
import re
import shutil

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

BASE = Path(__file__).resolve().parent / "Submission_Pack"
DATA = json.loads((BASE / "Sources/submission_data.json").read_text(encoding="utf-8"))
TITLE = "BioNutriGPT An Auditable Pipeline for Cardiometabolic Screening and Constraint Preserving Conversational Meal Planning"


def new_doc(short, title=None):
    d=Document(); sec=d.sections[0]
    for st in d.styles:
        for border in st.element.xpath(".//w:pBdr"):
            border.getparent().remove(border)
    sec.page_width=Inches(8.5); sec.page_height=Inches(11)
    sec.top_margin=sec.bottom_margin=Inches(.8)
    sec.left_margin=sec.right_margin=Inches(.85)
    for name in ("Normal","Title","Subtitle","Heading 1","Heading 2","Heading 3","Caption"):
        st=d.styles[name];st.font.name="Times New Roman";st.font.color.rgb=RGBColor(0,0,0)
        st.font.size=Pt(11.5)
    d.styles["Normal"].paragraph_format.space_after=Pt(6)
    d.styles["Normal"].paragraph_format.line_spacing=1.12
    d.styles["Title"].font.size=Pt(17);d.styles["Title"].font.bold=True
    d.styles["Title"].paragraph_format.space_after=Pt(14)
    for name,size in (("Heading 1",14),("Heading 2",12),("Heading 3",11.5)):
        d.styles[name].font.size=Pt(size);d.styles[name].font.bold=True
        d.styles[name].paragraph_format.space_before=Pt(12)
        d.styles[name].paragraph_format.space_after=Pt(6)
        d.styles[name].paragraph_format.keep_with_next=True
    d.styles["Caption"].font.size=Pt(10)
    p=sec.header.paragraphs[0];p.text="BioNutriGPT | Research draft";p.style=d.styles["Normal"]
    for r in p.runs:r.font.size=Pt(9)
    p=sec.footer.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    r=p.add_run();r.font.size=Pt(9)
    field=OxmlElement("w:fldSimple");field.set(qn("w:instr"),"PAGE");r._r.addnext(field)
    d.core_properties.author="";d.core_properties.last_modified_by=""
    d.core_properties.title=short;d.core_properties.subject="Journal-neutral research draft"
    if title:d.add_paragraph(title,"Title")
    return d


def text(p, value):
    # Preserve actual clickable source links instead of raw Markdown syntax.
    pattern=r"(\[([^\]]+)\]\((https?://[^)]+)\)|\*\*([^*]+)\*\*|`([^`]+)`|\*([^*]+)\*)"
    cursor=0
    for m in re.finditer(pattern,value):
        p.add_run(value[cursor:m.start()])
        if m.group(2):
            rel=p.part.relate_to(m.group(3),"http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",is_external=True)
            h=OxmlElement("w:hyperlink");h.set(qn("r:id"),rel)
            run=OxmlElement("w:r");props=OxmlElement("w:rPr");color=OxmlElement("w:color");color.set(qn("w:val"),"1D526C");props.append(color);run.append(props)
            t=OxmlElement("w:t");t.text=m.group(2);run.append(t);h.append(run);p._p.append(h)
        else:
            r=p.add_run(m.group(4) or m.group(5) or m.group(6));r.bold=bool(m.group(4));r.italic=bool(m.group(6))
        cursor=m.end()
    p.add_run(value[cursor:])


def table(d, headers, rows, widths=None, size=9.5):
    t=d.add_table(rows=1, cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    if widths is None:widths=[6.8/len(headers)]*len(headers)
    for col,w in zip(t.columns,widths):col.width=Inches(w)
    pr=t._tbl.tblPr
    borders=OxmlElement("w:tblBorders")
    for tag in ("top","left","bottom","right","insideH","insideV"):
        e=OxmlElement("w:"+tag);e.set(qn("w:val"),"single");e.set(qn("w:sz"),"4");e.set(qn("w:color"),"D9D9D9");borders.append(e)
    pr.append(borders)
    for i,head in enumerate(headers):t.rows[0].cells[i].text=str(head)
    repeat=OxmlElement("w:tblHeader");t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for row in rows:
        cells=t.add_row().cells
        for i,v in enumerate(row):cells[i].text=str(v)
    for ri,row in enumerate(t.rows):
        # Keep each record intact, while allowing the whole table to paginate.
        cant=OxmlElement("w:cantSplit");row._tr.get_or_add_trPr().append(cant)
        for ci,cell in enumerate(row.cells):
            cell.width=Inches(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cp=cell._tc.get_or_add_tcPr();margins=OxmlElement("w:tcMar")
            for side,val in (("top",60),("bottom",60),("left",75),("right",75)):
                e=OxmlElement("w:"+side);e.set(qn("w:w"),str(val));e.set(qn("w:type"),"dxa");margins.append(e)
            cp.append(margins)
            if ri==0:
                shade=OxmlElement("w:shd");shade.set(qn("w:fill"),"E8EEF1");cp.append(shade)
            for p in cell.paragraphs:
                p.paragraph_format.space_after=Pt(1);p.paragraph_format.space_before=Pt(1)
                p.paragraph_format.line_spacing=1.0
                if len(str(cell.text))<17:p.alignment=WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:r.font.name="Times New Roman";r.font.size=Pt(size);r.bold=ri==0
    spacer=d.add_paragraph()
    spacer.paragraph_format.space_before=Pt(0)
    spacer.paragraph_format.space_after=Pt(0)
    spacer.paragraph_format.line_spacing=Pt(1)
    spacer.add_run().font.size=Pt(1)
    return t


def caption(d, value):
    p=d.add_paragraph(value,"Caption");p.paragraph_format.keep_with_next=True


def figure(d,name,label,width=6.5):
    p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
    r=p.add_run();r.add_picture(str(BASE/"Figures"/(name+".png")),width=Inches(width))
    props=p._p.xpath(".//wp:docPr")
    for prop in props:prop.set("descr",label)
    p=d.add_paragraph(label,"Caption");p.paragraph_format.keep_with_next=False


def math_bounds(d):
    p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    om=OxmlElement("m:oMath")
    def run(parent,t):
        r=OxmlElement("m:r");e=OxmlElement("m:t");e.text=t;r.append(e);parent.append(r)
    def sub(parent,b,s):
        e=OxmlElement("m:sSub");base=OxmlElement("m:e");run(base,b);ss=OxmlElement("m:sub");run(ss,s);e.extend([base,ss]);parent.append(e)
    sub(om,"L","k");run(om," ≤ ")
    n=OxmlElement("m:nary");props=OxmlElement("m:naryPr");ch=OxmlElement("m:chr");ch.set(qn("m:val"),"∑");props.append(ch)
    hide=OxmlElement("m:supHide");hide.set(qn("m:val"),"1");props.append(hide);n.append(props)
    lo=OxmlElement("m:sub");run(lo,"i");n.append(lo);n.append(OxmlElement("m:sup"))
    body=OxmlElement("m:e");sub(body,"a","ki");run(body," ");sub(body,"q","i");n.append(body);om.append(n)
    run(om," ≤ ");sub(om,"U","k");p._p.append(om)


def manuscript():
    d=new_doc("BioNutriGPT manuscript")
    lines=(BASE/"Sources/Main_Manuscript.md").read_text(encoding="utf-8").splitlines();i=0;tn=0
    captions=["Table 1 Cohort and outcome counts audited from the current files",
              "Table 2 Current catalog characteristics",
              "Table 3 Saved unweighted Stage 1 temporal holdout performance",
              "Table 4 Historical 29 entry optimizer benchmark using seven scored bounds",
              "Table 5 LASI profile acceptance across three dietary modes",
              "Table 6 Matched v4 conversational comparison on 12 selected participants"]
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith("```mermaid"):
            while i<len(lines)-1:
                i+=1
                if lines[i].startswith("```"):break
            figure(d,"Figure_1_Architecture","Figure 1 Four stage workflow with controlled revisions returned to Stage 3",width=5.65)
        elif line.startswith("# "):
            d.add_paragraph(TITLE,"Title")
        elif line.startswith("##"):
            if line.startswith("### 3.3"):
                figure(d,"Figure_3_Current_Catalog_Coverage","Figure 3 Rule roles and catalog completeness before preference filtering")
            level=min(3,len(line)-len(line.lstrip("#"))-1)
            label=re.sub(r"^\d+(?:\.\d+)*\.?\s*","",line.lstrip("# "))
            label=re.sub(r"[^\w\s]"," ",label);label=" ".join(label.split())
            d.add_heading(label,level=max(1,level))
        elif line.startswith("|"):
            chunk=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                cells=[c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[:\- ]+",c) for c in cells):chunk.append(cells)
                i+=1
            caption(d,captions[tn]);tn+=1
            widths={1:[2.7,1.0,1.05,1.05,1.0],2:[5.2,1.6],3:[3.2,1.2,1.2,1.2],4:[2.8,1.0,1.1,.9,1.0],5:[2.5,.9,1.05,1.35,1.0],6:[3.6,1.6,1.6]}[tn]
            table(d,chunk[0],chunk[1:],widths=widths,size=9.5)
            if tn==3:figure(d,"Figure_2_Stage1_Saved_Discrimination","Figure 2 Saved discrimination estimates on the NHANES temporal holdout")
            if tn==6:figure(d,"Figure_4_Verified_Stage4_Comparison","Figure 4 Measured v4 matched protocol success and conditional same input state serving change")
            continue
        elif line.startswith("`L_k"):
            math_bounds(d)
        else:
            if re.match(r"^\d+\. ",line):
                # Keep explicitly numbered reference and body records as authored.
                p=d.add_paragraph();text(p,line)
            else:p=d.add_paragraph();text(p,line)
        i+=1
    path=BASE/"Main_Manuscript_DRAFT.docx";d.save(path)


def supplementary():
    d=new_doc("BioNutriGPT supplementary material","BioNutriGPT Supplementary Material")
    d.add_paragraph("This supplement documents the current implementation and the provenance of saved results. It does not introduce new model evaluations or human studies. Historical catalogs and current rules are kept separate.")
    d.add_heading("Table S1 Model feature order",1)
    d.add_paragraph("The 87 predictors below follow the training-code contract. Units and transformations must be interpreted using the supplied preprocessing artifacts. This table is not a patient upload template.")
    table(d,["Position","Modality","Feature code"],[[r["position"],r["modality"],r["feature"]] for r in DATA["features"]],[.65,1.15,5.0],size=10)
    d.add_heading("Table S2 Nutrient roles and coverage",1)
    d.add_paragraph("The current rule version has 30 active fields, seven report-only fields and two separate exposure fields. The source links are contextual references, not endorsements of every project threshold. Numeric patient-specific bounds are intentionally not invented here; the exact rule implementation and role JSON accompany this pack.")
    table(d,["Nutrient code","Unit","Planning role","Records with declared gaps"],[[r["nutrient"],r["unit"],r["planning_role"].replace("_"," "),sum(r["nutrient"] in f["unknown_nutrients"] for f in DATA["catalog"])] for r in DATA["rules"]],[1.4,.6,3.0,1.8],size=9.5)
    d.add_heading("Rule interpretation notes",2)
    for r in DATA["rules"]:
        p=d.add_paragraph();p.add_run(r["nutrient"]+" ").bold=True;p.add_run(r["review_note"]+" ");text(p,"[Reference]("+r["source_url"]+")")
    d.add_heading("Table S3 Current food catalog",1)
    d.add_paragraph("Catalog sourced-cooked-v5-veg-nonveg contains 49 records: 20 INDB recipe variants and 29 USDA proxies. The entries below are standardized catalog portions; names do not establish equivalence to every household recipe. All entries retain possible unknown allergens. The source-family column abbreviates provenance and must be read with the catalog artifact manifest.")
    table(d,["Food ID","Catalog name","Portion g","Source"],[[r["food_id"],r["name"],r["serving_weight_g"],"INDB" if r["data_source"].startswith("INDB") else "USDA proxy"] for r in DATA["catalog"]],[1.65,3.35,.7,1.1],size=9)
    d.add_heading("Table S4 Evidence versions",1)
    table(d,["Evidence","Catalog and scope","Interpretation"],[
        ["Current descriptive audit","49 entries; 30 active fields","43 records without declared active gaps before other filters"],
        ["Historical optimizer benchmark","29 entries; 20 cases; seven scored nutrients","Not a current strict acceptance study"],
        ["Historical conversational benchmark","31 entries; 8 scenarios; 16 turns; 2 base profiles","Predates the latest chatbot repairs"],
        ["Previous repair verification","11 chatbot checks plus 13 production checks","Historical 24-check pass; current chatbot rerun 10/11"],
        ["LASI optimizer pilot","100 profiles; 300 mode-specific solves","Accepted 91/100, 96/100 and 98/100"],
        ["Final LASI conversation v4","36 distinct participants; 78 turns per guarded variant","76/78 full protocol passes; 29 valid revisions and 11 justified refusals"],
        ["Matched v4 Qwen comparison","12 participants; 24 turns per system","24/24 versus 2/24 protocol passes; 9/9 versus 0/9 feasible proposals"]],[1.7,2.6,2.5],size=10)
    d.add_heading("LASI adapter and evaluation limitations",1)
    d.add_paragraph("The LASI sample is balanced and unweighted, not nationally representative performance evidence. The adapter uses observed anthropometry and reported diagnoses, assumed 0.85/0.10 routing scores and 2800/2600 mg sodium/potassium baselines. Stage 1 and the full counterfactual search are not evaluated. The verified v4 study covers four disease strata, with frozen requests, complete-data eligibility, full-precision nutrient coefficients, canonical validator rules and separate timeout/refusal scoring. Earlier diabetes-only v2 studies remain historical. Only aggregate v4 summaries are packaged; participant-level inputs and detailed clinical logs remain local.")
    d.add_heading("Table S5 Verified v4 endpoints and conditional uncertainty",1)
    table(d,["Endpoint","Measured result","Interpretation"],[
        ["Full guarded protocol","76/78; 29/30 oracle-feasible proposals","Two timeout-related failures remain"],
        ["Matched protocol","24/24 guarded; 2/24 Qwen","12 selected distinct participants, not population evidence"],
        ["Same input state ablation","7.72 versus 16.56; n=25","53.38% lower mean serving change among successful pairs"],
        ["Paired difference interval","8.84; 95% bootstrap interval 6.72 to 11.12","Scenario-level resampling; successful-pair conditional"],
        ["Audit trail","24 prompts verified; four stage checkpoints","Recorded model revision, full raw outputs and unchanged source hashes"]],[1.7,2.3,2.8],size=9.5)
    d.add_heading("Evaluation exclusions and unresolved analyses",1)
    d.add_paragraph("The assigned-count 100-case LLM report is excluded as empirical evidence. Survey-weighted uncertainty, calibration-split independence and matched comparator training budgets need reconciliation. No measured calibration curve is reconstructed from aggregate summary scores. No current-version optimizer or real-LLM benchmark is generated during document preparation.")
    d.add_heading("Computational formulation and failure behavior",1)
    d.add_paragraph("The production solver uses integer portion counts, binary food selection, dietary-mode category constraints and patient-specific active nutrient intervals. Active interval deviations are fixed to zero. Revision cost includes absolute serving changes, while independent validation checks the candidate before acceptance. A rejected revision preserves the previously accepted basket.")
    math_bounds(d)
    d.add_paragraph("Allergy declarations persist at patient level. Known matching foods are excluded; incomplete allergen information means numerical acceptance is not allergen safety certification. An accepted proposal must match the server's pending current-version proposal. Optional free-text chatbot rewriting is disabled; the separate report compiler remains implemented.")
    d.add_heading("Reproduction and access",1)
    d.add_paragraph("The Reproducibility folder contains a curated source snapshot, observed environment inventory, artifact hashes and local verification instructions. Full NHANES tables, original food workbooks, patient session records and model weights are not included. The declared remote URL is not a verified frozen release. Data permissions and clean-environment execution remain author tasks.")
    d.save(BASE/"Supplementary_Material_DRAFT.docx")


def supporting():
    folder=BASE/"Author_Completion"
    d=new_doc("BioNutriGPT title page author form","BioNutriGPT Title Page")
    d.add_paragraph("Author completion form. This is not a completed title page for upload.")
    p=d.add_paragraph();p.add_run("Article title ").bold=True;p.add_run(TITLE)
    for label in ("Author names in agreed order","Affiliation for each author","Department and institution","City and country","Corresponding author name","Corresponding author email","Postal address if required","ORCID identifiers if available"):
        p=d.add_paragraph();p.add_run(label+"\n").bold=True;p.add_run("________________________________________________________________")
    d.save(folder/"Title_Page_AUTHOR_COMPLETION.docx")

    d=new_doc("BioNutriGPT cover letter draft","BioNutriGPT Cover Letter Draft")
    d.add_paragraph("Journal-neutral draft. Select the journal and obtain author confirmations before sending.")
    d.add_paragraph("Dear Editor,")
    d.add_paragraph("We present the manuscript entitled “"+TITLE+"” for consideration as a computational research article. BioNutriGPT connects laboratory-assisted cardiometabolic screening, explicit nutrient rules, discrete meal optimization and controlled conversational revisions.")
    d.add_paragraph("The central research focus is preservation of declared constraints and saved preferences across repeated meal-plan changes. The work separates observed patient inputs, revision proposals and accepted plan versions, with independent numerical acceptance checks and traceable food provenance.")
    d.add_paragraph("The manuscript distinguishes saved screening results, historical studies and the verified v4 LASI conversational evaluation. In its matched 24-turn comparison, the guarded system passed 24/24 protocol turns versus 2/24 for the specified Qwen baseline. Same-input-state successful pairs showed less serving change with the penalty. These are selected-case computational results; the manuscript does not claim Indian screening validity, clinical benefit, allergen-free certification or overall Transformer superiority. Further prediction and optimizer comparisons remain necessary.")
    d.add_paragraph("The journal-specific suitability paragraph, author signature and required originality, author-approval and competing-submission statements must be completed after confirmation. No such declarations are assumed in this draft.")
    d.add_paragraph("Corresponding author and contact details: to be supplied by the authors.")
    d.save(folder/"Cover_Letter_DRAFT.docx")

    d=new_doc("BioNutriGPT highlights draft","BioNutriGPT Research Highlights")
    highlights=["Meal revisions return to optimization before numerical acceptance.",
                "Observed intake stays separate from proposed and accepted plans.",
                "Saved restrictions persist across revisions and version rollback.",
                "Food provenance and missing nutrient fields remain explicit."]
    d.add_paragraph("Draft highlights. Adapt the number and character limits to the chosen journal.")
    for h in highlights:d.add_paragraph(h,style="List Bullet")
    d.save(BASE/"Highlights_DRAFT.docx")

    d=new_doc("BioNutriGPT author declarations form","BioNutriGPT Author Declarations")
    d.add_paragraph("Complete these fields with your co-authors or supervisor. Public data do not automatically establish an institutional ethics exemption. Do not sign or upload this form while answers remain unknown.")
    fields=[("Authors and contributions","Give each author's name and actual CRediT roles; confirm author order and final manuscript approval."),
            ("Funding","List confirmed funding and funder roles. If no external funding was received, state that only after checking."),
            ("Competing interests","Collect a declaration from every author. Do not assume that all authors have none."),
            ("Ethics determination","Record the institution, committee and approval, waiver, exemption or not-required determination as applicable. Confirm the wording with your institution."),
            ("Data and code availability","Supply a working public release link or controlled access instructions, version identifier and applicable licenses."),
            ("Registration and protocol","Provide genuine registration details, or state that the study was not registered. A draft protocol is not preregistration."),
            ("Consent and new participant activity","Confirm whether any new participant or expert data were collected and the applicable consent requirements."),
            ("Originality and simultaneous submission","Confirm manuscript originality, related preprints and absence of incompatible simultaneous submissions before using the journal's declaration."),
            ("AI assistance disclosure","AI assistance was used to help draft and organize this submission pack. Confirm the final description of tools and tasks and the chosen journal's disclosure requirements; authors must verify and take responsibility for submitted content.")]
    for title,body in fields:
        d.add_heading(title,2);d.add_paragraph(body);d.add_paragraph("Author response: __________________________________________________")
    d.save(folder/"Declarations_AUTHOR_CONFIRMATION.docx")


AI_ROWS=[
 ("1","Study title","Title","Partial: add the study population if required for the prediction focus."),
 ("2","Abstract","Abstract","Drafted with verified v4 results; final journal limits and prediction uncertainty pending."),
 ("3","Clinical context","Introduction and limitations","Partial: intended users and inequalities need fuller treatment."),
 ("4","Study aims","Introduction","Drafted: distinguish development, evaluation and software aims."),
 ("5","Data origins","Methods Cohorts","Partial: exact acquisition dates and representativeness audit."),
 ("6","Participant selection","Methods Cohorts","Partial: executable full inclusion/exclusion flow pending."),
 ("7","Data preparation","Methods Cohorts; preprocessing report","Partial: audit fitted preprocessing boundaries."),
 ("8","Outcome measurement","Methods Cohorts","Partial: ascertainment completeness and blinding details."),
 ("9","Predictor definitions","Methods Cohorts; Supplement S1","Partial: measurement and transformation details."),
 ("10","Sample justification","Methods Cohorts","Pending: justify analysis sample and precision."),
 ("11","Missing values","Methods; preprocessing report","Partial: quantify by split and group."),
 ("12","Analysis methods","Methods Architecture and Evaluation","Partial: matched tuning, survey design and calibration repair."),
 ("13","Outcome imbalance","Training source code","Pending: report weighting/loss and its calibration implications."),
 ("14","Fairness methods","Limitations","Pending: specify subgroup methods without claiming fairness."),
 ("15","Prediction outputs","Methods Screening model","Partial: deployed probabilities and any threshold rationale."),
 ("16","Dataset differences","Methods Cohorts; Discussion","Partial: formal split-wise covariate comparison."),
 ("17","Ethics","Declarations","Author confirmation required."),
 ("18","Open science","Declarations; Reproducibility","Partial: funding, conflicts, registration, access and release pending."),
 ("19","Public involvement","Declarations","No completed involvement documented; author confirmation."),
 ("20","Participant results","Cohort table; Supplement","Partial: full flow and subgroup characteristics."),
 ("21","Analysis denominators","Cohort table","Partial: events within training and validation subsets."),
 ("22","Model access","Methods; source snapshot","Partial: frozen executable release and access terms."),
 ("23","Predictive performance","Results Stage 1","Partial: validated confidence intervals and subgroups."),
 ("24","Updated models","Methods Evaluation","Pending: recalibration independence and deployed model audit."),
 ("25","Interpretation","Discussion","Drafted; revise after current experiments."),
 ("26","Study limitations","Discussion","Drafted; final review after analysis."),
 ("27","Implementation scope","Discussion and next evaluation","Partial: user expertise, bad inputs and external testing.")]

LLM_ROWS=[
 ("1","Study title","Title","Partial: state exact evaluated LLM role in final title if central."),
 ("2","Abstract","Abstract","Partial: distinguish deterministic chat and LLM compiler."),
 ("3","Use context","Introduction","Drafted: computational prototype, not clinical deployment."),
 ("4","Objectives","Introduction","Drafted: specify LLM evaluation versus tool-layer evaluation."),
 ("5","Data provenance","Methods LASI evaluation; v4 manifest","Current cases frozen and audited; selected initially feasible profiles."),
 ("6","Model methods","Methods LASI evaluation; v4 aggregates","Revision, prompts, settings, GPU and checkpoints recorded; access terms pending."),
 ("7","Output assessment","Methods Evaluation; Results historical chat","Partial: joint task success, denominators and assessor details."),
 ("8","Annotation","Historical scripted scenarios","Pending applicability decision; no new human annotation claimed."),
 ("9","Prompt development","Benchmark source","Partial: prompt selection data and separation from test cases."),
 ("10","Summarization","Separate report compiler","Pending applicability decision for retained compiler study."),
 ("11","Instruction tuning","Current prototype","No project LLM tuning documented; confirm applicability."),
 ("12","Compute","Methods and matched v4 results","Warm-model timings measured; cold start and application latency differ."),
 ("13","Open science","Declarations; Reproducibility","Partial: release, disclosures and access terms."),
 ("14","Public involvement","Declarations","No completed involvement documented; confirm with authors."),
 ("15","Study data results","Methods LASI and Results","36 distinct participants, four strata; initial-feasibility selection disclosed."),
 ("16","Model or system details","Methods Stage 4","Partial: separate compiler, prompt-only and tool-backed roles."),
 ("17","Evaluation results","Results LASI v4; Supplement S5","Verified counts, paired analysis and conditional uncertainty; wider evaluation pending."),
 ("18","Interpretation","Discussion","Drafted; contextualize historical-only comparison."),
 ("19","Limitations","Discussion","Drafted; no human acceptability or clinical benefit claimed.")]


def reporting():
    for name,title,rows,url in [("TRIPOD_AI_Reporting_Map_DRAFT","BioNutriGPT TRIPOD AI Reporting Map",AI_ROWS,"https://www.tripod-statement.org/wp-content/uploads/2019/12/TRIPODAI_checklist.pdf"),
                                ("TRIPOD_LLM_Reporting_Map_DRAFT","BioNutriGPT TRIPOD LLM Reporting Map",LLM_ROWS,"https://www.tripod-statement.org/wp-content/uploads/2025/01/TRIPOD-LLM-Checklist.pdf")]:
        d=new_doc(title,title)
        d.add_paragraph("Working main-item coverage map, not a completed official checklist or compliance certificate. Labels are abbreviated and paraphrased. Use the official checklist for its full subitems and applicability, then add final manuscript page numbers after the analysis and journal formatting are complete.")
        p=d.add_paragraph();text(p,"[Official checklist]("+url+")")
        table(d,["Item","Topic","Draft location","Status and remaining work"],rows,[.45,1.25,1.9,3.2],size=9.5)
        d.save(BASE/"Reporting"/(name+".docx"))


def main():
    manuscript();supplementary();supporting();reporting()
    print("Created 8 Word drafts and author-completion forms.")


if __name__ == "__main__":main()
