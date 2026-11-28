from pathlib import Path
import sys,re
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_submission_documents as b
out=Path(__file__).resolve().parent/'Publication_Candidate_2026-11-24'
d=b.new_doc('BioNutriGPT publication candidate');lines=(out/'Manuscript.md').read_text(encoding='utf-8').splitlines();i=0;tn=0
while i<len(lines):
 line=lines[i].strip()
 if line.startswith('```mermaid'):
  i+=1
  while i<len(lines) and not lines[i].startswith('```'):i+=1
  b.figure(d,'Figure_1_Architecture','Figure 1 Four stage workflow with validated meal revisions',width=5.6)
 elif line.startswith('# '): d.add_paragraph('BioNutriGPT An Auditable Pipeline for Cardiometabolic Screening and Constraint Preserving Conversational Meal Planning','Title')
 elif line.startswith('##'):
  level=min(3,len(line)-len(line.lstrip('#'))-1);d.add_heading(line.lstrip('# '),level)
 elif line.startswith('|'):
  rows=[]
  while i<len(lines) and lines[i].strip().startswith('|'):
   cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
   if not all(re.fullmatch('[: -]+',x) for x in cells):rows.append(cells)
   i+=1
  tn+=1;b.caption(d,f'Table {tn} '+rows[0][0]);b.table(d,rows[0],rows[1:],size=9);continue
 elif line:
  b.text(d.add_paragraph(),line)
 i+=1
d.save(out/'Manuscript.docx');print('Saved manuscript DOCX',tn,'tables')
