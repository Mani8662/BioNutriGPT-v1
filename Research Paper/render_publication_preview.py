from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image,ImageDraw
p=Path('Research Paper/Publication_Candidate_2026-11-24');q=p/'_QA';q.mkdir(exist_ok=True);doc=convert_from_path(str(p/'Manuscript.pdf'),dpi=55);print('Pages',len(doc))
for start in range(0,len(doc),8):
 sheet=Image.new('RGB',(1200,1700),'#ccc')
 for i,page in enumerate(list(doc)[start:start+8]):
  im=page.copy();im.thumbnail((295,810));x=(i%4)*300;y=(i//4)*850;sheet.paste(im,(x,y+25));ImageDraw.Draw(sheet).text((x+5,y+5),str(start+i+1),fill='black')
 sheet.save(q/f'sheet_{start}.png')
