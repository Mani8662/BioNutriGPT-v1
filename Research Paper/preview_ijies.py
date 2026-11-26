from pathlib import Path
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image,ImageDraw
out=Path('Research Paper/IJIES_Final');qa=out/'_QA';qa.mkdir(exist_ok=True)
for n in ['BioNutriGPT_IJIES','_Sources/IJIES_Official_Template']:
 p=out/(n+'.pdf');r=PdfReader(p);print(n,len(r.pages),'pages');print('Last page',r.pages[-1].extract_text()[-220:]);pages=convert_from_path(str(p),dpi=65)
 for start in range(0,len(pages),6):
  sheet=Image.new('RGB',(1500,1450),'#bbb');dr=ImageDraw.Draw(sheet)
  for i,im in enumerate(pages[start:start+6]):
   im.thumbnail((485,680));x=i%3*500;y=i//3*725;sheet.paste(im,(x,y+25));dr.text((x+5,y+5),str(start+i+1),fill='black')
  sheet.save(qa/(Path(n).name+'_'+str(start)+'.png'))
