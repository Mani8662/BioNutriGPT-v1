"""Build IJIES manuscript from the retained official template; no scientific runs."""
from pathlib import Path
from copy import deepcopy
import re, json, hashlib
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Cm, Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT=Path(__file__).resolve().parent/'IJIES_Final'
FIG=OUT/'Figures'; FIG.mkdir(exist_ok=True)
FONT=Path('C:/Windows/Fonts')
def font(size,bold=False):return ImageFont.truetype(str(FONT/('timesbd.ttf' if bold else 'times.ttf')),size)

def figures():
    im=Image.new('RGB',(1800,930),'white');d=ImageDraw.Draw(im)
    boxes=[(55,190,420,440),(495,190,860,440),(935,190,1300,440),(1375,190,1740,440)]
    labels=[('Screening','87 measured/derived inputs','Three calibrated scores'),('Nutrient search','Rules and joint projection','Hypothetical targets'),('Meal optimization','Integer portions','Hard nutrient constraints'),('Conversation','Inspect or request revision','Accept or roll back')]
    for box,lines in zip(boxes,labels):
        d.rounded_rectangle(box,radius=12,fill='#eff4f8',outline='#243d50',width=3)
        for j,line in enumerate(lines):
            f=font(33,j==0);d.text(((box[0]+box[2])/2,box[1]+50+j*62),line,font=f,fill='black',anchor='mm')
    for a,b in zip(boxes,boxes[1:]):
        d.line((a[2]+5,315,b[0]-8,315),fill='black',width=4);d.polygon([(b[0]-8,315),(b[0]-25,305),(b[0]-25,325)],fill='black')
    d.text((900,80),'Observed patient inputs remain separate from proposed meals',font=font(38,True),fill='black',anchor='mm')
    d.rounded_rectangle((935,550,1740,710),radius=12,outline='#243d50',width=3)
    d.text((1337,595),'Independent basket validator',font=font(36,True),fill='black',anchor='mm')
    d.text((1337,655),'Recalculate totals and restrictions before acceptance',font=font(31),fill='black',anchor='mm')
    d.line((1117,440,1117,545),fill='black',width=4);d.polygon([(1117,545),(1107,527),(1127,527)],fill='black')
    d.line((1557,550,1557,445),fill='black',width=4);d.polygon([(1557,445),(1547,463),(1567,463)],fill='black')
    d.line((1557,190,1557,130,1117,130,1117,185),fill='#3b6078',width=4)
    d.polygon([(1117,185),(1107,167),(1127,167)],fill='#3b6078')
    d.text((1337,110),'Revision request',font=font(27),fill='black',anchor='mm')
    d.text((900,825),'Food catalog and provenance supply coefficients to optimization and validation',font=font(35),fill='black',anchor='mm')
    im.save(FIG/'architecture.png',dpi=(300,300))
    im=Image.new('RGB',(1800,1050),'white');d=ImageDraw.Draw(im)
    names=['Nominal','10% removed','20% removed','No dairy','No legumes','5% composition','10% composition']
    aa=[299,296,224,200,100,236,192];bb=[299,296,224,200,100,215,159]
    left,top,right,bottom=390,130,1640,930
    for tick in [0,50,100,150,200,250,300]:
        x=left+(right-left)*tick/300;d.line((x,top,x,bottom),fill='#ddd',width=2);d.text((x,bottom+30),str(tick),font=font(28),fill='black',anchor='mm')
    for i,(name,a,b) in enumerate(zip(names,aa,bb)):
        y=top+i*112+20;d.text((left-20,y+30),name,font=font(32),fill='black',anchor='rm')
        for j,(v,c) in enumerate([(a,'#274c68'),(b,'#a75a25')]):
            yy=y+j*40;x=left+(right-left)*v/300;d.rectangle((left,yy,x,yy+30),fill=c);d.text((x+8,yy+15),str(v),font=font(25),fill='black',anchor='lm')
    d.rectangle((500,40,535,70),fill='#274c68');d.text((550,55),'Bounds-based A',font=font(32),fill='black',anchor='lm')
    d.rectangle((1030,40,1065,70),fill='#a75a25');d.text((1080,55),'Target-aware B',font=font(32),fill='black',anchor='lm')
    d.text((1030,1010),'Compliant baskets out of 300 participant-mode cells',font=font(32),fill='black',anchor='mm')
    im.save(FIG/'robustness.png',dpi=(300,300))
    (FIG/'figure_data.json').write_text(json.dumps({'scenarios':names,'bounds_based':aa,'target_aware':bb,'denominator':300},indent=2))

def border(p,side):
    pr=p._p.get_or_add_pPr();bs=OxmlElement('w:pBdr');b=OxmlElement('w:'+side);b.set(qn('w:val'),'single');b.set(qn('w:sz'),'6');bs.append(b);pr.append(bs)

def rich(p,text):
    parts=re.split(r'(\*\*.*?\*\*)',text)
    for t in parts:
        r=p.add_run(t[2:-2] if t.startswith('**') else t);r.bold=t.startswith('**')

def table(doc,rows):
    n=len(rows[0]); widths={3:[8.0,4.55,4.55],4:[4.1,5.4,3.8,3.8],5:[5.5,2.9,2.9,2.9,2.9]}.get(n,[17.1/n]*n)
    if rows[0][0]=='Partition':widths=[5.5,2.9,2.9,2.9,2.9]
    if rows[0][0]=='Frozen scorer':widths=[3.4,6.1,3.7,3.9]
    if rows[0][0]=='Scenario':widths=[5.0,3.2,3.2,5.7]
    t=doc.add_table(rows=0,cols=n);t.autofit=False;t.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for c,w in zip(t.columns,widths):c.width=Cm(w)
    for ri,row in enumerate(rows):
        cells=t.add_row().cells
        trpr=cells[0]._tc.getparent().get_or_add_trPr();cant=OxmlElement('w:cantSplit');trpr.append(cant)
        if ri==0:trpr.append(OxmlElement('w:tblHeader'))
        for ci,(c,v,w) in enumerate(zip(cells,row,widths)):
            c.width=Cm(w);p=c.paragraphs[0];p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.keep_with_next=ri<len(rows)-1
            p.paragraph_format.line_spacing=Pt(11.5);p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3)
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER
            r=p.add_run(v);r.font.size=Pt(10);r.bold=ri==0
            if ri==0:
                sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'EEEEEE');c._tc.get_or_add_tcPr().append(sh)
    pr=t._tbl.tblPr;bs=OxmlElement('w:tblBorders')
    for name in ['top','bottom','insideH']:
        e=OxmlElement('w:'+name);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');bs.append(e)
    pr.append(bs)
    doc.add_paragraph().paragraph_format.line_spacing=Pt(4)

def build(source,dest):
    template=OUT/'_Sources/IJIES_Official_Template.docx';doc=Document(template)
    logo = next((deepcopy(p._p) for p in doc.paragraphs[:2] if p._p.xpath('.//w:drawing') or p._p.xpath('.//w:pict')), None)
    for child in list(doc._element.body):
        if child.tag!=qn('w:sectPr'):doc._element.body.remove(child)
    for sec in doc.sections:
        sec.page_width=Cm(21);sec.page_height=Cm(29.7);sec.top_margin=sec.bottom_margin=Cm(2.54);sec.left_margin=sec.right_margin=Cm(1.905)
        sec.header_distance=Cm(1.5);sec.footer_distance=Cm(1.75)
        cols=sec._sectPr.find(qn('w:cols'));cols.set(qn('w:num'),'1')
        for hf in [sec.header,sec.footer,sec.first_page_header,sec.first_page_footer]:
            for child in list(hf._element):hf._element.remove(child)
        p=sec.footer.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.RIGHT
        fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');p._p.append(fld)
    for name in ['Title','Heading 1','Heading 2','Caption']:
        if name not in doc.styles:doc.styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH)
    for st in doc.styles:
        if st.type==WD_STYLE_TYPE.PARAGRAPH:
            st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)
    st=doc.styles['Normal'];st.font.size=Pt(11);pf=st.paragraph_format;pf.line_spacing=Pt(12.5);pf.space_after=pf.space_before=Pt(0);pf.first_line_indent=Pt(16.5);pf.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    for name,size in [('Title',14),('Heading 1',12),('Heading 2',11),('Caption',10)]:
        st=doc.styles[name];st.font.size=Pt(size);st.font.bold=name!='Caption';pf=st.paragraph_format;pf.first_line_indent=Pt(0);pf.line_spacing=Pt(size+2);pf.space_before=Pt(9);pf.space_after=Pt(6);pf.keep_with_next=name!='Caption';pf.alignment=WD_ALIGN_PARAGRAPH.CENTER if name in ['Title','Caption'] else WD_ALIGN_PARAGRAPH.LEFT
    doc.styles['Title'].paragraph_format.space_before=Pt(10.5);doc.styles['Title'].paragraph_format.space_after=Pt(14)
    if logo is not None:
        doc._element.body.insert(0,logo)
    lines=source.read_text(encoding='utf-8').splitlines();i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('# '):
            doc.add_paragraph(line[2:],'Title')
            if source.name=='Manuscript.md':
                for row in ['Dandetikar Mani Kumar*          Aerra Surya Teja          Bhargava Chandra Myskar',
                            'Chandramahanthi Sai Mani Balaji          Katumulla Abhiram Reddy          Penta Anuj']:
                    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.line_spacing=Pt(13)
                    r=p.add_run(row);r.bold=True;r.font.size=Pt(11)
                p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.space_before=Pt(6);p.paragraph_format.line_spacing=Pt(13)
                r=p.add_run('Department of Computer Science and Engineering, Neil Gogte Institute of Technology,\nHyderabad, Telangana 500088, India');r.italic=True;r.font.size=Pt(11)
                p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.space_after=Pt(10)
                r=p.add_run('* Corresponding author’s Email: dandetikarmanikumar@gmail.com');r.font.size=Pt(10.5)
        elif line.startswith('### '):doc.add_paragraph(line[4:],'Heading 2')
        elif line.startswith('## '):doc.add_paragraph(line[3:],'Heading 1')
        elif line.startswith('!['):
            name=re.search(r'\((.+)\)',line).group(1);p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.line_spacing=1.0;p.paragraph_format.keep_with_next=True;p.add_run().add_picture(str(OUT/name),width=Cm(16.6))
        elif line.startswith('Table ') or line.startswith('Figure '):
            p=doc.add_paragraph(line,'Caption');p.paragraph_format.keep_with_next=line.startswith('Table ')
        elif line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',c) for c in row):rows.append(row)
                i+=1
            table(doc,rows);continue
        else:
            p=doc.add_paragraph();rich(p,line)
            if line.startswith('**Abstract:') or line.startswith('**Keywords:'):
                p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.line_spacing=Pt(11.5)
                for r in p.runs:r.font.size=Pt(10)
                border(p,'top' if line.startswith('**Abstract:') else 'bottom')
                p.paragraph_format.space_after=Pt(6)
            elif re.match(r'^\[\d+\]',line):
                p.paragraph_format.left_indent=Cm(.7);p.paragraph_format.first_line_indent=Cm(-.7);p.paragraph_format.space_after=Pt(5);p.paragraph_format.keep_together=True
                venues=['Smart Health','Nutrition','PLOS Digital Health','Advances in Neural Information Processing Systems','International Journal of Intelligent Engineering and Systems','Conference on Fairness, Accountability, and Transparency','39th International Conference on Machine Learning','BMJ','Nature Medicine','International Journal of Epidemiology','Current Developments in Nutrition','Dietary Guidelines for Indians']
                # Style journal/book names and et al. without altering bibliographic text.
                p.clear(); pattern='('+'|'.join(re.escape(v) for v in sorted(venues,key=len,reverse=True))+r'|et al\.)'
                for part in re.split(pattern,line):
                    r=p.add_run(part);r.italic=part in venues or part=='et al.'
            elif re.search(r'\(\d\)$',line):
                p.paragraph_format.left_indent=Cm(.5);p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.space_before=p.paragraph_format.space_after=Pt(7)
        i+=1
    doc.core_properties.author='';doc.core_properties.last_modified_by='';doc.core_properties.title='BioNutriGPT';doc.save(dest)

if __name__=='__main__':
    figures();build(OUT/'Manuscript.md',OUT/'BioNutriGPT_IJIES.docx');build(OUT/'Supplementary_Methods.md',OUT/'Supplementary_Methods.docx')
    print('IJIES manuscript and supplementary methods built')
