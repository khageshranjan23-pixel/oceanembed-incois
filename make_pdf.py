from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import re

with open("Architecture_Modification_Guide.md", "r") as f:
    text = f.read()

# Very basic markdown to reportlab conversion
doc = SimpleDocTemplate("OceanEmbed_Architecture_Guide.pdf", pagesize=letter)
styles = getSampleStyleSheet()
normal = styles["Normal"]
h1 = styles["Heading1"]
h2 = styles["Heading2"]
bullet = ParagraphStyle(name='Bullet', parent=styles['Normal'], leftIndent=20, spaceAfter=5)

story = []

for line in text.split('\n'):
    line = line.strip()
    if not line or line == '---':
        story.append(Spacer(1, 0.1*inch))
        continue
    
    # Handle bold
    line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
    # Handle backticks
    line = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', line)

    if line.startswith('# '):
        story.append(Paragraph(line[2:], h1))
    elif line.startswith('## '):
        story.append(Paragraph(line[3:], h2))
    elif line.startswith('* '):
        story.append(Paragraph(line, bullet))
    else:
        story.append(Paragraph(line, normal))

doc.build(story)
