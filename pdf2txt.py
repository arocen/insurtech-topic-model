# transfer reference paper from pdf to txt

import pdfplumber
import os
from dotenv import load_dotenv

load_dotenv()
file_address = os.environ.get("pdf_to_extract10")
res_pdf = pdfplumber.open(file_address)

# 提取文本信息
all_text = ""
for page in res_pdf.pages:
    text = page.extract_text()
    all_text = all_text + "\n" + text

# 保存文本内容
txt_path = file_address[:-3] + "txt"
with open(txt_path, 'a', encoding='UTF-8', errors='ignore') as f:
    f.write(all_text)