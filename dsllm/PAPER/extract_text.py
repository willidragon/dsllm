from PyPDF2 import PdfReader

# Extract text from the original large PDF
input_pdf = "research/dsllm/dsllm/PAPER/4049_SensorLLM_Aligning_Large_.pdf"
output_file = "sensorllm_full_text.txt"

reader = PdfReader(input_pdf)
text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Extracted text from {input_pdf} to {output_file}") 