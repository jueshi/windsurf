import PyPDF2

def pdf_to_text(pdf_path, txt_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        with open(txt_path, 'w', encoding='utf-8') as text_file:
            for page in reader.pages:
                text_file.write(page.extract_text())

pdf_to_text(r'C:\Users\JueShi\OneDrive - Astera Labs, Inc\Documents\Taurus\Register Maps\Taurus3\dwc_112g_ethernet_lrm_phy_tsmc5ff_x4ns_reference.pdf', 
'dwc_112g_ethernet_lrm_phy_tsmc5ff_x4ns_reference.txt'
)