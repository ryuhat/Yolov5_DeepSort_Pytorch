import gradio as gr

def display_pdf(txt_path):
    with open(txt_path, 'r') as file:
        pdf_path = file.read().strip()
    return pdf_path

inputs = gr.inputs.Textbox(lines=1, label="Text File Path")
output = gr.outputs.File(label="PDF Display")

title = "PDF Viewer"
description = "Enter the path to a text file containing the PDF file path."

gr.Interface(fn=display_pdf, inputs=inputs, outputs=output, title=title, description=description).launch()
