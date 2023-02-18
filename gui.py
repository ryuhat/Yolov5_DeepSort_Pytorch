import gradio as gr

def predict_video(model, source, show):
    # Put your predict.py script code here
    # ...
    # return predicted video
    iface = gr.Interface(
        predict_video,
        [
            gr.inputs.Textbox(label="Model"),
            gr.inputs.Textbox(label="Source"),
            gr.inputs.Checkbox(label="Show", default=True)
        ],
        gr.outputs.Video(label="Predicted Video")
    )

if __name__ == '__main__':
    iface.launch()