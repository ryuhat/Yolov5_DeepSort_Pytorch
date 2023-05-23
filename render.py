import gradio as gr

def show_video(text_path):
    video_path = f"./videos/{text_path}.mp4"
    return video_path

iface = gr.Interface(
    fn=show_video,
    inputs="text",
    outputs=gr.outputs.Video(type="mp4"),
    title="Video Viewer",
    description="Display a video based on a text path.",
    examples=[["733"]],
)

iface.launch()

