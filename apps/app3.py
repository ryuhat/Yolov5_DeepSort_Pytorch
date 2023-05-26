import gradio as gr
from pathlib import Path
import subprocess

def run_speed_command(video_path, weights_path):
    video_path = Path(video_path)
    command = f"python run.py --source {video_path}"
    # command = f"python run.py --source {video_path} --yolo-weights {weights_path}"
    print("Executing command:", command)
    subprocess_result = subprocess.run(command, shell=True, check=True, capture_output=True)
    print("Command output:", subprocess_result)
    output_video_path = f"./runs/track/exp65/{video_path.stem}.mp4"
    with open(output_video_path, "rb") as f:
        video_bytes = f.read()
    print("Returning video bytes:", video_bytes[:10], "...")
    return (video_bytes, "mp4")


input_video = gr.inputs.Textbox(label="Input Video Path (*.mp4)", default="./videos/733.mp4")
input_weights = gr.inputs.Textbox(label="Path to weights file")

output_video = gr.outputs.Video(type="bytes", label="Output Video")

title = "Speed Prediction"
description = "Predict the speed of an object in a video using a PyTorch model."

examples = [["./videos/733.mp4", "./weights/best.py"]]

gradio_app = gr.Interface(run_speed_command,
                          inputs=[input_video, input_weights],
                          outputs=output_video,
                          title=title,
                          description=description,
                          examples=examples)

if __name__ == "__main__":
    gradio_app.launch()
