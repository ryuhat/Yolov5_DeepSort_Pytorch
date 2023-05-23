import gradio as gr
from pathlib import Path
import subprocess

def get_latest_exp_number():
    track_folder = "./runs/track"
    track_path = Path(track_folder)
    exp_folders = [f for f in track_path.iterdir() if f.is_dir() and f.name.startswith("exp") and f.name[3:].isdigit()]
    if not exp_folders:
        return 0
    latest_exp_folder = max(exp_folders, key=lambda f: int(f.name[3:]))
    latest_exp_number = int(latest_exp_folder.name[3:]) + 1
    return latest_exp_number

def run_speed_command(input_type, video_path, video_upload, weights_path):
    if input_type == "Video Path":
        video_path = Path(video_path)
    else:
        video_path = Path(video_upload.name)
        video_upload.save(video_path)
    latest_exp_number = get_latest_exp_number()
    output_video_path = f"./runs/track/exp{latest_exp_number}/{video_path.stem}.mp4"
    output_image_path = f"./runs/track/exp{latest_exp_number}/tracks/velocity.jpg"
    command = f"python run.py --source {video_path} --yolo-weights {weights_path} --save-vid --save-txt"
    print("Executing command:", command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ""
    while True:
        line = process.stdout.readline().decode().strip()
        if line == "" and process.poll() is not None:
            break
        output += line + "\n"
        print(line)
    process.wait()
    with open(output_video_path, "rb") as f:
        video_data = f.read()
    return video_data, output_image_path, output

input_type = gr.inputs.Radio(["Video Path", "Upload"], label="Input", default="Video Path")
input_video_path = gr.inputs.Textbox(label="Video Path (*.mp4)", default="./videos/733.mp4", placeholder="Enter video path")
input_video_upload = gr.inputs.Video(type="mp4", label="Upload Video", source="upload")
input_weights = gr.inputs.Textbox(label="Path to weights file", default="./weights/best.pt")

output_video = gr.outputs.Video(type="mp4", label="Output Video")
output_image = gr.outputs.Image(type="pil", label="Velocity Image")
output_text = gr.outputs.Textbox(label="Process Output")

title = "Speed Prediction"
description = "Predict the speed of an object in a video using a PyTorch model."

examples = [["./videos/733.mp4", "./weights/best.pt"]]

gradio_app = gr.Interface(run_speed_command,
                          inputs=[input_type, input_video_path, input_video_upload, input_weights],
                          outputs=[output_video, output_image, output_text],
                          title=title,
                          description=description,
                          examples=examples)

if __name__ == "__main__":
    gradio_app.launch()
