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
    latest_exp_number = int(latest_exp_folder.name[3:])+1
    return latest_exp_number

def run_speed_command(video_path, weights_path):
    video_path = Path(video_path)
    latest_exp_number = get_latest_exp_number()
    output_video_path = f"./runs/track/exp{latest_exp_number}/{video_path.stem}.mp4"
    output_pdf_path = f"./runs/track/exp{latest_exp_number}/tracks/velocity.pdf"
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
    return output_video_path, output_image_path, output_pdf_path, output  # Updated return statement

input_video = gr.inputs.Textbox(label="Input Video Path (*.mp4)", default="./videos/733.mp4")
input_weights = gr.inputs.Textbox(label="Path to weights file", default="./weights/best.pt")

output_video = gr.outputs.Video(type="mp4", label="Speed Estimated Video")
output_image = gr.outputs.Image(type="pil", label="Velocity Plot") 
output_pdf = gr.outputs.File(label="Velocity Plot PDF") 
output_text = gr.outputs.Textbox(label="Process Output")

title = "Speed Prediction"
description = "Predict the speed of an object in a video using a PyTorch model."

examples = [["./videos/733.mp4", "./weights/best.pt"]]

gradio_app = gr.Interface(run_speed_command,
                          inputs=[input_video, input_weights],
                          outputs=[output_video, output_image, output_pdf, output_text],
                          title=title,
                          description=description,
                          examples=examples)

if __name__ == "__main__":
    # gradio_app.launch(share=True)
    gradio_app.launch()
