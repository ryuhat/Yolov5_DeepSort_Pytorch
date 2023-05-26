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

def run_speed_command(video_path, weights_path, tracking_method, reid_weights, iou_thres):
    video_path = Path(video_path)
    latest_exp_number = get_latest_exp_number()
    output_video_path = f"./runs/track/exp{latest_exp_number}/{video_path.stem}.mp4"
    output_pdf_path = f"./runs/track/exp{latest_exp_number}/tracks/velocity.pdf"
    output_image_path = f"./runs/track/exp{latest_exp_number}/tracks/velocity.jpg"
    command = f"python run.py --source {video_path} --yolo-weights {weights_path} --tracking-method {tracking_method} --reid-weights {reid_weights} --iou-thres {iou_thres} --save-vid --save-txt"
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

# input_video = gr.inputs.Textbox(label="Input Video Path (*.mp4)", default="./videos/733.mp4")
input_video = gr.inputs.Dropdown(
    choices=["./videos/733.mp4", "./videos/20141009/Dat_034.m4v"],
    label="Input Video Path (*.mp4, *.m4v, ...)",
    default="./videos/733.mp4"
)

input_weights = gr.inputs.Textbox(label="Path to weights file", default="./weights/best.pt")

input_weights = gr.inputs.Dropdown(
    choices=["./weights/best.pt", "./weights/best-seg.pt"],
    label="Path to weights file",
    default="./weights/best-seg.pt"
)

# input_tracking_method = gr.inputs.Textbox(label="Tracking Method", default="bytetrack")
input_tracking_method = gr.inputs.Dropdown(
    choices=["bytetrack", "strongsort"],
    label="Tracking Method",
    default="bytetrack"
)

input_reid_weights = gr.inputs.Dropdown(
    choices=['osnet_x0_25_msmt17.pt', 'osnet_x0_25_msmt17.pt'],
    label="ReID Weight",
    default='osnet_x0_25_msmt17.pt'
)

input_iou_thres = gr.inputs.Slider(label="IOU Threshold", minimum=0.0, maximum=1.0, default=0.5, step=0.1)

output_video = gr.outputs.Video(type="mp4", label="Velocity Estimated Video")
output_image = gr.outputs.Image(type="pil", label="Velocity Plot (per frame)") 
output_pdf = gr.outputs.File(label="Velocity Plot PDF") 
output_text = gr.outputs.Textbox(label="Process Output")

title = "Velocity Estimation"
description = "Estimate the velocity of an object in a video using a PyTorch model. This model utilizes object detection with YOLOv8 and object tracking techniques."

examples = [["./videos/733.mp4", "./weights/best.pt"]]

gradio_app = gr.Interface(run_speed_command,
                          inputs=[input_video, input_weights, input_tracking_method, input_reid_weights, input_iou_thres],
                          outputs=[output_video, output_image, output_pdf, output_text],
                          title=title,
                          description=description,
                          examples=examples)

if __name__ == "__main__":
    gradio_app.launch()
