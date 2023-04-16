import gradio as gr
import subprocess

# def run_speed_command(video_path, weights_path):
#     # command = ["python", "speed.py", video_path, weights_path]
#     command = f"python speed.py --source {video_path} --yolo-weights {weights_path} --save-vid --save-txt"
#     subprocess.run(command, check=True)
#     with open(video_path, "rb") as f:
#         video_bytes = f.read()
#     return video_bytes, "file"

# def run_speed_command(video_path, weights_path):
#     command = f"python speed.py --source {video_path} --yolo-weights {weights_path} --save-vid --save-txt"
#     print("Executing command:", command)
#     subprocess_result = subprocess.run(command, check=True, capture_output=True)
#     print("Command output:", subprocess_result)
#     with open(video_path, "rb") as f:
#         video_bytes = f.read()
#     print("Returning video bytes:", video_bytes[:10], "...")
#     return video_bytes, "file"

def run_speed_command(video_path, weights_path):
    command = f"python speed.py --source {video_path} --yolo-weights {weights_path} --save-vid --save-txt"
    print("Executing command:", command)
    subprocess_result = subprocess.run(command, check=True, capture_output=True)
    print("Command output:", subprocess_result)
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    print("Returning video bytes:", video_bytes[:10], "...")
    return (video_bytes, "mp4")


input_video = gr.inputs.Video(type="mp4")
input_weights = gr.inputs.Textbox(label="Path to weights file")

output_video = gr.outputs.Video(type="file", label="Output Video")

title = "Speed Prediction"
description = "Predict the speed of an object in a video using a PyTorch model."
examples = [["./videos/733.mp4", "./yolov8/runs/detect/train15/weights/best.pt"]]

gradio_app = gr.Interface(run_speed_command, 
                        inputs=[input_video, input_weights], 
                        outputs=output_video, 
                        title=title, 
                        description=description,
                        examples=examples)
gradio_app.launch()

