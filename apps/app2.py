import gradio as gr
import subprocess

def generate_video(input_video_path):
    command = ["python", "run.py", "--source", input_video_path, "--yolo-weights", "./weights/best.py"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    output = ""
    for line in process.stdout:
        output += line
    process.wait()
    return output

def video_generator():
    input_video = gr.inputs.Textbox(label="Input Video Path (*.mp4)", default="./videos/733.mp4")
    output_video = gr.outputs.Video(type="mp4", label="Output Video")
    terminal_output = gr.outputs.Textbox(label="Terminal Output")

    def generate_and_display_video(input_video_path):
        output_video_path = generate_video(input_video_path)
        return output_video_path, "Terminal output:\n" + output_video_path

    return gr.Interface(fn=generate_and_display_video, inputs=input_video, outputs=[output_video, terminal_output])

if __name__ == "__main__":
    video_gen = video_generator()
    video_gen.launch()
