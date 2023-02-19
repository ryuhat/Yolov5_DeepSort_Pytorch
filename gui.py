import argparse
import gradio as gr
import subprocess

def detect(input_video, weights_path="./yolov8/runs/detect/train15/weights/best.pt"):
    output_video = "output.mp4"
    command = f"python speed.py --source {input_video.name} --yolo-weights {weights_path} --save-vid {output_video} --save-txt"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output_video

def main(args):
    input_video = gr.inputs.Video(type="file", label="Input video")
    weights_path = gr.inputs.Textbox(label="Weights path (YOLOv8)", default="./yolov8/runs/detect/train15/weights/best.pt")
    output_video = gr.outputs.Video(label="Output video")

    if args.share:
        iface = gr.Interface(detect, [input_video, weights_path], output_video, title="YOLOv8 StrongSORT Velocity Estimation").launch(share=True)
    else:
        iface = gr.Interface(detect, [input_video, weights_path], output_video, title="YOLOv8 StrongSORT Velocity Estimation")
    iface.launch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Enable sharing of the interface")
    args = parser.parse_args()
    main(args)





# import gradio as gr
# import subprocess

# def detect(input_video, weights_path):
#     output_video = "output.mp4"
#     command = f"python speed.py --source {input_video.name} --yolo-weights {weights_path} --save-vid {output_video} --save-txt"
#     process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
#     output, error = process.communicate()
#     return output_video

# input_video = gr.inputs.Video(type="mp4", label="Input video")
# weights_path = gr.inputs.Textbox(label="Weights path")
# output_video = gr.outputs.Video(label="Output video")

# gr.Interface(detect, [input_video, weights_path], output_video, title="YOLOv8 StrongSORT Velocity Estimation").launch()


# import gradio as gr
# import cv2

# def process_video(input_video):
#     cap = cv2.VideoCapture(input_video.name)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     output_video = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # Process the frame here
#         processed_frame = frame
#         output_video.write(processed_frame)

#     cap.release()
#     output_video.release()

#     return "output.mp4"

# inputs = [gr.inputs.Video(type="mp4")]
# outputs = gr.outputs.Video(type="mp4")

# gr.Interface(process_video, inputs, outputs).launch()
