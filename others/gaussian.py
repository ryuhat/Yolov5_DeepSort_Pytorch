import cv2
import gradio as gr

def gaussian_filter(input_video, kernel_size):
    # Load video
    video = cv2.VideoCapture(input_video.name)

    # Define codec and video writer
    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter("output.mp4", fourcc, fps, frame_size)

    # Process each frame in video
    while True:
        ret, frame = video.read()
        if not ret:
            break
        filtered_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        out.write(filtered_frame)

    # Release resources
    video.release()
    out.release()

    return "output.mp4"

inputs = [gr.inputs.File(label="Select a video file"), 
          gr.inputs.Dropdown(label="Kernel Size", choices=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21], default=3)]

outputs = gr.outputs.File(label="Output video")

gr.Interface(fn=gaussian_filter, inputs=inputs, outputs=outputs, title="Gaussian Filter on Video").launch()

