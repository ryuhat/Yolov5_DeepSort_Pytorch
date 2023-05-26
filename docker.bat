docker build --tag yolov8-speed .
docker run -it -d --gpus all --name yolov8-speed -p 8502:8502 yolov8-speed