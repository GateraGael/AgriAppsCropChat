from ultralytics import YOLO

model = YOLO('yolov8m-cls.pt')

result = model.train(data = './data/RicePlantDeficiencyImgs', epochs = 10, batch = 128)