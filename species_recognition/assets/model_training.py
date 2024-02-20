from ultralytics import YOLO

model = YOLO('yolov8m-cls.pt')

result = model.train(data = './data/plantnet_300K/plantnet_300K/images', epochs = 10, batch = 128)