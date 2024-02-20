from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # build a new model from scratch

results = model.train(data='./data/PlantDoc.v4i.yolov8/data.yaml', epochs=100, batch=8)