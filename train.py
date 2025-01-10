from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-seg.pt")
model = YOLO("yolo11n.pt")

train_results = model.train(
    data="switch-det.yaml",  # path to dataset YAML
    epochs=60,  # number of training epochs
    imgsz=640,  # training images size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch=16,
    workers=0,
)

# # Evaluate model performance on the validation set
metrics = model.val()
