from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
import argparse
import torch
from pprint import pprint
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torchvision.transforms import ToTensor
import cv2

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", "-i", type = str, default = "./faster_rcnn/input.jpg", help = "Path to the input image")
    parser.add_argument("--model-path", "-m", type = str, default = "./faster_rcnn/checkpoints/best.pt", help = "Path to the models")
    parser.add_argument("--threshold", "-t", type = float, default = 0.5, help = "Detection threshold")
    parser.add_argument("--output-path", "-o", type = str, default = "./faster_rcnn/output.jpg", help = "Path to the result image")

    return parser.parse_args()

def inference(args):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    saved_data = torch.load(args.model_path)
    classes = saved_data["classes"]
    
    #preprocess the image
    img = Image.open(args.input_path)
    img = ToTensor()(img)
    img = img[None, :, :, :]

    #Load the model
    model = fasterrcnn_mobilenet_v3_large_320_fpn()
    #modify the last layer of the model
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes))
    model.to(device)
    model.load_state_dict(saved_data["model"])
    
    #Inference
    model.eval()
    with torch.no_grad():
        #forward pass
        img = img.to(device)
        predictions = model(img)
        
        boxes = predictions[0]["boxes"].tolist()
        labels = predictions[0]["labels"].tolist()
        scores = predictions[0]["scores"].tolist()

        #Filter the bounding boxes
        valid_boxes = []
        valid_labels = []
        valid_scores = []
        for box, label, score in zip(boxes, labels, scores):
            if score >= args.threshold:
                valid_boxes.append(box)
                valid_labels.append(label)
                valid_scores.append(score)

    #Show the result
    img = cv2.imread(args.input_path)
    for box, label, score in zip(valid_boxes, valid_labels, valid_scores):
        x1, y1, x2, y2 = [int(box[i]) for i in range(4)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color = (0, 200, 0), thickness = 2)
        cv2.putText(img, f"{classes[label]}: {score * 100: .2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
    
    cv2.imwrite(args.output_path, img)
    cv2.imshow("Output", img)
    cv2.waitKey(0)
    
if __name__ == "__main__":
    args = get_args()
    inference(args)