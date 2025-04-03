from torchvision.datasets import VOCDetection
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
import torch
from pprint import pprint
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import argparse
from tqdm import tqdm
from torch.optim import Adam
import os
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection import MeanAveragePrecision

class Dataset(VOCDetection):
    def __init__(self, root, year, image_set, download, transform = None):
        super().__init__(root, year, image_set, download, transform)
        self.classes = ['background','aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        #Fix the label to match the input form of model
        target = {"boxes" :[],
                  "labels": []}
        for obj in label["annotation"]["object"]:
            target["boxes"].append([
                int(obj["bndbox"]["xmin"]),
                int(obj["bndbox"]["ymin"]),
                int(obj["bndbox"]["xmax"]),
                int(obj["bndbox"]["ymax"])
            ])
            target["labels"].append(self.classes.index(obj["name"]))
        target["boxes"] = torch.FloatTensor(target["boxes"])
        target["labels"] = torch.LongTensor(target["labels"])

        return image, target

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", "-d", type = str, default = "./datasets", help = "Path to the dataset")
    parser.add_argument("--num-epochs", "-n", type = int, default = 100, help = "Number of epochs")
    parser.add_argument("--batch-size", "-b", type = int, default = 4, help = "Number of images in a batch")
    parser.add_argument("--lr", "-l", type = float, default = 1e-4, help = "Learning rate of the optimizer")
    parser.add_argument("--checkpoint-path", "-c", type = str, default = "./faster_rcnn/checkpoints", help = "Path to save the models")
    parser.add_argument("--tensorboard-path", "-t", type = str, default = "./faster_rcnn/tensorboard", help = "Path to the tensorboard")
    parser.add_argument("--resume-training", "-r", type = bool, default = True, help = "Continue training from previous model or not")
    parser.add_argument("--patience", "-p", type = int, default = 10, help = "Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--min-delta", "-md", type = float, default = 0.001, help = "Minimum change in MAP to qualify as an improvement")

    return parser.parse_args()

def train(args):
    #hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path, exist_ok=True)
    if not os.path.exists(args.tensorboard_path):
        os.makedirs(args.tensorboard_path, exist_ok=True)
    writer = SummaryWriter(args.tensorboard_path)

    #preprocess the data 
    train_set = Dataset(root = os.path.join(args.dataset, "VOC_train"),
                        year = "2012",
                        image_set = "train",
                        download = False,
                        transform = ToTensor())
    train_loader = DataLoader(dataset = train_set,
                              batch_size = args.batch_size,
                              shuffle = True,
                              drop_last = True,
                              num_workers = 4,
                              collate_fn = lambda batch: zip(*batch))
    test_set = Dataset(root = os.path.join(args.dataset, "VOC_test"),
                       year = "2012",
                       image_set = "val",
                       download = False,
                       transform = ToTensor())
    test_loader = DataLoader(dataset = test_set,
                             batch_size = args.batch_size,
                             shuffle = False,
                             drop_last = False,
                             num_workers = 4,
                             collate_fn = lambda batch: zip(*batch))

    #model 
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
    #modify the last layer of the model
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = len(train_set.classes)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    
    #optimizer
    optimizer = Adam(params = model.parameters(), lr = args.lr)

    # Early stopping variables
    patience = args.patience
    min_delta = args.min_delta
    counter = 0
    best_MAP = 0
    start_epoch = 0

    if args.resume_training:
        try:
            saved_data = torch.load(os.path.join(args.checkpoint_path, "last.pt"))
            model.load_state_dict(saved_data["model"])
            optimizer.load_state_dict(saved_data["optimizer"])
            start_epoch = saved_data["epoch"]
            best_MAP = saved_data["best_MAP"]
            counter = saved_data["counter"]
            print(f"Resuming from epoch {start_epoch} with best MAP: {best_MAP:.4f}")
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch")
    
    #training
    for epoch_id in range(start_epoch, args.num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, colour = "green")
        model.train()
        
        #training phase
        for batch_id, (images, targets) in enumerate(progress_bar):
            #Send data to GPU
            images = [image.to(device) for image in images]
            targets = [{
                "boxes": target["boxes"].to(device),
                "labels": target["labels"].to(device)
            } for target in targets]
            
            #Calculate loss
            losses = model(images, targets)
            loss = losses["loss_classifier"] + losses["loss_box_reg"] + losses["loss_objectness"] + losses["loss_rpn_box_reg"]

            total_loss += loss.item()
            avg_loss = total_loss / (batch_id + 1)
            progress_bar.set_description(f"Epoch: {epoch_id + 1}/{args.num_epochs}, Avg_loss: {avg_loss: .4f}, Device: {device}")
            writer.add_scalar("Train/Loss", avg_loss, epoch_id * len(train_loader) + batch_id)

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #validation phase
        progress_bar = tqdm(test_loader, colour = "yellow")
        MAP_metric = MeanAveragePrecision()
        model.eval()

        with torch.no_grad():
            for batch_id, (images, targets) in enumerate(progress_bar):
                #Send data to GPU
                images = [image.to(device) for image in images]
                targets = [{
                    "boxes": target["boxes"].to(device),
                    "labels": target["labels"].to(device)
                } for target in targets]
                
                #forward pass
                predictions = model(images)

                #Calculate mean avg precision
                MAP_metric.update(predictions, targets)
        
        results = MAP_metric.compute()
        MAP_value = results["map_50"]
        current_MAP = MAP_value.item()
        print(f"MAP: {current_MAP:.4f}")
        writer.add_scalar("Validation/MAP", current_MAP, epoch_id)
        
        # Early stopping logic
        if current_MAP > best_MAP + min_delta:
            print(f"MAP improved from {best_MAP:.4f} to {current_MAP:.4f}")
            best_MAP = current_MAP
            counter = 0
            
            # Save best model
            best_model_path = os.path.join(args.checkpoint_path, "best.pt")
            saved_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_id + 1,
                "best_MAP": best_MAP,
                "classes": train_set.classes,
                "counter": counter
            }
            torch.save(saved_data, best_model_path)
        else:
            counter += 1
            print(f"MAP did not improve. Counter: {counter}/{patience}")
            
            if counter >= patience:
                print(f"Early stopping triggered after {epoch_id + 1} epochs")
                break
        
        # Save checkpoint
        last_model_path = os.path.join(args.checkpoint_path, "last.pt")
        saved_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch_id + 1,
            "best_MAP": best_MAP,
            "classes": train_set.classes,
            "counter": counter
        }
        torch.save(saved_data, last_model_path)
        
        # Log early stopping status
        writer.add_scalar("EarlyStopping/Counter", counter, epoch_id)

    print(f"Training completed. Best MAP: {best_MAP:.4f}")
    writer.close()

if __name__ == "__main__":
    args = get_args()
    train(args)