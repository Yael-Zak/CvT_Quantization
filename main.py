import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor
from torchvision import transforms
from torchvision.datasets import ImageNet, ImageFolder
from transformers import AutoImageProcessor, CvtForImageClassification, CvtModel
import argparse
from utils import collate_fn, get_deepest_folder
import os
from Quantization import quantize_model

# Load arguments needed - in kaggle this doesn't work, write arguments like in c
parser = argparse.ArgumentParser()
parser.add_argument('--image_directory', default=r'C:\Users\matan\Desktop\CvT_huggingface\Validation', type=str, help='path to directory of ImageNet test')
parser.add_argument('--Output_path', default=r'./outputs', type=str, help='path to directory of ImageNet test')
parser.add_argument('--name', default=r'baseline_predictions.txt', type=str, help='path to directory of ImageNet test')
parser.add_argument('--small_dataset', default=r'C:\Users\matan\Desktop\CvT_huggingface\small_test', type=str, help='path to directory of ImageNet test')
parser.add_argument('--small_dataset_flag', default=False, type=str, help='run om small test for debug')
parser.add_argument('--state_dict', default='./models/quantized_model', type=str, help='state dict to load and run the model on')
parser.add_argument('--to_quantize', default=True, type=str, help='state dict to load and run the model on')

args = parser.parse_args()

# creating path for output if none exist
if not os.path.exists(args.Output_path):
    os.makedirs(args.Output_path)

if args.small_dataset_flag:
    image_directory = args.small_dataset
else:
    image_directory = args.image_directory

# Define transformations for the images
transform = transforms.Compose([
    Resize((224, 224)),
    ToTensor()
])

# Create an instance of ImageNet dataset
dataset = ImageFolder(root=image_directory, transform=transform)

# Create a DataLoader to load the images in batches
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Initialize the Cvt model and image processor
image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
model = CvtForImageClassification.from_pretrained("microsoft/cvt-13", output_hidden_states=True, return_dict=True)
if args.to_quantize:
    quantized_state_dict = quantize_model(model, compute=True)

# load new state model into model
model.load_state_dict(quantized_state_dict)

# Move model and data to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a list to store the predictions
predictions = []

# Perform inference on the dataset
model.eval()
with torch.no_grad():
    for index, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        inputs = image_processor(images, return_tensors="pt")
        logits = model(**inputs).logits
        predicted_labels = torch.argsort(logits, dim=1, descending=True)[:, :5]  # Get top 5 predicted labels
        predictions.extend(predicted_labels.tolist())

        if index % 50 == 0:
            print("Processed {} images out of 100,000".format(index))

# Create a list of image filenames sorted alphabetically
deepest_folder = get_deepest_folder(image_directory)
image_filenames = sorted(os.listdir(deepest_folder)) # don't need that in the end

# Write the predictions to a text file in the required format
output_file = os.path.join(args.Output_path, args.name)
with open(output_file, "w") as file:
    for prediction, image_filename in zip(predictions, image_filenames):
        labels = " ".join(str(label) for label in prediction)
        line = f"{labels}\n"
        file.write(line)

print("Finished script! Check the predictions.txt file for the results.")
