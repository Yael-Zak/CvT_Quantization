import os
from transformers import CvtConfig, CvtModel
from transformers import AutoImageProcessor, CvtForImageClassification
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import argparse

## load arguments needed
parser = argparse.ArgumentParser()
parser.add_argument('--image_directory', default=r'C:\Users\matan\Desktop\CvT\ILSVRC2012_img_test_v10102019\test',
                    type=str, help='path to directory of ImageNet test')
args = parser.parse_args()

# Create an instance of ImageFolder class from torchvision using the image directory
dataset = ImageFolder(root=args.image_directory)

# Create a DataLoader to load the images in batches
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Initialize the Cvt model and image processor
# configuration = CvtConfig() - probably not needed
# model = CvtModel(configuration) - probably not needed

image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
model = CvtForImageClassification.from_pretrained("microsoft/cvt-13")

# Define a list to store the predictions
predictions = []

# Perform inference on the dataset
model.eval()
with torch.no_grad():
    for images, _ in dataloader:
        inputs = image_processor(images, return_tensors="pt")
        logits = model(**inputs).logits
        predicted_labels = torch.argsort(logits, dim=1, descending=True)[:, :5]  # Get top 5 predicted labels
        predictions.extend(predicted_labels.tolist())

# Create a list of image filenames sorted alphabetically
image_filenames = sorted(os.listdir(args.image_directory))

# Write the predictions to a text file in the required format
output_file = "predictions.txt"
with open(output_file, "w") as file:
    for prediction, image_filename in zip(predictions, image_filenames):
        labels = " ".join(str(label) for label in prediction)
        line = f"{labels}\n"
        file.write(line)
