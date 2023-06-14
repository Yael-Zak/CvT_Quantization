import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor
from torchvision import transforms
from torchvision.datasets import ImageNet
from transformers import AutoImageProcessor, CvtForImageClassification
import argparse
from utils import collate_fn

# Load arguments needed
parser = argparse.ArgumentParser()
parser.add_argument('--image_directory', default=r'C:\Users\matan\Desktop\CvT_huggingface\ILSVRC2012_img_test_v10102019\test',
                    type=str, help='path to directory of ImageNet test')
parser.add_argument('--small_image_directory', default=r'C:\Users\matan\Desktop\CvT_huggingface\ILSVRC2012_img_test_v10102019\small_test',
                    type=str, help='path to directory of 30 ImageNet test images to use when debugging')
parser.add_argument('--debug_mode', default=True, type=bool, help='flag for debug - load small dataset instead')
args = parser.parse_args()

if args.debug_mode:
    image_directory = args.small_image_directory
else:
    image_directory = args.image_directory

# Define transformations for the images
transform = transforms.Compose([
    Resize((224, 224)),
    ToTensor()
])

# Create an instance of ImageNet dataset
dataset = ImageNet(root=image_directory, split='val', transform=transform)

# Create a DataLoader to load the images in batches
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Initialize the Cvt model and image processor
image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
model = CvtForImageClassification.from_pretrained("microsoft/cvt-13")

# Move model and data to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a list to store the predictions
predictions = []

# Perform inference on the dataset
model.eval()
with torch.no_grad():
    for images in dataloader:
        images = images.to(device)
        inputs = image_processor(images, return_tensors="pt")
        logits = model(**inputs).logits
        predicted_labels = torch.argsort(logits, dim=1, descending=True)[:, :5]  # Get top 5 predicted labels
        predictions.extend(predicted_labels.tolist())

# Create a list of image filenames sorted alphabetically
image_filenames = sorted(os.listdir(image_directory))

# Write the predictions to a text file in the required format
output_file = "predictions.txt"
with open(output_file, "w") as file:
    for prediction, image_filename in zip(predictions, image_filenames):
        labels = " ".join(str(label) for label in prediction)
        line = f"{labels}\n"
        file.write(line)

print("Finished script! Check the predictions.txt file for the results.")
