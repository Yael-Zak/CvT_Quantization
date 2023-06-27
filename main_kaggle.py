# this is what you need to import to kaggle to run in notebookes
# each line of # will seperate between code blocks, just copy them into blocks

####################################### block 1 - imports #######################################
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor
from torchvision import transforms
from torchvision.datasets import ImageNet, ImageFolder
from transformers import AutoImageProcessor, CvtForImageClassification, CvtModel
import os
import time
####################################### block 2 - functions #######################################


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels


def quantize_model(model, layer_type_to_quantize='convolution_projection', save=False, compute=False):
    """
    Args:
        model: model to quantize
        layer_type_to_quantize: type of layer to quantize from the state_dict
        save: should you save the state dict (bool)
        compute: should you compute Mb saved in the process (not really says anything)

    Returns:

    """
    base_model_state_dict = model.state_dict()
    new_state_dict = {}
    layers_quantized = {}
    dict_for_size = []
    for layer_name, layer_params in base_model_state_dict.items():
        if layer_type_to_quantize in layer_name:
            new_state_dict[layer_name] = layer_params.to(torch.int8)
            layers_quantized[layer_name] = new_state_dict[layer_name].numel()
        else:
            new_state_dict[layer_name] = layer_params
        dict_for_size.append(new_state_dict[layer_name].numel())

    if save:
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        quantized_model_path = os.path.join(model_dir, "quantized_model.pth")
        torch.save(new_state_dict, quantized_model_path)
        print("Quantized model state dict saved at:", quantized_model_path)

    if compute:
        original_model_size = sum(dict_for_size)
        num_of_Mb = 2*original_model_size/(1024**2)
        print("original size in Mb: " + str(num_of_Mb))

        quantized_num_parameters = sum(layers_quantized.values())
        num_of_Mb = 4*quantized_num_parameters/(1024**2)
        print("Number of Mb saved: " + str(num_of_Mb))

    return new_state_dict


def calculate_accuracy(predictions_file, labels_file):
    with open(predictions_file, 'r') as pred_file, open(labels_file, 'r') as labels_file:
        predictions = [int(line.strip()) for line in pred_file.readlines()]
        labels = [int(line.strip()) for line in labels_file.readlines()]

    total_samples = len(labels)
    correct_predictions = sum([1 for pred, label in zip(predictions, labels) if pred == label])

    accuracy = (correct_predictions / total_samples) * 100
    return accuracy


def craete_predictions_file(output_file_path, predictions):
    with open(output_file_path, "w") as file:
        for prediction in predictions:
            labels = " ".join(str(label) for label in prediction)
            line = f"{labels}\n"
            file.write(line)


############################# block 3 - arguments #############################
image_directory = '0'
to_quantize = True
output_path = '1'
############################# block 4 - create a prediction file ############################
# load the label file

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
if to_quantize:
    quantized_state_dict = quantize_model(model, compute=True)
    # load new state model into model
    model.load_state_dict(quantized_state_dict)

# Move model and data to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Device used is: " + str(device))

# Perform inference on the dataset
model.eval()
start_time = time.time() # start timer
with torch.no_grad():
    for index, images in enumerate(dataloader):
        images = images.to(device)
        inputs = image_processor(images, return_tensors="pt")
        logits = model(**inputs).logits
        predicted_labels = torch.argsort(logits, dim=1, descending=True)[:, :1]  # Get top 1 predicted labels (originally 5)

        if index % 50 == 0:
            print("Processed {} images out of 50,000".format(index*32))


end_time = time.time() # end timer
# Calculate the elapsed time in minutes
elapsed_time_minutes = (end_time - start_time) / 60

# Print the elapsed time
print(f"Elapsed Time: {elapsed_time_minutes:.4f} minutes")
craete_predictions_file(output_path, predicted_labels)
print("Finished :)")
############################# block 5 - check accuracy ############################

accuracy_score = calculate_accuracy("predictions.txt", "labels.txt")
print(f"Accuracy Score: {accuracy_score:.2f}%")
