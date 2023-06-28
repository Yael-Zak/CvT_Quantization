import os
import random
import csv
import shutil


def sample_validation_set(validation_folder, labels_file, output_csv):
    # Read labels from the file
    with open(labels_file, 'r') as file:
        labels = [int(line.strip()) for line in file.readlines()]

    # Get a list of all image names in the validation folder
    image_names = sorted(os.listdir(validation_folder))

    # Sample 10 images for each label
    sampled_images = []
    for label in set(labels):
        indices = [i for i, x in enumerate(labels) if x == label]
        sampled_indices = random.sample(indices, k=10)
        sampled_images.extend([image_names[i] for i in sampled_indices])

    # Create and save the CSV file
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Label'])
        for image in sampled_images:
            label = labels[image_names.index(image)]
            writer.writerow([image, label])


def create_validation_folder(output_folder, csv_file, source_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the CSV file
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            image = row[0]
            label = row[1]

            # Create a folder for the label if it doesn't exist
            label_folder = os.path.join(output_folder, f"{label}")
            os.makedirs(label_folder, exist_ok=True)

            # Copy the image to the label folder
            source_path = os.path.join(source_folder, image)
            destination_path = os.path.join(label_folder, image)
            shutil.copyfile(source_path, destination_path)

            # Optionally, you can also copy the label file to the label folder
            # if it exists and you want to keep track of the labels.
            label_file = f"{label}.txt"
            label_source_path = os.path.join(source_folder, label_file)
            label_destination_path = os.path.join(label_folder, label_file)
            if os.path.isfile(label_source_path):
                shutil.copyfile(label_source_path, label_destination_path)


# for some reason relative paths didn't work - try however you want
validation_path = r'C:\Users\matan\Desktop\CvT_huggingface\ILSVRC2012_img_val'
labels_path = r'C:\Users\matan\Desktop\CvT_huggingface\CvT_Quantization\ILSVRC2012_validation_ground_truth.txt'
output_csv_path = r'C:\Users\matan\Desktop\CvT_huggingface\CvT_Quantization\outputs\validation.csv'
output_path = r'imagenet_val_small'

sample_validation_set(validation_folder=validation_path, labels_file=labels_path, output_csv=output_csv_path)
create_validation_folder(output_folder=output_path, csv_file=output_csv_path, source_folder=validation_path)
