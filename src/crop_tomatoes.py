"""
This script prepares the tomato image dataset for the classification model.

The original Laboro Tomato dataset contains large farm images with multiple
tomatoes in each image. The annotation JSON files tell us where each tomato
is located and what ripeness class it belongs to.

This script:
1. Reads the annotation files.
2. Finds each tomato using its bounding box.
3. Crops the tomato from the original image.
4. Saves the cropped image into one of three folders:
   - green
   - turning
   - ripe

The processed images will later be used to train a CNN model.
"""

import json
import os
from PIL import Image


# Original dataset folder
DATA_FOLDER = "data/laboro_big"

# Folder where cropped tomato images will be saved
OUTPUT_FOLDER = "data/processed"


# Change the original class names to simpler names
class_names = {
    "b_green": "green",
    "b_half_ripened": "turning",
    "b_fully_ripened": "ripe"
}


def crop_tomatoes(split):

    # Paths for images and annotations
    annotation_file = os.path.join(
        DATA_FOLDER,
        "annotations",
        split + ".json"
    )

    image_folder = os.path.join(
        DATA_FOLDER,
        split
    )

    # Open annotation JSON file
    with open(annotation_file, "r") as file:
        data = json.load(file)


    # Connect category IDs with class names
    categories = {}

    for category in data["categories"]:
        category_id = category["id"]
        original_name = category["name"]

        categories[category_id] = class_names[original_name]


    # Connect image IDs with image information
    images = {}

    for image in data["images"]:
        image_id = image["id"]
        images[image_id] = image


    # Count how many tomatoes are saved
    counts = {
        "green": 0,
        "turning": 0,
        "ripe": 0
    }


    # Go through all tomato annotations
    for annotation in data["annotations"]:

        image_id = annotation["image_id"]
        image_info = images[image_id]

        file_name = image_info["file_name"]
        image_path = os.path.join(image_folder, file_name)

        category_id = annotation["category_id"]
        tomato_class = categories[category_id]


        # Get tomato location
        x, y, width, height = annotation["bbox"]

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + width)
        y2 = int(y + height)


        # Open image
        image = Image.open(image_path)

        # Crop tomato
        tomato = image.crop((x1, y1, x2, y2))


        # Create output folder
        save_folder = os.path.join(
            OUTPUT_FOLDER,
            split,
            tomato_class
        )

        os.makedirs(save_folder, exist_ok=True)


        # Create file name
        annotation_id = annotation["id"]

        save_name = (
            str(annotation_id)
            + "_"
            + tomato_class
            + ".jpg"
        )

        save_path = os.path.join(
            save_folder,
            save_name
        )


        # Save cropped tomato image
        tomato.save(save_path)

        counts[tomato_class] += 1


    # Print result
    print(split + " complete")

    print("green:", counts["green"])
    print("turning:", counts["turning"])
    print("ripe:", counts["ripe"])


# Process training images
crop_tomatoes("train")

# Process test images
crop_tomatoes("test")
