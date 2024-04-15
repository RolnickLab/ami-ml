"""
Author: Aditya Jain
Date last modified: November 6, 2023
About: Test the yolo annotations visually on a few raw images
"""

import os
import cv2
import random
import json

# User-input variables
yolo_data_dir = "/home/mila/a/aditya.jain/scratch/cvpr2024_data/ami-traps-dataset"
output_dir = "/home/mila/a/aditya.jain/scratch/cvpr2024_data/eg-visual-annotations"


def test_annotations(data_dir: str, output_dir: str, n: int=10):
    """Main function to draw the annotations on raw images"""

    image_dir = os.path.join(data_dir, "images")

    # Draw annotations over n random images
    for _ in range(n):
        image_filename = random.choice(os.listdir(image_dir))
        label_filename = os.path.splitext(image_filename)[0] + ".txt"
        label_file = open(os.path.join(data_dir, "labels", label_filename), "r")

        # Get image dimensions
        image = cv2.imread(os.path.join(image_dir, image_filename))
        img_height, img_width = image.shape[0], image.shape[1]

        # Iterate over each annotation separately
        for line in label_file:
            label_id, x, y, w, h = (
                int(line.split()[0]),
                float(line.split()[1]),
                float(line.split()[2]),
                float(line.split()[3]),
                float(line.split()[4]),
            )

            # Get class name for display
            class_list = json.load(open(os.path.join(data_dir, "notes.json")))[
                "categories"
            ]
            for class_entry in class_list:
                if class_entry["id"] == label_id:
                    label_name = class_entry["name"]
                    break

            # Process coordinates and lengths for use in CV2 draw function
            crop_width, crop_height = int(w * img_width), int(h * img_height)
            x_start = int((x - w / 2) * img_width)
            y_start = int((y - h / 2) * img_height)

            # Draw bounding box and corresponding label
            image = cv2.rectangle(
                image,
                (x_start, y_start),
                (x_start + crop_width, y_start + crop_height),
                (0, 0, 1),
                2,
            )
            cv2.putText(
                image,
                label_name,
                (x_start, y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 1),
                2,
            )

        # Write image to disk
        cv2.imwrite(os.path.join(output_dir, image_filename), image)


if __name__ == "__main__":
    test_annotations(yolo_data_dir, output_dir)
