"""
Show one example image from each tomato ripeness class.

The three classes are:
- green
- turning
- ripe
"""

import glob
from PIL import Image
import matplotlib.pyplot as plt


classes = ["green", "turning", "ripe"]


for i, class_name in enumerate(classes):

    # Find images in each class folder
    image_files = glob.glob(
        "data/processed/train/" + class_name + "/*.jpg"
    )

    # Open the first image
    image = Image.open(image_files[0])

    # Create one of the three image spaces
    plt.subplot(1, 3, i + 1)

    # Show the image
    plt.imshow(image)

    # Add the class name
    plt.title(class_name.capitalize())

    # Hide x and y axes
    plt.axis("off")


plt.tight_layout()

# Save the image for the README
plt.savefig(
    "tomato_examples.png",
    dpi=300
)

# Show the image
plt.show()
