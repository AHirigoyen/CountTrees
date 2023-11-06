"""
Usage:
    visualize_data <root_directory> <annotations>
"""
from docopt import docopt
import cv2
import os
import pandas as pd


def draw_bounding_boxes(image_path, boxes):
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    image_name = os.path.basename(image_path).split('.')[0]
    print(boxes)
    for box in boxes:
        class_id, x, y, width, height = box
        # Draw the bounding box
        color = (0, 255, 0)  # BGR color format (green)
        thickness = 2
        cv2.rectangle(image, (x - width // 2, y - height // 2), (x + width // 2, y + height // 2),color, thickness)
        label = str(class_id) #labels[int(class_id)]
        cv2.putText(image, label, (x - width // 2, y - height // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, thickness)

    
    cv2.putText(image, image_name, (0, 15),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                         (255, 0, 0),
                         thickness)
    return image


def convert_bbox(xmin, ymin, xmax, ymax):
    width = xmax - xmin
    height = ymax - ymin
    
    x = xmin + width//2
    y = ymin + height//2
    
    return (x, y, width, height)

def get_images_labels(root_dir, annotations):

    images = annotations['image_path'].unique()

    labels = []
    for img in images:
        df_c = annotations[annotations['image_path']==img].copy()

        labels_img = []
        for index, row in df_c.iterrows():
            box = convert_bbox(row.xmin, row.ymin, row.xmax, row.ymax)
            labels_img.append([row.label, *box])
        
        labels.append(labels_img)

    images = [os.path.join(root_dir, i) for i in images]
    return list(zip(images,labels))





# Function to display images and navigate using arrow keys
def visualize_images(root_dir, annotations):
    image_files = get_images_labels(root_dir, annotations)

    current_index = 0

    while True:
        image_path, labels = image_files[current_index]
        image = draw_bounding_boxes(image_path, labels)

        if image is not None:
            cv2.imshow('Image Viewer', image)
            key = cv2.waitKeyEx(0)
            # Handle keyboard inputs
            if key == ord('q'):
                break  # Quit the viewer
            elif key == 65361:  # Left arrow key
                current_index = (current_index - 1) % len(image_files)  # Previous image
            elif key == 65363:  # Right arrow key
                current_index = (current_index + 1) % len(image_files)  # Next image

        else:
            print(f"Error loading image: {image_path}")

    cv2.destroyAllWindows()


def main():
    arguments = docopt(__doc__)
    root_dir = arguments['<root_directory>']
    annotations = arguments['<annotations>']

    annotations = pd.read_csv(annotations)
    
    visualize_images(root_dir, annotations)

    


if __name__ == "__main__":
    main()