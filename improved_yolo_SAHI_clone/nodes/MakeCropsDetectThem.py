import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from ..components.CropComponent import CropComponent


class MakeCropsDetectThem:
    """
    Class implementing cropping and passing crops through a neural network
    for detection/segmentation.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the PyTorch model.
        imgsz (int): Size of the input image for inference PyTorch.
        conf (float): Confidence threshold for detections PyTorch.
        iou (float): IoU threshold for non-maximum suppression.
        classes_list (List[int] or None): List of classes to filter detections. If None,
                                          all classes are considered. Defaults to None.
        shape_x (int): Size of the crop in the x-coordinate.
        shape_y (int): Size of the crop in the y-coordinate.
        overlap_x (int): Percentage of overlap along the x-axis.
        overlap_y (int): Percentage of overlap along the y-axis.
        show_crops (bool): Whether to visualize the cropping.
        resize_initial_size (bool): Whether to resize the results to the original
                                    image size (ps: slow operation).

    Attributes:
        model: PyTorch model loaded from the specified path.
        image (np.ndarray): Input image BGR.
        imgsz (int): Size of the input image for inference.
        conf (float): Confidence threshold for detections.
        iou (float): IoU threshold for non-maximum suppression.
        classes_list (List[int] or None): List of classes to filter detections. If None,
                                          all classes are considered. Defaults to None.
        shape_x (int): Size of the crop in the x-coordinate.
        shape_y (int): Size of the crop in the y-coordinate.
        overlap_x (int): Percentage of overlap along the x-axis.
        overlap_y (int): Percentage of overlap along the y-axis.
        crops (list): List to store the CropElement objects.
        show_crops (bool): Whether to visualize the cropping.
        resize_initial_size (bool): Whether to resize the results to the original
                                    image size (ps: slow operation).
        class_names (list): List containing class names of the PyTorch model.
    """

    def __init__(
        self,
        image_path: str,
        model_path: str,
        imgsz=640,
        conf=0.5,
        iou=0.7,
        classes_list=None,
        shape_x=700,
        shape_y=700,
        overlap_x=25,
        overlap_y=25,
        show_crops=False,
        resize_initial_size=False,
    ) -> None:
        self.model = torch.load(model_path)  # Load the PyTorch model
        self.image = cv2.imread(image_path)  # Input image
        self.imgsz = imgsz  # Size of the input image for inference
        self.conf = conf  # Confidence threshold for detections
        self.iou = iou  # IoU threshold for non-maximum suppression
        self.classes_list = classes_list  # Classes to detect
        self.shape_x = shape_x  # Size of the crop in the x-coordinate
        self.shape_y = shape_y  # Size of the crop in the y-coordinate
        self.overlap_x = overlap_x  # Percentage of overlap along the x-axis
        self.overlap_y = overlap_y  # Percentage of overlap along the y-axis
        self.crops = []  # List to store the CropElement objects
        self.show_crops = show_crops  # Whether to visualize the cropping
        self.resize_initial_size = resize_initial_size  # slow operation !
        self.class_names = ["class1", "class2", "class3"]  # Placeholder for class names

        self.crops = self.get_crops_xy(
            self.image,
            shape_x=self.shape_x,
            shape_y=self.shape_y,
            overlap_x=self.overlap_x,
            overlap_y=self.overlap_y,
            show=self.show_crops,
        )
        self._detect_objects()

    def get_crops_xy(
        self,
        image_full,
        shape_x: int,
        shape_y: int,
        overlap_x=25,
        overlap_y=25,
        show=False,
    ):
        """Preprocessing of the image. Generating crops with overlapping."""
        cross_koef_x = 1 - (overlap_x / 100)
        cross_koef_y = 1 - (overlap_y / 100)

        data_all_crops = []

        y_steps = int((image_full.shape[0] - shape_y) / (shape_y * cross_koef_y)) + 1
        x_steps = int((image_full.shape[1] - shape_x) / (shape_x * cross_koef_x)) + 1

        y_new = round((y_steps-1) * (shape_y * cross_koef_y) + shape_y)
        x_new = round((x_steps-1) * (shape_x * cross_koef_x) + shape_x)
        image_innitial = image_full.copy()
        image_full = cv2.resize(image_full, (x_new, y_new))

        if show:
            plt.figure(figsize=[x_steps*0.9, y_steps*0.9])

        count = 0
        for i in range(y_steps):
            for j in range(x_steps):
                x_start = int(shape_x * j * cross_koef_x)
                y_start = int(shape_y * i * cross_koef_y)

                # Check for residuals
                if x_start + shape_x > image_full.shape[1]:
                    print('Error in generating crops along the x-axis')
                    continue
                if y_start + shape_y > image_full.shape[0]:
                    print('Error in generating crops along the y-axis')
                    continue

                im_temp = image_full[y_start:y_start + shape_y, x_start:x_start + shape_x]

                # Display the result:
                if show:
                    plt.subplot(y_steps, x_steps, i * x_steps + j + 1)
                    plt.imshow(cv2.cvtColor(im_temp.copy(), cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                count += 1

                data_all_crops.append(CropComponent(
                    source_image=image_innitial,
                    source_image_resized=image_full,
                    crop=im_temp,
                    number_of_crop=count,
                    x_start=x_start,
                    y_start=y_start,
                ))

        if show:
            plt.show()
            print('Number of generated images:', count)

        return data_all_crops

    def _detect_objects(self):
        """
        Method to detect objects in each crop using PyTorch.

        This method iterates through each crop and performs inference using
        the PyTorch model.
        """
        transform = ToTensor()

        for crop in self.crops:
            image_tensor = transform(crop.crop).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(image_tensor)
            # Process outputs and store results
