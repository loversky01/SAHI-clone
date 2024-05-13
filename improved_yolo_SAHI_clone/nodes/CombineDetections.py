import torch
import numpy as np
from .MakeCropsDetectThem import MakeCropsDetectThem


class CombineDetection:
    """
    Class implementing combining masks/boxes from multiple crops + NMS (Non-Maximum Suppression).

    Args:
        element_crops (MakeCropsDetectThem): Object containing crop information.
        nms_threshold (float): IoU/IoS threshold for non-maximum suppression.
        match_metric (str): Matching metric, either 'IOU' or 'IOS'.
        intelligent_sorter (bool): Enable sorting by area and rounded confidence parameter.
            If False, sorting will be done only by confidence (usual nms). (Default True)

    Attributes:
        conf_treshold (float): Confidence threshold of yolov8.
        class_names (dict): Dictionary containing class names pf yolov8 model.
        crops (list): List to store the CropElement objects.
        image (np.ndarray): Source image in BGR.
        nms_threshold (float): IOU/IOS threshold for non-maximum suppression.
        match_metric (str): Matching metric (IOU/IOS).
        intelligent_sorter (bool): Flag indicating whether sorting by area and confidence parameter is enabled.
        detected_conf_list_full (list): List of detected confidences.
        detected_xyxy_list_full (list): List of detected bounding boxes.
        detected_masks_list_full (list): List of detected masks.
        detected_cls_id_list_full (list): List of detected class IDs.
        detected_cls_names_list_full (list): List of detected class names.
        filtered_indices (list): List of indices after non-maximum suppression.
        filtered_confidences (list): List of confidences after non-maximum suppression.
        filtered_boxes (list): List of bounding boxes after non-maximum suppression.
        filtered_classes_id (list): List of class IDs after non-maximum suppression.
        filtered_classes_names (list): List of class names after non-maximum suppression.
        filtered_masks (list): List of filtered (after nms) masks if segmentation is enabled.
    """

    def __init__(
        self,
        element_crops: MakeCropsDetectThem,
        nms_threshold=0.3,
        match_metric='IOS',
        intelligent_sorter=True
    ) -> None:
        self.conf_treshold = element_crops.conf
        self.class_names = element_crops.class_names_dict
        self.crops = element_crops.crops  # List to store the CropElement objects
        if element_crops.resize_initial_size:
            self.image = element_crops.crops[0].source_image
        else:
            self.image = element_crops.crops[0].source_image_resized

        self.nms_threshold = nms_threshold  # IOU or IOS threshold for NMS
        self.match_metric = match_metric
        self.intelligent_sorter = intelligent_sorter # enable sorting by area and confidence parameter

        # combined detections of all patches
        (
            self.detected_conf_list_full,
            self.detected_xyxy_list_full,
            self.detected_masks_list_full,
            self.detected_cls_id_list_full
        ) = self.combinate_detections(crops=self.crops)

        self.detected_cls_names_list_full = [
            self.class_names[value] for value in self.detected_cls_id_list_full
        ]  # make str list

        # Invoke the NMS for segmentation masks method for filtering predictions
        if len(self.detected_masks_list_full)>0:

            self.filtered_indices = self.nms(
                self.detected_conf_list_full,
                self.detected_xyxy_list_full,
                self.match_metric,
                self.nms_threshold,
                self.detected_masks_list_full,
                intelligent_sorter=self.intelligent_sorter
            )  # for instance segmentation
        else:
            # Invoke the NMS method for filtering prediction
            self.filtered_indices = self.nms(
                self.detected_conf_list_full,
                self.detected_xyxy_list_full,
                self.match_metric,
                self.nms_threshold,
                intelligent_sorter=self.intelligent_sorter
            )  # for detection

        # Apply filtering (nms output indexes) to the prediction lists
        self.filtered_confidences = [self.detected_conf_list_full[i] for i in self.filtered_indices]
        self.filtered_boxes = [self.detected_xyxy_list_full[i] for i in self.filtered_indices]
        self.filtered_classes_id = [self.detected_cls_id_list_full[i] for i in self.filtered_indices]
        self.filtered_classes_names = [self.detected_cls_names_list_full[i] for i in self.filtered_indices]

        if element_crops.segment:
            self.filtered_masks = [self.detected_masks_list_full[i] for i in self.filtered_indices]
        else:
            self.filtered_masks = []

    def combinate_detections(self, crops):
        """
        Combine detections from multiple crop elements.

        Args:
            crops (list): List of CropElement objects.

        Returns:
            tuple: Tuple containing lists of detected confidences, bounding boxes,
                masks, and class IDs.
        """
        detected_conf = []
        detected_xyxy = []
        detected_masks = []
        detected_cls = []

        for crop in crops:
            detected_conf.extend(crop.detected_conf)
            detected_xyxy.extend(crop.detected_xyxy_real)
            detected_masks.extend(crop.detected_masks_real)
            detected_cls.extend(crop.detected_cls)

        return detected_conf, detected_xyxy, detected_masks, detected_cls

    @staticmethod
    def intersect_over_union(mask, masks_list):
        """
        Compute Intersection over Union (IoU) scores for a given mask against a list of masks.

        Args:
            mask (np.ndarray): Binary mask to compare.
            masks_list (list of np.ndarray): List of binary masks for comparison.

        Returns:
            torch.Tensor: IoU scores for each mask in masks_list compared to the input mask.
        """
        iou_scores = []
        for other_mask in masks_list:
            # Compute intersection and union
            intersection = np.logical_and(mask, other_mask).sum()
            union = np.logical_or(mask, other_mask).sum()
            # Compute IoU score, avoiding division by zero
            iou = intersection / union if union != 0 else 0
            iou_scores.append(iou)
        return torch.tensor(iou_scores)

    @staticmethod
    def intersect_over_smaller(mask, masks_list):
        """
        Compute Intersection over Smaller area scores for a given mask against a list of masks.

        Args:
            mask (np.ndarray): Binary mask to compare.
            masks_list (list of np.ndarray): List of binary masks for comparison.

        Returns:
            torch.Tensor: IoU scores for each mask in masks_list compared to the input mask, calculated over the smaller area.
        """
        ios_scores = []
        for other_mask in masks_list:
            # Compute intersection and area of smaller mask
            intersection = np.logical_and(mask, other_mask).sum()
            smaller_area = min(mask.sum(), other_mask.sum())
            # Compute IoU score over smaller area, avoiding division by zero
            ios = intersection / smaller_area if smaller_area != 0 else 0
            ios_scores.append(ios)
        return torch.tensor(ios_scores)

    def nms(
        self,
        confidences: list,
        boxes: list,
        match_metric,
        nms_threshold,
        masks=None,
        intelligent_sorter=False,
    ):
        """
        Apply non-maximum suppression to avoid detecting too many
        overlapping bounding boxes for a given object.

        Args:
            confidences (list): List of confidence scores.
            boxes (list): List of bounding boxes.
            match_metric (str): Matching metric, either 'IOU' or 'IOS'.
            nms_threshold (float): The threshold for match metric.
            masks (list, optional): List of masks. Defaults to None.
            intelligent_sorter (bool, optional): intelligent sorter

        Returns:
            list: List of filtered indexes.
        """
        if len(boxes) == 0:
            return []

        # Convert lists to tensors
        boxes = torch.tensor(boxes)
        confidences = torch.tensor(confidences)

        # Extract coordinates for every prediction box present
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # Calculate area of every box
        areas = (x2 - x1) * (y2 - y1)

        # Sort the prediction boxes according to their confidence scores or intelligent_sorter mode
        if intelligent_sorter:
            # Sort the prediction boxes according to their round confidence scores and area sizes
            order = torch.tensor(
                sorted(
                    range(len(confidences)),
                    key=lambda k: (round(confidences[k].item(), 1), areas[k]),
                    reverse=False,
                )
            )
        else:
            order = confidences.argsort()
        # Initialise an empty list for filtered prediction boxes
        keep = []

        while len(order) > 0:
            # Extract the index of the prediction with highest score
            idx = order[-1]

            # Push the index in filtered predictions list
            keep.append(idx.tolist())

            # Remove the index from the list
            order = order[:-1]

            # If there are no more boxes, break
            if len(order) == 0:
                break

            # Select coordinates of BBoxes according to the indices
            xx1 = torch.index_select(x1, dim=0, index=order)
            xx2 = torch.index_select(x2, dim=0, index=order)
            yy1 = torch.index_select(y1, dim=0, index=order)
            yy2 = torch.index_select(y2, dim=0, index=order)

            # Find the coordinates of the intersection boxes
            xx1 = torch.max(xx1, x1[idx])
            yy1 = torch.max(yy1, y1[idx])
            xx2 = torch.min(xx2, x2[idx])
            yy2 = torch.min(yy2, y2[idx])

            # Find height and width of the intersection boxes
            w = xx2 - xx1
            h = yy2 - yy1

            # Take max with 0.0 to avoid negative width and height
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)

            # Find the intersection area
            inter = w * h

            # Find the areas of BBoxes
            rem_areas = torch.index_select(areas, dim=0, index=order)

            if match_metric == "IOU":
                # Find the union of every prediction with the prediction
                union = (rem_areas - inter) + areas[idx]
                # Find the IoU of every prediction
                match_metric_value = inter / union

            elif match_metric == "IOS":
                # Find the smaller area of every prediction with the prediction
                smaller = torch.min(rem_areas, areas[idx])
                # Find the IoU of every prediction
                match_metric_value = inter / smaller

            else:
                raise ValueError("Unknown matching metric")

            # If masks are provided and IoU based on bounding boxes is greater than 0,
            # calculate IoU for masks and keep the ones with IoU < nms_threshold
            if masks is not None and torch.any(match_metric_value > 0):

                mask_mask = match_metric_value > 0

                order_2 = order[mask_mask]
                filtered_masks = [masks[i] for i in order_2]

                if match_metric == "IOU":
                    mask_iou = self.intersect_over_union(masks[idx], filtered_masks)
                    mask_mask = mask_iou > nms_threshold

                elif match_metric == "IOS":
                    mask_ios = self.intersect_over_smaller(masks[idx], filtered_masks)
                    mask_mask = mask_ios > nms_threshold
                # create a tensor of incidences to delete in tensor order
                order_2 = order_2[mask_mask]
                inverse_mask = ~torch.isin(order, order_2)

                # Keep only those order values that are not contained in order_2
                order = order[inverse_mask]

            else:
                # Keep the boxes with IoU/IoS less than threshold
                mask = match_metric_value < nms_threshold

                order = order[mask]

        return keep
