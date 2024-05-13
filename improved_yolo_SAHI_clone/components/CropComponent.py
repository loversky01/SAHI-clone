import numpy as np
import cv2


class CropComponent:
    def __init__(
            self,
            source_image: np.array,
            source_image_resized: np.array,
            crop: np.array,
            number_of_crop: int,
            x_start: int,
            y_start: int
    ) -> None:
        """
        Khởi tạo đối tượng CropElement.

        Tham số:
        - source_image: Ảnh gốc.
        - source_image_resized: Ảnh gốc đã được resize thành bội số của kích thước cắt.
        - crop: Phần cắt cụ thể.
        - number_of_crop: Số thứ tự của phần cắt, từ trái qua phải, từ trên xuống dưới.
        - x_start: Tọa độ X của góc trên cùng bên trái.
        - y_start: Tọa độ Y của góc trên cùng bên trái.
        """
        self.source_image = source_image
        self.source_image_resized = source_image_resized
        self.crop = crop
        self.number_of_crop = number_of_crop
        self.x_start = x_start
        self.y_start = y_start
        self.detected_conf = None
        self.detected_cls = None
        self.detected_xyxy = None
        self.detected_masks = None
        self.detected_xyxy_real = None
        self.detected_masks_real = None

    def calculate_inference(self, model, imgsz=640, conf=0.35, iou=0.7, segment=False, classes_list=None):
        """
        Thực hiện dự đoán sử dụng mô hình cung cấp.

        Tham số:
        - model: Mô hình nhận dạng đối tượng.
        - imgsz: Kích thước ảnh đầu vào.
        - conf: Ngưỡng độ tin cậy.
        - iou: Ngưỡng IOU.
        - segment: Có thực hiện phân đoạn không.
        - classes_list: Danh sách các lớp cần nhận diện.
        """
        predictions = model.predict(self.crop, imgsz=imgsz, conf=conf, iou=iou, classes=classes_list, verbose=False)
        pred = predictions[0]
        self.detected_xyxy = pred.boxes.xyxy.cpu().int().tolist()
        self.detected_cls = pred.boxes.cls.cpu().int().tolist()
        self.detected_conf = pred.boxes.conf.cpu().numpy()
        if segment and self.detected_cls and len(self.detected_cls) != 0:
            self.detected_masks = pred.masks.data.cpu().numpy()

    def calculate_real_values(self):
        """
        Tính toán giá trị thực của hộp giới hạn và mặt nạ trong source_image_resized.
        """
        x_start_global = self.x_start  # Tọa độ X toàn cục của phần cắt
        y_start_global = self.y_start  # Tọa độ Y toàn cục của phần cắt

        # Tính toán tọa độ hộp giới hạn thực dựa trên thông tin vị trí của phần cắt
        self.detected_xyxy_real = np.array(self.detected_xyxy) + np.array(
            [[x_start_global, y_start_global, x_start_global, y_start_global]])

        if self.detected_masks is not None:
            # Kiểm tra sự tồn tại của mask trước khi thực hiện resize
            self.detected_masks_real = [
                cv2.resize(mask, (self.source_image_resized.shape[1], self.source_image_resized.shape[0]),
                           interpolation=cv2.INTER_NEAREST) for mask in self.detected_masks]

    def resize_results(self):
        """
        Thay đổi kích thước tọa độ hộp giới hạn và mặt nạ từ source_image_resized sang kích thước của source_image.
        """
        scale_x = self.source_image.shape[1] / self.source_image_resized.shape[1]
        scale_y = self.source_image.shape[0] / self.source_image_resized.shape[0]

        # Thay đổi kích thước tọa độ hộp giới hạn
        self.detected_xyxy_real[:, [0, 2]] *= scale_x
        self.detected_xyxy_real[:, [1, 3]] *= scale_y

        if self.detected_masks_real is not None:
            # Thay đổi kích thước mặt nạ
            self.detected_masks_real = [cv2.resize(mask, (self.source_image.shape[1], self.source_image.shape[0]),
                                                   interpolation=cv2.INTER_NEAREST) for mask in
                                        self.detected_masks_real]
