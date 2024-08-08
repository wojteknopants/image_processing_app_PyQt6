#Pytania:
# adjust brightness: V+shift czy V*factor?
# zjechanie saturacji do zera a potem uzycie negatywu (w RGB), usuwa informacje o kanale H. Tracimy kolor bezpowrotnie
# contrast linear: V*factor? tak jak brightness?
# contrast_A 1.2 -> contrast_A 1.4 = contrast_A 1.2*1.4 czy contrast_A 1.4? edytowac obecny wynik? czy przechowywac i edytowac oryginal?
# jesli przechowywac oryginal to contrast_B 1.5 = oryginal * contrast_B 1.5 czy contrast_A 1.4 * contrast_B 1.5? nadpisywac oryginal przy zmianie metody?
# arytmetyka: czemu takie dziwne wychodzi

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QSizePolicy, QFileDialog, QMessageBox, QSlider, QStatusBar, QToolBar, QCheckBox, QRadioButton, QButtonGroup, QSpinBox, QDoubleSpinBox, QTextEdit, QDialog, QSpacerItem
)
from PyQt6.QtGui import QPixmap, QImage, QAction
from PyQt6.QtCore import Qt
import numpy as np
from PIL import Image
import pyqtgraph as pg

from scipy.ndimage import convolve
from scipy.signal import convolve2d
import cv2
import time

class WatershedUtils():

    def __init__(self):
        pass


    def gaussian_blur(self, image, kernel_size, sigma):
        kernel_maker = KernelMaker()
        kernel_maker.set_gaussian_kernel(kernel_size, kernel_size, sigma)
        kernel = kernel_maker.get_kernel()
        return convolve(image, kernel)

    def compute_gradient(self, image):
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        Gx = convolve2d(image, sobel_x, mode='same', boundary='symm')
        Gy = convolve2d(image, sobel_y, mode='same', boundary='symm')

        gradient_magnitude = np.sqrt(Gx**2 + Gy**2)
        gradient_direction = np.arctan2(Gy, Gx) * 180 / np.pi

        return gradient_magnitude, gradient_direction

    def watershed_by_immersion(self, pil_image_hsv):
        
        # Convert the Pillow image to RGB and then to grayscale
        pil_image_rgb = pil_image_hsv.convert('RGB')
        gray_image = pil_image_rgb.convert('L')
        image = np.array(gray_image)
        
        # Gaussian blur to reduce noise
        blurred = self.gaussian_blur(image, 5, 1)
        
        # Compute gradient magnitude and direction
        gradient_magnitude, _ = self.compute_gradient(blurred)
        gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)
        
        # Watershed initialization
        hmin = gradient_magnitude.min()
        hmax = gradient_magnitude.max()
        
        lab = np.full(gradient_magnitude.shape, -1)  # Initialize label image with init (-1)
        dist = np.zeros(gradient_magnitude.shape, dtype=int)  # Initialize distance image
        curlab = 0  # Initialize current label
        queue = []  # FIFO queue
        
        def neighbors(p):
            x, y = p
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < gradient_magnitude.shape[0] and 0 <= ny < gradient_magnitude.shape[1]:
                        yield (nx, ny)
        
        # Sort pixels by increasing order of grey values
        sorted_pixels = np.argsort(gradient_magnitude, axis=None)
        sorted_pixels = np.unravel_index(sorted_pixels, gradient_magnitude.shape)
        
        # Start Flooding
        for h in range(hmin, hmax + 1):
            current_level_pixels = zip(*np.where(gradient_magnitude == h))
            for p in current_level_pixels:
                lab[p] = -2  # Mask pixel
                for q in neighbors(p):
                    if lab[q] > 0 or lab[q] == 0:
                        dist[p] = 1
                        queue.append(p)
                        break

            curdist = 1
            queue.append((-1, -1))  # Add fictitious pixel
            
            while queue:
                p = queue.pop(0)
                if p == (-1, -1):
                    if not queue:
                        break
                    else:
                        queue.append((-1, -1))
                        curdist += 1
                        p = queue.pop(0)
                        
                for q in neighbors(p):
                    if dist[q] < curdist and (lab[q] > 0 or lab[q] == 0):
                        if lab[q] > 0:
                            if lab[p] == -2 or lab[p] == 0:
                                lab[p] = lab[q]
                            elif lab[p] != lab[q]:
                                lab[p] = 0
                        elif lab[p] == -2:
                            lab[p] = 0
                    elif lab[q] == -2 and dist[q] == 0:
                        dist[q] = curdist + 1
                        queue.append(q)
            
            # Detect and process new minima at level h
            current_level_pixels = zip(*np.where(gradient_magnitude == h))
            for p in current_level_pixels:
                dist[p] = 0
                if lab[p] == -2:
                    curlab += 1
                    queue.append(p)
                    lab[p] = curlab
                    while queue:
                        q = queue.pop(0)
                        for r in neighbors(q):
                            if lab[r] == -2:
                                queue.append(r)
                                lab[r] = curlab
        
        # Convert the label image to RGB for visualization
        label_rgb = np.zeros((*lab.shape, 3), dtype=np.uint8)
        unique_labels = np.unique(lab)
        for label in unique_labels:
            if label == -1:
                color = [0, 0, 0]  # Background
            elif label == 0:
                color = [255, 0, 0]  # Watershed
            else:
                color = np.random.randint(0, 255, 3)  # Random color for each label
            label_rgb[lab == label] = color
        
        # Convert back to Pillow image
        pil_label_rgb = Image.fromarray(label_rgb)
        pil_label_hsv = pil_label_rgb.convert('HSV')
        
        return pil_label_hsv

class CannyDialog(QDialog):
    def __init__(self, parent=None, current_image_hsv=None):
        super().__init__(parent)
        self.setWindowTitle('Canny Edge gradient_magnitudeection')

        self.current_image_hsv = current_image_hsv  # Store the current image
        self.current_image_rgb = current_image_hsv.convert('RGB')  # Convert to RGB for processing

        # Create layout
        layout = QVBoxLayout()

        # Smoothing parameters
        self.kernel_size_label = QLabel("Gaussian Kernel Size:")
        self.kernel_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.kernel_size_slider.setRange(1, 16)
        self.kernel_size_slider.setValue(2)
        self.kernel_size_slider.setSingleStep(1)
        self.kernel_size_slider.setTickInterval(1)
        self.kernel_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.kernel_size_value_label = QLabel("3")

        self.sigma_label = QLabel("Sigma:")
        self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_slider.setRange(1, 100)
        self.sigma_slider.setValue(10)
        self.sigma_slider.setSingleStep(1)
        self.sigma_slider.setTickInterval(10)
        self.sigma_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sigma_value_label = QLabel("1.0")

        # Hysteresis thresholds
        self.low_threshold_label = QLabel("Low Threshold:")
        self.low_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.low_threshold_slider.setRange(0, 255)
        self.low_threshold_slider.setValue(50)
        self.low_threshold_value_label = QLabel("50")

        self.high_threshold_label = QLabel("High Threshold:")
        self.high_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.high_threshold_slider.setRange(0, 255)
        self.high_threshold_slider.setValue(150)
        self.high_threshold_value_label = QLabel("150")

        # Buttons
        self.preview_button = QPushButton("Preview")
        self.apply_button = QPushButton("Apply")
        self.cancel_button = QPushButton("Cancel")

        # Layouts
        layout.addWidget(self.kernel_size_label)
        layout.addWidget(self.kernel_size_slider)
        layout.addWidget(self.kernel_size_value_label)

        layout.addWidget(self.sigma_label)
        layout.addWidget(self.sigma_slider)
        layout.addWidget(self.sigma_value_label)

        layout.addWidget(self.low_threshold_label)
        layout.addWidget(self.low_threshold_slider)
        layout.addWidget(self.low_threshold_value_label)

        layout.addWidget(self.high_threshold_label)
        layout.addWidget(self.high_threshold_slider)
        layout.addWidget(self.high_threshold_value_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.preview_button)
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect signals
        self.kernel_size_slider.valueChanged.connect(self.update_parameters)
        self.sigma_slider.valueChanged.connect(self.update_parameters)
        self.low_threshold_slider.valueChanged.connect(self.update_parameters)
        self.high_threshold_slider.valueChanged.connect(self.update_parameters)

        self.preview_button.clicked.connect(self.preview_canny)
        self.apply_button.clicked.connect(self.on_apply_clicked)
        self.cancel_button.clicked.connect(self.reject)

        self.processed_image = None

    def update_parameters(self):
        self.kernel_size_value_label.setText(str(2* self.kernel_size_slider.value() -1))
        self.sigma_value_label.setText(f"{self.sigma_slider.value() / 10.0:.1f}")
        self.low_threshold_value_label.setText(str(self.low_threshold_slider.value()))
        self.high_threshold_value_label.setText(str(self.high_threshold_slider.value()))

    def gaussian_blur(self, image, kernel_size, sigma):
        kernel_maker = KernelMaker()
        kernel_maker.set_gaussian_kernel(kernel_size, kernel_size, sigma)
        kernel = kernel_maker.get_kernel()
        return convolve(image, kernel)
    
    def compute_gradient(self, image):
        
        # Sobel operators for x and y directions
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        # Compute gradients using convolve2d
        Gx = convolve2d(image, sobel_x, mode='same', boundary='symm')
        Gy = convolve2d(image, sobel_y, mode='same', boundary='symm')

        # Compute gradient magnitude and direction
        gradient_magnitude = np.sqrt(Gx**2 + Gy**2)
        gradient_direction = np.arctan2(Gy, Gx) * 180 / np.pi  # Convert to degrees

        return gradient_magnitude, gradient_direction

    def non_maximum_suppression(self, gradient_magnitude, gradient_direction):
        
        gmax = np.zeros(gradient_magnitude.shape)
        angle = gradient_direction

        # Normalize angle to range [0, 180)
        angle = angle % 180

        # Padding to avoid boundary issues
        padded_gradient = np.pad(gradient_magnitude, ((1, 1), (1, 1)), mode='constant')
        padded_angle = np.pad(angle, ((1, 1), (1, 1)), mode='constant')

        for i in range(1, gradient_magnitude.shape[0] + 1):
            for j in range(1, gradient_magnitude.shape[1] + 1):
                # 0 degrees
                if (padded_angle[i, j] >= 0 and padded_angle[i, j] < 22.5) or (padded_angle[i, j] >= 157.5 and padded_angle[i, j] < 180):
                    if padded_gradient[i, j] >= padded_gradient[i, j + 1] and padded_gradient[i, j] >= padded_gradient[i, j - 1]:
                        gmax[i - 1, j - 1] = padded_gradient[i, j]
                # 45 degrees
                elif (padded_angle[i, j] >= 22.5 and padded_angle[i, j] < 67.5):
                    if padded_gradient[i, j] >= padded_gradient[i - 1, j + 1] and padded_gradient[i, j] >= padded_gradient[i + 1, j - 1]:
                        gmax[i - 1, j - 1] = padded_gradient[i, j]
                # 90 degrees
                elif (padded_angle[i, j] >= 67.5 and padded_angle[i, j] < 112.5):
                    if padded_gradient[i, j] >= padded_gradient[i - 1, j] and padded_gradient[i, j] >= padded_gradient[i + 1, j]:
                        gmax[i - 1, j - 1] = padded_gradient[i, j]
                # 135 degrees
                elif (padded_angle[i, j] >= 112.5 and padded_angle[i, j] < 157.5):
                    if padded_gradient[i, j] >= padded_gradient[i - 1, j - 1] and padded_gradient[i, j] >= padded_gradient[i + 1, j + 1]:
                        gmax[i - 1, j - 1] = padded_gradient[i, j]

        # Remove false positive outliers and normalize
        #high_value = np.percentile(gmax, 98)
        #gmax = np.clip(gmax, 0, high_value)
        # gmax = (gmax - gmax.min()) / (gmax.max() - gmax.min()) * 255

        # Calculate the range of the gmax values
        value_range = gmax.max() - gmax.min()

        # Only perform the normalization if the range is non-zero
        if value_range != 0:
            gmax = (gmax - gmax.min()) / value_range * 255
        else:
            # If all values are the same, set gmax to 0 (or any appropriate value, depending on your application)
            gmax = np.zeros_like(gmax)

        return gmax
    
    def double_threshold(self, nms_image, low_threshold, high_threshold):
        thres  = np.zeros(nms_image.shape)
        strong = 1 * 255
        weak   = 0.2 * 255
        mmax = np.max(nms_image)
        lo = (low_threshold/255)*mmax
        hi = (high_threshold/255)*mmax
        #lo, hi = 0.1 * mmax,0.8 * mmax
        strongs = []
        weaks = []
        for i in range(nms_image.shape[0]):
            for j in range(nms_image.shape[1]):
                px = nms_image[i][j]
                if px >= hi:
                    thres[i][j] = strong
                    strongs.append((i, j))
                elif px >= lo:
                    thres[i][j] = weak

        return thres, weak, strong

    def edge_tracking_by_hysteresis(self, image, weak, strong=255):
    
        strong_i, strong_j = np.where(image == strong)
        strongs = list(zip(strong_i, strong_j))
        
        
        # Initialize result image with zeros
        result_image = np.zeros_like(image)
        image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
        
        while strongs:
            i, j = strongs.pop()
            
            weak_i, weak_j = np.where(image[i-1:i+1, j-1:j+1] == weak)
            result_image[i, j] = strong
            for wi, wj in zip(weak_i, weak_j):
                strongs.append((wi, wj))

        return result_image

    def apply_canny(self):
        kernel_size = 2 * self.kernel_size_slider.value() -1
        sigma = self.sigma_slider.value() / 10.0
        low_threshold = self.low_threshold_slider.value()
        high_threshold = self.high_threshold_slider.value()

        # Convert the image to grayscale
        gray_image = self.current_image_rgb.convert('L')
        gray_array = np.array(gray_image)

        # Apply Gaussian blur
        blurred = self.gaussian_blur(gray_array, kernel_size, sigma)
        img = Image.fromarray(blurred.astype(np.uint8), 'L')
        img.save("1-Gauss.png")

        # Compute gradients
        gradient_magnitude, gradient_direction = self.compute_gradient(blurred)

        # Non-maximum suppression
        nms_image = self.non_maximum_suppression(gradient_magnitude, gradient_direction)
        nms_image_normalized = (nms_image / nms_image.max()) * 255  # Normalize for visualization
        img = Image.fromarray(nms_image_normalized.astype(np.uint8), 'L')
        img.save("2-NMS.png")

        # Double threshold
        threshold_image, weak, strong = self.double_threshold(nms_image, low_threshold, high_threshold)
        img = Image.fromarray(threshold_image.astype(np.uint8), 'L')
        img.save("3-Threshold.png")

        # Edge tracking by hysteresis
        final_image = self.edge_tracking_by_hysteresis(threshold_image, weak, strong)
        img = Image.fromarray(final_image.astype(np.uint8), 'L')
        img.save("4-Edge_tracking.png")

        final_image_pil = img.convert('HSV')
        return final_image_pil
    
    def apply_canny_cv2(self):
        kernel_size = 2 * self.kernel_size_slider.value() - 1
        sigma = self.sigma_slider.value() / 10.0
        low_threshold = self.low_threshold_slider.value()
        high_threshold = self.high_threshold_slider.value()

        # Convert the image to grayscale
        gray_image = self.current_image_rgb.convert('L')

        # Apply Gaussian blur using KernelMaker
        kernel_maker = KernelMaker()
        kernel_maker.set_gaussian_kernel(kernel_size, kernel_size, sigma)
        blurred_image = kernel_maker.perform_convolution(gray_image)
        blurred_array = np.array(blurred_image.convert('L'))
        img = Image.fromarray(blurred_array.astype(np.uint8), 'L')
        img.save("1-Gauss.png")

        # Apply Canny edge detection using OpenCV
        edges = cv2.Canny(blurred_array.astype(np.uint8), low_threshold, high_threshold)
        img = Image.fromarray(edges.astype(np.uint8), 'L')
        img.save("2-Canny.png")

        final_image_pil = img.convert('HSV')
        return final_image_pil
        


        


    def preview_canny(self):
        self.processed_image = self.apply_canny()
        self.parent().display_image(self.processed_image)

    def on_apply_clicked(self):
        self.processed_image = self.apply_canny_cv2()
        self.accept()

    def get_processed_image(self):
        return self.processed_image
    
class BinarizationDialog(QDialog):
    def __init__(self, parent=None, current_image_hsv=None):
        super().__init__(parent)
        self.setWindowTitle('Binarization')

        self.current_image_hsv = current_image_hsv  # Store the current image
        self.processed_image = None

        # Convert to grayscale
        self.current_image_gray = self.convert_to_grayscale(self.current_image_hsv)

        # Create layout
        layout = QVBoxLayout()

        # Slider for manual threshold adjustment
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(128)
        self.threshold_slider.valueChanged.connect(self.update_binarization)
        layout.addWidget(QLabel("Binarization Threshold:"))
        layout.addWidget(self.threshold_slider)

        # Label to show current threshold value
        self.threshold_label = QLabel("Threshold: 128")
        layout.addWidget(self.threshold_label)

        # Button for setting threshold with Otsu's algorithm
        self.otsu_button = QPushButton("Set with Otsu Algo")
        self.otsu_button.clicked.connect(self.apply_otsu_threshold)
        layout.addWidget(self.otsu_button)

        # Buttons for apply and cancel
        button_layout = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.apply_button)
        

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Connect signals
        self.apply_button.clicked.connect(self.on_apply_clicked)
        self.cancel_button.clicked.connect(self.reject)

    def convert_to_grayscale(self, hsv_image):
        rgb_image = hsv_image.convert('RGB')
        gray_image = rgb_image.convert('L')  # Convert to grayscale
        return gray_image

    def update_binarization(self):
        threshold = self.threshold_slider.value()
        self.threshold_label.setText(f"Threshold: {threshold}")
        self.processed_image = self.binarize_image(self.current_image_gray, threshold)
        self.parent().display_image(self.processed_image)

    def apply_otsu_threshold(self):
        threshold = self.calculate_otsu_threshold(self.current_image_gray)
        self.threshold_slider.setValue(threshold)
        self.threshold_label.setText(f"Threshold: {threshold}")
        self.update_binarization()

    def binarize_image(self, gray_image, threshold):
        gray_array = np.array(gray_image)
        binary_array = (gray_array > threshold) * 255
        binary_image = Image.fromarray(binary_array.astype(np.uint8), 'L')
        return binary_image.convert('HSV')

    def calculate_otsu_threshold(self, gray_image):
        gray_array = np.array(gray_image)
        hist, bin_edges = np.histogram(gray_array, bins=256, range=(0, 255))
        total_pixels = gray_array.size
        sum_total = np.dot(np.arange(256), hist)
        sum_background, weight_background, weight_foreground, var_max, threshold = 0, 0, 0, 0, 0

        for i in range(256):
            weight_background += hist[i]
            if weight_background == 0:
                continue
            weight_foreground = total_pixels - weight_background
            if weight_foreground == 0:
                break
            sum_background += i * hist[i]
            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground
            var_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
            if var_between > var_max:
                var_max = var_between
                threshold = i
        return threshold

    def on_apply_clicked(self):
        self.processed_image = self.binarize_image(self.current_image_gray, self.threshold_slider.value())
        self.accept()

    def get_processed_image(self):
        return self.processed_image

    def get_action_name(self):
        return "Applied Binarization"
    
class KernelMaker:
    def __init__(self):
        self.kernel = None

    def set_custom_kernel(self, custom_kernel):
        
        if not isinstance(custom_kernel, np.ndarray):
            raise TypeError("Custom kernel must be a numpy array.")
        
        if custom_kernel.ndim != 2:
            raise ValueError("Custom kernel must be a 2D array.")
        
        self.kernel = custom_kernel

    def set_gaussian_kernel(self, size_x, size_y, sigma=1.0):
        
        center_x = (size_x - 1) / 2
        center_y = (size_y - 1) / 2

        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma**2)) * 
                         np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2)),
            (size_y, size_x)
        )
        self.kernel = kernel / np.sum(kernel)

    def set_log_kernel(self, size_x=None, size_y=None, sigma=1.0):
        if size_x is None or size_y is None:
            size_x = size_y = int(np.ceil(sigma * 6))
            if size_x % 2 == 0:
                size_x += 1
            if size_y % 2 == 0:
                size_y += 1
        
        K = 1.6
        center_x = (size_x - 1) / 2
        center_y = (size_y - 1) / 2

        gauss1 = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma**2)) * 
                         np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2)),
            (size_y, size_x)
        )
        gauss2 = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * (K * sigma)**2)) * 
                         np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (K * sigma)**2)),
            (size_y, size_x)
        )

        kernel = gauss1 - gauss2
        self.kernel = kernel / np.sum(kernel)

    def set_mean_blur_kernel(self, size_x=3, size_y=3):
        
        self.kernel = np.ones((size_y, size_x)) / (size_x * size_y)

    def set_sharpen_kernel(self):
        """
        Set a sharpening kernel.
        """
        self.kernel = np.array([[0, -1, 0], 
                                [-1, 5, -1], 
                                [0, -1, 0]])

    def set_sobel_kernel(self, axis):
        
        if axis == 'y':
            self.kernel = np.array([[-1, 0, 1], 
                                    [-2, 0, 2], 
                                    [-1, 0, 1]])
        elif axis == 'x':
            self.kernel = np.array([[-1, -2, -1], 
                                    [0, 0, 0], 
                                    [1, 2, 1]])
        else:
            raise ValueError("Axis must be 'x' or 'y'.")

    def set_prewitt_kernel(self, axis):
        
        if axis == 'y':
            self.kernel = np.array([[-1, 0, 1], 
                                    [-1, 0, 1], 
                                    [-1, 0, 1]])
        elif axis == 'x':
            self.kernel = np.array([[-1, -1, -1], 
                                    [0, 0, 0], 
                                    [1, 1, 1]])
        else:
            raise ValueError("Axis must be 'x' or 'y'.")

    def set_roberts_kernel(self, axis):
        
        if axis == 'y':
            self.kernel = np.array([[1, 0], 
                                    [0, -1]])
        elif axis == 'x':
            self.kernel = np.array([[0, 1], 
                                    [-1, 0]])
        else:
            raise ValueError("Axis must be 'x' or 'y'.")

    def set_laplacian_kernel(self):
        
        self.kernel = np.array([[0, 1, 0], 
                                [1, -4, 1], 
                                [0, 1, 0]])

    def get_kernel(self):
        
        return self.kernel

    def perform_convolution(self, image):
        
        if self.kernel is None:
            raise ValueError("No kernel has been set.")
        
        # Convert image to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert image to numpy array
        image_array = np.array(image)

        # Perform convolution on each channel separately
        channels = []
        
        for i in range(3):
            channel = image_array[..., i]
            convoluted_channel = convolve2d(channel, self.kernel, mode='same', boundary='symm')
            channels.append(convoluted_channel)
    
        # Stack channels back into an image
        convoluted_image_array = np.stack(channels, axis=-1)
        
        # Clip values to the valid range [0, 255] and convert to uint8
        convoluted_image_array = np.clip(convoluted_image_array, 0, 255).astype(np.uint8)
        
        # Convert back to Pillow Image and then to HSV
        convoluted_image = Image.fromarray(convoluted_image_array, 'RGB').convert('HSV')
        return convoluted_image

class ConvolutionFilterDialog(QDialog):
    def __init__(self, parent=None, current_image_hsv=None):
        super().__init__(parent)
        self.setWindowTitle('Convolution Filters')

        self.current_image_hsv = current_image_hsv  # Store the current image
        self.kernel_maker = KernelMaker()
        self.selected_filter_name = ""

        # Create layout
        layout = QVBoxLayout()

        # Filter selection radio buttons
        self.radio_group = QButtonGroup(self)
        self.radio_mean_blur = QRadioButton("Mean Blur")
        self.radio_sharpen = QRadioButton("Sharpen")
        self.radio_gaussian = QRadioButton("Gaussian Blur")
        self.radio_log = QRadioButton("LoG")
        self.radio_sobel_h = QRadioButton("Sobel (H)")
        self.radio_sobel_v = QRadioButton("Sobel (V))")
        self.radio_prewitt_h = QRadioButton("Prewitt (H))")
        self.radio_prewitt_v = QRadioButton("Prewitt (V)")
        self.radio_roberts_h = QRadioButton("Roberts (H)")
        self.radio_roberts_v = QRadioButton("Roberts (V)")
        self.radio_laplace = QRadioButton("Laplacian")
        self.radio_custom = QRadioButton("Custom")


        self.radio_group.addButton(self.radio_mean_blur)
        self.radio_group.addButton(self.radio_gaussian)
        self.radio_group.addButton(self.radio_sharpen)
        self.radio_group.addButton(self.radio_custom)
        self.radio_group.addButton(self.radio_sobel_h)
        self.radio_group.addButton(self.radio_sobel_v)
        self.radio_group.addButton(self.radio_prewitt_h)
        self.radio_group.addButton(self.radio_prewitt_v)
        self.radio_group.addButton(self.radio_roberts_h)
        self.radio_group.addButton(self.radio_roberts_v)
        self.radio_group.addButton(self.radio_laplace)
        self.radio_group.addButton(self.radio_log)
        self.radio_group.buttonClicked.connect(self.on_filter_selected)


        # Gaussian and LoG parameter fields
        self.size_x_label = QLabel("X:")
        self.size_x_spinbox = QSpinBox()
        self.size_x_spinbox.setRange(1, 100)
        self.size_x_spinbox.setValue(3)
        self.size_x_spinbox.setMinimumWidth(50)
        self.size_y_label = QLabel("Y:")
        self.size_y_spinbox = QSpinBox()
        self.size_y_spinbox.setRange(1, 100)
        self.size_y_spinbox.setValue(3)
        self.size_y_spinbox.setMinimumWidth(50)
        self.sigma_label = QLabel("Sigma:")
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setRange(0.1, 100.0)
        self.sigma_spinbox.setValue(1.0)
        self.sigma_spinbox.setMinimumWidth(50)
        self.calculate_button = QPushButton("Calculate Kernel")
        self.calculate_button.clicked.connect(self.update_kernel_display)

        # Create HBoxes for parameters
        self.label_parameters = QLabel("Parameters")
        self.hbox_parameters = QHBoxLayout()
        self.hbox_parameters.addWidget(self.size_x_label)
        self.hbox_parameters.addWidget(self.size_x_spinbox)
        self.hbox_parameters.addWidget(self.size_y_label)
        self.hbox_parameters.addWidget(self.size_y_spinbox)
        self.hbox_parameters.addWidget(self.sigma_label)
        self.hbox_parameters.addWidget(self.sigma_spinbox)
        self.hbox_parameters.addWidget(self.calculate_button)

        # Main VBox layout
        layout.addWidget(self.radio_mean_blur)
        layout.addWidget(self.radio_gaussian)
        layout.addWidget(self.radio_sharpen)
        layout.addWidget(self.radio_sobel_h)
        layout.addWidget(self.radio_sobel_v)
        layout.addWidget(self.radio_prewitt_h)
        layout.addWidget(self.radio_prewitt_v)
        layout.addWidget(self.radio_roberts_h)
        layout.addWidget(self.radio_roberts_v)
        layout.addWidget(self.radio_laplace)
        layout.addWidget(self.radio_log)
        layout.addWidget(self.radio_custom)
        layout.addWidget(self.label_parameters)
        layout.addLayout(self.hbox_parameters)
        self.label_custom = QLabel("Editing Enabled - type your kernel below")
        layout.addWidget(self.label_custom)

        # Kernel display field
        self.kernel_display = QTextEdit()
        self.kernel_display.setReadOnly(True)
        layout.addWidget(self.kernel_display)

        # Buttons
        button_layout = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.on_apply_clicked)
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.on_preview_clicked)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.preview_button)
        button_layout.addWidget(self.apply_button)
        layout.addLayout(button_layout)
        self.setLayout(layout)


        self.kernel = None
        self.processed_image = None

        self.update_ui_elements()


    def on_filter_selected(self, button):
        # Update UI based on selected filter
        self.update_ui_elements()
        if self.radio_gaussian.isChecked() or self.radio_log.isChecked() or self.radio_custom.isChecked() or self.radio_mean_blur.isChecked():
            return
        self.update_kernel_display()

    def update_ui_elements(self):
        if self.radio_gaussian.isChecked() or self.radio_log.isChecked():

            self.label_parameters.setVisible(True)
            self.size_x_label.setVisible(True)
            self.size_x_spinbox.setVisible(True)
            self.size_y_label.setVisible(True)
            self.size_y_spinbox.setVisible(True)
            self.sigma_label.setVisible(True)
            self.sigma_spinbox.setVisible(True)
            self.calculate_button.setVisible(True)

            self.label_custom.setVisible(False)
            self.kernel_display.setReadOnly(True)

        elif self.radio_custom.isChecked():

            self.label_parameters.setVisible(False)
            self.size_x_label.setVisible(False)
            self.size_x_spinbox.setVisible(False)
            self.size_y_label.setVisible(False)
            self.size_y_spinbox.setVisible(False)
            self.sigma_label.setVisible(False)
            self.sigma_spinbox.setVisible(False)
            self.calculate_button.setVisible(False)

            self.label_custom.setVisible(True)
            self.kernel_display.setReadOnly(False)

        elif self.radio_mean_blur.isChecked():

            self.label_parameters.setVisible(True)
            self.size_x_label.setVisible(True)
            self.size_x_spinbox.setVisible(True)
            self.size_y_label.setVisible(True)
            self.size_y_spinbox.setVisible(True)
            self.sigma_label.setVisible(False)
            self.sigma_spinbox.setVisible(False)
            self.calculate_button.setVisible(True)

            self.label_custom.setVisible(False)
            self.kernel_display.setReadOnly(True)

        else:

            self.label_parameters.setVisible(False)
            self.size_x_label.setVisible(False)
            self.size_x_spinbox.setVisible(False)
            self.size_y_label.setVisible(False)
            self.size_y_spinbox.setVisible(False)
            self.sigma_label.setVisible(False)
            self.sigma_spinbox.setVisible(False)
            self.calculate_button.setVisible(False)

            self.label_custom.setVisible(False)
            self.kernel_display.setReadOnly(True)
            


    def update_kernel_display(self):
        size_x = self.size_x_spinbox.value()
        size_y = self.size_y_spinbox.value()
        sigma = self.sigma_spinbox.value()

        if self.radio_gaussian.isChecked():
            self.kernel_maker.set_gaussian_kernel(size_x, size_y, sigma)
            self.selected_filter_name = "Gaussian Blur"
        elif self.radio_log.isChecked():
            self.kernel_maker.set_log_kernel(size_x, size_y, sigma)
            self.selected_filter_name = "LoG"
        elif self.radio_sobel_h.isChecked():
            self.kernel_maker.set_sobel_kernel('x')
            self.selected_filter_name = "Sobel (H)"
        elif self.radio_sobel_v.isChecked():
            self.kernel_maker.set_sobel_kernel('y')
            self.selected_filter_name = "Sobel (V)"
        elif self.radio_prewitt_h.isChecked():
            self.kernel_maker.set_prewitt_kernel('x')
            self.selected_filter_name = "Prewitt (H)"
        elif self.radio_prewitt_v.isChecked():
            self.kernel_maker.set_prewitt_kernel('y')
            self.selected_filter_name = "Prewitt (V)"
        elif self.radio_roberts_h.isChecked():
            self.kernel_maker.set_roberts_kernel('x')
            self.selected_filter_name = "Roberts (H)"
        elif self.radio_roberts_v.isChecked():
            self.kernel_maker.set_roberts_kernel('y')
            self.selected_filter_name = "Roberts (V)"
        elif self.radio_laplace.isChecked():
            self.kernel_maker.set_laplacian_kernel()
            self.selected_filter_name = "Laplace"
        elif self.radio_mean_blur.isChecked():
            self.kernel_maker.set_mean_blur_kernel(size_x, size_y)
            self.selected_filter_name = "Mean Blur"
        elif self.radio_sharpen.isChecked():
            self.kernel_maker.set_sharpen_kernel()
            self.selected_filter_name = "Sharpen"

        current_kernel = self.kernel_maker.get_kernel()
        self.kernel_display.setText(np.array2string(current_kernel, formatter={'float_kind': lambda x: "%.2f" % x}, separator=', '))

    def on_preview_clicked(self):
        if self.current_image_hsv is None:
            QMessageBox.warning(self, "Warning", "No image to edit.")
            return
        
        # Set the kernel based on the selected filter
        self.update_kernel_display()
        # Perform convolution using the current kernel on the image
        self.processed_image = self.kernel_maker.perform_convolution(self.current_image_hsv)
        
        # Display the processed image in the main application
        self.parent().display_image(self.processed_image)

    def on_apply_clicked(self):
        if self.current_image_hsv is None:
            QMessageBox.warning(self, "Warning", "No image to edit.")
            return
        self.processed_image = self.kernel_maker.perform_convolution(self.current_image_hsv)
        self.accept()

    def get_action_name(self):
        return f"Applied {self.selected_filter_name} Filter"

    def get_processed_image(self):
        return self.processed_image

class AppState:
    def __init__(self, current_image_hsv=None, original_image_hsv=None, saturation_value=0, brightness_value=0, linear_contrast_value=0, exponential_contrast_value=0, action=None, who_edited_V = None):
        self.current_image_hsv = current_image_hsv
        self.original_image_hsv = original_image_hsv
        self.saturation_value = saturation_value
        self.brightness_value = brightness_value
        self.linear_contrast_value = linear_contrast_value
        self.exponential_contrast_value = exponential_contrast_value
        self.action = action
        self.who_edited_V = who_edited_V

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Image Processing App')
        self.resize(800, 600)  # Set initial size of the window

        # Variables
        self.current_image_hsv = None
        self.original_image_hsv = None
        self.undo_stack = []
        self.is_slider_active = False
        self.who_edited_V = None

        # Menubar
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')  
        upload_action = file_menu.addAction('Upload Image')
        upload_action.triggered.connect(self.upload_image)
        save_action = file_menu.addAction('Save Image')
        save_action.triggered.connect(self.save_image)

        # Image Arithmetics menu
        arithmetics_menu = menubar.addMenu('Image Arithmetics')
        addition_action = arithmetics_menu.addAction('Addition')
        addition_action.triggered.connect(lambda: self.arithmetic_operation("Addition"))
        subtraction_action = arithmetics_menu.addAction('Subtraction')
        subtraction_action.triggered.connect(lambda: self.arithmetic_operation("Subtraction"))
        product_action = arithmetics_menu.addAction('Product')
        product_action.triggered.connect(lambda: self.arithmetic_operation("Product"))

        # Convolution filters menu
        convolution_menu = menubar.addMenu('Convolution Filters')
        convolution_action = QAction('Open Convolution Filters window', self)
        convolution_action.triggered.connect(self.open_convolution_filter_dialog)
        convolution_menu.addAction(convolution_action)

        # Binarization menu
        binarization_menu = menubar.addMenu('Binarization')
        binarization_action = QAction('Open Binarization window', self)
        binarization_action.triggered.connect(self.open_binarization_dialog)
        binarization_menu.addAction(binarization_action)
        # Canny menu
        canny_menu = menubar.addMenu('Canny')
        canny_action = QAction('Open Canny window', self)
        canny_action.triggered.connect(self.open_canny_dialog)
        canny_menu.addAction(canny_action)

        # Main Widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Left Column (25% width)
        left_column_widget = QWidget()
        left_column_layout = QVBoxLayout()
        left_column_widget.setLayout(left_column_layout)

        #left_column_label = QLabel('L_Column')
        #left_column_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        #left_column_label.setStyleSheet('background-color: lightblue;')  # Just to visualize the layout
        #left_column_layout.addWidget(left_column_label)

        vertical_spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        left_column_layout.addItem(vertical_spacer)

        self.vincent_soille_button = QPushButton("Try Watershed")
        self.vincent_soille_button.clicked.connect(self.apply_vincent_soille)
        left_column_layout.addWidget(self.vincent_soille_button)

        # Create slider for linear contrast
        self.linear_contrast_label = QLabel("Linear Contrast: a=0.0")
        self.linear_contrast_label.setToolTip("(HSV) V = 1 / [1 + e^(-a(V-0.5))] ")
        left_column_layout.addWidget(self.linear_contrast_label)

        self.linear_contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.linear_contrast_slider.setRange(-100, 600) 
        self.linear_contrast_slider.setValue(0)  # default value representing no change = 0
        self.linear_contrast_slider.setEnabled(False)
        self.linear_contrast_slider.valueChanged.connect(self.adjust_linear_contrast)
        self.linear_contrast_slider.sliderPressed.connect(self.slider_pressed)
        self.linear_contrast_slider.sliderReleased.connect(self.slider_released_linear_contrast)
        left_column_layout.addWidget(self.linear_contrast_slider)

        # Create buttons for logarithmic contrast adjustment
        self.logarithmic_contrast_label = QLabel("Logarithmic Contrast")
        self.logarithmic_contrast_label.setToolTip("(HSV) V = log(V + 1) or V = exp(V)")
        left_column_layout.addWidget(self.logarithmic_contrast_label)
        
        logarithmic_contrast_buttons_layout = QHBoxLayout()
        self.logarithmic_contrast_minus_button = QPushButton("-")
        self.logarithmic_contrast_minus_button.clicked.connect(lambda: self.adjust_logarithmic_contrast('inverse'))
        logarithmic_contrast_buttons_layout.addWidget(self.logarithmic_contrast_minus_button)

        self.logarithmic_contrast_plus_button = QPushButton("+")
        self.logarithmic_contrast_plus_button.clicked.connect(lambda: self.adjust_logarithmic_contrast('add'))
        logarithmic_contrast_buttons_layout.addWidget(self.logarithmic_contrast_plus_button)

        left_column_layout.addLayout(logarithmic_contrast_buttons_layout)

        # Create slider for exponential contrast
        self.exponential_contrast_label = QLabel("Polynomial Contrast: gamma=1.0")
        self.exponential_contrast_label.setToolTip("(HSV) V = V ** gamma")

        left_column_layout.addWidget(self.exponential_contrast_label)

        self.exponential_contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.exponential_contrast_slider.setRange(-100, 100)
        self.exponential_contrast_slider.setValue(0)  # default value representing gamma = 1.0
        self.exponential_contrast_slider.setEnabled(False)
        self.exponential_contrast_slider.valueChanged.connect(self.adjust_exponential_contrast)
        self.exponential_contrast_slider.sliderPressed.connect(self.slider_pressed)
        self.exponential_contrast_slider.sliderReleased.connect(self.slider_released_exponential_contrast)
        left_column_layout.addWidget(self.exponential_contrast_slider)

        # Apply monochromatic button
        apply_monochromatic_button = QPushButton("Apply Monochromatic")
        apply_monochromatic_button.clicked.connect(self.apply_monochromatic)
        left_column_layout.addWidget(apply_monochromatic_button)

        # Invert color (HSV) button
        invert_color_button = QPushButton("Invert Color (HSV)")
        invert_color_button.clicked.connect(self.invert_color_hsv)
        left_column_layout.addWidget(invert_color_button)

        # Apply negative (RGB) button
        apply_negative_button = QPushButton("Apply Negative (RGB)")
        apply_negative_button.clicked.connect(self.apply_negative_rgb)
        left_column_layout.addWidget(apply_negative_button)

        # Brightness label and slider
        self.brightness_label = QLabel("Brightness: 0")
        left_column_layout.addWidget(self.brightness_label)

        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setEnabled(False)
        self.brightness_slider.valueChanged.connect(self.adjust_brightness)
        self.brightness_slider.sliderPressed.connect(self.slider_pressed)
        self.brightness_slider.sliderReleased.connect(self.slider_released_brightness)
        left_column_layout.addWidget(self.brightness_slider)

        # Saturation label and slider
        self.saturation_label = QLabel("Saturation: 0")
        left_column_layout.addWidget(self.saturation_label)

        self.saturation_slider = QSlider(Qt.Orientation.Horizontal)
        self.saturation_slider.setRange(-100, 100)
        self.saturation_slider.setValue(0)
        self.saturation_slider.setEnabled(False)
        #self.saturation_slider.valueChanged.connect(lambda: self.saturation_label.setText(f"Saturation: {self.saturation_slider.value()}"))
        self.saturation_slider.valueChanged.connect(self.adjust_saturation)
        self.saturation_slider.sliderPressed.connect(self.slider_pressed)
        self.saturation_slider.sliderReleased.connect(self.slider_released_saturation)
        left_column_layout.addWidget(self.saturation_slider)

        # Histogram Section
        histogram_label = QLabel("Histogram")
        left_column_layout.addWidget(histogram_label)

        hbox_checkboxes = QHBoxLayout()
        self.checkbox_r = QCheckBox("R")
        self.checkbox_g = QCheckBox("G")
        self.checkbox_b = QCheckBox("B")
        self.checkbox_v = QCheckBox("V")
        self.checkbox_v.setChecked(True)  # Set V as the default checked option

        # Connect checkboxes to the update function
        self.checkbox_r.stateChanged.connect(self.update_histogram)
        self.checkbox_g.stateChanged.connect(self.update_histogram)
        self.checkbox_b.stateChanged.connect(self.update_histogram)
        self.checkbox_v.stateChanged.connect(self.update_histogram)

        # Add checkboxes to the hbox layout
        hbox_checkboxes.addWidget(self.checkbox_r)
        hbox_checkboxes.addWidget(self.checkbox_g)
        hbox_checkboxes.addWidget(self.checkbox_b)
        hbox_checkboxes.addWidget(self.checkbox_v)

        left_column_layout.addLayout(hbox_checkboxes)

        # Apply Histogram Equalization button
        equalize_histogram_button = QPushButton("Equalize Histogram")
        equalize_histogram_button.clicked.connect(self.equalize_histogram)
        left_column_layout.addWidget(equalize_histogram_button)

        # Add populated left column to main widget
        main_layout.addWidget(left_column_widget, 1)  # 1 part of the total space
        
        # Right Column (75% width)
        right_column_widget = QWidget()
        right_column_layout = QVBoxLayout()
        right_column_widget.setLayout(right_column_layout)
        right_column_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Upper item of right column (75% height)
        self.upper_item_label = QLabel('Upload something')
        self.upper_item_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.upper_item_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.upper_item_label.setStyleSheet('background-color: lightgreen;')  # Just to visualize the layout
        self.upper_item_label.setScaledContents(True)
        right_column_layout.addWidget(self.upper_item_label, 3)  # 3 parts of the total space
        
        # Lower item of right column (25% height)
        # Initialize PyQtGraph plot widget for histogram
        self.lower_item_histogram = pg.PlotWidget()
        self.lower_item_histogram.setBackground('w')
        self.lower_item_histogram.setXRange(0, 255)
        self.lower_item_histogram.setYRange(0, 1)
        self.lower_item_histogram.showGrid(x=True, y=True)
        #self.lower_item_histogram.setLabel('left', 'Frequency')
        #self.lower_item_histogram.setLabel('bottom', 'Pixel Value')
        #self.lower_item_histogram.setTitle("Histogram")
        #self.lower_item_histogram.setLimits(xMin=0, xMax=255, yMin=0, yMax=1)
        self.lower_item_histogram.setMouseEnabled(True, False)
        right_column_layout.addWidget(self.lower_item_histogram, 1)  # 1 part of the total space
        
        # Add populated right column to main widget
        main_layout.addWidget(right_column_widget, 3)  # 3 parts of the total space

        # Bottom Toolbar
        self.bottom_toolbar = QToolBar()
        self.bottom_toolbar.setMovable(False)
        self.bottom_toolbar.setFloatable(False)
        self.bottom_toolbar.setStyleSheet("background-color: #D6CADD;")  # Set background color to pale violet
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.bottom_toolbar)

        # Undo action for bottom toolbar
        self.undo_action = QAction('Undo', self)
        self.undo_action.triggered.connect(self.undo)
        self.update_undo_tooltip()
        self.bottom_toolbar.addAction(self.undo_action)
        # We cannot set style directly on QAction, so we need to access the QToolButton associated with it
        undo_button_widget = self.bottom_toolbar.widgetForAction(self.undo_action)
        undo_button_widget.setStyleSheet("padding: 0 10px; margin: 1 3px; background-color: #ede1ec")  # Add padding and margin and color

        # Status tip label
        self.status_tip_label = QLabel("")
        self.bottom_toolbar.addWidget(self.status_tip_label)

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.pbm *.pgm *.ppm *.pnm);;All Files (*)")
        if file_name:
            self.original_image_hsv = Image.open(file_name).convert("HSV")
            self.current_image_hsv = self.original_image_hsv.copy()
            self.display_image(self.current_image_hsv)
            self.save_state("Uploaded Image")
            self.saturation_slider.setEnabled(True)  # Enable the slider
            self.brightness_slider.setEnabled(True)  # Enable the slider
            self.linear_contrast_slider.setEnabled(True) # Enable the slider
            self.exponential_contrast_slider.setEnabled(True) # Enable the slider

    def save_image(self):
        if self.current_image_hsv is None:
            QMessageBox.warning(self, "Warning", "No image to save.")
            return
        
        file_name, _ = QFileDialog.getSaveFileName(self, 
            "Save Image File", "", 
            "PPM Files (*.ppm);;All Files (*)")
        if file_name:
            rgb_image = self.current_image_hsv.convert('RGB')
            rgb_image.save(file_name, format='PPM')

    def display_image(self, hsv_image):
        rgb_image = hsv_image.convert('RGB')
        qpixmap = self.pil_to_qpixmap(rgb_image)
        if qpixmap:
            scaled_pixmap = qpixmap.scaled(self.upper_item_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.upper_item_label.setPixmap(scaled_pixmap)

        self.update_histogram()

    def pil_to_qpixmap(self, pil_image):
        image_np = np.array(pil_image)
        height, width, channels = image_np.shape
        bytes_per_line = channels * width
        qimage = QImage(image_np.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimage)
    
    def apply_vincent_soille(self):
        result_image = self.watershed_cv2(self.current_image_hsv)
        self.display_image(result_image)
    
    def watershed_cv2(self, pil_image_hsv):
        # Convert Pillow image to RGB and then to numpy array
        pil_image_rgb = pil_image_hsv.convert('RGB')
        image = np.array(pil_image_rgb)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Otsu's thresholding
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labeling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Mark the unknown region with zero
        markers[unknown == 255] = 0

        # Apply watershed
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [255, 0, 0]

        # Convert back to Pillow image
        result_image = Image.fromarray(image)
        result_image_hsv = result_image.convert('HSV')

        return result_image_hsv

    def open_canny_dialog(self):
        dialog = CannyDialog(self, self.current_image_hsv)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.current_image_hsv = dialog.get_processed_image()
            self.save_state("Applied Canny Edge Detection")
            self.display_image(self.current_image_hsv)
        else:
            self.display_image(self.current_image_hsv)
    
    def open_binarization_dialog(self):
        dialog = BinarizationDialog(self, self.current_image_hsv)
        result = dialog.exec()
        if result == 1:
            self.current_image_hsv = dialog.get_processed_image()
            self.original_image_hsv = self.current_image_hsv.copy()
            action_name = dialog.get_action_name()
            self.save_state(action_name)
            self.display_image(self.current_image_hsv)
        else:
            self.display_image(self.current_image_hsv)

    def open_convolution_filter_dialog(self):
        dialog = ConvolutionFilterDialog(self, self.current_image_hsv)
        result = dialog.exec()
        
        if result == 1:
            self.current_image_hsv = dialog.get_processed_image()
            self.original_image_hsv = self.current_image_hsv.copy()
            action_name = dialog.get_action_name() 
            self.save_state(action_name)
            self.display_image(self.current_image_hsv)
        else:
            self.display_image(self.current_image_hsv)

    def update_histogram(self):
        if self.current_image_hsv is None:
            return

        selected_channels = self.get_selected_channels()
        if not selected_channels:
            return

        self.lower_item_histogram.clear()

        if any(channel in selected_channels for channel in ['R', 'G', 'B']):
            rgb_image = self.current_image_hsv.convert('RGB')
            rgb_array = np.array(rgb_image)

            if 'R' in selected_channels:
                R_values = rgb_array[..., 0].flatten()
                hist, bins = np.histogram(R_values, bins=256, range=(0, 255))
                hist = hist / hist.max()
                bins = bins[:-1]
                self.lower_item_histogram.plot(bins, hist, stepMode=False, fillLevel=0, brush=(255, 0, 0, 150), pen='r', name='R')  # Red color for R

            if 'G' in selected_channels:
                G_values = rgb_array[..., 1].flatten()
                hist, bins = np.histogram(G_values, bins=256, range=(0, 255))
                hist = hist / hist.max()
                bins = bins[:-1]
                self.lower_item_histogram.plot(bins, hist, stepMode=False, fillLevel=0, brush=(0, 255, 0, 150), pen='g', name='G')  # Green color for G

            if 'B' in selected_channels:
                B_values = rgb_array[..., 2].flatten()
                hist, bins = np.histogram(B_values, bins=256, range=(0, 255))
                hist = hist / hist.max()
                bins = bins[:-1]
                self.lower_item_histogram.plot(bins, hist, stepMode=False, fillLevel=0, brush=(0, 0, 255, 150), pen='b', name='B')  # Blue color for B

        if 'V' in selected_channels:
            hsv_array = np.array(self.current_image_hsv)
            V_values = hsv_array[..., 2].flatten()
            hist, bins = np.histogram(V_values, bins=256, range=(0, 255))
            hist = hist / hist.max()  # Normalize the histogram
            bins = bins[:-1]  # Remove the last bin edge
            self.lower_item_histogram.plot(bins, hist, stepMode=False, fillLevel=0, brush=(150, 150, 150, 150), pen='d', name='V')  # Grey color for V

    def histogram_equalization(self, image_hsv):
        # Convert PIL Image to numpy array
        hsv_array = np.array(image_hsv)
        
        # Separate the V channel
        v_channel = hsv_array[:, :, 2]
        
        # Compute the histogram of the V channel
        hist, bins = np.histogram(v_channel.flatten(), 256, [0, 256])
        
        # Compute the cumulative distribution function (CDF) (dystrybuanta)
        cdf = hist.cumsum()
        cdf_normalized = cdf * (255 / cdf[-1])
        
        # Use the normalized CDF to map the original V values to equalized values
        equalized_v = np.interp(v_channel.flatten(), bins[:-1], cdf_normalized)
        
        # Reshape and replace the V channel in the HSV array
        hsv_array[:, :, 2] = equalized_v.reshape(v_channel.shape)
        
        # Convert back to PIL Image and return
        return Image.fromarray(hsv_array, "HSV")
    
    def get_selected_channels(self):
        selected_channels = []
        if self.checkbox_r.isChecked():
            selected_channels.append('R')
        if self.checkbox_g.isChecked():
            selected_channels.append('G')
        if self.checkbox_b.isChecked():
            selected_channels.append('B')
        if self.checkbox_v.isChecked():
            selected_channels.append('V')
        return selected_channels

    

    
    def update_original_V_with_current_V(self):
        
        if self.original_image_hsv is None or self.current_image_hsv is None:
            QMessageBox.warning(self, "Warning", "No image to update V dimension.")
            return
        
        # Convert both images to numpy arrays
        original_hsv_array = np.array(self.original_image_hsv)
        current_hsv_array = np.array(self.current_image_hsv)
        
        # Update the V dimension in original with the V dimension from current
        original_hsv_array[..., 2] = current_hsv_array[..., 2]
        
        # Convert back to PIL image and update variable
        self.original_image_hsv = Image.fromarray(original_hsv_array, "HSV")

    def reset_V_sliders_except(self, exception=None):
        
        if exception != 'brightness':
            self.brightness_slider.blockSignals(True)
            self.brightness_slider.setValue(0)
            self.brightness_label.setText(f"Brightness: {0}")
            self.brightness_slider.blockSignals(False)
        
        if exception != 'linear_contrast':
            self.linear_contrast_slider.blockSignals(True)
            self.linear_contrast_slider.setValue(0) 
            self.linear_contrast_label.setText(f"Linear Contrast: a={0.00:.2f}")
            self.linear_contrast_slider.blockSignals(False)
        
        if exception != 'exponential_contrast':
            self.exponential_contrast_slider.blockSignals(True)
            self.exponential_contrast_slider.setValue(0)  
            self.exponential_contrast_label.setText(f"Polynomial Contrast: gamma={1.00:.2f}")
            self.exponential_contrast_slider.blockSignals(False)

    def arithmetic_operation(self, operation):
        if self.original_image_hsv is None:
            QMessageBox.warning(self, "Warning", "No base image loaded.")
            return

        file_name, _ = QFileDialog.getOpenFileName(self, 
            f"Open Image File for {operation}", "", 
            "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.pbm *.pgm *.ppm *.pnm);;All Files (*)")
        if not file_name:
            return

        second_image_hsv = Image.open(file_name).convert("HSV")
        result_image_hsv = self.perform_arithmetic(self.original_image_hsv, second_image_hsv, operation)
        self.current_image_hsv = result_image_hsv
        self.original_image_hsv = result_image_hsv
        self.display_image(self.current_image_hsv)
        self.save_state(f"Arithmetic Operation: {operation}")

    def perform_arithmetic(self, image1, image2, operation):
        rgb_image_1 = image1.convert('RGB')
        rgb_image_2 = image2.convert('RGB')
        arr1 = np.array(rgb_image_1)
        arr2 = np.array(rgb_image_2)

        if operation == "Addition":
            result_arr = np.clip(arr1 + arr2, 0, 255)
        elif operation == "Subtraction":
            result_arr = np.clip(arr1 - arr2, 0, 255)
        elif operation == "Product":
            result_arr = np.clip(arr1 * arr2 // 255, 0, 255)

        return Image.fromarray(result_arr.astype('uint8'), 'HSV')
    
    def equalize_histogram(self):
        if self.current_image_hsv is None:
            QMessageBox.warning(self, "Warning", "No image to apply histogram equalization.")
            return
        
        self.current_image_hsv = self.histogram_equalization(self.current_image_hsv)
        self.display_image(self.current_image_hsv)
        self.save_state("Applied Histogram Equalization")
    
    def adjust_linear_contrast(self):
        # Linear contrast adjustment: V = 1 / 1 + e^(-a(V-0.5))

        a = self.linear_contrast_slider.value()/100
        self.linear_contrast_label.setText(f"Linear Contrast: a={a:.2f}")

        if self.current_image_hsv is None or self.original_image_hsv is None:
            QMessageBox.warning(self, "Warning", "No image to adjust linear contrast.")
            return

        if self.who_edited_V != 'linear_contrast':
            self.update_original_V_with_current_V()
            self.reset_V_sliders_except('linear_contrast')
            
        
        hsv_array = np.array(self.current_image_hsv)
        original_hsv_array = np.array(self.original_image_hsv)
        
        adjusted_v_channel = self.adjustable_sigmoid_contrast(original_hsv_array[..., 2], a)
        hsv_array[..., 2] = np.clip(adjusted_v_channel, 0, 255)
        self.current_image_hsv = Image.fromarray(hsv_array, "HSV")
        self.display_image(self.current_image_hsv)
        self.who_edited_V = 'linear_contrast'
        
    def adjustable_sigmoid_contrast(self, v_channel, a=1):
        
        # Normalize the input
        x = v_channel / 255.0
        
        # Finetuning
        a=a/2

        # Define the steepness of the sigmoid curve, 10*a works good
        steepness = 10 * a

        # Calculate the sigmoid component with the defined steepness
        sigmoid = 1 / (1 + np.exp(-steepness * (x - 0.5)))

        # Compute alpha to manage blending based on the absolute value of a
        alpha = np.tanh(np.abs(a))

        # Blend the original and sigmoid values based on a
        output = (1 - alpha) * x + alpha * sigmoid

        # Scale back to 0-255 and convert to integer type
        return (output * 255).astype(np.uint8)
    
    
    
    def adjust_logarithmic_contrast(self, operation):
        # Logarithmic contrast adjustment: V = c * log(V + 1)
        # After testing slider for c, it was just set to 1
        if self.current_image_hsv is None or self.original_image_hsv is None:
            QMessageBox.warning(self, "Warning", "No image to adjust logarithmic contrast.")
            return        

        hsv_array = np.array(self.current_image_hsv)
        hsv_normalized = hsv_array / 255.0

        if operation == 'add':
            hsv_normalized[..., 2] = np.clip(np.log2(1+hsv_normalized[..., 2]), 0, 1)
            action_text = "Added Logarithmic Contrast"
        elif operation == 'inverse':
            hsv_normalized[..., 2] = np.clip((2 ** (hsv_normalized[..., 2]) - 1), 0, 1)
            action_text = "Inversed Logarithmic Contrast"
        else:
            return

        hsv_array = (hsv_normalized * 255).astype(np.uint8)
        self.current_image_hsv = Image.fromarray(hsv_array, "HSV")

        self.update_original_V_with_current_V
        self.reset_V_sliders_except()
        

        self.display_image(self.current_image_hsv)
        self.save_state(action_text)


    def adjust_exponential_contrast(self):
        # Exponential contrast adjustment: V = V ** gamma

        gamma = 1 + self.exponential_contrast_slider.value() / 100.0
        self.exponential_contrast_label.setText(f"Polynomial Contrast: gamma={gamma:.2f}")

        if self.current_image_hsv is None or self.original_image_hsv is None:
            QMessageBox.warning(self, "Warning", "No image to adjust exponential contrast.")
            return

        if self.who_edited_V != 'exponential_contrast':
            self.update_original_V_with_current_V()
            self.reset_V_sliders_except('exponential_contrast')
            
        hsv_array = np.array(self.current_image_hsv)
        original_hsv_array = np.array(self.original_image_hsv)
        hsv_array[..., 2] = np.clip(255 * (original_hsv_array[..., 2] / 255) ** gamma, 0, 255)
        self.current_image_hsv = Image.fromarray(hsv_array, "HSV")
        self.display_image(self.current_image_hsv)
        self.who_edited_V = 'exponential_contrast'
        


    def apply_monochromatic(self):
        if self.current_image_hsv is None:
            QMessageBox.warning(self, "Warning", "No image to apply monochromatic effect.")
            return

        hsv_array = np.array(self.current_image_hsv)

        # Convert to grayscale by setting saturation to 0
        hsv_array[..., 1] = 0
        monochromatic_image_hsv = Image.fromarray(hsv_array, 'HSV')
        self.current_image_hsv = monochromatic_image_hsv
        self.display_image(self.current_image_hsv)

        # Set saturation slider to 0 to reflect current state
        self.saturation_slider.blockSignals(True) 
        self.saturation_slider.setValue(-100)
        self.saturation_slider.blockSignals(False)
        self.saturation_label.setText(f"Saturation: {-100}")

        self.save_state("Applied Monochromatic")

    def invert_color_hsv(self):
        if self.current_image_hsv is None:
            QMessageBox.warning(self, "Warning", "No image to apply color inversion.")
            return

        hsv_array = np.array(self.current_image_hsv)
        # Invert the Hue (H) channel by adding 128 (180 degrees equivalent in 0-255 range)
        hsv_array[:, :, 0] = (hsv_array[:, :, 0] + 128) % 256  
        inverted_image_hsv = Image.fromarray(hsv_array, 'HSV')
        self.current_image_hsv = inverted_image_hsv
        self.display_image(self.current_image_hsv)
        self.save_state("Inverted Color (HSV)")

    def apply_negative_rgb(self):
        if self.current_image_hsv is None:
            QMessageBox.warning(self, "Warning", "No image to apply negative.")
            return

        rgb_image = self.current_image_hsv.convert('RGB')
        arr = np.array(rgb_image)
        negative_arr = 255 - arr
        negative_image = Image.fromarray(negative_arr, "RGB")
        
        self.current_image_hsv = negative_image.convert("HSV")
        self.original_image_hsv = self.current_image_hsv.copy()
        self.display_image(self.current_image_hsv)
        self.save_state("Applied Negative (RGB)")

    def adjust_saturation(self):
        if self.current_image_hsv is None or self.original_image_hsv is None:
            QMessageBox.warning(self, "Warning", "No image to adjust saturation.")
            return

        value = self.saturation_slider.value()
        self.saturation_label.setText(f"Saturation: {value}")

        hsv_array_orig = np.array(self.original_image_hsv)
        factor = (value + 100) / 100
        new_saturation = np.clip((hsv_array_orig[:, :, 1] * factor), 0, 255) 

        hsv_array_current = np.array(self.current_image_hsv)
        hsv_array_current[:, :, 1] = new_saturation.astype(np.uint8)
        self.current_image_hsv = Image.fromarray(hsv_array_current, "HSV")
        self.display_image(self.current_image_hsv)

    def adjust_brightness(self):
        if self.current_image_hsv is None or self.original_image_hsv is None:
            QMessageBox.warning(self, "Warning", "No image to adjust brightness.")
            return
        
        # Check if the last method was us (brightness) or not
        if self.who_edited_V != 'brightness':
            # Update the V dimension of original_image_hsv to the current state, so we won't cancel out previous changes to V (e.g. contrast)
            self.update_original_V_with_current_V()
            self.reset_V_sliders_except('brightness')
            

        value = self.brightness_slider.value()
        self.brightness_label.setText(f"Brightness: {value}")

        hsv_array_orig = np.array(self.original_image_hsv)
        #factor = (value + 100) / 100
        shift = value/100 * 255
        #new_brightness = np.clip((hsv_array_orig[:, :, 2] * factor), 0, 255) 
        new_brightness = np.clip((hsv_array_orig[:, :, 2] + shift), 0, 255) 

        hsv_array_current = np.array(self.current_image_hsv)
        hsv_array_current[:, :, 2] = new_brightness.astype(np.uint8)
        self.current_image_hsv = Image.fromarray(hsv_array_current, "HSV")
        self.display_image(self.current_image_hsv)
        self.who_edited_V = 'brightness'

    def slider_pressed(self):
        self.is_slider_active = True

    def slider_released_linear_contrast(self):
        if self.is_slider_active:
            a = 1 + self.linear_contrast_slider.value() / 100.0
            self.adjust_linear_contrast()  # Apply the adjustment only when the slider is released
            self.save_state(f"Adjusted Linear Contrast: a={a:.2f}")
            self.is_slider_active = False

    def slider_released_exponential_contrast(self):
        if self.is_slider_active:
            gamma = 1 + self.exponential_contrast_slider.value() / 100.0
            self.adjust_exponential_contrast()  # Apply the adjustment only when the slider is released
            self.save_state(f"Adjusted Polynomial Contrast: gamma={gamma:.2f}")
            self.is_slider_active = False

    def slider_released_saturation(self):
        if self.is_slider_active:
            value = self.saturation_slider.value()
            self.adjust_saturation()  # Apply the adjustment only when the slider is released
            self.save_state(f"Adjusted Saturation: {value}")
            self.is_slider_active = False

    def slider_released_brightness(self):
        if self.is_slider_active:
            value = self.brightness_slider.value()
            self.adjust_brightness()  # Apply the adjustment only when the slider is released
            self.save_state(f"Adjusted Brightness: {value}")
            self.is_slider_active = False

    def undo(self):
        if len(self.undo_stack) > 1:
            self.undo_stack.pop()
            state = self.undo_stack[-1]
            self.load_state(state)

        else:
            QMessageBox.warning(self, "Warning", "No more actions to undo.")

    def update_undo_tooltip(self):
        tooltip_text = "Undo Stack:\n" + "\n".join([state.action for state in self.undo_stack])
        self.bottom_toolbar.setToolTip(tooltip_text)

    def save_state(self, action):
        if len(self.undo_stack) >= 10:
            self.undo_stack.pop(0)

        state_copy = AppState(
            current_image_hsv=self.current_image_hsv.copy() if self.current_image_hsv else None,
            original_image_hsv=self.original_image_hsv.copy() if self.original_image_hsv else None,
            saturation_value=self.saturation_slider.value(),
            brightness_value=self.brightness_slider.value(),
            linear_contrast_value=self.exponential_contrast_slider.value(),
            exponential_contrast_value=self.exponential_contrast_slider.value(),
            action=action,
            who_edited_V=self.who_edited_V
        )

        self.undo_stack.append(state_copy)
        self.status_tip_label.setText(action)  # Show the latest action in the status tip label
        self.update_undo_tooltip()
    
    def load_state(self, state):
        self.current_image_hsv = state.current_image_hsv
        self.original_image_hsv = state.original_image_hsv
        self.saturation_slider.blockSignals(True)  # Block signals to prevent triggering valueChanged event
        self.brightness_slider.blockSignals(True)  # Block signals to prevent triggering valueChanged event
        self.linear_contrast_slider.blockSignals(True)
        self.exponential_contrast_slider.blockSignals(True)
        self.saturation_slider.setValue(state.saturation_value)
        self.brightness_slider.setValue(state.brightness_value)
        self.linear_contrast_slider.setValue(state.linear_contrast_value)
        self.exponential_contrast_slider.setValue(state.exponential_contrast_value)
        self.saturation_slider.blockSignals(False)
        self.brightness_slider.blockSignals(False)
        self.linear_contrast_slider.blockSignals(False)
        self.exponential_contrast_slider.blockSignals(False)
        self.saturation_label.setText(f"Saturation: {state.saturation_value}")
        self.brightness_label.setText(f"Brightness: {state.brightness_value}")
        self.linear_contrast_label.setText(f"Linear Contrast: a={state.linear_contrast_value:.2f}")
        self.exponential_contrast_label.setText(f"Polynomial Contrast: gamma={state.exponential_contrast_value:.2f}")
        self.status_tip_label.setText(state.action)
        self.who_edited_V = state.who_edited_V
        self.update_undo_tooltip()
        self.display_image(self.current_image_hsv)

            
            


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
