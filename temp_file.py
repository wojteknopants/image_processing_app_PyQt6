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

