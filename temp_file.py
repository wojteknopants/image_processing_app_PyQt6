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
