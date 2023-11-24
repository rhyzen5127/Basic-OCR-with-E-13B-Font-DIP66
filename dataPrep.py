from PIL import Image
import numpy as np
import math
import cmath
import sys
import random
import os
import re
class ImageManager:
    global img
    global data
    global original
    global width
    global height
    global bitDepth
    #attributes
    width = None
    height = None
    bitDepth = None
    img = None
    data = None
    original = None
    def read(self, fileName):
        global img
        global data
        global original
        global width
        global height
        global bitDepth
        img = Image.open(fileName)
        data = np.array(img)
        original = np.copy(data)
        width = data.shape[0]
        height = data.shape[1]

        mode_to_bpp = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "YCbCr": 24, "LAB": 24, "HSV": 24, "I": 32, "F": 32}
        bitDepth = mode_to_bpp[img.mode]

        self.img = img
        self.data = data
        self.original = original
        self.width = width
        self.height = height
        self.bitDepth = bitDepth

        print("Image %s with %s x %s pixels (%s bits per pixel) has been read!" % (img.filename, width, height, bitDepth))


    def write(self, fileName):
        global img
        img = Image.fromarray(data)
        try:
            img.save(fileName)
        except:
            print("Write file error")
        else:
            print("Image %s has been written!" % (fileName))

    def restoreToOriginal(self):
        global data
        width = original.shape[0]
        height = original.shape[1]
        data = np.zeros([width, height, 3])
        data = np.copy(original)

    def resize(self, newX, newY):
        """
        Resizes the image to newX width and newY height.
        """
        global img
        global data
        if img is not None and newX > 0 and newY > 0:
            # Calculate scale factors
            scaleX = newX / self.width
            scaleY = newY / self.height

            # New image data array
            new_data = np.zeros((newX, newY, data.shape[2]), dtype=data.dtype)

            for x in range(newX):
                for y in range(newY):
                    # Mapping the new coordinates to the original image coordinates
                    original_x = min(int(x / scaleX), self.width - 1)
                    original_y = min(int(y / scaleY), self.height - 1)

                    # Copy the pixel value
                    new_data[x, y] = data[original_x, original_y]

            # Update global variables
            data = new_data
            img = Image.fromarray(data)
            print(f"Image has been resized to {newX}x{newY} pixels!")
        else:
            print("Invalid new dimensions or no image loaded.")

    def resizeNearestNeighbour(self, scaleX, scaleY):
        global data
        global width
        global height
        newWidth = (int)(round(width * scaleX))
        newHeight = (int)(round(height * scaleY))

        data_temp = np.zeros([width, height, 3])
        data_temp = data.copy()

        data = np.resize(data, [newWidth, newHeight, 3])

        for y in range(newHeight):
            for x in range(newWidth):
                xNearest = (int)(round(x / scaleX))
                yNearest = (int)(round(y / scaleY))

                xNearest = width - 1 if xNearest >= width else xNearest
                xNearest = 0 if xNearest < 0 else xNearest

                yNearest = height - 1 if yNearest >= height else yNearest
                yNearest = 0 if yNearest < 0 else yNearest

                data[x, y, :] = data_temp[xNearest, yNearest, :]

    def resizeBilinear(self, scaleX, scaleY):
        global data
        global width
        global height
        newWidth = (int)(round(width * scaleX))
        newHeight = (int)(round(height * scaleY))

        data_temp = np.zeros([width, height, 3])
        data_temp = data.copy()

        data = np.resize(data, [newWidth, newHeight, 3])

        for y in range(newHeight):
            for x in range(newWidth):
                oldX = x / scaleX
                oldY = y / scaleY

                #get 4 coordinates
                x1 = min((int)(np.floor(oldX)), width - 1)
                y1 = min((int)(np.floor(oldY)), height - 1)
                x2 = min((int)(np.ceil(oldX)), width - 1)
                y2 = min((int)(np.ceil(oldY)), height - 1)

                #get colours
                color11 = np.array(data_temp[x1, y1, :])
                color12 = np.array(data_temp[x1, y2, :])
                color21 = np.array(data_temp[x2, y1, :])
                color22 = np.array(data_temp[x2, y2, :])

                #interpolate x
                P1 = (x2 - oldX) * color11 + (oldX - x1) * color21
                P2 = (x2 - oldX) * color12 + (oldX - x1) * color22

                if x1 == x2:
                    P1 = color11
                    P2 = color22

                #interpolate y
                P = (y2 - oldY) * P1 + (oldY - y1) * P2

                if y1 == y2:
                    P = P1

                P = np.round(P)

                data[x, y, :] = P
    
    def adjustSharpness(self, amount):
        """
        GPT 4.0
        Adjusts the sharpness of the image.
        :param amount: The amount of sharpness to apply. Higher values mean more sharpness.
        """

        # Step 1: Create a blurred version of the image
        blurredImage = self.applyGaussianBlur()

        # Step 2: Create the unsharp mask by subtracting the blurred image from the original
        unsharpMask = self.data - blurredImage

        # Step 3: Add the unsharp mask to the original image, scaled by the amount of sharpness
        self.data = np.clip(self.data + amount * unsharpMask, 0, 255)

    def applyGaussianBlur(self, kernelSize=5):
        """
        GPT 4.0
        Applies a simple averaging blur (as a stand-in for Gaussian blur) to the image.
        :param kernelSize: Size of the averaging kernel. Default is 5.
        :return: Blurred image data.
        """
        kernel = np.ones((kernelSize, kernelSize), np.float32) / (kernelSize**2)
        return self.convolve(kernel)

    def convolve(self, kernel):
        """
        GPT 4.0
        Applies convolution with the given kernel to the image.
        :param kernel: The convolution kernel.
        :return: The convolved image data.
        """
        kernelHeight, kernelWidth = kernel.shape
        paddedImage = np.pad(self.data, ((kernelHeight // 2, kernelHeight // 2), 
                                         (kernelWidth // 2, kernelWidth // 2), 
                                         (0, 0)), mode='constant', constant_values=0)

        # Assuming self.data is a numpy array of shape (height, width, channels)
        height, width, channels = self.data.shape
        blurredImage = np.zeros_like(self.data)

        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    blurredImage[y, x, c] = (kernel * paddedImage[y:y+kernelHeight, x:x+kernelWidth, c]).sum()

        return blurredImage

    def convertToGrayscale(self):
        global data
        for y in range(height):
            for x in range(width):
                data[x, y, 0] = int(0.2989 * data[x, y, 0] + 0.5870 * data[x, y, 1] + 0.1140 * data[x, y, 2])
                data[x, y, 1] = int(0.2989 * data[x, y, 0] + 0.5870 * data[x, y, 1] + 0.1140 * data[x, y, 2])
                data[x, y, 2] = int(0.2989 * data[x, y, 0] + 0.5870 * data[x, y, 1] + 0.1140 * data[x, y, 2])

    def thresholding(self, threshold):

        global data
        self.convertToGrayscale()

        for y in range(height):
            for x in range(width):
                gray = data[x, y, 0]
                gray = 0 if gray < threshold else 255
                data[x, y, 0] = gray
                data[x, y, 1] = gray
                data[x, y, 2] = gray
    
    def otsuThreshold(self):
        global data
        self.convertToGrayscale()
        histogram = np.zeros(256)
        for y in range(height):
            for x in range(width):
                histogram[data[x, y, 0]] += 1
        histogramNorm = np.zeros(len(histogram))
        pixelNum = width * height
        for i in range(len(histogramNorm)):
            histogramNorm[i] = histogram[i] / pixelNum
        histogramCS = np.zeros(len(histogram))
        histogramMean = np.zeros(len(histogram))
        for i in range(len(histogramNorm)):
            if (i == 0):
                histogramCS[i] = histogramNorm[i]
                histogramMean[i] = 0
            else:
                histogramCS[i] = histogramCS[i - 1] + histogramNorm[i]
                histogramMean[i] = histogramMean[i - 1] + histogramNorm[i] * i

        globalMean = histogramMean[len(histogramMean) - 1]
        max = sys.float_info.min
        maxVariance = sys.float_info.min
        countMax = 0
        for i in range(len(histogramCS)):
            if (histogramCS[i] < 1 and histogramCS[i] > 0):
                variance = ((globalMean * histogramCS[i] - histogramMean[i]) ** 2) / (histogramCS[i] * (1 - histogramCS[i]))
            if (variance > maxVariance):
                maxVariance = variance
                max = i
                countMax = 1
            elif (variance == maxVariance):
                countMax = countMax + 1
                max = ((max * (countMax - 1)) + i) / countMax
        self.thresholding(round(max))

    def crop(self, x1, y1, x2, y2):
        
        global img
        global data
        if img is not None:
            if 0 <= x1 < self.width and 0 <= y1 < self.height and 0 <= x2 < self.width and 0 <= y2 < self.height:
                img = img.crop((x1, y1, x2, y2))
                data = np.array(img)
                print(f"Image has been cropped to coordinates ({x1}, {y1}), ({x2}, {y2})!")
            else:
                print("Invalid crop coordinates.")
        else:
            print("No image loaded to crop.")

    def rotate(self, degree):
        """
        Rotates the image by a specified degree between 0 and 360, using bilinear interpolation.
        """
        global img, data
        if img is not None:
            radians = math.radians(degree)
            cos_angle = math.cos(radians)
            sin_angle = math.sin(radians)

            new_height = int(abs(self.height * cos_angle) + abs(self.width * sin_angle))
            new_width = int(abs(self.width * cos_angle) + abs(self.height * sin_angle))

            new_data = np.zeros((new_width, new_height, data.shape[2]), dtype=data.dtype)

            cx, cy = self.width // 2, self.height // 2
            new_cx, new_cy = new_width // 2, new_height // 2

            for x in range(new_width):
                for y in range(new_height):
                    # Inverse mapping to original coordinates
                    original_x = (x - new_cx) * cos_angle + (y - new_cy) * sin_angle + cx
                    original_y = -(x - new_cx) * sin_angle + (y - new_cy) * cos_angle + cy

                    # Bilinear interpolation
                    if 0 <= original_x < self.width - 1 and 0 <= original_y < self.height - 1:
                        int_x, int_y = int(original_x), int(original_y)
                        delta_x, delta_y = original_x - int_x, original_y - int_y

                        # Interpolation weights
                        top_left = (1 - delta_x) * (1 - delta_y)
                        top_right = delta_x * (1 - delta_y)
                        bottom_left = (1 - delta_x) * delta_y
                        bottom_right = delta_x * delta_y

                        # Interpolate each channel
                        for c in range(data.shape[2]):
                            tl = data[int_x, int_y, c]
                            tr = data[int_x + 1, int_y, c]
                            bl = data[int_x, int_y + 1, c]
                            br = data[int_x + 1, int_y + 1, c]

                            interpolated_value = (tl * top_left + tr * top_right + bl * bottom_left + br * bottom_right)
                            new_data[x, y, c] = np.clip(interpolated_value, 0, 255)

            data = new_data
            img = Image.fromarray(data)
            print(f"Image has been manually rotated by {degree} degrees!")
        else:
            print("No image loaded to rotate.")

    def getGrayscaleHistogram(self):
        self.convertToGrayscale()
        histogram = np.array([0] * 256)

        for y in range(height):
            for x in range(width):
                histogram[data[x, y, 0]] += 1

        self.restoreToOriginal()
        return histogram

    def getContrast(self):
        contrast = 0.0
        histogram = self.getGrayscaleHistogram()
        avgIntensity = 0.0
        pixelNum = width * height

        for i in range(len(histogram)):
            avgIntensity += histogram[i] * i

        avgIntensity /= pixelNum

        for y in range(height):
            for x in range(width):
                contrast += (data[x, y, 0] - avgIntensity) ** 2

        contrast = (contrast / pixelNum) ** 0.5

        return contrast
    
    def adjustContrast(self, contrast):
        global data
        currentContrast = self.getContrast()

        histogram = self.getGrayscaleHistogram()
        avgIntensity = 0.0
        pixelNum = width * height

        for i in range(len(histogram)):
            avgIntensity += histogram[i] * i

        avgIntensity /= pixelNum
        min = avgIntensity - currentContrast
        max = avgIntensity + currentContrast
        newMin = avgIntensity - currentContrast - contrast / 2
        newMax = avgIntensity + currentContrast + contrast / 2
        newMin = 0 if newMin < 0 else newMin
        newMax = 0 if newMax < 0 else newMax
        newMin = 255 if newMin > 255 else newMin
        newMax = 255 if newMax > 255 else newMax

        if (newMin > newMax):
            temp = newMax
            newMax = newMin
            newMin = temp

        contrastFactor = (newMax - newMin) / (max - min)

        for y in range(height):
            for x in range(width):
                r = data[x, y, 0]
                g = data[x, y, 1]
                b = data[x, y, 2]
                contrast += (data[x, y, 0] - avgIntensity) ** 2
                r = (int)((r - min) * contrastFactor + newMin)
                r = 255 if r > 255 else r
                r = 0 if r < 0 else r

                g = (int)((g - min) * contrastFactor + newMin)
                g = 255 if g > 255 else g
                g = 0 if g < 0 else g
                b = (int)((b - min) * contrastFactor + newMin)
                b = 255 if b > 255 else b
                b = 0 if b < 0 else b

                data[x, y, 0] = r
                data[x, y, 1] = g
                data[x, y, 2] = b


path = 'C:/Users/RhyzenWorkspace/OneDrive/ocr/test/'

im = ImageManager()
im.read(f'{path}FinalDIP66.bmp')

im.convertToGrayscale()
im.rotate(-31.2)
im.adjustSharpness(5)
im.write(f'{path}rotate.bmp')

im.thresholding(200)
im.write(f'{path}threshold.bmp')

im.read(f'{path}threshold.bmp')
im.crop(313, 317, 355, 373)
im.write(f'{path}0.png')

im.read(f'{path}0.png')
im.resize(28, 28)
im.write(f'{path}0.png')

im.read(f'{path}threshold.bmp')
im.crop(364, 317, 394, 373)
im.write(f'{path}1.png')

im.read(f'{path}1.png')
im.resize(28, 28)
im.write(f'{path}1.png')

im.read(f'{path}threshold.bmp')
im.crop(394, 320, 436, 376)
im.write(f'{path}2.png')

im.read(f'{path}2.png')
im.resize(28, 28)
im.write(f'{path}2.png')

im.read(f'{path}threshold.bmp')
im.crop(436, 320, 471, 376)
im.write(f'{path}3.png')


im.read(f'{path}3.png')
im.resize(28, 28)
im.write(f'{path}3.png')

im.read(f'{path}threshold.bmp')
im.crop(471, 320, 510, 376)
im.write(f'{path}4.png')

im.read(f'{path}4.png')
im.resize(28, 28)
im.write(f'{path}4.png')

im.read(f'{path}threshold.bmp')
im.crop(510, 320, 543, 376)
im.write(f'{path}5.png')

im.read(f'{path}5.png')
im.resize(28, 28)
im.write(f'{path}5.png')

# im = ImageManager()
# im.read("C:/Users/Rhyzen/OneDrive/ocr/test/FinalDIP66.bmp")

# im.rotate(-31.2)
# im.write('C:/Users/Rhyzen/OneDrive/ocr/test/rotate.bmp')

# crop_coordinates = [(313, 317, 355, 373), (364, 317, 394, 373), 
#                     (394, 317, 436, 373), (436, 317, 471, 373),
#                     (471, 317, 510, 373), (510, 317, 543, 373)]

# for i, (x1, y1, x2, y2) in enumerate(crop_coordinates):
#     im.read("C:/Users/Rhyzen/OneDrive/ocr/test/rotate.bmp")

#     im.crop(x1, y1, x2, y2)

#     im.resize(28, 28)

#     file_path = f"C:/Users/Rhyzen/OneDrive/ocr/test/{i}.bmp"
#     im.write(file_path)