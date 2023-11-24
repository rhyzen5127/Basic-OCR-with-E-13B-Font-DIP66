from PIL import Image
import numpy as np
import math
import cmath
import sys
import random
import os
import re

class FrequencyDomainManager:
  data = None
  width = None
  height = None
  imgWidth = None
  imgHeight = None
  original = None
  im = None

  def __init__(self, im):
    self.im = im
    self.imgWidth = im.data.shape[0]
    self.imgHeight = im.data.shape[1]
    self.width = self.nextPowerOf2(self.imgWidth)
    self.height = self.nextPowerOf2(self.imgHeight)
    self.data = np.zeros((self.width, self.height), dtype=np.complex_)

    for y in range(self.imgHeight):
      for x in range(self.imgWidth):
        gray = im.data[x, y, 0]
        self.data[x, y] = gray + 0j

    self.fft2d(False)
    self.shifting()

    self.original = np.copy(self.data)

  def shifting(self):
    halfWidth = self.width // 2
    halfHeight = self.height // 2

    self.data = np.roll(self.data, halfHeight, axis = 0)
    self.data = np.roll(self.data, halfWidth, axis = 1)

  def nextPowerOf2(self, a):
    b = 1
    while (b < a):
      b = b << 1
    return b

  def fft(self, x):

    n = self.nextPowerOf2(len(x))

    # base case
    if (n == 1):
      return x

    # radix 2 Cooley-Tukey FFT
    # even terms
    evenFFT = np.array(self.fft(x[0::2]), dtype=np.complex_)

    # odd terms
    oddFFT = np.array(self.fft(x[1::2]), dtype=np.complex_)

    # compute FFT
    factor = np.array([math.cos(-2 * k * math.pi / n) + math.sin(-2 * k * math.pi / n) * 1j for k in range(n // 2)], dtype=np.complex_)
    factor = factor * oddFFT
    return [evenFFT[k] + factor[k] for k in range(n // 2)] + [evenFFT[k] - factor[k] for k in range(n // 2)]

  def fft2d(self, invert):
    # horizontal first
    if (not invert):
      self.data = [self.fft(row) for row in self.data]
    else:
      self.data = [self.ifft(row) for row in self.data]

    self.data = np.transpose(self.data)
    # then vertical
    if (not invert):
      self.data = [self.fft(row) for row in self.data]
    else:
      self.data = [self.ifft(row) for row in self.data]

    self.data = np.transpose(self.data)

  def writeSpectrumLogScaled(self, fileName):
    temp = np.zeros((self.height, self.width, 3))
    spectrum = np.absolute(self.data)
    max = np.max(spectrum)
    min = np.min(spectrum)
    min = 0 if min < 1 else math.log10(min)
    max = 0 if max < 1 else math.log10(max)

    for y in range(self.height):
      for x in range(self.width):
        spectrumV = spectrum[x, y]
        spectrumV = 0 if spectrumV < 1 else math.log10(spectrumV)
        spectrumV = ((spectrumV - min) * 255 / (max - min))

        spectrumV = 255 if spectrumV > 255 else spectrumV
        spectrumV = 0 if spectrumV < 0 else spectrumV

        temp[x, y, 0] = spectrumV
        temp[x, y, 1] = spectrumV
        temp[x, y, 2] = spectrumV

    img = Image.fromarray(temp.astype(np.uint8))
    try:
      img.save(fileName)
    except:
      print("Write file error")
    else:
      print("Image %s has been written!" % (fileName))

  def writePhase(self, fileName):

    temp = np.zeros((self.height, self.width, 3))

    phase = np.angle(self.data)

    max = np.max(phase)
    min = np.min(phase)

    for y in range(self.height):
      for x in range(self.width):
        phaseV = phase[x,y]
        phaseV = ((phaseV - min) * 255 / (max - min))

        phaseV = 255 if phaseV > 255 else phaseV
        phaseV = 0 if phaseV < 0 else phaseV

        temp[x, y, 0] = phaseV
        temp[x, y, 1] = phaseV
        temp[x, y, 2] = phaseV

    img = Image.fromarray(temp.astype(np.uint8))
    try:
      img.save(fileName)
    except:
      print("Write file error")
    else:
      print("Image %s has been written!" % (fileName))

  def ifft(self, x):
    n = len(x)

    # conjugate then fft for the inverse
    x = np.conjugate(x)
    x = self.fft(x)
    x = np.conjugate(x)

    x = x / n

    return x

  def getInverse(self):

    self.shifting()
    self.fft2d(True)

    dataRe = np.real(self.data)
    for y in range(self.height):
      for x in range(self.width):
        color = dataRe[x, y]
        color = 255 if color > 255 else color
        color = 0 if color < 0 else color
        self.im.data[x, y, 0] = color
        self.im.data[x, y, 1] = color
        self.im.data[x, y, 2] = color

  def ILPF(self, radius):
    if (radius <= 0 or radius > min(self.width/2, self.height/2)):
      print("INVALID Radius!")
      return

    centerX = self.width // 2
    centerY = self.height // 2

    for y in range(self.height):
      for x in range(self.width):
        if ((x - centerX) ** 2 + (y - centerY) ** 2 > radius ** 2):
          self.data[x, y] = 0 + 0j

  def IHPF(self, radius):
    if (radius <= 0 or radius > min(self.width/2, self.height/2)):
      print("INVALID Radius!")
      return

    centerX = self.width // 2
    centerY = self.height // 2

    for y in range(self.height):
      for x in range(self.width):
        if ((x - centerX) ** 2 + (y - centerY) ** 2 <= radius ** 2):
          self.data[x, y] = 0 + 0j

class StructuringElement:
  elements = None
  width = 0
  height = 0
  origin = None
  ignoreElements = None
  def __init__(self, width, height, origin):
    self.width = width
    self.height = height
    if (origin.real < 0 or origin.real >= width or origin.imag < 0 or origin.imag >= height):
      self.origin = complex(0, 0)
    else:
      self.origin = origin
    self.elements = np.zeros([width, height])
    self.ignoreElements = []


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

    def convertToRed(self):
        global data
        for y in range(height):
            for x in range(width):
                data[x, y, 1] = 0
                data[x, y, 2] = 0

    def convertToGreen(self):
        global data
        for y in range(height):
            for x in range(width):
                data[x, y, 0] = 0
                data[x, y, 2] = 0

    def convertToBlue(self):
        global data
        for y in range(height):
            for x in range(width):
                data[x, y, 0] = 0
                data[x, y, 1] = 0

    def convertToGrayscale(self):
        global data
        for y in range(height):
            for x in range(width):
                data[x, y, 0] = int(0.2989 * data[x, y, 0] + 0.5870 * data[x, y, 1] + 0.1140 * data[x, y, 2])
                data[x, y, 1] = int(0.2989 * data[x, y, 0] + 0.5870 * data[x, y, 1] + 0.1140 * data[x, y, 2])
                data[x, y, 2] = int(0.2989 * data[x, y, 0] + 0.5870 * data[x, y, 1] + 0.1140 * data[x, y, 2])

    def restoreToOriginal(self):
        global data
        width = original.shape[0]
        height = original.shape[1]
        data = np.zeros([width, height, 3])
        data = np.copy(original)

    def setRGB(self, x, y, R, G, B):
        self.data[y, x, 0] = R
        self.data[y, x, 1] = G
        self.data[y, x, 2] = B

    def adjustBrightness(self, brightness):
        global data
        for y in range(height):
            for x in range(width):
                r = data[x, y, 0]
                g = data[x, y, 1]
                b = data[x, y, 2]
                r = r + brightness
                r = 255 if r > 255 else r
                r = 0 if r < 0 else r

                g = g + brightness
                g = 255 if g > 255 else g
                g = 0 if g < 0 else g
                b = b + brightness
                b = 255 if b > 255 else b
                b = 0 if b < 0 else b
                data[x, y, 0] = r
                data[x, y, 1] = g
                data[x, y, 2] = b

    def invert(self):
        global data
        for y in range(height):
            for x in range(width):
                r = data[x, y, 0]
                g = data[x, y, 1]
                b = data[x, y, 2]
                r = 255 - r
                g = 255 - g
                b = 255 - b
                data[x, y, 0] = r
                data[x, y, 1] = g
                data[x, y, 2] = b

    def getHistogram(self):
        histogram = np.array([0] * 256)

        for y in range(height):
            for x in range(width):
                histogram[data[x, y, 0]] += 1

        self.restoreToOriginal()
        return histogram

    def getGrayscaleHistogram(self):
        self.convertToGrayscale()
        histogram = np.array([0] * 256)

        for y in range(height):
            for x in range(width):
                histogram[data[x, y, 0]] += 1

        self.restoreToOriginal()
        return histogram

    def writeHistogramToCSV(self, histogram, fileName):
        histogram.tofile(fileName,sep=',',format='%s')

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

    def adjustGamma(self, gamma):
        global data
        for y in range(height):
            for x in range(width):
                r = data[x, y, 0]
                g = data[x, y, 1]
                b = data[x, y, 2]

                r /= 255.0
                g /= 255.0
                b /= 255.0

                r = (int)((r**gamma) * 255)
                r = min(255, max(0, r))

                g = (int)((g**gamma) * 255)
                g = min(255, max(0, g))

                b = (int)((b**gamma) * 255)
                b = min(255, max(0, b))

                data[x, y, 0] = r
                data[x, y, 1] = g
                data[x, y, 2] = b

    def averagingFilter(self, size):
        global data
        if (size % 2 == 0):
            print("Size Invalid: must be odd number!")
            return
        data_zeropaded = np.zeros([width + int(size/2) * 2, height + int(size/2) * 2, 3])
        data_zeropaded[int(size/2):width + int(size/2), int(size/2):height + int(size/2), :] = data
        for y in range(int(size/2), int(size/2) + height):
            for x in range(int(size/2), int(size/2) + width):
                subData = data_zeropaded[x - int(size/2):x + int(size/2) + 1, y - int(size/2):y + int(size/2) + 1, :]
                avgRed = np.mean(subData[:,:,0:1])
                avgGreen = np.mean(subData[:,:,1:2])
                avgBlue = np.mean(subData[:,:,2:3])
                avgRed = 255 if avgRed > 255 else avgRed
                avgRed = 0 if avgRed < 0 else avgRed
                avgGreen = 255 if avgGreen > 255 else avgGreen
                avgGreen = 0 if avgGreen < 0 else avgGreen
                avgBlue = 255 if avgBlue > 255 else avgBlue
                avgBlue = 0 if avgBlue < 0 else avgBlue
                data[x - int(size/2), y - int(size/2), 0] = avgRed
                data[x - int(size/2), y - int(size/2), 1] = avgGreen
                data[x - int(size/2), y - int(size/2), 2] = avgBlue

    def medianFilter(self, size):
        global data

        if (size % 2 == 0):
            print("Size Invalid: must be odd number!")
            return

        paddedData = np.zeros([width + size - 1, height + size - 1, 3], dtype=np.uint8)
        paddedData[int((size-1)/2):width + int((size-1)/2), int((size-1)/2):height + int((size-1)/2), :] = data

        output = np.zeros_like(data)

        for y in range(height):
            for x in range(width):
                subData = paddedData[x:x + size, y:y + size, :]

                medRed = np.median(subData[:, :, 0])
                medGreen = np.median(subData[:, :, 1])
                medBlue = np.median(subData[:, :, 2])

                output[x, y, 0] = int(medRed)
                output[x, y, 1] = int(medGreen)
                output[x, y, 2] = int(medBlue)

        data = output

    def unsharpMasking(self, size, k=1):
        global data
        global original

        if size not in [3, 7, 15]:
            print("Size Invalid: Only 3x3, 7x7, or 15x15 are allowed!")
            return
        filteredData = np.copy(data)
        self.averagingFilter(size)
        filteredData = data
        data = np.copy(original)

        detail_mask = data - filteredData

        sharpened = data + k * detail_mask
        sharpened = np.clip(sharpened, 0, 255)

        data = sharpened.astype(np.uint8)

    def getFrequencyDomain(self):
        self.convertToGrayscale()
        fft = FrequencyDomainManager(self)
        self.restoreToOriginal()
        return fft

    def addSaltNoise(self, percent):
        global data
        noOfPX = height * width
        noiseAdded = (int)(percent * noOfPX)
        whiteColor = 255
        for i in range(noiseAdded):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            data[x, y, 0] = whiteColor
            data[x, y, 1] = whiteColor
            data[x, y, 2] = whiteColor

    def addPepperNoise(self, percent):
        global data
        noOfPX = height * width
        noiseAdded = (int)(percent * noOfPX)
        blackColor = 0
        for i in range(noiseAdded):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            data[x, y, 0] = blackColor
            data[x, y, 1] = blackColor
            data[x, y, 2] = blackColor

    def addSaltAndPepperNoise(self, percent):
        global data
        noOfPX = height * width

        noiseAdded = (int)(percent * noOfPX / 100)

        whiteColor = 255
        blackColor = 0

        for i in range(noiseAdded):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            data[x, y, 0] = whiteColor
            data[x, y, 1] = whiteColor
            data[x, y, 2] = whiteColor

        for i in range(noiseAdded):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            data[x, y, 0] = blackColor
            data[x, y, 1] = blackColor
            data[x, y, 2] = blackColor


    def addUniformNoise(self, percent, distribution):
        global data
        noOfPX = height * width
        noiseAdded = (int)(percent * noOfPX)
        for i in range(noiseAdded):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            gray = data[x, y, 0]

            gray += (random.randint(0, distribution * 2 - 1) - distribution)

            gray = 255 if gray > 255 else gray
            gray = 0 if gray < 0 else gray

            data[x, y, 0] = gray
            data[x, y, 1] = gray
            data[x, y, 2] = gray

    def contraharmonicFilter(self, size, Q):
        global data
        if (size % 2 == 0):
            print("Size Invalid: must be odd number!")
            return

        data_temp = np.zeros([width, height, 3])
        data_temp = data.copy()
        for y in range(height):
            for x in range(width):

                sumRedAbove = 0
                sumGreenAbove = 0
                sumBlueAbove = 0
                sumRedBelow = 0
                sumGreenBelow = 0
                sumBlueBelow = 0

                subData = data_temp[x - int(size/2):x + int(size/2) + 1, y - int(size/2):y + int(size/2) + 1, :].copy()
                subData = subData ** (Q + 1)
                sumRedAbove = np.sum(subData[:,:,0:1], axis=None)
                sumGreenAbove = np.sum(subData[:,:,1:2], axis=None)
                sumBlueAbove = np.sum(subData[:,:,2:3], axis=None)

                subData = data_temp[x - int(size/2):x + int(size/2) + 1, y - int(size/2):y + int(size/2) + 1, :].copy()
                subData = subData ** Q
                sumRedBelow = np.sum(subData[:,:,0:1], axis=None)
                sumGreenBelow = np.sum(subData[:,:,1:2], axis=None)
                sumBlueBelow = np.sum(subData[:,:,2:3], axis=None)

                if (sumRedBelow != 0): sumRedAbove /= sumRedBelow
                sumRedAbove = 255 if sumRedAbove > 255 else sumRedAbove
                sumRedAbove = 0 if sumRedAbove < 0 else sumRedAbove
                if (math.isnan(sumRedAbove)): sumRedAbove = 0
                if (sumGreenBelow != 0): sumGreenAbove /= sumGreenBelow
                sumGreenAbove = 255 if sumGreenAbove > 255 else sumGreenAbove
                sumGreenAbove = 0 if sumGreenAbove < 0 else sumGreenAbove

                if (sumBlueBelow != 0): sumBlueAbove /= sumBlueBelow
                sumBlueAbove = 255 if sumBlueAbove > 255 else sumBlueAbove
                sumBlueAbove = 0 if sumBlueAbove < 0 else sumBlueAbove
                if (math.isnan(sumBlueAbove)): sumBlueAbove = 0
                data[x, y, 0] = sumRedAbove
                data[x, y, 1] = sumGreenAbove
                data[x, y, 2] = sumBlueAbove

    def alphaTrimmedFilter(self, size, d):
        global data
        if (size % 2 == 0):
            print("Size Invalid: must be odd number!")
            return

        data_zeropaded = np.zeros([width + int(size/2) * 2, height + int(size/2) * 2, 3])
        data_zeropaded[int(size/2):width + int(size/2), int(size/2):height + int(size/2), :] = data

        for y in range(height):
            for x in range(width):

                subData = data_zeropaded[x:x + size + 1, y:y + size + 1, :]
                sortedRed = np.sort(subData[:,:,0:1], axis=None)
                sortedGreen = np.sort(subData[:,:,1:2], axis=None)
                sortedBlue = np.sort(subData[:,:,2:3], axis=None)
                r = np.mean(sortedRed[int(d/2) : size * size - int(d/2) + 1])
                r = 255 if r > 255 else r
                r = 0 if r < 0 else RuntimeError

                g = np.mean(sortedGreen[int(d/2) : size * size - int(d/2) + 1])
                g = 255 if g > 255 else g
                g = 0 if g < 0 else g
                b = np.mean(sortedBlue[int(d/2) : size * size - int(d/2) + 1])
                b = 255 if b > 255 else b
                b = 0 if b < 0 else b
                data[x, y, 0] = r
                data[x, y, 1] = g
                data[x, y, 2] = b

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

    def linearSpatialFilter(self, kernel, size):
        global data
        if (size % 2 ==0):
            print("Size Invalid: must be odd number!")
            return

        data_zeropaded = np.zeros([width + int(size/2) * 2, height + int(size/2) * 2, 3])
        data_zeropaded[int(size/2):width + int(size/2), int(size/2):height + int(size/2), :] = data

        for y in range(int(size/2), int(size/2) + height):
            for x in range(int(size/2), int(size/2) + width):
                subData = data_zeropaded[x - int(size/2):x + int(size/2) + 1, y - int(size/2):y + int(size/2) + 1, :]

                sumRed = np.sum(np.multiply(subData[:,:,0:1].flatten(), kernel))
                sumGreen = np.sum(np.multiply(subData[:,:,1:2].flatten(), kernel))
                sumBlue = np.sum(np.multiply(subData[:,:,2:3].flatten(), kernel))

                sumRed = 255 if sumRed > 255 else sumRed
                sumRed = 0 if sumRed < 0 else sumRed

                sumGreen = 255 if sumGreen > 255 else sumGreen
                sumGreen = 0 if sumGreen < 0 else sumGreen

                sumBlue = 255 if sumBlue > 255 else sumBlue
                sumBlue = 0 if sumBlue < 0 else sumBlue

                data[x - int(size/2), y - int(size/2), 0] = sumRed
                data[x - int(size/2), y - int(size/2), 1] = sumGreen
                data[x - int(size/2), y - int(size/2), 2] = sumBlue

    def cannyEdgeDetector(self, lower, upper):
        global data
        #Step 1 - Apply 5 x 5 Gaussian filter
        gaussian = [2.0 / 159.0, 4.0 / 159.0, 5.0 / 159.0, 4.0 / 159.0, 2.0 / 159.0,
        4.0 / 159.0, 9.0 / 159.0, 12.0 / 159.0, 9.0 / 159.0, 4.0 / 159.0,
        5.0 / 159.0, 12.0 / 159.0, 15.0 / 159.0, 12.0 / 159.0, 5.0 / 159.0,
        4.0 / 159.0, 9.0 / 159.0, 12.0 / 159.0, 9.0 / 159.0, 4.0 / 159.0,
        2.0 / 159.0, 4.0 / 159.0, 5.0 / 159.0, 4.0 / 159.0, 2.0 / 159.0]
        self.linearSpatialFilter(gaussian, 5)
        self.convertToGrayscale()

        #Step 2 - Find intensity gradient
        sobelX = [ 1, 0, -1,
                    2, 0, -2,
                    1, 0, -1]
        sobelY = [ 1, 2, 1,
                0, 0, 0,
                -1, -2, -1]
        magnitude = np.zeros([width, height])
        direction = np.zeros([width, height])
        data_zeropaded = np.zeros([width + 2, height + 2, 3])
        data_zeropaded[1:width + 1, 1:height + 1, :] = data
        for y in range(1, height + 1):
            for x in range(1, width + 1):
                gx = 0
                gy = 0
                subData = data_zeropaded[x - 1:x + 2, y - 1:y + 2, :]
                gx = np.sum(np.multiply(subData[:,:,0:1].flatten(), sobelX))
                gy = np.sum(np.multiply(subData[:,:,0:1].flatten(), sobelY))
                magnitude[x - 1, y - 1] = math.sqrt(gx * gx + gy * gy)
                direction[x - 1, y - 1] = math.atan2(gy, gx) * 180 / math.pi

        #Step 3 - Nonmaxima Suppression
        gn = np.zeros([width, height])
        for y in range(1, height + 1):
            for x in range(1, width + 1):
                targetX = 0
                targetY = 0
                #find closest direction
                if (direction[x - 1, y - 1] <= -157.5):
                    targetX = 1
                    targetY = 0
                elif (direction[x - 1, y - 1] <= -112.5):
                    targetX = 1
                    targetY = -1
                elif (direction[x - 1, y - 1] <= -67.5):
                    targetX = 0
                    targetY = 1
                elif (direction[x - 1, y - 1] <= -22.5):
                    targetX = 1
                    targetY = 1
                elif (direction[x - 1, y - 1] <= 22.5):
                    targetX = 1
                    targetY = 0
                elif (direction[x - 1, y - 1] <= 67.5):
                    targetX = 1
                    targetY = -1
                elif (direction[x - 1, y - 1] <= 112.5):
                    targetX = 0
                    targetY = 1
                elif (direction[x - 1, y - 1] <= 157.5):
                    targetX = 1
                    targetY = 1
                else:
                    targetX = 1
                    targetY = 0

                if (y + targetY >= 0 and y + targetY < height and x + targetX >= 0 and x + targetX < width and magnitude[x - 1, y - 1] < magnitude[x + targetY - 1, y + targetX - 1]):
                    gn[x - 1, y - 1] = 0
                elif (y - targetY >= 0 and y - targetY < height and x - targetX >= 0 and x - targetX < width and magnitude[x - 1, y - 1] < magnitude[x - targetY - 1, y - targetX - 1]):
                    gn[x - 1, y - 1] = 0
                else:
                    gn[x - 1, y - 1] = magnitude[x - 1, y - 1]
        
        #set back first
        gn[x - 1, y - 1] = 255 if gn[x - 1, y - 1] > 255 else gn[x - 1, y - 1]
        gn[x - 1, y - 1] = 0 if gn[x - 1, y - 1] < 0 else gn[x - 1, y - 1]
        data[x - 1, y - 1, 0] = gn[x - 1, y - 1]
        data[x - 1, y - 1, 1] = gn[x - 1, y - 1]
        data[x - 1, y - 1, 2] = gn[x - 1, y - 1]

        #upper threshold checking with recursive
        for y in range(height):
            for x in range(width):
                if (data[x, y, 0] >= upper):
                    data[x, y, 0] = 255
                    data[x, y, 1] = 255
                    data[x, y, 2] = 255
                    self.hystConnect(x, y, lower)
        #clear unwanted values
        for y in range(height):
            for x in range(width):
                if (data[x, y, 0] != 255):
                    data[x, y, 0] = 0
                    data[x, y, 1] = 0
                    data[x, y, 2] = 0

    def hystConnect(self, x, y, threshold):
        global data
        for i in range(y - 1, y + 2):
            for j in range(x - 1, x + 2):
                if ((j < width) and (i < height) and
                    (j >= 0) and (i >= 0) and
                    (j != x) and (i != y)):
                    value = data[j, i, 0]
                    if (value != 255):
                        if (value >= threshold):
                            data[j, i, 0] = 255
                            data[j, i, 1] = 255
                            data[j, i, 2] = 255
                            self.hystConnect(j, i, threshold)
                        else:
                            data[j, i, 0] = 0
                            data[j, i, 1] = 0
                            data[j, i, 2] = 0

    # def houghTransform(self, percent):
    # global data
    # #The image should be converted to edge map first

    # #Work out how the hough space is quantized
    # numOfTheta = 720
    # thetaStep = math.pi / numOfTheta

    # highestR = int(round(max(width, height) * math.sqrt(2)))

    # centreX = int(width / 2)
    # centreY = int(height / 2)

    # print("Hough array w: %s height: %s" % (numOfTheta, (2*highestR)))

    # #Create the hough array and initialize to zero
    # houghArray = np.zeros([numOfTheta, 2*highestR])

    # #Step 1 - find each edge pixel
    # #Find edge points and vote in array
    # for y in range(3, height - 3):
    #   for x in range(3, width - 3):
    #     pointColor = data[x, y, 0]
    #     if (pointColor != 0):
    #       #Edge pixel found
    #       for i in range(numOfTheta):
    #         #Step 2 - Apply the line equation and update hough array
    #         #Work out the r values for each theta step
    #         r = int((x - centreX) * math.cos(i * thetaStep) + (y - centreY) * math.sin(i * thetaStep))

    #         #Move all values into positive range for display purposes
    #         r = r + highestR
    #         if (r < 0 or r >= 2 * highestR):
    #           continue

    #         #Increment hough array
    #         houghArray[i, r] = houghArray[i, r] + 1

    # #Step 3 - Apply threshold to hough array to find line
    # #Find the max hough value for the thresholding operation
    # maxHough = np.amax(houghArray)

    # #Set the threshold limit
    # threshold = percent * maxHough
    # #Step 4 - Draw lines

    # # Search for local peaks above threshold to draw
    # for i in range(numOfTheta):
    #   for j in range(2 * highestR):
    #     #only consider points above threshold
    #     if (houghArray[i, j] >= threshold):
    #     # see if local maxima
    #       draw = True
    #       peak = houghArray[i, j]

    #       for k in range(-1, 2):
    #         for l in range(-1, 2):
    #         #not seeing itself
    #           if (k == 0 and l == 0):
    #             continue

    #           testTheta = i + k
    #           testOffset = j + l

    #           if (testOffset < 0 or testOffset >= 2*highestR):
    #             continue
    #           if (testTheta < 0):
    #             testTheta = testTheta + numOfTheta
    #           if (testTheta >= numOfTheta):
    #             testTheta = testTheta - numOfTheta
    #           if (houghArray[testTheta][testOffset] > peak):
    #             #found bigger point
    #             draw = False
    #             break

    #       #point found is not local maxima
    #       if (not(draw)):
    #         continue

    #       #if local maxima, draw red back
    #       tsin = math.sin(i*thetaStep)
    #       tcos = math.cos(i*thetaStep)

    #       if (i <= numOfTheta / 4 or i >= (3 * numOfTheta) / 4):
    #         for y in range(height):
    #           #vertical line
    #           x = int((((j - highestR) - ((y - centreY) * tsin)) / tcos) + centreX)

    #           if(x < width and x >= 0):
    #             data[x, y, 0] = 255
    #             data[x, y, 1] = 0
    #             data[x, y, 2] = 0
    #       else:
    #         for x in range(width):
    #         #horizontal line
    #           y = int((((j - highestR) - ((x - centreX) * tcos)) / tsin) + centreY)

    #           if(y < height and y >= 0):
    #             data[x, y, 0] = 255
    #             data[x, y, 1] = 0
    #             data[x, y, 2] = 0

    def ADIAbsolute(self, sequences, threshold, step):
        global data
        data_temp = np.zeros([width, height, 3])
        data_temp = np.copy(data)
        data[data > 0] = 0
        for n in range(len(sequences)):
            #read file
            otherImage = Image.open(sequences[n])
            otherData = np.array(otherImage)

            for y in range(height):
                for x in range(width):
                    dr = int(data_temp[x, y, 0]) - int(otherData[x, y, 0])
                    dg = int(data_temp[x, y, 1]) - int(otherData[x, y, 1])
                    db = int(data_temp[x, y, 2]) - int(otherData[x, y, 2])
                    dGray = int(round((0.2126*dr) + int(0.7152*dg) + int(0.0722*db)))

                    if (abs(dGray) > threshold):
                        newColor = data[x, y, 0] + step
                        newColor = 255 if newColor > 255 else newColor
                        newColor = 0 if newColor < 0 else newColor
                        data[x, y, 0] = newColor
                        data[x, y, 1] = newColor
                        data[x, y, 2] = newColor

    def negativeADIAbsolute(self, sequences, threshold, step):
        global data
        data_temp = np.zeros([width, height, 3])
        data_temp = np.copy(data)
        data[data > 0] = 0
        for n in range(len(sequences)):
            #read file
            otherImage = Image.open(sequences[n])
            otherData = np.array(otherImage)

            for y in range(height):
                for x in range(width):
                    dr = int(data_temp[x, y, 0]) - int(otherData[x, y, 0])
                    dg = int(data_temp[x, y, 1]) - int(otherData[x, y, 1])
                    db = int(data_temp[x, y, 2]) - int(otherData[x, y, 2])
                    dGray = int(round((0.2126*dr) + int(0.7152*dg) + int(0.0722*db)))

                    if (dGray < -threshold):
                        newColor = data[x, y, 0] + step
                        newColor = 255 if newColor > 255 else newColor
                        newColor = 0 if newColor < 0 else newColor
                        data[x, y, 0] = newColor
                        data[x, y, 1] = newColor
                        data[x, y, 2] = newColor

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
        Manually rotates the image by a specified degree between 0 and 360.
        """
        global img
        global data
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
                    # Map the new coordinates to the original image
                    original_x = int((x - new_cx) * cos_angle + (y - new_cy) * sin_angle + cx)
                    original_y = int(-(x - new_cx) * sin_angle + (y - new_cy) * cos_angle + cy)

                    if 0 <= original_x < self.width and 0 <= original_y < self.height:
                        new_data[x, y] = data[original_x, original_y]

            data = new_data
            img = Image.fromarray(data)
            print(f"Image has been manually rotated by {degree} degrees!")
        else:
            print("No image loaded to rotate.")