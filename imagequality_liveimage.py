import numpy as np
from ctypes import *
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from random import random
import time
nullfmt = NullFormatter()

# define image
def generate_image(height, width, xpos, ypos):
    w = width

    x = np.linspace(-10, 10, 1200)
    min = -10*(1600/1200)
    max = 10*(1600/1200)
    y = np.linspace(min, max, 1600)
    xv, yv = np.meshgrid(x, y)
    z = height * np.exp(-((xv - xpos)**2 + (yv - ypos)**2) / w**2)
    return z


def quality(image):
    plt.figure(1)
    plt.clf()
    sumvertical = np.sum(image, 0)
    xvert = range(np.shape(image)[1])
    xvert = np.array(xvert)

    sumhoriz = np.sum(image, 1)
    yhoriz = range(np.shape(image)[0])
    yhoriz = np.array(yhoriz)

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_x = [left, bottom_h, width, 0.2]
    rect_y = [left_h, bottom, 0.2, height]

    # plt.figure(1, figsize=(8, 8))

    axCenter = plt.axes(rect_scatter)
    axhoriz = plt.axes(rect_x)
    axvert = plt.axes(rect_y)

    axCenter.imshow(image, origin='lower', cmap='jet', aspect='auto')

    # plot on top
    axhoriz.plot(xvert, sumvertical, color='b')
    # plot the gaussian approximation
    centroid = np.sum(xvert * sumvertical) / np.sum(sumvertical)
    maxtop = np.max(sumvertical)
    indexstartleft, indexstartright = xvert[0], xvert[-1]
    while sumvertical[indexstartleft] < maxtop / 2:
        indexstartleft += 1
    while sumvertical[indexstartright] < maxtop / 2:
        indexstartright -= 1
    width = indexstartright - indexstartleft
    gaussfittop = maxtop*np.exp(-4 * np.log(2) * (xvert-centroid)**2/width**2)
    #calculate rmse
    rmsetop = np.sum(((gaussfittop - sumvertical)**2)/len(gaussfittop))
    axhoriz.text(int(centroid), 0, 'RMSE: {}\nWidth:{}'.format(np.round(rmsetop, 2), width), color='red')

    axhoriz.plot(xvert, gaussfittop, color='g')
    axhoriz.set_xlim(0, 1200)
    axhoriz.legend(['Signal', 'Gaussian Fit'])
    # axhoriz.xaxis.tick_top()

    # plot on the side
    axvert.plot(sumhoriz, yhoriz, color='b')
    # plot the gaussian approximation
    centroid = np.sum(yhoriz * sumhoriz) / np.sum(sumhoriz)
    maxtop = np.max(sumhoriz)
    indexstartleft, indexstartright = yhoriz[0], yhoriz[-1]
    while sumhoriz[indexstartleft] < maxtop / 2:
        indexstartleft += 1
    while sumhoriz[indexstartright] < maxtop / 2:
        indexstartright -= 1
    width = indexstartright - indexstartleft
    gaussfitside = maxtop*np.exp(-4 * np.log(2) * (yhoriz-centroid)**2/width**2)
    #rmse side
    rmseside = np.sum(((gaussfitside - sumhoriz) ** 2) / len(gaussfitside))
    axvert.text(0, int(centroid), 'RMSE: {}\nWidth:{}'.format(np.round(rmseside, 2), width), color='red')
    axvert.plot(gaussfitside, yhoriz, color='g')
    axvert.set_ylim(0, 1600)
    axvert.legend(['Signal', 'Gaussian Fit'])
    # axvert.yaxis.tick_right()

    axhoriz.xaxis.set_major_formatter(nullfmt)
    axvert.yaxis.set_major_formatter(nullfmt)
    plt.pause(0.0001)
    # plt.show()


def setup_camera(mode):
    if mode == 'Trigger' or mode == 'Freerun':
        pass
    else:
        raise ValueError("mode must be set to 'Trigger' or 'Freerun'")

    global mydll
    global hCamera
    global pbyteraw
    global dwBufferSize
    global dwNumberOfByteTrans
    global dwFrameNo
    global dwMilliseconds
    global threshhold

    # create parameters for camera
    im_height = 1200
    im_width = 1600

    if mode == 'Trigger':
        dwTransferBitsPerPixel = 4
        dwBufferSize = im_height * im_width * 2
        pbyteraw = np.zeros((im_height, im_width), dtype=np.uint16)
        triggermode = 2049

    elif mode == 'Freerun':
        dwTransferBitsPerPixel = 1
        dwBufferSize = im_height * im_width
        pbyteraw = np.zeros((im_height, im_width), dtype=np.uint8)
        triggermode = 0
    else:
        dwTransferBitsPerPixel = None
        dwBufferSize = None
        pbyteraw = None
        triggermode = None


    dwNumberOfByteTrans = c_uint32()
    dwFrameNo = c_uint32()
    dwMilliseconds = 3000

    #  set up camera capture
    mydll = windll.LoadLibrary('StTrgApi.dll')
    hCamera = mydll.StTrg_Open()
    print('hCamera id:', hCamera)

    mydll.StTrg_SetTransferBitsPerPixel(hCamera, dwTransferBitsPerPixel)
    mydll.StTrg_SetScanMode(hCamera, 0, 0, 0, 0, 0)
    mydll.StTrg_SetGain(hCamera, 0)
    mydll.StTrg_SetDigitalGain(hCamera, 64)
    mydll.StTrg_SetExposureClock(hCamera, 200000)
    mydll.StTrg_SetClock(hCamera, 0, 0)
    mydll.StTrg_SetTriggerMode(hCamera, triggermode)
    mydll.StTrg_SetTriggerTiming(hCamera, 0, 0)
    mydll.StTrg_SetIOPinDirection(hCamera, 0)
    mydll.StTrg_SetIOPinPolarity(hCamera, 0)
    mydll.StTrg_SetIOPinMode(hCamera, 0, 16)


def take_image():
    # mydll.StTrg_TakeRawSnapShot(hCamera, pbyteraw.ctypes.data_as(POINTER(c_int16)),
    #                             dwBufferSize, pointer(dwNumberOfByteTrans), pointer(dwFrameNo), dwMilliseconds)
    # image = np.rot90(pbyteraw, 1)

    return generate_image(height=1 + random(), width=1 + random(), xpos=10*(random()-0.5), ypos=10*(random()-0.5))
    # return image


setup_camera(mode='Freerun')

plt.ion()
plt.figure(1, figsize=(8, 8))
while True:
    im_out = take_image()
    im_out = im_out + take_image()
    im_out = im_out + take_image()
    im_out = im_out + take_image()
    im_out = im_out + take_image()
    im_out = im_out + take_image()
    im_out = im_out + take_image()
    quality(im_out)


# plt.imshow(im_out)
# plt.show()

# imag = image(height=1, width=1, xpos=0, ypos=0)
# imag = imag + image(height=1, width=1, xpos=2, ypos=2)
# quality(imag)