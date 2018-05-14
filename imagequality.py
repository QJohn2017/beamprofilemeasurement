import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
nullfmt = NullFormatter()

# define image
def image(height, width, xpos, ypos):
    w = width

    x = np.linspace(-10, 10, 1200)
    min = -10*(1600/1200)
    max = 10*(1600/1200)
    y = np.linspace(min, max, 1600)
    xv, yv = np.meshgrid(x, y)
    z = height * np.exp(-((xv - xpos)**2 + (yv - ypos)**2) / w**2)
    return z

def quality(image):
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

    plt.figure(1, figsize=(8, 8))

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

    plt.show()

imag = image(height=1, width=1, xpos=0, ypos=0)
imag = imag + image(height=1, width=1, xpos=2, ypos=2)
quality(imag)