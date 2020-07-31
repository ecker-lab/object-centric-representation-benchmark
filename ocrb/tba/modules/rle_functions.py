import numpy as np
import matplotlib.pyplot as plt

RESOLUTION = 64
BOX = (0, 10, 20, 20)


def box_to_rle(box, res):
    '''
    Transforms a bounding box into an RLE code string
    '''
    rle_code = str([(res * (box[1] + row) + box[0], box[2]) for row in range(box[3])]
                    ).strip('[]').replace('(','').replace(')','').replace(',','')

    return rle_code


def rle_to_pixels(rle_code, res):
    '''
    Transforms an RLE code string into a list of pixels
    '''
    rle_code = [int(i) for i in rle_code.split()]
    pixels = [(pixel_position % res, pixel_position // res)
                 for start, length in list(zip(rle_code[0:-1:2], rle_code[1::2]))
                 for pixel_position in range(start, start + length)]
    return pixels


rle = box_to_rle(BOX, RESOLUTION)
pixels = rle_to_pixels(rle, RESOLUTION)

canvas = np.zeros((RESOLUTION, RESOLUTION))
canvas[tuple(zip(*pixels))] = 1
plt.imshow(canvas)
plt.show()
