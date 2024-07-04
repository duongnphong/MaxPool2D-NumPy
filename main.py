import cv2
import numpy as np

from helpers import visualize
from maxpool2d import maxpool2d


def main():
    # Load image
    im = cv2.imread("assets/cat.webp")
    im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_LINEAR)
    img = np.array(im)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get each channel of the image
    channel = np.moveaxis(img, -1, 0)

    # Perform 1st maxpool2d
    maxpool1 = maxpool2d(image=img, kernel_size=2)
    # Perform 2nd maxpool2d
    maxpool2 = maxpool2d(image=maxpool1, kernel_size=2)
    # Perform 2nd maxpool2d
    maxpool3 = maxpool2d(image=maxpool2, kernel_size=2)

    # Plot the result of each conv2d
    visualize(
        image=img,
        in_channels=channel,
        out_channels_1=maxpool1,
        out_channels_2=maxpool2,
        out_channels_3=maxpool3,
    )


if __name__ == "__main__":
    main()
