import cv2
import matplotlib.pyplot as plt
import numpy as np


def add_padding(image, padding=0):
    if padding > 0:
        # Calculate the amount of padding needed on each side
        pad_amount = padding

        # Create an array for padded image
        padded_image = np.pad(
            image,
            ((pad_amount, pad_amount), (pad_amount, pad_amount), (0, 0)),
            mode="constant",
        )
    else:
        return image

    return padded_image


def maxpool(image, kernel_size: tuple, stride=None):
    stride = kernel_size[0] if stride is None else stride
    h, w, c = image.shape  # Height, width, and number of channels in the image
    output_height = (h - kernel_size[0]) // stride + 1
    output_width = (w - kernel_size[1]) // stride + 1
    output_channels = c  # Number of channels in the image

    output = np.zeros((output_height, output_width, output_channels))

    for channel in range(output_channels):
        for i in range(output_height):
            for j in range(output_width):
                # Extract subsection from channel 'channel' and (i*stride, j*stride) to ((i+1)*stride, (j+1)*stride)
                subsection = image[
                    i * stride : i * stride + kernel_size[0],
                    j * stride : j * stride + kernel_size[1],
                    channel,
                ]
                # Compute max value of the subsection and assign it to output
                output[i, j, channel] = np.max(subsection)

    return output


def visualize(image, in_channels, **out_channels):
    num_out_channels = len(out_channels)
    max_out_channels = max(out_channels[key].shape[2] for key in out_channels)

    fig, axes = plt.subplots(
        max(len(in_channels), max_out_channels) + 1,
        num_out_channels + 2,
        figsize=(12, 8),
        gridspec_kw={"wspace": 0.1, "hspace": 0.3},
    )

    # Calculate the starting indices for plotting
    channel_start_row = (max_out_channels - len(in_channels)) // 2
    conv_start_row = (len(in_channels) - max_out_channels) // 2

    # Plot img aligned with the second element of channel
    img_row = channel_start_row + 1
    axes[img_row, 0].imshow(image, cmap="gray")
    axes[img_row, 0].text(
        0.5,
        1.15,
        f"({image.shape[0]}, {image.shape[1]})",
        ha="center",
        va="center",
        transform=axes[img_row, 0].transAxes,
    )

    # Plot channel elements in the first column
    for i in range(len(in_channels)):
        cmap = None
        if i == 0:
            cmap = "Reds"
            axes[channel_start_row + i, 1].text(
                0.5,
                1.15,
                f"({in_channels[i].shape[0]}, {in_channels[i].shape[1]})",
                ha="center",
                va="center",
                transform=axes[channel_start_row + i, 1].transAxes,
            )
        elif i == 1:
            cmap = "Greens"
        elif i == 2:
            cmap = "Blues"
        axes[channel_start_row + i, 1].imshow(in_channels[i], cmap=cmap)

    # Plot conv elements for each out_channel
    for i, key in enumerate(out_channels):
        out_channel = out_channels[key]
        channel_start_row_2 = (max_out_channels - out_channel.shape[2]) // 2

        for j in range(out_channel.shape[2]):
            axes[channel_start_row_2 + j, i + 2].imshow(
                out_channel[:, :, j], cmap="gray"
            )

        # Display the shape on top of the column
        axes[channel_start_row_2, i + 2].text(
            0.5,
            1.15,
            f"({out_channel.shape[0]}, {out_channel.shape[1]})",
            ha="center",
            va="center",
            transform=axes[channel_start_row_2, i + 2].transAxes,
        )

        # Remove extra empty plots in the first column
        for j in range(channel_start_row_2):
            fig.delaxes(axes[j, i + 2])

        # Remove extra empty plots in the second column
        for j in range(channel_start_row_2 + out_channel.shape[2], max_out_channels):
            fig.delaxes(axes[j, i + 2])

    # Remove extra empty plots in the first column
    for i in range(channel_start_row):
        fig.delaxes(axes[i, 1])

    # Remove extra empty plots in the other columns
    for i in range(conv_start_row + max_out_channels, len(in_channels)):
        for j in range(1, num_out_channels + 2):
            fig.delaxes(axes[i, j])

    # Adjust spacing and layout
    # fig.suptitle("Image, Channel Elements, and Conv Layers", fontsize=16)
    for ax in axes.flat:
        ax.axis("off")

    plt.show()


# np.random.seed(42)
# a = np.random.rand(64, 64, 3)
# print(a)
# print("------")

# kernel_size = (2, 2)

# out = maxpool(image=a, kernel_size=kernel_size)
# print(out)

# print("-------------------")
# print(f"Shape of input: {a.shape}, Shape of output: {out.shape}")
