import torch
from torchvision.transforms import v2

import logging
log = logging.getLogger(__name__)

__all__ = ["Gray2RGB", "SetChannelsFirst", "Clamp", "Standardize", "Normalize", "Padding", "FillNan", "Scale", "CenterCrop"]


class Gray2RGB(torch.nn.Module):
    def __init__(self):
        """
        Convert a single channel image to a 3-channel image.
        """
        super(Gray2RGB, self).__init__()

    @staticmethod
    def forward(image):
        if len(image.shape) == 2 or 1 in image.shape:  # if single channel repeat it to make it 3-channel
            if len(image.shape) == 2:
                image = image.unsqueeze(0)
            ch_index = [i for i, j in enumerate(image.shape) if j == 1][0]
            repeats = [3 if i == ch_index else 1 for i in range(len(image.shape))]
            image = image.repeat(*repeats)
        return image


class SetChannelsFirst(torch.nn.Module):
    def __init__(self, channels_order: tuple = (2, 0, 1)):
        """
        Set the order of the channels in the image tensor.

        Parameters
        ----------
        channels_order: tuple of int (default=(2, 0, 1)). The first one will be the new first dimension, as long as it is the smallest.
        """
        super(SetChannelsFirst, self).__init__()
        self.channels_order = channels_order

    def forward(self, image):
        if len(image.shape) == 2:  # Single channel
            image = image.unsqueeze(2)
        if len(image.shape) == len(self.channels_order) and image.shape[self.channels_order[0]] < image.shape[self.channels_order[1]] and image.shape[self.channels_order[0]] < image.shape[self.channels_order[2]]:  # Channels as first dimension
            image = torch.permute(image, self.channels_order)
        return image


class Clamp(torch.nn.Module):
    def __init__(self, min_val: float = 0, max_val: float = 1):
        """
        Clamp the values of the image tensor between min_val and max_val.

        Parameters
        ----------
        min_val: float (default=0). The minimum value.
        max_val: float (default=1). The maximum value.
        """
        super(Clamp, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, image):
        image = torch.clamp(image, min=self.min_val, max=self.max_val)
        return image


class Standardize(torch.nn.Module):
    def __init__(self, mean: float = None, std: float = None):
        """
        Normalize the values of the image tensor to the range [-1, 1].

        Parameters
        ----------
        mean: float (default=None). The mean value.
        std: float (default=None). The standard deviation value.
        """
        super(Standardize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, image):
        image = image.float()
        if self.mean is None or self.std is None:
            mean = image.mean(axis=(-2, -1))
            std = image.std(axis=(-2, -1))
        else:
            mean = self.mean
            std = self.std
        image = v2.functional.normalize(image, mean=mean, std=std)
        return image


class Normalize(torch.nn.Module):
    def __init__(self, min_val: float = 0, max_val: float = 255):
        """
        Normalize the values of the image tensor to the range [0, 1].

        Parameters
        ----------
        min_val: float (default=None). The mean value.
        max_val: float (default=None). The standard deviation value.
        """
        super(Normalize, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, image):
        image = image.float()
        if self.min_val is None or self.max_val is None:
            min_val = torch.amin(image, dim=(-2, -1), keepdim=True)
            max_val = torch.amax(image, dim=(-2, -1), keepdim=True)
        else:
            min_val = torch.tensor(self.min_val)
            max_val = torch.tensor(self.max_val)
        image = image.sub(min_val).div(max_val.sub(min_val))
        return image


class Scale(torch.nn.Module):
    def __init__(self, scale: float = 1024, min_val: float = 0, max_val: float = 255, method: str = "positive"):
        """
        Scale the values of the image tensor.

        Parameters
        ----------
        scale: float (default=1). The scale value.
        """
        super(Scale, self).__init__()
        self.scale = scale
        self.min_val = min_val
        self.max_val = max_val
        self.method = method

    def forward(self, image):
        image = image.float()
        if self.min_val is None or self.max_val is None:
            min_val = torch.amin(image, dim=(-2, -1), keepdim=True)
            max_val = torch.amax(image, dim=(-2, -1), keepdim=True)
        else:
            min_val = torch.tensor(self.min_val)
            max_val = torch.tensor(self.max_val)
        image = image.sub(min_val).div(max_val.sub(min_val))
        if self.method == "symmetric":
            image = image.mul(2).sub(1).mul(self.scale)
        else:
            image = image.mul(self.scale)
        return image


class CenterCrop(torch.nn.Module):
    def __init__(self, size):
        """
        Center crop the image tensor.

        Parameters
        ----------
        size: int. The size of the crop.
        """
        super(CenterCrop, self).__init__()
        self.size = size

    def forward(self, image):
        h, w = image.shape[-2:]
        if self.size is not None:
            crop_size = self.size
        else:
            crop_size = min(h, w)
        startx = w // 2 - (crop_size // 2)
        starty = h // 2 - (crop_size // 2)
        return image[:, starty:starty + crop_size, startx:startx + crop_size]


class Padding(torch.nn.Module):
    def __init__(self, size=None, pad=None, mode='constant', value=0):
        """
        Pad the image tensor.

        Parameters
        ----------
        padding: int (default=0). The padding value.
        """
        super(Padding, self).__init__()
        self.size = size
        self.pad = pad
        self.mode = mode
        self.value = value

    def forward(self, image):
        if self.size is None:
            max_size = torch.max(torch.tensor(image.shape[-2:]))
            size = (max_size, max_size)
        else:
            size = self.size
        if self.pad is None:
            h, w = image.shape[-2:]
            h_padding, w_padding = (size[0]-h)//2, (size[1]-w)//2
            padding = (w_padding, w_padding, h_padding, h_padding)
        else:
            padding = self.pad
        image = torch.nn.functional.pad(image, pad=padding, mode=self.mode, value=self.value)
        return image


class FillNan(torch.nn.Module):
    def __init__(self, fill_value: float = 0):
        """
        Fill NaN values of the image tensor.

        Parameters
        ----------
        fill_value: float (default=0). The value to fill NaN values.
        """
        super(FillNan, self).__init__()
        self.fill_value = fill_value

    def forward(self, image):
        image = torch.nan_to_num(image, nan=self.fill_value)
        return image


if __name__ == "__main__":
    pass
