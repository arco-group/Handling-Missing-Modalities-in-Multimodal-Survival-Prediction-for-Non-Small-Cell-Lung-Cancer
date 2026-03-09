from typing import Union, List

import torch
from torchvision.transforms import v2

__all__ = ["RandomOrderApply", "RandomCrop", "RandomZoom"]

class RandomOrderApply(torch.nn.Module):
    def __init__(self, transforms: list, p: float = 0.3):
        """
        Apply a list of transformations in a random order.

        Parameters
        ----------
        transforms: list. A list of transformations.
        """
        super(RandomOrderApply, self).__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, *inputs):
        needs_unpacking = len(inputs) > 1

        if torch.rand(1) > self.p:
            return inputs if needs_unpacking else inputs[0]

        outputs = inputs if needs_unpacking else inputs[0]
        for idx in torch.randperm(len(self.transforms)):
            if torch.rand(1) > self.p:
                transform = self.transforms[idx]
                outputs = transform(*inputs)
                inputs = outputs if needs_unpacking else (outputs,)
        return outputs


class RandomCrop(torch.nn.Module):
    def __init__(self, size: Union[int, List[int]], p: float = 0.5, resize: bool = True, padding: int = 0, pad_if_needed: bool = False, fill: int = 0, padding_mode: str = "constant") -> None:
        """
        Crop the image tensor at a random location.
        Parameters
        ----------
        size
        padding
        pad_if_needed
        fill
        padding_mode
        """
        super(RandomCrop, self).__init__()
        if not isinstance(size, list):
            size = (size, size)
        self.size = size
        self.p = p
        self.resize = resize
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        if isinstance(size, float):
            size = (int(224*size), int(224*size))
        self.random_crop = v2.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed, fill=fill, padding_mode=padding_mode)

    def forward(self, image):
        if torch.rand(1) > self.p:
            return image
        if isinstance(self.size[0], float):
            size = int(image.shape[-2]*self.size[0]), int(image.shape[-1]*self.size[1])
        else:
            size = self.size
        params = self.random_crop.get_params(image, size)
        orig_size = list(image.shape)
        image = v2.functional.crop(image, *params)
        if self.resize:
            image = v2.functional.resize(image, orig_size[-2:], antialias=True)
        return image


class RandomZoom(torch.nn.Module):
    def __init__(self, fill: float = 0, side_range: List[float] = (0.8, 1.2), p: float = 0.3) -> None:
        super(RandomZoom, self).__init__()
        self.fill = fill
        self.side_range = side_range
        self.p = p

    def forward(self, image):
        # seleziona un valore randomico per lo zoom all'interno del range
        if torch.rand(1) > self.p:
            return image

        orig_h, orig_w = image.shape[-2:]
        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        new_width = int(orig_w * r)
        new_height = int(orig_h * r)

        if r < 1:
            image = v2.functional.pad(image, [(orig_h//2) - (orig_h - new_height)//2, (orig_h//2) - (orig_h - new_height)//2, (orig_w//2) - (orig_w - new_width)//2, (orig_w//2) + (orig_w - new_width)//2], self.fill)
            image = v2.functional.resize_image(image, [orig_h, orig_w], antialias=True)
        else:
            left = int((new_width - orig_w)//2)
            top = int((new_height - orig_h)//2)
            image = v2.functional.resize(image, [new_height, new_width], antialias=True)
            image = v2.functional.crop(image, top=top, left=left, height=orig_h, width=orig_w)
        return image


if __name__ == "__main__":
    pass
