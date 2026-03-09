import os
import nrrd
import scipy.io
import numpy as np
import tifffile
from PIL import Image
from CMC_utils.paths import add_extension, get_extension

__all__ = ["load_image", "save_image"]

## load

def _load_nrrd_image(path: str) -> np.array:
    """
    This function loads a nrrd image.

    Parameters:
        path: string.
            Path of the image to load.

    Returns:

    image: np.array.
        Image loaded.

    """
    image, _ = nrrd.read(path)
    return image


def _load_mat_image(path: str) -> np.array:
    """
    This function loads a mat image generated with matlab.

    Parameters:
        path: string.
            Path of the image to load.

    Returns:

    image: np.array.
        Image loaded.

    """
    mat = scipy.io.loadmat(path)
    return mat


def _load_tiff_image(path: str) -> np.array:
    """
    This function loads a tiff image.

    Parameters:
    """
    image = tifffile.imread(path)
    return image


def _load_image(path: str) -> np.array:
    """
    This function loads an image.

    Parameters:
        path: string.
            Path of the image to load.

    Returns:

    image: numpy.array.
        Image loaded.

    """
    image = Image.open(path)

    return np.array(image)


def load_image(path: str, **_) -> np.array:
    """
    This function loads an image, independently of the extension it has.

    Parameters:
        path: string.
            Path of the image to load.

    Returns:

    image: np.array.
        Image loaded.

    """
    extension = get_extension(path).lower()

    options = {"tif": _load_tiff_image, "tiff": _load_tiff_image, "png": _load_image, "jpg": _load_image, "jpeg": _load_image,
               "nrrd": _load_nrrd_image, "mat": _load_mat_image}
    image = options[extension](path)

    return image


## save
def _save_image_as_nrrd(image: np.array, path: str, **_) -> None:
    """
    This function saves a nrrd image.

    Parameters:
        image: np.array.
            Image to save.
        path: string
            Path where to save the image.

    Returns:

    None

    """
    nrrd.write(path, image)


def _save_image_as_tif(image: np.array, path: str, **kwargs) -> None:
    """
    This function saves a tif image.

    Parameters:
        image: np.array.
            Image to save.
        path: string.
            Path where to save the image.
        kwargs:
            Keyword arguments to pass on to the save function.

    Returns:

    None

    """
    # image = Image.fromarray(image)
    # image.save(path, "TIFF", **kwargs)
    tifffile.imwrite(path, image)


def _save_image_as_png(image: np.array, path: str, convert: str = None, **kwargs) -> None:
    """
    This function saves a png image.

    Parameters:
        image: np.array.
            Image to save.
        path: string.
            Path where to save the image.
        kwargs:
            Keyword arguments to pass on to the save function.

    Returns:

    None

    """
    image = Image.fromarray(image)
    if convert is not None:
        image = image.convert(convert)  # ("L")
    image.save(path, "png", **kwargs)


def _save_image_as_jpeg(image: np.array, path: str, convert: str = None, **kwargs) -> None:
    """
    This function saves a jpeg image.

    Parameters:
        image: numpy.array.
            Image to save.
        path: string.
            Path where to save the image.
        kwargs:
            Keyword arguments to pass on to the save function.

    Returns:

    None

    """
    image = Image.fromarray(image)
    if convert is not None:
        image = image.convert(convert)  # ("L")
    image.save(path, "JPEG", **kwargs)


def save_image(image: np.array, image_name: str, directory_path: str, extension: str = None, dtype: str = None, **kwargs) -> None:
    """
    This function saves an image independetly from the extension it has.

    Parameters:
        image: numpy.array.
            Image to save.
        image_name: string.
            Name of the file to save.
        directory_path: string.
            Path to the directory where to save the image.
        extension: string.
            Extension to use.
        dtype: string, default "float32".
            Type to be applied to the image before being saved.
        kwargs:
            Keyword arguments to pass on to the save function.

    Returns:

    None

    """
    if extension is not None:
        image_name = add_extension(image_name, extension)
    else:
        extension = get_extension(image_name).lower()

    assert extension is not None, "Image extension not provided"

    path = os.path.join(directory_path, image_name)

    if dtype is not None:
        image = image.astype(dtype)

    options = {"tif": _save_image_as_tif, "tiff": _save_image_as_tif, "png": _save_image_as_png,
               "jpg": _save_image_as_jpeg, "jpeg": _save_image_as_jpeg, "nrrd": _save_image_as_nrrd}
    options[extension](image, path, **kwargs)


if __name__ == "__main__":
    pass
