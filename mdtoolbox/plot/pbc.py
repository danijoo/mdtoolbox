import numpy as np


def pbc_mask(coords, min_dev=3):
    """Removes jumps from the coordinates by masking values with high slope

    Parameters
    ----------
    coords: ndarray
    min_dev: float
        Higher numbers mask more agressivly

    """
    # calculate slope between all points
    abs_d_data = np.abs(np.diff(coords))
    # mask all point with a slope that deviates too much from mean
    if len(coords.shape) == 1:
        mask = np.hstack(
            [abs_d_data > abs_d_data.mean() + min_dev * abs_d_data.std(), [False]]
        )
    else:
        mask = np.empty(coords.shape)
        for i in range(mask.shape[0]):
            mask[i] = np.hstack(
                [abs_d_data[i] > abs_d_data[i].mean() + min_dev * abs_d_data[i].std(),
                 [False]]
            )

    # return the masked output
    return np.ma.MaskedArray(coords, mask)
