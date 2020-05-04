import numpy as np
import pandas as pd
from astropy.wcs import WCS
import torch
import torchvision.utils as vutils


def get_corners(row):
    corners = np.array([[row.ra1, row.dec1], [row.ra2, row.dec2], [row.ra3, row.dec3], [row.ra4, row.dec4]])
    min_ra, min_dec = np.min(corners, axis=0)
    max_ra, max_dec = np.max(corners, axis=0)
    return min_ra, max_ra, min_dec, max_dec


def make_wcs(row):
    w = WCS(naxis=2)
    w.wcs.cd = np.array([[row.cd1_1, row.cd1_2], [row.cd2_1, row.cd2_2]])
    w.wcs.crval = [row.crval1, row.crval2]
    w.wcs.crpix = [row.crpix1, row.crpix2]
    w.wcs.ctype = [row.ctype1, row.ctype2]
    w.wcs.set_pv([(2, 1, row.pv2_1), (2, 2, row.pv2_2), (2, 3, row.pv2_3), (2, 4, row.pv2_4), (2, 5, row.pv2_5)])
    return w


def filter_dataframe(df, column, min_val, max_val):
    """Filter a dataframe with column values between min and max, inclusive"""
    df_filt = df[df[column] >= min_val]
    return df_filt[df_filt[column] <= max_val]


def rand_beta(n):
    return int(np.random.beta(2, 2) * n)


def nsqrt(arr):
    """Take square root, where sqrt(-x) = -sqrt(x) rather than i*sqrt(x)"""
    arr = np.array(arr)
    return np.sqrt(np.abs(arr)) * np.where(arr > 0, 1, -1)


def make_grid_transpose(tensor, nrow=8, range=(0, 1), num=64, sqrt=True, label=None):
    """Wrapper around torchvision.utils.make_grid
    Args:
        tensor: list or 4D tensor of images
        nrow: number of images per row
        range: tuple of min/max values to normalize images between. Set to None to determine min/max from tensor
        num: max number of images to plot
        sqrt: whether to sqrt the image
        label: if not None, an overlapping pixel source to plot
    """
    grid = vutils.make_grid(tensor[:num], nrow=nrow, padding=2, normalize=True, range=range).cpu()
    if sqrt:
        grid = np.sqrt(grid)

    if label is not None:
        stars = vutils.make_grid(label[:num], padding=2, normalize=True, range=range).cpu()
        # grid[0] = ((grid[0] + stars[0]).clamp_(*range)  # only add to red channel)
        for y, z in stars[0].nonzero():
            grid[:, y, z] = torch.tensor([1, 0, 0])

    return np.transpose(grid, (1, 2, 0))

