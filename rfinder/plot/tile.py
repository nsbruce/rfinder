from typing import List, Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import numpy.typing as npt

from rfinder.environment import load_env
from rfinder.types import Box

from .utils import add_center_dot_patch, add_rect_patch


def tile(
    ax: plt.Axes,
    tile: npt.NDArray[np.float_],
    boxesA: Optional[List[Box]] = None,
    boxesB: Optional[List[Box]] = None,
    title: Optional[str] = None,
) -> None:
    """Plots a single tile

    Args:
        ax (plt.Axes): Axes object on which to plot
        tile (npt.NDArray[np.float_]): Tile to plot
        boxesA (Optional[List[Box]], optional): Boxes to overlay. Defaults to None.
        boxesB (Optional[List[Box]], optional): Other boxes to overlay. Defaults to
        None.
        title (Optional[str], optional): Figure title. Defaults to None.
    """
    if title:
        ax.set_title(title)

    ax.imshow(tile, aspect="equal", origin="lower")
    if boxesA:
        add_center_dot_patch(ax, boxesA, "white")
        add_rect_patch(ax, boxesA, "white")
    if boxesB:
        add_center_dot_patch(ax, boxesB, "red")
        add_rect_patch(ax, boxesB, "red")


def boxes(
    ax: plt.Axes,
    boxesA: Optional[List[Box]] = None,
    boxesB: Optional[List[Box]] = None,
    title: Optional[str] = None,
) -> None:
    """Plots boxes on a blank tile

    Args:
        ax (plt.Axes): Axes object on which to plot
        boxesA (Optional[List[Box]], optional): Boxes to overlay. Defaults to None.
        boxesB (Optional[List[Box]], optional): More boxes to overlay. Defaults to None.
        title (Optional[str], optional): Axes title. Defaults to None.
    """
    env = load_env()
    tile_dim = int(env["TILE_DIM"])

    if title:
        ax.set_title(title)
    ax.set_xlim([0, tile_dim])
    ax.set_ylim([0, tile_dim])
    ax.set_aspect("equal")
    if boxesA:
        add_center_dot_patch(ax, boxesA, "black")
        add_rect_patch(ax, boxesA, "black")
    if boxesB:
        add_center_dot_patch(ax, boxesB, "red")
        add_rect_patch(ax, boxesB, "red")
