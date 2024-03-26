import pathlib

from .metrics import batch_intersection_union, batch_pix_accuracy

project_root = pathlib.Path(__file__).resolve().parents[2]

__all__ = ['project_root']
