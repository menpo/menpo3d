from __future__ import division
import numpy as np

from menpo.image import Image
from menpo.feature import gradient as fast_gradient, no_op


# ----------- INTERFACES -----------
class LucasKanadeBaseInterface(object):
    r"""
    Base interface for Lucas-Kanade optimization of 3DMMs.

    """
    def __init__(self, transform, template, sampling=None):
        self.transform = transform
        self.template = template

    def warp_jacobian(self):
        r"""
        Computes the ward jacobian.

        :type: ``(n_dims, n_pixels, n_params)`` `ndarray`
        """
        return 0

    def warp(self, image):
        r"""
        Warps an image into the template's mask.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The input image to be warped.

        Returns
        -------
        warped_image : `menpo.image.Image` or subclass
            The warped image.
        """
        return 0

    def warped_images(self, image, shapes):
        r"""
        Given an input test image and a list of shapes, it warps the image
        into the shapes. This is useful for generating the warped images of a
        fitting procedure.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The input image to be warped.
        shapes : `list` of `menpo.shape.PointCloud`
            The list of shapes in which the image will be warped. The shapes
            are obtained during the iterations of a fitting procedure.

        Returns
        -------
        warped_images : `list` of `menpo.image.MaskedImage`
            The warped images.
        """
        return 0

    def gradient(self, image):
        r"""
        Computes the gradient of an image and vectorizes it.

        """

    def steepest_descent_images(self, nabla, dW_dp):
        r"""
        Computes the steepest descent images, i.e. the product of the gradient
        and the warp jacobian.
        """

    def algorithm_result(self, image, shapes, shape_parameters,
                         appearance_parameters=None, initial_shape=None,
                         gt_shape=None, costs=None):
        """"""
        return 0


# ----------- ALGORITHMS -----------
class LucasKanade(object):
    r"""
    Abstract class for a Lucas-Kanade optimization algorithm.

    Parameters
    ----------

    eps : `float`, optional
        Value for checking the convergence of the optimization.
    """
    def __init__(self, eps=10**-5):
        self.eps = eps
        self._precompute()

    @property
    def template(self):
        r"""
        Returns the template of the 3DMM (usually the mean of the 3DMM).

        :type: `menpo.image.Image` or subclass
        """
        return self.interface.template

    def _precompute(self):
        return 0


class Simultaneous(LucasKanade):
    r"""
    Abstract class for defining Simultaneous AAM optimization algorithms.
    """
    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            return_costs=False, map_inference=False):
        r"""
        Execute the optimization algorithm.
        """
        # return algorithm result


class SimultaneousForwardAdditive(Simultaneous):
    r"""
    Simultaneous Forward Additive algorithm.
    """
    def _compute_jacobian(self):
        # return forward jacobian
        return 0

    def _update_warp(self):
        # update warp based on forward addition
        return 0

    def __str__(self):
        return "Simultaneous Forward Additive Algorithm"




