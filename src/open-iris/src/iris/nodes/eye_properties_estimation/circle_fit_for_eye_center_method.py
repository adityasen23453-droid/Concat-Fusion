from typing import Tuple

import cv2
import numpy as np
from pydantic import Field

from iris.io.class_configs import Algorithm
from iris.io.dataclasses import EyeCenters, GeometryPolygons
from iris.io.errors import EyeCentersEstimationError


class CircleFitEyeCenterMethod(Algorithm):
    """Estimate pupil and iris centers using a robust circle-fitting approach.

    This algorithm estimates the center of a pupil or iris polygon using:

    1. an initial least-squares circle fit to all polygon points,
    2. trimming of outlier points based on radial residuals using a dynamic threshold,
    3. a second circle fit on the retained inliers.

    LIMITATIONS:
    This method assumes that the pupil and iris contours are reasonably well
    approximated by circles in image space. It is therefore most appropriate
    when off-gaze and strong perspective distortion have already been filtered out.
    """

    class Parameters(Algorithm.Parameters):
        """Default parameters for circle-fit eye center algorithm."""

        mad_scale: float = Field(..., gt=0.0)

    __parameters_type__ = Parameters

    def __init__(
        self,
        mad_scale: float = 3.0,
    ) -> None:
        """Assign parameters.

        Args:
            mad_scale (float, optional): Scale factor used in the dynamic inlier
                threshold: median(residuals) + mad_scale * MAD(residuals).
                Defaults to 3.0.
        """
        super().__init__(mad_scale=mad_scale)

    def run(self, geometries: GeometryPolygons) -> EyeCenters:
        """Estimate pupil and iris centers.

        Args:
            geometries (GeometryPolygons): Pupil and iris geometry polygons.

        Returns:
            EyeCenters: Estimated pupil and iris center coordinates.
        """
        pupil_center_x, pupil_center_y = self._calculate_circle_fit_center(geometries.pupil_array.astype(np.float64))
        iris_center_x, iris_center_y = self._calculate_circle_fit_center(geometries.iris_array.astype(np.float64))

        return EyeCenters(
            pupil_x=pupil_center_x,
            pupil_y=pupil_center_y,
            iris_x=iris_center_x,
            iris_y=iris_center_y,
        )

    def _calculate_circle_fit_center(self, polygon: np.ndarray) -> Tuple[float, float]:
        """Estimate the center of a polygon using fit -> trim outliers -> refit.

        Args:
            polygon (np.ndarray): Polygon points of shape (N, 2) representing a
                contour that is approximately circular.

        Raises:
            EyeCentersEstimationError: Raised if the polygon is invalid or if a
                valid circle cannot be fit.

        Returns:
            Tuple[float, float]: Estimated center coordinates (x, y).
        """
        if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
            raise EyeCentersEstimationError("Polygon must have shape (N, 2) with at least 3 points")

        pts = polygon.astype(np.float64, copy=False)

        # Initial least-squares circle fit
        x = pts[:, 0]
        y = pts[:, 1]

        A = np.column_stack([x, y, np.ones_like(x)]).astype(np.float64)
        b = (-(x * x + y * y)).reshape(-1, 1).astype(np.float64)

        ok, coeffs = cv2.solve(A, b, flags=cv2.DECOMP_SVD)
        if not ok:
            raise EyeCentersEstimationError("Circle fit failed")

        a, b_, c = coeffs.ravel()

        cx = -a / 2.0
        cy = -b_ / 2.0

        r_sq = cx * cx + cy * cy - c
        if r_sq <= 0:
            raise EyeCentersEstimationError("Failed to fit a valid circle to the polygon")

        r = np.sqrt(r_sq)
        center = np.array([cx, cy], dtype=np.float64)

        # Trim outliers using dynamic radial residual threshold
        d = np.linalg.norm(pts - center, axis=1)
        residuals = np.abs(d - r)

        median_residual = np.median(residuals)
        mad = np.median(np.abs(residuals - median_residual))

        if mad == 0:
            keep_mask = residuals <= median_residual
        else:
            threshold = median_residual + self.params.mad_scale * mad
            keep_mask = residuals <= threshold

        inliers = pts[keep_mask]

        # Ensure enough points remain to refit
        if len(inliers) < 3:
            keep_idx = np.argsort(residuals)[:3]
            inliers = pts[keep_idx]

        # Refit circle on inliers
        x = inliers[:, 0]
        y = inliers[:, 1]

        A = np.column_stack([x, y, np.ones_like(x)])
        b = -(x * x + y * y)

        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        a, b_, c = coeffs
        cx = -a / 2.0
        cy = -b_ / 2.0

        r_sq = cx * cx + cy * cy - c
        if r_sq <= 0:
            raise EyeCentersEstimationError("Failed to refit a valid circle to the inlier polygon points")

        return float(cx), float(cy)
