import math

import numpy as np
import pytest

from iris.io.dataclasses import GeometryPolygons
from iris.nodes.eye_properties_estimation.circle_fit_for_eye_center_method import CircleFitEyeCenterMethod
from tests.unit_tests.utils import generate_arc


@pytest.fixture
def algorithm() -> CircleFitEyeCenterMethod:
    return CircleFitEyeCenterMethod(mad_scale=3.0)


def test_estimation_on_mock_example(algorithm: CircleFitEyeCenterMethod) -> None:
    pupil_radius = 25.0
    iris_radius = 100.0
    eyeball_radius = 400.0
    pupil_center_x, pupil_center_y = 95.0, 145.0
    iris_center_x, iris_center_y = 100.0, 155.0

    mock_polygons = GeometryPolygons(
        pupil_array=generate_arc(pupil_radius, pupil_center_x, pupil_center_y, 0.0, 2 * np.pi),
        iris_array=generate_arc(iris_radius, iris_center_x, iris_center_y, 0.0, 2 * np.pi),
        eyeball_array=generate_arc(eyeball_radius, iris_center_x, iris_center_y, 0.0, 2 * np.pi),
    )

    result = algorithm(mock_polygons)

    assert math.isclose(result.pupil_x, pupil_center_x, rel_tol=0.1)
    assert math.isclose(result.pupil_y, pupil_center_y, rel_tol=0.1)
    assert math.isclose(result.iris_x, iris_center_x, rel_tol=0.1)
    assert math.isclose(result.iris_y, iris_center_y, rel_tol=0.1)
