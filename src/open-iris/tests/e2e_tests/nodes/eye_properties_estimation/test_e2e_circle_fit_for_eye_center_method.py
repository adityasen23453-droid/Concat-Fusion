import math
import os
import pickle
from typing import Any

import pytest

from iris.nodes.eye_properties_estimation.circle_fit_for_eye_center_method import CircleFitEyeCenterMethod


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks", "eye_center_method")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


@pytest.fixture
def algorithm() -> CircleFitEyeCenterMethod:
    return CircleFitEyeCenterMethod(mad_scale=3.0)


def test_e2e_bisectors_method_algorithm(algorithm: CircleFitEyeCenterMethod) -> None:
    mock_polygons = load_mock_pickle(name="geometry_polygons")
    expected_result = load_mock_pickle(name="circle_fit_method_e2e_expected_result")

    result = algorithm(geometries=mock_polygons)

    assert math.isclose(result.pupil_x, expected_result.pupil_x)
    assert math.isclose(result.pupil_y, expected_result.pupil_y)
    assert math.isclose(result.iris_x, expected_result.iris_x)
    assert math.isclose(result.iris_y, expected_result.iris_y)
