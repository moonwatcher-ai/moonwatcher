import pytest

from moonwatcher.utils.bbox_utils import box_xywh_abs_to_xyxy_abs


@pytest.mark.parametrize(
    "input_box, expected_output",
    [
        ([1, 2, 3, 4], [1.0, 2.0, 4.0, 6.0]),
        ([0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0]),
        ([-1, -2, -3, -4], [-1.0, -2.0, -4.0, -6.0]),
        ([1.5, 2.5, 3.5, 4.5], [1.5, 2.5, 5.0, 7.0]),
        (
            [1000000, 2000000, 3000000, 4000000],
            [1000000.0, 2000000.0, 4000000.0, 6000000.0],
        ),
        ([100, 200, 50, 75], [100.0, 200.0, 150.0, 275.0]),
        ([100, 200, 0, 0], [100.0, 200.0, 100.0, 200.0]),
        ([100, 200, -50, -75], [100.0, 200.0, 50.0, 125.0]),
        ([100.5, 200.5, 50.25, 75.75], [100.5, 200.5, 150.75, 276.25]),
        ([1e9, 1e9, 1e9, 1e9], [1e9, 1e9, 2e9, 2e9]),
    ],
)
def test_box_xywh_abs_to_xyxy_abs(input_box, expected_output):
    assert box_xywh_abs_to_xyxy_abs(input_box) == expected_output
