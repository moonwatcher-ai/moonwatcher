from typing import List


def box_xywh_abs_to_xyxy_abs(box_xywh_abs: List[float]):
    x1_abs, y1_abs, w_abs, h_abs = box_xywh_abs
    x2_abs = x1_abs + w_abs
    y2_abs = y1_abs + h_abs
    box_xyxy_abs = [x1_abs, y1_abs, x2_abs, y2_abs]
    box_xyxy_abs = [float(item) for item in box_xyxy_abs]
    return box_xyxy_abs
