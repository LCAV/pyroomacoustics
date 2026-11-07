"""
evaluate.py 의 evaluate_position() / evaluate_candidates() 테스트

실행 방법:  pytest soundstage/test_evaluate.py
"""

import pytest

from demo_basic import make_listener_grid
from evaluate import evaluate_candidates, evaluate_position, make_listening_points

# 테스트용 기본 방: demo_basic.py와 같은 10 x 8 x 3 m
ROOM = (10.0, 8.0, 3.0)

# 반환 딕셔너리에 꼭 있어야 하는 키
REQUIRED_KEYS = {"levels_db", "mean_db", "std_db", "min_db",
                 "mean_linear", "std_linear", "cv_linear"}


def test_listening_points_match_demo_basic():
    """make_listening_points()가 demo_basic.py의 12개 격자와 같은 좌표여야 한다."""
    points = make_listening_points()
    grid = make_listener_grid()
    assert len(points) == 12
    for i, (x, y, z) in enumerate(points):
        assert (x, y, z) == pytest.approx(tuple(grid[:, i]))
        assert z == pytest.approx(1.2)


def test_result_has_required_keys_and_lengths():
    """필요한 키가 모두 있고, levels_db 길이가 청취 지점 수와 같아야 한다."""
    points = make_listening_points()
    result = evaluate_position(ROOM, (5.0, 0.5, 2.5), points)
    assert REQUIRED_KEYS <= set(result.keys())
    assert len(result["levels_db"]) == len(points)


def test_std_db_is_not_negative():
    """표준편차는 정의상 0 이상이어야 한다."""
    result = evaluate_position(ROOM, (2.0, 3.0, 2.5), make_listening_points())
    assert result["std_db"] >= 0
    assert result["std_linear"] >= 0


def test_single_point_has_zero_std():
    """청취 지점이 한 곳뿐이면 흩어질 것이 없으므로 std_db는 0이다."""
    result = evaluate_position(ROOM, (5.0, 0.5, 2.5), [(5.0, 4.0, 1.2)])
    assert result["std_db"] == pytest.approx(0.0)
    assert result["min_db"] == pytest.approx(result["mean_db"])


@pytest.mark.parametrize("speaker, listeners", [
    ((11.0, 4.0, 2.5), [(5.0, 4.0, 1.2)]),   # 스피커가 x 방향으로 방 밖
    ((5.0, 4.0, 0.0), [(5.0, 4.0, 1.2)]),    # 스피커가 바닥 위(경계)
    ((5.0, 4.0, 2.5), [(5.0, -1.0, 1.2)]),   # 청취 지점이 방 밖
    ((5.0, 4.0, 2.5), [(5.0, 4.0, 3.5)]),    # 청취 지점이 천장 위
])
def test_outside_room_raises(speaker, listeners):
    """스피커나 청취 지점이 방 밖이면 ValueError가 나야 한다."""
    with pytest.raises(ValueError, match="방 밖"):
        evaluate_position(ROOM, speaker, listeners)


def test_center_is_more_uniform_than_corner():
    """방 한가운데 위쪽 스피커가 구석 스피커보다 균일해야(std_db가 작아야) 한다."""
    points = make_listening_points()
    center = evaluate_position(ROOM, (5.0, 4.0, 2.5), points)
    corner = evaluate_position(ROOM, (0.5, 0.5, 2.5), points)
    assert center["std_db"] < corner["std_db"]


def test_evaluate_candidates_keeps_order_and_position():
    """evaluate_candidates는 후보 순서대로 position과 평가 항목을 함께 돌려준다."""
    cands = [(1.0, 1.0, 2.5), (5.0, 4.0, 2.5)]
    results = evaluate_candidates(ROOM, cands, make_listening_points())
    assert len(results) == 2
    for cand, r in zip(cands, results):
        assert r["position"] == cand
        assert REQUIRED_KEYS <= set(r.keys())
