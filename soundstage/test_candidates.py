"""
candidates.py 의 generate_candidates() 테스트

실행 방법:  pytest soundstage/test_candidates.py
"""

import pytest

from candidates import generate_candidates

# 테스트용 기본 방: demo_basic.py와 같은 10 x 8 x 3 m
ROOM = (10.0, 8.0, 3.0)


def test_all_candidates_inside_room():
    """모든 후보가 벽에서 min_wall_distance 이상 안쪽, 지정한 높이에 있어야 한다."""
    margin = 0.7
    cands = generate_candidates(ROOM, step=0.3, min_wall_distance=margin, height=2.5)
    assert len(cands) > 0
    for x, y, z in cands:
        assert margin <= x <= ROOM[0] - margin
        assert margin <= y <= ROOM[1] - margin
        assert z == 2.5


def test_forbidden_zones_excluded():
    """금지 구역 안(경계 포함)의 점은 후보에 없어야 한다."""
    zones = [(4.0, 0.0, 6.0, 2.0), (0.0, 6.0, 3.0, 8.0)]
    cands = generate_candidates(ROOM, step=0.5, min_wall_distance=0.5, height=2.0,
                                forbidden_zones=zones)
    assert len(cands) > 0
    for x, y, _ in cands:
        for x_min, y_min, x_max, y_max in zones:
            assert not (x_min <= x <= x_max and y_min <= y <= y_max)

    # 금지 구역이 없을 때보다 후보가 줄어들어야 한다 (실제로 뭔가 걸러졌는지 확인)
    all_cands = generate_candidates(ROOM, step=0.5, min_wall_distance=0.5, height=2.0)
    assert len(cands) < len(all_cands)


@pytest.mark.parametrize("limit", [1, 2, 7, 20, 79, 80, 1000])
def test_max_candidates_limit(limit):
    """max_candidates를 주면 후보 수가 그 값 이하이고, 원래 후보 중에서만 골라야 한다."""
    full = generate_candidates(ROOM, step=1.0, min_wall_distance=0.5, height=2.5)  # 80개
    limited = generate_candidates(ROOM, step=1.0, min_wall_distance=0.5, height=2.5,
                                  max_candidates=limit)
    assert len(limited) <= limit
    assert len(limited) == min(limit, len(full))
    assert len(set(limited)) == len(limited)        # 같은 점이 두 번 뽑히지 않음
    assert set(limited) <= set(full)                # 원래 후보에 있던 점만


def test_max_candidates_spread_evenly():
    """줄인 후보가 방 앞쪽과 뒤쪽 양 끝을 모두 포함해야 한다 (한쪽에 몰리지 않음)."""
    limited = generate_candidates(ROOM, step=1.0, min_wall_distance=0.5, height=2.5,
                                  max_candidates=10)
    ys = [y for _, y, _ in limited]
    assert min(ys) == 0.5
    assert max(ys) == 7.5


@pytest.mark.parametrize("kwargs", [
    dict(room_dims=(0, 8, 3)),                          # 방 가로 0
    dict(room_dims=(10, -1, 3)),                        # 방 세로 음수
    dict(room_dims=(10, 8, 0)),                         # 방 높이 0
    dict(room_dims=(10, 8)),                            # 치수 2개뿐
    dict(room_dims=(10, "팔", 3)),                       # 숫자가 아님
    dict(step=0),                                       # 격자 간격 0
    dict(step=-0.5),                                    # 격자 간격 음수
    dict(min_wall_distance=-0.1),                       # 벽 거리 음수
    dict(min_wall_distance=4.0),                        # 세로 8m에서 양쪽 4m → 공간 없음
    dict(height=3.0),                                   # 천장 높이와 같음
    dict(height=3.5),                                   # 천장보다 높음
    dict(height=0),                                     # 바닥
    dict(forbidden_zones=[(6.0, 0.0, 4.0, 2.0)]),       # x_min > x_max
    dict(forbidden_zones=[(1.0, 2.0, 3.0)]),            # 좌표 3개뿐
    dict(max_candidates=0),                             # 상한 0
    dict(max_candidates=2.5),                           # 정수가 아님
])
def test_invalid_input_raises(kwargs):
    """잘못된 입력이면 ValueError가 나야 한다."""
    args = dict(room_dims=ROOM, step=1.0, min_wall_distance=0.5, height=2.5)
    args.update(kwargs)
    with pytest.raises(ValueError):
        generate_candidates(**args)


def test_error_message_is_korean():
    """오류 메시지가 무엇이 잘못됐는지 한국어로 알려 줘야 한다."""
    with pytest.raises(ValueError, match="천장"):
        generate_candidates(ROOM, step=1.0, min_wall_distance=0.5, height=3.0)
