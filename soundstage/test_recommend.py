"""
recommend.py 의 select_best() / recommend_placement() 테스트

select_best 테스트는 시뮬레이션 없이 손으로 만든 "가짜 평가 결과"를 써서 빠르게 돈다.
recommend_placement 테스트만 실제 pyroomacoustics 계산을 한 번 한다.

실행 방법:  pytest soundstage/test_recommend.py
"""

import pytest

from candidates import generate_candidates
from evaluate import evaluate_position, make_listening_points
from recommend import default_baseline_position, recommend_placement, select_best


def fake(name, std_db, min_db):
    """테스트용 가짜 평가 결과. position 대신 이름표를 넣어 누가 뽑혔는지 알아보기 쉽게 한다."""
    return {"position": name, "std_db": std_db, "min_db": min_db}


# 기준 배치: std_db 4.0 dB, 가장 작은 자리 -20 dB
BASELINE = fake("기준", std_db=4.0, min_db=-20.0)

# 가짜 후보 5개
RESULTS = [
    fake("A", std_db=1.0, min_db=-25.0),   # 가장 균일하지만 너무 작게 들림 → 조건 미달
    fake("B", std_db=2.0, min_db=-19.0),   # 조건 통과, 통과 후보 중 가장 균일
    fake("C", std_db=3.0, min_db=-18.0),   # 조건 통과
    fake("D", std_db=2.5, min_db=-20.0),   # min_db가 기준과 딱 같음 → 통과(>=)
    fake("E", std_db=5.0, min_db=-15.0),   # 조건 통과, 기준보다 덜 균일
]


def test_failing_candidate_is_not_selected():
    """std_db가 가장 작아도 최소 음향 수준을 못 넘으면 최적으로 뽑히면 안 된다."""
    rec = select_best(RESULTS, BASELINE)
    assert rec["best"]["position"] != "A"
    assert all(r["position"] != "A" for r in rec["top"])
    assert rec["threshold_db"] == pytest.approx(-20.0)
    assert rec["num_candidates"] == 5
    assert rec["num_passed"] == 4          # A만 탈락, D는 경계값이라 통과


def test_best_has_smallest_std_among_passed():
    """최적 배치의 std_db는 통과한 후보들 중 가장 작아야 하고, top은 오름차순이어야 한다."""
    rec = select_best(RESULTS, BASELINE)
    passed = [r for r in RESULTS if r["min_db"] >= rec["threshold_db"]]
    assert rec["best"]["std_db"] == min(r["std_db"] for r in passed)
    assert rec["best"]["position"] == "B"
    assert [r["position"] for r in rec["top"]] == ["B", "D", "C", "E"]
    assert rec["baseline"] is BASELINE


def test_improvement_percent_matches_hand_calculation():
    """개선율 = (4.0 - 2.0) / 4.0 * 100 = 50 %"""
    rec = select_best(RESULTS, BASELINE)
    assert rec["improvement_percent"] == pytest.approx(50.0)


def test_top_n_limits_length():
    """top_n개까지만 돌려준다."""
    rec = select_best(RESULTS, BASELINE, top_n=2)
    assert [r["position"] for r in rec["top"]] == ["B", "D"]


def test_margin_db_relaxes_threshold():
    """margin_db=6이면 기준값이 -26 dB로 내려가 A도 통과하고 최적이 된다."""
    rec = select_best(RESULTS, BASELINE, margin_db=6.0)
    assert rec["threshold_db"] == pytest.approx(-26.0)
    assert rec["best"]["position"] == "A"
    assert rec["num_passed"] == 5


def test_no_candidate_passes_raises_value_error():
    """통과 후보가 없으면 기준값과 후보 중 최고 min_db가 들어간 ValueError가 나야 한다."""
    with pytest.raises(ValueError) as info:
        select_best(RESULTS, BASELINE, min_level_db=-10.0)
    message = str(info.value)
    assert "-10.00" in message      # 기준값
    assert "-15.00" in message      # 후보 중 최고 min_db (E)


def test_explicit_min_level_db_is_used_as_threshold():
    """min_level_db를 직접 주면 기준 배치·margin과 상관없이 그 값이 기준이 된다."""
    rec = select_best(RESULTS, BASELINE, min_level_db=-18.5, margin_db=100.0)
    assert rec["threshold_db"] == pytest.approx(-18.5)
    # min_db >= -18.5 인 후보는 C(-18), E(-15)뿐 → 둘 중 std_db가 작은 C
    assert rec["num_passed"] == 2
    assert rec["best"]["position"] == "C"


def test_empty_results_raises_value_error():
    with pytest.raises(ValueError):
        select_best([], BASELINE)


# ---------------------------------------------------------------------
# 실제 시뮬레이션을 돌리는 테스트 (작은 후보 집합으로 빠르게)
# ---------------------------------------------------------------------

def test_recommend_placement_end_to_end():
    room = (10.0, 8.0, 3.0)
    # 2 m 간격 + 금지 구역 → 후보 수를 줄여 빠르게 돈다
    cands = generate_candidates(room, 2.0, 0.5, 2.5,
                                forbidden_zones=[(4.0, 0.0, 6.0, 2.0)])
    listeners = make_listening_points()

    # baseline_pos 자동 계산: x=가로 절반, y=0.5, z=후보 높이
    assert default_baseline_position(room, cands) == pytest.approx((5.0, 0.5, 2.5))

    rec = recommend_placement(room, cands, listeners, top_n=3)

    # 기준 배치는 금지 구역 안이라 후보 격자에는 없지만, 따로 평가되어 들어 있어야 한다
    assert rec["baseline"]["position"] == pytest.approx((5.0, 0.5, 2.5))
    assert (5.0, 0.5, 2.5) not in cands
    direct = evaluate_position(room, (5.0, 0.5, 2.5), listeners)
    assert rec["baseline"]["std_db"] == pytest.approx(direct["std_db"])

    # 선택 결과의 기본 성질
    assert rec["num_candidates"] == len(cands)
    assert 1 <= rec["num_passed"] <= len(cands)
    assert rec["threshold_db"] == pytest.approx(rec["baseline"]["min_db"])
    assert rec["best"]["min_db"] >= rec["threshold_db"]
    assert rec["best"]["position"] in cands
    assert rec["best"] is rec["top"][0]
    assert len(rec["top"]) <= 3
    stds = [r["std_db"] for r in rec["top"]]
    assert stds == sorted(stds)
