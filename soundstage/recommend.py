"""
soundstage 이슈 4: 최적 스피커 배치 추천 + 기준 배치와 비교

이슈 3(evaluate.py)에서 후보 위치마다 균일도(std_db)를 계산했다.
여기서는 그 결과 중 "가장 좋은 자리"를 하나 골라 추천한다.

고르는 규칙
1) 최소 음향 수준 조건:
   후보의 min_db(가장 작게 들리는 자리의 dB)가 기준값 이상이어야 한다.
   기준값을 따로 주지 않으면 "기준 배치의 min_db - margin_db"를 쓴다.
   → 뜻: "맨 뒷자리처럼 가장 작게 들리는 자리가 지금(기준 배치)보다 나빠지면 안 된다."
   (균일도만 보면, 모든 자리가 똑같이 '작게' 들리는 자리를 고를 수도 있어서 이 조건이 필요하다)
2) 조건을 통과한 후보 중 std_db가 가장 작은(= 가장 고르게 들리는) 후보를 추천한다.
3) 기준 배치(지금 스피커가 있는 자리)와 비교해 균일도가 몇 % 좋아졌는지 계산한다.

실행 방법:  python soundstage/recommend.py
"""

import math

from demo_basic import (
    ROOM_WIDTH, ROOM_LENGTH, ROOM_HEIGHT,
    FS, WALL_ABSORPTION, MAX_ORDER,
)
from candidates import (
    generate_candidates,
    DEFAULT_STEP, DEFAULT_MIN_WALL_DISTANCE, DEFAULT_HEIGHT, DEFAULT_FORBIDDEN_ZONES,
)
from evaluate import evaluate_candidates, evaluate_position, make_listening_points

# 기준 배치("전면 중앙 상단")에서 스피커가 앞벽(y=0)으로부터 떨어진 거리 [m]
BASELINE_FRONT_DISTANCE = 0.5


def select_best(results, baseline, min_level_db=None, margin_db=0.0, top_n=5):
    """
    평가 결과 목록에서 최적 배치를 고른다. (시뮬레이션 계산 없이 숫자만 비교하는 순수 함수)

    매개변수
    --------
    results : evaluate_candidates()가 돌려준 딕셔너리 리스트
              (각 딕셔너리에 "std_db", "min_db"가 있어야 한다)
    baseline : 기준 배치의 평가 딕셔너리 1개 (evaluate_position()의 결과 등)
    min_level_db : 최소 음향 수준 기준값 [dB]. None이면 baseline["min_db"] - margin_db
    margin_db : min_level_db가 None일 때 기준 배치보다 이만큼은 작아져도 봐준다 [dB]
    top_n : 통과 후보 중 상위 몇 개를 돌려줄지

    반환값 (딕셔너리)
    ------
    best                : 추천(최적) 후보 딕셔너리
    top                 : 통과 후보를 std_db 오름차순으로 정렬한 상위 top_n개
    baseline            : 입력한 기준 배치 딕셔너리 그대로
    threshold_db        : 실제로 사용한 최소 음향 수준 기준값 [dB]
    num_candidates      : 전체 후보 수
    num_passed          : 조건을 통과한 후보 수
    improvement_percent : 균일도 개선율 [%]
                          = (기준 std_db - 최적 std_db) / 기준 std_db * 100
                          (양수면 기준보다 더 균일해졌다는 뜻)
    results             : 입력한 후보 전체 평가 결과 (시각화 히트맵에서 사용, 이슈 5에서 추가)
    """
    # -----------------------------------------------------------------
    # 1) 입력 검사
    # -----------------------------------------------------------------
    results = list(results)
    if len(results) == 0:
        raise ValueError("평가 결과(results)가 비어 있습니다. 후보를 하나 이상 평가해 주세요.")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError(f"top_n은 1 이상의 정수여야 합니다. (받은 값: {top_n!r})")

    # -----------------------------------------------------------------
    # 2) 최소 음향 수준 기준값 정하기
    # -----------------------------------------------------------------
    if min_level_db is None:
        # 기준 배치의 "가장 작게 들리는 자리" 크기에서 margin만큼 봐준 값
        threshold = float(baseline["min_db"]) - float(margin_db)
    else:
        # 사용자가 직접 준 값이 있으면 그것을 그대로 쓴다 (margin_db는 무시)
        threshold = float(min_level_db)
    if not math.isfinite(threshold):
        raise ValueError(f"최소 음향 수준 기준값이 올바른 숫자가 아닙니다. (계산된 값: {threshold})")

    # -----------------------------------------------------------------
    # 3) 조건 통과 후보만 남기고, std_db가 작은 순으로 정렬
    # -----------------------------------------------------------------
    passed = [r for r in results if r["min_db"] >= threshold]

    if len(passed) == 0:
        best_min = max(r["min_db"] for r in results)
        raise ValueError(
            f"최소 음향 수준 조건(min_db >= {threshold:.2f} dB)을 통과한 후보가 하나도 없습니다. "
            f"후보 중 가장 큰 min_db는 {best_min:.2f} dB입니다. "
            f"min_level_db를 낮추거나 margin_db를 키워 보세요."
        )

    # sorted는 값이 같으면 원래 순서를 유지하므로(안정 정렬), 동점이면 앞쪽 후보가 먼저 온다
    ranked = sorted(passed, key=lambda r: r["std_db"])
    best = ranked[0]

    # -----------------------------------------------------------------
    # 4) 기준 배치 대비 균일도 개선율
    # -----------------------------------------------------------------
    base_std = float(baseline["std_db"])
    if base_std > 0:
        improvement = (base_std - best["std_db"]) / base_std * 100
    else:
        # 기준 배치가 이미 완벽히 균일(std_db=0)하면 더 나아질 수 없으므로 0 %로 둔다
        improvement = 0.0

    return {
        "best": best,
        "top": ranked[:top_n],
        "baseline": baseline,
        "threshold_db": threshold,
        "num_candidates": len(results),
        "num_passed": len(passed),
        "improvement_percent": float(improvement),
        "results": results,
    }


def default_baseline_position(room_dims, candidates):
    """
    기준 배치 "전면 중앙 상단" 위치를 계산한다.

    x = 방 가로의 절반 (가운데)
    y = 앞벽(y=0)에서 BASELINE_FRONT_DISTANCE(0.5 m)
    z = 후보들의 설치 높이와 같은 값 (같은 높이끼리 비교해야 공정하므로)
    """
    candidates = list(candidates)
    if len(candidates) == 0:
        raise ValueError("후보 위치가 비어 있어 기준 배치 높이를 정할 수 없습니다.")
    heights = {float(c[2]) for c in candidates}
    if len(heights) != 1:
        raise ValueError(
            f"후보들의 설치 높이가 서로 다릅니다: {sorted(heights)}. "
            f"기준 배치 높이를 정할 수 없으니 baseline_pos를 직접 넣어 주세요."
        )
    return (float(room_dims[0]) / 2, BASELINE_FRONT_DISTANCE, heights.pop())


def recommend_placement(room_dims, candidates, listening_points, baseline_pos=None,
                        absorption=WALL_ABSORPTION, max_order=MAX_ORDER, fs=FS,
                        min_level_db=None, margin_db=0.0, top_n=5):
    """
    후보 전체와 기준 배치를 평가한 뒤 select_best()로 최적 배치를 고른다.

    baseline_pos가 None이면 default_baseline_position()으로 "전면 중앙 상단"을 쓴다.
    기준 배치는 후보 격자에 없을 수도 있으므로(예: 금지 구역 안) 따로 평가한다.

    반환값: select_best()의 반환 딕셔너리
            (baseline 딕셔너리에는 "position" 키도 들어 있다)
    """
    candidates = list(candidates)
    if baseline_pos is None:
        baseline_pos = default_baseline_position(room_dims, candidates)

    # 1) 기준 배치 평가 (evaluate_candidates의 결과와 모양을 맞추려고 position을 붙인다)
    base_result = evaluate_position(room_dims, baseline_pos, listening_points,
                                    absorption=absorption, max_order=max_order, fs=fs)
    baseline = {"position": tuple(float(v) for v in baseline_pos), **base_result}

    # 2) 후보 전체 평가
    results = evaluate_candidates(room_dims, candidates, listening_points,
                                  absorption=absorption, max_order=max_order, fs=fs)

    # 3) 숫자만 비교해서 최적 배치 고르기
    return select_best(results, baseline, min_level_db=min_level_db,
                       margin_db=margin_db, top_n=top_n)


def _fmt_pos(pos):
    """(x, y, z) 좌표를 '(5.0, 0.5, 2.5) m' 모양 글자로 바꾼다."""
    return "(" + ", ".join(f"{v:.1f}" for v in pos) + ") m"


# 이 파일을 직접 실행했을 때만 아래 예제가 실행된다
if __name__ == "__main__":
    import time

    room_dims = (ROOM_WIDTH, ROOM_LENGTH, ROOM_HEIGHT)
    cands = generate_candidates(room_dims, DEFAULT_STEP, DEFAULT_MIN_WALL_DISTANCE,
                                DEFAULT_HEIGHT, forbidden_zones=DEFAULT_FORBIDDEN_ZONES)
    listeners = make_listening_points()

    start = time.perf_counter()
    rec = recommend_placement(room_dims, cands, listeners)
    elapsed = time.perf_counter() - start

    base = rec["baseline"]
    best = rec["best"]
    line = "=" * 72

    print(line)
    print(" 스피커 최적 배치 추천 결과")
    print(line)
    print(f"방 크기: {ROOM_WIDTH} x {ROOM_LENGTH} x {ROOM_HEIGHT} m, 청취 지점 {len(listeners)}개, "
          f"흡음률 {WALL_ABSORPTION}, max_order {MAX_ORDER}")
    print()
    print("[기준 배치] 전면 중앙 상단")
    print(f"  위치    : {_fmt_pos(base['position'])}")
    print(f"  std_db  : {base['std_db']:.3f} dB   (작을수록 균일)")
    print(f"  min_db  : {base['min_db']:.2f} dB  (가장 작게 들리는 자리)")
    print()
    print("[조건] 최소 음향 수준")
    print(f"  기준값  : min_db >= {rec['threshold_db']:.2f} dB")
    print(f"  후보    : 전체 {rec['num_candidates']}개 중 {rec['num_passed']}개 통과")
    print()
    print("[추천 배치]")
    print(f"  위치    : {_fmt_pos(best['position'])}")
    print(f"  std_db  : {best['std_db']:.3f} dB")
    print(f"  min_db  : {best['min_db']:.2f} dB  "
          f"(기준 대비 {best['min_db'] - base['min_db']:+.2f} dB)")
    print()
    print(f"[균일도 개선율] {rec['improvement_percent']:.1f} %  "
          f"(std_db {base['std_db']:.3f} → {best['std_db']:.3f} dB)")
    print()
    print(f"[조건 통과 후보 상위 {len(rec['top'])}개] (std_db 작은 순)")
    print(f"{'순위':>4} | {'x':>5} {'y':>5} {'z':>5} | {'std_db':>7} {'mean_db':>8} "
          f"{'min_db':>8} | {'개선율%':>7}")
    print("-" * 72)
    for rank, r in enumerate(rec["top"], start=1):
        x, y, z = r["position"]
        gain = (base["std_db"] - r["std_db"]) / base["std_db"] * 100
        print(f"{rank:>4} | {x:5.1f} {y:5.1f} {z:5.1f} | {r['std_db']:7.3f} "
              f"{r['mean_db']:8.2f} {r['min_db']:8.2f} | {gain:7.1f}")
    print("-" * 72)
    print(f"계산 시간: {elapsed:.2f}초")
