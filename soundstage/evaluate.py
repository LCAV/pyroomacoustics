"""
soundstage 이슈 3: 후보 위치별 음향 균일도 평가

스피커를 어떤 후보 위치에 놓았을 때, 청취 지점(관객 자리)들에서 소리 크기가
얼마나 "고르게" 들리는지를 숫자로 평가한다.

- 각 청취 지점의 소리 크기 = RIR 에너지(샘플 제곱의 합)  ← demo_basic.py와 같은 방식
- 소리 크기를 dB(데시벨)로 바꾼 뒤, 지점들 사이의 표준편차(std_db)를 균일도 점수로 쓴다.
  std_db가 작을수록 모든 자리에서 비슷한 크기로 들린다 = 더 균일하다.

실행 방법:  python soundstage/evaluate.py
"""

import time

import numpy as np                 # 숫자 배열 계산용 라이브러리
import pyroomacoustics as pra      # 방 음향 시뮬레이션 라이브러리

# demo_basic.py의 설정과 청취 지점 격자를 그대로 가져온다.
# (숫자를 복사하면 한쪽만 바뀌었을 때 값이 어긋나므로 직접 import한다)
from demo_basic import (
    ROOM_WIDTH, ROOM_LENGTH, ROOM_HEIGHT,
    FS, WALL_ABSORPTION, MAX_ORDER,
    make_listener_grid,
)
from candidates import generate_candidates

# 에너지가 0일 때 log10(0) = -무한대 오류를 막기 위해 더하는 아주 작은 값
EPS = 1e-12


def make_listening_points():
    """
    demo_basic.py와 똑같은 12개 청취 지점(3열 x 4행, 높이 1.2 m)을
    [(x, y, z), ...] 리스트로 돌려준다.

    demo_basic.make_listener_grid()는 (3, 지점 수) 모양 배열을 주므로,
    전치(.T)해서 지점 하나당 (x, y, z) 한 줄이 되도록 바꾼다.
    """
    grid = make_listener_grid()
    return [tuple(float(v) for v in point) for point in grid.T]


def _check_inside(pos, room_dims, name):
    """좌표가 방 안에 있는지 확인하고, 밖(또는 벽 위)이면 ValueError를 낸다."""
    pos = np.asarray(pos, dtype=float)
    size = np.asarray(room_dims, dtype=float)
    if pos.shape != (3,):
        raise ValueError(f"{name} 좌표는 (x, y, z) 3개 숫자여야 합니다. (받은 값: {pos.tolist()})")
    # 모든 축에서 0 < 좌표 < 방 크기 이어야 방 "내부"이다
    if np.any(pos <= 0) or np.any(pos >= size):
        raise ValueError(
            f"{name} 좌표 {tuple(pos.tolist())} 가 방 밖에 있습니다. "
            f"(방 크기: {tuple(size.tolist())} m, 각 좌표는 0보다 크고 방 크기보다 작아야 합니다)"
        )
    return pos


def evaluate_position(room_dims, speaker_pos, listening_points,
                      absorption=WALL_ABSORPTION, max_order=MAX_ORDER, fs=FS):
    """
    스피커 1개를 speaker_pos에 두었을 때 청취 지점들의 소리 크기와 균일도를 계산한다.

    매개변수
    --------
    room_dims : (가로, 세로, 높이) 방 치수 [m]
    speaker_pos : (x, y, z) 스피커 위치 [m]
    listening_points : [(x, y, z), ...] 청취 지점 목록
    absorption : 벽의 에너지 흡음률 (0~1)
    max_order : 최대 반사 차수
    fs : 샘플링 주파수 [Hz]

    반환값 (딕셔너리)
    ------
    levels_db   : 지점별 소리 크기 [dB] 리스트
    mean_db     : dB 평균
    std_db      : dB 표준편차  ← 균일도 점수 (작을수록 균일)
    min_db      : 가장 작게 들리는 지점의 dB
    mean_linear : 에너지(선형 값) 평균
    std_linear  : 에너지(선형 값) 표준편차
    cv_linear   : 변동계수 = std_linear / mean_linear (평균 대비 흩어진 정도)
    """
    # -----------------------------------------------------------------
    # 1) 입력 검사: 방 크기, 스피커, 청취 지점이 올바른지 확인
    # -----------------------------------------------------------------
    room_dims = np.asarray(room_dims, dtype=float)
    if room_dims.shape != (3,) or np.any(room_dims <= 0):
        raise ValueError(f"room_dims는 0보다 큰 (가로, 세로, 높이) 3개 숫자여야 합니다. "
                         f"(받은 값: {room_dims.tolist()})")

    speaker = _check_inside(speaker_pos, room_dims, "스피커")

    points = [_check_inside(p, room_dims, f"청취 지점 {i}")
              for i, p in enumerate(listening_points)]
    if len(points) == 0:
        raise ValueError("청취 지점이 하나 이상 있어야 합니다.")

    # -----------------------------------------------------------------
    # 2) 방 만들기 (demo_basic.py와 같은 설정)
    # -----------------------------------------------------------------
    material = pra.Material(energy_absorption=float(absorption))
    room = pra.ShoeBox(
        room_dims.tolist(),
        fs=fs,
        materials=material,
        max_order=max_order,
        air_absorption=False,   # 공기 흡음은 끔 (단순화)
        ray_tracing=False,      # 레이 트레이싱 끔 (빠른 계산)
    )

    # -----------------------------------------------------------------
    # 3) 스피커와 청취 지점(마이크) 배치 후 RIR 계산
    # -----------------------------------------------------------------
    room.add_source(speaker.tolist())
    # pyroomacoustics는 (3, 지점 수) 모양을 원하므로 전치(.T)한다
    room.add_microphone_array(np.array(points).T)
    room.compute_rir()

    # -----------------------------------------------------------------
    # 4) 지점별 에너지 → dB 변환
    # -----------------------------------------------------------------
    # room.rir[m][0] : m번 청취 지점, 0번(유일한) 스피커의 RIR
    energies = np.array([np.sum(room.rir[m][0] ** 2) for m in range(len(points))])
    # dB = 10 * log10(에너지). 에너지가 0이어도 오류가 나지 않게 EPS를 더한다.
    levels_db = 10 * np.log10(energies + EPS)

    # -----------------------------------------------------------------
    # 5) 통계 계산
    # -----------------------------------------------------------------
    mean_linear = float(np.mean(energies))
    std_linear = float(np.std(energies))
    # 평균이 0이면 나눌 수 없으므로 0으로 둔다
    cv_linear = std_linear / mean_linear if mean_linear > 0 else 0.0

    return {
        "levels_db": [float(v) for v in levels_db],
        "mean_db": float(np.mean(levels_db)),
        "std_db": float(np.std(levels_db)),
        "min_db": float(np.min(levels_db)),
        "mean_linear": mean_linear,
        "std_linear": std_linear,
        "cv_linear": float(cv_linear),
    }


def evaluate_candidates(room_dims, candidates, listening_points,
                        absorption=WALL_ABSORPTION, max_order=MAX_ORDER, fs=FS):
    """
    후보 위치 목록(이슈 2의 결과) 각각을 evaluate_position()으로 평가한다.

    반환값: [{"position": (x, y, z), "levels_db": ..., "std_db": ..., ...}, ...]
            (입력한 후보 순서 그대로)
    """
    results = []
    for pos in candidates:
        result = evaluate_position(room_dims, pos, listening_points,
                                   absorption=absorption, max_order=max_order, fs=fs)
        # "position"을 맨 앞에 두고 나머지 평가 항목을 이어 붙인다
        results.append({"position": tuple(float(v) for v in pos), **result})
    return results


def _print_table(title, rows):
    """평가 결과 몇 개를 표로 출력한다."""
    print(title)
    print(f"{'순위':>4} | {'x':>5} {'y':>5} {'z':>5} | {'std_db':>7} {'mean_db':>8} "
          f"{'min_db':>8} | {'cv_linear':>9}")
    print("-" * 68)
    for rank, r in rows:
        x, y, z = r["position"]
        print(f"{rank:>4} | {x:5.1f} {y:5.1f} {z:5.1f} | {r['std_db']:7.3f} "
              f"{r['mean_db']:8.2f} {r['min_db']:8.2f} | {r['cv_linear']:9.3f}")
    print()


# 이 파일을 직접 실행했을 때만 아래 예제가 실행된다
if __name__ == "__main__":
    room_dims = (ROOM_WIDTH, ROOM_LENGTH, ROOM_HEIGHT)

    # 후보 생성 설정: candidates.py의 실행 예제(__main__)와 같은 값
    # (그 값들은 __main__ 블록 안에 있어 import할 수 없으므로 여기 다시 적는다)
    STEP = 1.0                      # 1 m 간격 격자
    MIN_WALL_DISTANCE = 0.5         # 벽에서 최소 0.5 m
    HEIGHT = 2.5                    # 스피커 높이 (demo_basic.SPEAKER_POS의 z)
    FORBIDDEN = [(4.0, 0.0, 6.0, 2.0)]   # 앞쪽 중앙 교탁/스크린 자리

    cands = generate_candidates(room_dims, STEP, MIN_WALL_DISTANCE, HEIGHT,
                                forbidden_zones=FORBIDDEN)
    listeners = make_listening_points()

    print(f"방 크기: {room_dims} m, fs={FS} Hz, 흡음률={WALL_ABSORPTION}, "
          f"max_order={MAX_ORDER}")
    print(f"청취 지점 {len(listeners)}개, 금지 구역 {FORBIDDEN}")
    print()

    start = time.perf_counter()
    results = evaluate_candidates(room_dims, cands, listeners)
    elapsed = time.perf_counter() - start

    # std_db가 작은 순(= 균일한 순)으로 정렬
    ranked = sorted(results, key=lambda r: r["std_db"])
    n = len(ranked)

    _print_table("[가장 균일한 5개] (std_db 작은 순)",
                 [(i + 1, r) for i, r in enumerate(ranked[:5])])
    _print_table("[가장 불균일한 5개] (std_db 큰 순)",
                 [(n - i, ranked[n - 1 - i]) for i in range(min(5, n))])

    # 참고: 선형 변동계수(cv_linear)로 정렬했을 때의 1위와 비교
    by_cv = sorted(results, key=lambda r: r["cv_linear"])
    print(f"cv_linear 기준 1위: {by_cv[0]['position']}  "
          f"(std_db 기준 1위: {ranked[0]['position']})")
    print(f"전체 후보 수: {n}개, 평가에 걸린 시간: {elapsed:.2f}초")
