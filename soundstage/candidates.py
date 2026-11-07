"""
soundstage 이슈 2: 스피커를 놓을 "후보 위치"를 만드는 함수

방 바닥을 바둑판(격자)처럼 일정 간격으로 나누고, 각 칸의 점을 스피커 후보 위치로 삼는다.
- 벽에 너무 가까운 점은 뺀다. (min_wall_distance)
- 설치할 수 없는 구역(문, 교탁, 기둥 등)에 들어간 점은 뺀다. (forbidden_zones)
- 후보가 너무 많으면 방 전체에 고르게 퍼지도록 일정 간격으로 골라 개수를 줄인다. (max_candidates)
- 모든 후보는 같은 높이(height)에 놓인다.

실행 방법:  python soundstage/candidates.py
"""

import math

# 스피커 높이는 demo_basic.py의 스피커 위치(SPEAKER_POS)의 z값을 그대로 쓴다.
from demo_basic import SPEAKER_POS

# =====================================================================
# 기본 설정 상수
# candidates.py / evaluate.py / recommend.py 의 실행 예제가 모두 이 값을 같이 쓴다.
# (여러 파일에 같은 숫자를 따로 적어 두면 한쪽만 바뀌었을 때 결과가 어긋나므로
#  여기 한 곳에만 적고, 다른 파일은 import해서 쓴다)
# =====================================================================
DEFAULT_STEP = 1.0                  # 격자 간격 [m]: 1 m마다 후보 하나
DEFAULT_MIN_WALL_DISTANCE = 0.5     # 벽에서 최소 0.5 m 떨어진 곳에만 후보를 둔다
DEFAULT_HEIGHT = SPEAKER_POS[2]     # 스피커 설치 높이 [m] (demo_basic.py와 같은 2.5 m)
# 설치 금지 구역 예시: 앞쪽 중앙의 교탁/스크린 자리 (x 4~6 m, y 0~2 m)
# 튜플로 둔 이유: 리스트와 달리 실수로 내용을 바꿀(append 등) 수 없어서 상수로 안전하다.
DEFAULT_FORBIDDEN_ZONES = ((4.0, 0.0, 6.0, 2.0),)


def _to_number(value, name):
    """값을 실수(float)로 바꾼다. 숫자가 아니거나 무한대/NaN이면 오류를 낸다."""
    # True/False도 파이썬에서는 숫자(1/0)로 취급되므로 실수로 넣은 것을 막는다
    if isinstance(value, bool):
        raise ValueError(f"{name}에는 숫자를 넣어야 합니다. (받은 값: {value!r})")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}에는 숫자를 넣어야 합니다. (받은 값: {value!r})")
    if not math.isfinite(number):
        raise ValueError(f"{name}에 무한대나 NaN은 넣을 수 없습니다. (받은 값: {value!r})")
    return number


def _grid_positions(length, margin, step):
    """
    한 축(가로 또는 세로)에서 격자점 좌표 목록을 만든다.

    margin 부터 (length - margin) 까지 step 간격으로 점을 찍는다.
    예) length=10, margin=0.5, step=1 → 0.5, 1.5, 2.5, ..., 9.5 (10개)
    """
    usable = length - 2 * margin  # 양쪽 벽 여유를 빼고 남은 실제 사용 가능한 길이
    # 사용 가능한 길이 안에 step이 몇 번 들어가는지 + 시작점 1개
    # (1e-9는 0.1+0.2=0.30000000000000004 같은 소수점 오차 때문에 점 하나를 놓치지 않게 하는 여유)
    count = int(math.floor(usable / step + 1e-9)) + 1
    # round(..., 9): 9.499999999 같은 지저분한 값을 9.5로 정리
    return [round(margin + i * step, 9) for i in range(count)]


def _in_zone(x, y, zone):
    """점 (x, y)가 직사각형 구역 안(경계 포함)에 있으면 True."""
    x_min, y_min, x_max, y_max = zone
    # 경계선 위의 점도 "구역 안"으로 본다. (금지 구역에 딱 붙은 자리도 피하는 쪽이 안전)
    return x_min <= x <= x_max and y_min <= y <= y_max


def _thin_evenly(points, max_count):
    """
    점 목록에서 max_count개를 "일정한 간격"으로 골라 돌려준다.

    예) 점 10개에서 4개를 고르면 0, 3, 6, 9번째 점을 고른다.
    격자점은 앞줄→뒷줄 순서로 저장되어 있으므로, 번호를 고르게 뽑으면
    방의 앞쪽부터 뒤쪽까지 골고루 후보가 남는다.
    """
    n = len(points)
    if max_count >= n:
        return list(points)
    if max_count == 1:
        return [points[n // 2]]  # 하나만 남길 때는 목록 한가운데 점
    # 0번부터 (n-1)번까지를 (max_count-1)개 구간으로 똑같이 나눈 위치의 번호
    indices = [round(i * (n - 1) / (max_count - 1)) for i in range(max_count)]
    return [points[i] for i in indices]


def generate_candidates(room_dims, step, min_wall_distance, height,
                        forbidden_zones=None, max_candidates=None):
    """
    스피커를 놓을 수 있는 후보 위치 목록을 만든다.

    매개변수
    --------
    room_dims : (가로, 세로, 높이) 방 치수 [m]
    step : 격자 간격 [m]. 작을수록 후보가 촘촘하고 많아진다.
    min_wall_distance : 옆벽(앞/뒤/좌/우)에서 최소로 떨어져야 하는 거리 [m]
    height : 스피커 설치 높이 [m]. 모든 후보의 z 좌표가 이 값이 된다.
    forbidden_zones : 설치 금지 직사각형 목록 [(x_min, y_min, x_max, y_max), ...]
                      None이면 금지 구역 없음.
    max_candidates : 후보 수 상한. None이면 제한 없음.

    반환값
    ------
    [(x, y, z), ...] 형태의 후보 좌표 리스트
    """
    # -----------------------------------------------------------------
    # 1) 입력 검사: 잘못된 값이면 바로 이해하기 쉬운 오류를 낸다
    # -----------------------------------------------------------------
    try:
        dims = list(room_dims)
    except TypeError:
        raise ValueError("room_dims는 (가로, 세로, 높이) 3개 숫자로 넣어야 합니다.")
    if len(dims) != 3:
        raise ValueError(
            f"room_dims는 (가로, 세로, 높이) 3개 숫자여야 합니다. (받은 개수: {len(dims)}개)"
        )
    width = _to_number(dims[0], "방 가로(room_dims[0])")
    length = _to_number(dims[1], "방 세로(room_dims[1])")
    room_h = _to_number(dims[2], "방 높이(room_dims[2])")
    for name, value in (("가로", width), ("세로", length), ("높이", room_h)):
        if value <= 0:
            raise ValueError(f"방 {name}는 0보다 커야 합니다. (받은 값: {value} m)")

    step = _to_number(step, "격자 간격(step)")
    if step <= 0:
        raise ValueError(f"격자 간격(step)은 0보다 커야 합니다. (받은 값: {step} m)")

    margin = _to_number(min_wall_distance, "벽과의 최소 거리(min_wall_distance)")
    if margin < 0:
        raise ValueError(
            f"벽과의 최소 거리(min_wall_distance)는 0 이상이어야 합니다. (받은 값: {margin} m)"
        )
    # 양쪽 벽에서 margin씩 떼면 남는 공간이 있어야 한다
    if 2 * margin >= width or 2 * margin >= length:
        raise ValueError(
            f"벽과의 최소 거리 {margin} m가 너무 큽니다. "
            f"방({width} x {length} m)의 양쪽 벽에서 {margin} m씩 떼면 남는 공간이 없습니다."
        )

    z = _to_number(height, "설치 높이(height)")
    if z <= 0 or z >= room_h:
        raise ValueError(
            f"설치 높이(height)는 바닥(0 m)보다 높고 천장({room_h} m)보다 낮아야 합니다. "
            f"(받은 값: {z} m)"
        )

    zones = []
    for i, zone in enumerate(forbidden_zones or []):
        try:
            values = list(zone)
        except TypeError:
            raise ValueError(f"금지 구역 {i}번은 (x_min, y_min, x_max, y_max) 4개 숫자여야 합니다.")
        if len(values) != 4:
            raise ValueError(
                f"금지 구역 {i}번은 (x_min, y_min, x_max, y_max) 4개 숫자여야 합니다. "
                f"(받은 개수: {len(values)}개)"
            )
        x_min, y_min, x_max, y_max = (
            _to_number(v, f"금지 구역 {i}번의 좌표") for v in values
        )
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(
                f"금지 구역 {i}번 {tuple(values)}: x_min < x_max, y_min < y_max 이어야 합니다."
            )
        zones.append((x_min, y_min, x_max, y_max))

    if max_candidates is not None:
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
            raise ValueError(
                f"후보 수 상한(max_candidates)은 정수여야 합니다. (받은 값: {max_candidates!r})"
            )
        if max_candidates < 1:
            raise ValueError(
                f"후보 수 상한(max_candidates)은 1 이상이어야 합니다. (받은 값: {max_candidates})"
            )

    # -----------------------------------------------------------------
    # 2) 격자점 만들기: x, y 각각의 좌표 목록을 만든 뒤 모든 조합을 만든다
    # -----------------------------------------------------------------
    xs = _grid_positions(width, margin, step)
    ys = _grid_positions(length, margin, step)

    candidates = []
    for y in ys:          # 앞줄(y 작은 쪽)부터 뒷줄 순서로
        for x in xs:      # 각 줄에서 왼쪽부터 오른쪽 순서로
            # 금지 구역 중 하나라도 걸리면 이 점은 건너뛴다
            if any(_in_zone(x, y, zone) for zone in zones):
                continue
            candidates.append((x, y, z))

    # -----------------------------------------------------------------
    # 3) 후보가 너무 많으면 고르게 솎아 내기
    # -----------------------------------------------------------------
    if max_candidates is not None:
        candidates = _thin_evenly(candidates, max_candidates)

    return candidates


# 이 파일을 직접 실행했을 때만 아래 예제가 실행된다
if __name__ == "__main__":
    # 방 치수와 스피커 높이는 demo_basic.py에서 그대로 가져온다.
    # (숫자를 복사해 두면 한쪽만 바뀌었을 때 어긋나므로 직접 import한다)
    from demo_basic import ROOM_WIDTH, ROOM_LENGTH, ROOM_HEIGHT

    room_dims = (ROOM_WIDTH, ROOM_LENGTH, ROOM_HEIGHT)
    # 후보 생성 설정은 파일 맨 위의 DEFAULT_... 상수를 쓴다
    STEP = DEFAULT_STEP
    MIN_WALL_DISTANCE = DEFAULT_MIN_WALL_DISTANCE
    HEIGHT = DEFAULT_HEIGHT
    FORBIDDEN = list(DEFAULT_FORBIDDEN_ZONES)   # 출력 모양을 예전과 같게 리스트로

    cands = generate_candidates(room_dims, STEP, MIN_WALL_DISTANCE, HEIGHT,
                                forbidden_zones=FORBIDDEN)

    print(f"방 크기: {ROOM_WIDTH} x {ROOM_LENGTH} x {ROOM_HEIGHT} m")
    print(f"격자 간격={STEP} m, 벽 여유={MIN_WALL_DISTANCE} m, "
          f"설치 높이={HEIGHT} m, 금지 구역={FORBIDDEN}")
    print(f"후보 개수: {len(cands)}")
    print()
    print(f"{'번호':>4} | {'x':>6} {'y':>6} {'z':>6}")
    print("-" * 30)
    for i, (x, y, z) in enumerate(cands):
        print(f"{i:>4} | {x:6.2f} {y:6.2f} {z:6.2f}")
    print("-" * 30)

    # 같은 조건에 후보 수 상한을 20개로 걸었을 때
    limited = generate_candidates(room_dims, STEP, MIN_WALL_DISTANCE, HEIGHT,
                                  forbidden_zones=FORBIDDEN, max_candidates=20)
    print(f"max_candidates=20 적용 시 후보 개수: {len(limited)}")
    print("  " + ", ".join(f"({x:g},{y:g})" for x, y, _ in limited))
