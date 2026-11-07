"""
soundstage 첫 단계 예제: 직사각형 방에서 기본 RIR(Room Impulse Response) 계산 확인

- 스피커(음원) 1개, 청취 지점(마이크) 여러 개를 방에 배치한다.
- 각 청취 지점의 RIR 에너지(샘플 제곱의 합)를 구해 표로 출력한다.
- 지점 간 에너지의 평균/표준편차를 출력한다. (표준편차가 작을수록 균일)

실행 방법:  python soundstage/demo_basic.py
"""

import numpy as np                 # 숫자 배열 계산용 라이브러리
import pyroomacoustics as pra      # 방 음향 시뮬레이션 라이브러리

# =====================================================================
# 1) 설정 상수 (나중에 5공학관 5420 강의실 실측값으로 바꿀 부분)
# =====================================================================

# 방 치수 [m] : 가로(x), 세로(y), 높이(z)
ROOM_WIDTH = 10.0    # x 방향 길이
ROOM_LENGTH = 8.0    # y 방향 길이 (y=0 쪽 벽을 "방 앞쪽"으로 본다)
ROOM_HEIGHT = 3.0    # z 방향 길이 (바닥 z=0, 천장 z=ROOM_HEIGHT)

# 샘플링 주파수 [Hz] : 1초를 몇 개의 숫자(샘플)로 나타낼지
FS = 16000

# 벽의 에너지 흡음률 (0이면 소리를 전부 반사, 1이면 전부 흡수)
WALL_ABSORPTION = 0.3

# 이미지 소스 기법의 최대 반사 차수 (벽에 몇 번 튕긴 소리까지 계산할지)
# 값이 작을수록 계산이 빠르다. 3이면 기본 확인용으로 충분하다.
MAX_ORDER = 3

# 스피커 위치 [m] : 방 앞쪽(y가 작은 쪽) 벽 근처, 가로 중앙, 위쪽
SPEAKER_POS = [ROOM_WIDTH / 2, 0.5, 2.5]

# 청취 지점(마이크) 격자 설정
LISTENER_HEIGHT = 1.2   # 앉은 사람 귀 높이 정도 [m]
GRID_COLS = 3           # x 방향 열 개수
GRID_ROWS = 4           # y 방향 행 개수
WALL_MARGIN = 1.0       # 옆벽/뒷벽에서 최소 이만큼 떨어뜨림 [m]
FRONT_GAP = 2.0         # 첫 줄은 앞벽(스피커 쪽)에서 이만큼 떨어뜨림 [m]


def make_listener_grid():
    """청취 지점 좌표를 격자 형태로 만들어 (3, 지점 수) 배열로 돌려준다."""
    # x 좌표: 왼쪽 벽 여유 ~ 오른쪽 벽 여유 사이를 GRID_COLS개로 균등 분할
    xs = np.linspace(WALL_MARGIN, ROOM_WIDTH - WALL_MARGIN, GRID_COLS)
    # y 좌표: 앞쪽 간격 ~ 뒷벽 여유 사이를 GRID_ROWS개로 균등 분할
    ys = np.linspace(FRONT_GAP, ROOM_LENGTH - WALL_MARGIN, GRID_ROWS)

    points = []
    for y in ys:              # 앞줄부터 뒷줄 순서로
        for x in xs:          # 각 줄에서 왼쪽부터 오른쪽 순서로
            points.append([x, y, LISTENER_HEIGHT])

    # pyroomacoustics는 (차원, 마이크 수) 모양을 원하므로 전치(.T)한다
    return np.array(points).T


def check_inside_room(pos, name):
    """좌표가 방 안에 있는지 확인하고, 밖에 있으면 오류를 낸다."""
    room_size = np.array([ROOM_WIDTH, ROOM_LENGTH, ROOM_HEIGHT])
    pos = np.asarray(pos)
    # 모든 축에서 0 < 좌표 < 방 크기 이어야 방 내부
    if np.any(pos <= 0) or np.any(pos >= room_size):
        raise ValueError(f"{name} 좌표 {pos} 가 방 밖에 있습니다.")


def main():
    # -----------------------------------------------------------------
    # 2) 방 만들기
    # -----------------------------------------------------------------
    # 모든 벽(6면)에 같은 흡음률을 가진 재질을 지정한다.
    # (float로 넣어야 주파수와 무관한 '평평한' 재질로 인식된다)
    material = pra.Material(energy_absorption=float(WALL_ABSORPTION))

    # 직사각형(신발상자 모양) 방을 만든다.
    room = pra.ShoeBox(
        [ROOM_WIDTH, ROOM_LENGTH, ROOM_HEIGHT],  # 방 크기
        fs=FS,                                   # 샘플링 주파수
        materials=material,                      # 벽 재질
        max_order=MAX_ORDER,                     # 최대 반사 차수
        air_absorption=False,                    # 공기 흡음은 끔 (단순화)
        ray_tracing=False,                       # 레이 트레이싱 끔 (빠른 계산)
    )

    # -----------------------------------------------------------------
    # 3) 스피커와 청취 지점 배치
    # -----------------------------------------------------------------
    check_inside_room(SPEAKER_POS, "스피커")        # 스피커가 방 안인지 확인
    room.add_source(SPEAKER_POS)                    # 스피커(음원) 1개 추가

    mic_positions = make_listener_grid()            # 격자 좌표 생성 (3 x N)
    for i in range(mic_positions.shape[1]):         # 각 지점이 방 안인지 확인
        check_inside_room(mic_positions[:, i], f"청취 지점 {i}")
    room.add_microphone_array(mic_positions)        # 청취 지점(마이크) 추가

    # -----------------------------------------------------------------
    # 4) RIR 계산
    # -----------------------------------------------------------------
    # 계산 결과는 room.rir[마이크 번호][음원 번호] 에 1차원 배열로 저장된다.
    room.compute_rir()

    # -----------------------------------------------------------------
    # 5) 지점별 RIR 에너지 계산 및 표 출력
    # -----------------------------------------------------------------
    n_mics = mic_positions.shape[1]
    energies = np.zeros(n_mics)                     # 에너지를 담을 빈 배열
    for m in range(n_mics):
        rir = room.rir[m][0]                        # m번 지점, 0번(유일한) 스피커
        energies[m] = np.sum(rir ** 2)              # 샘플 제곱의 합 = 에너지

    # 스피커와 각 지점 사이의 직선 거리 (결과 해석용)
    distances = np.linalg.norm(
        mic_positions - np.array(SPEAKER_POS)[:, None], axis=0
    )

    print(f"방 크기: {ROOM_WIDTH} x {ROOM_LENGTH} x {ROOM_HEIGHT} m, "
          f"fs={FS} Hz, 흡음률={WALL_ABSORPTION}, max_order={MAX_ORDER}")
    print(f"스피커 위치: {SPEAKER_POS}")
    print()
    print(f"{'지점':>4} | {'x':>6} {'y':>6} {'z':>6} | {'거리[m]':>7} | {'에너지':>10}")
    print("-" * 52)
    for m in range(n_mics):
        x, y, z = mic_positions[:, m]
        print(f"{m:>4} | {x:6.2f} {y:6.2f} {z:6.2f} | "
              f"{distances[m]:7.2f} | {energies[m]:10.6f}")
    print("-" * 52)

    # -----------------------------------------------------------------
    # 6) 균일도 지표: 평균과 표준편차
    # -----------------------------------------------------------------
    mean_e = np.mean(energies)     # 지점들의 평균 에너지
    std_e = np.std(energies)       # 지점 간 흩어진 정도 (작을수록 균일)
    print(f"에너지 평균     : {mean_e:.6f}")
    print(f"에너지 표준편차 : {std_e:.6f}")
    print(f"변동계수(std/mean): {std_e / mean_e:.3f}")

    # 에너지가 전부 0이면 무언가 잘못된 것이므로 알려 준다
    if np.allclose(energies, 0):
        print("경고: 모든 에너지가 0입니다. 배치나 설정을 확인하세요.")


# 이 파일을 직접 실행했을 때만 main()을 호출한다
if __name__ == "__main__":
    main()
