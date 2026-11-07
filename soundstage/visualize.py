"""
soundstage 이슈 5: 배치 결과 2D 히트맵 시각화 (matplotlib)

recommend.py의 recommend_placement() 결과를 그림 한 장으로 정리한다.

  (왼쪽)   후보 위치마다 균일도 점수(std_db)를 색으로 칠한 히트맵
  (가운데) 기준 배치일 때 청취 지점 12곳의 음향 수준(dB)
  (오른쪽) 추천 배치일 때 청취 지점 12곳의 음향 수준(dB)

그림 좌표 약속 (방을 천장에서 내려다본 평면도)
  - x: 방 가로 (왼쪽 → 오른쪽)
  - y: 방 세로. 앞벽(y=0, 스크린/교탁 쪽)이 그림의 "위쪽"에 오도록 y축을 뒤집는다.

실행 방법:
  python soundstage/visualize.py          → soundstage/output/report.png 저장
  python soundstage/visualize.py --show   → 저장 + 창 띄우기
"""

import os

import matplotlib
from matplotlib import colors as mcolors
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# 색 지도(colormap) 선택
#   - 히트맵: RdYlBu_r → 값이 작으면 파랑(좋음, 균일), 크면 빨강(나쁨, 불균일)
#   - 청취 지점: viridis → 작게 들리면 어두운 보라, 크게 들리면 밝은 노랑
HEATMAP_CMAP = "RdYlBu_r"
LEVEL_CMAP = "viridis"

# Windows/맥/리눅스에서 흔히 쓰는 한글 폰트 (앞에 있을수록 먼저 시도)
KOREAN_FONTS = ("Malgun Gothic", "AppleGothic", "NanumGothic")


def setup_korean_font():
    """
    그림 속 한글이 네모(□)로 깨지지 않도록 한글 폰트를 설정한다.

    - 설치된 폰트 중 KOREAN_FONTS에 있는 것을 찾아 쓴다. (Windows는 'Malgun Gothic')
    - 하나도 없으면 matplotlib 기본 폰트를 그대로 쓴다. (오류를 내지 않는다)
    - 한글 폰트에는 '−'(유니코드 마이너스) 글자가 없는 경우가 많아서,
      axes.unicode_minus=False로 평범한 '-'를 쓰게 한다. (음수 눈금이 깨지지 않게)

    반환값: 실제로 설정한 폰트 이름 (못 찾았으면 None)
    """
    matplotlib.rcParams["axes.unicode_minus"] = False
    try:
        installed = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        # 폰트 목록을 못 읽는 특이한 환경이어도 그림은 그려져야 하므로 조용히 넘어간다
        return None
    for name in KOREAN_FONTS:
        if name in installed:
            matplotlib.rcParams["font.family"] = name
            return name
    return None


# 이 파일을 import 하기만 해도 한글 폰트가 설정되도록 한 번 호출해 둔다
setup_korean_font()


# ---------------------------------------------------------------------
# 작은 도우미 함수들
# ---------------------------------------------------------------------
def _new_ax(figsize=(6, 5)):
    """
    ax를 안 받았을 때 새 그림과 축을 만든다.

    pyplot 대신 Figure를 직접 만들면 화면(창)이 전혀 필요 없다.
    → 서버나 테스트처럼 모니터가 없는 환경에서도 그림을 그리고 저장할 수 있다.
    """
    fig = Figure(figsize=figsize, layout="constrained")
    return fig.add_subplot()


def _draw_room(ax, room_dims):
    """방 테두리를 그리고, 앞벽(y=0)이 위로 오도록 축을 정리한다."""
    width, length = float(room_dims[0]), float(room_dims[1])
    ax.add_patch(Rectangle((0, 0), width, length, fill=False,
                           edgecolor="black", linewidth=1.5, zorder=1))
    pad = 0.4  # 테두리 바깥 여백 [m]
    ax.set_xlim(-pad, width + pad)
    ax.set_ylim(length + pad, -pad)   # (큰 값, 작은 값) 순서로 주면 y축이 뒤집힌다 → y=0이 위쪽
    ax.set_aspect("equal")            # 1 m가 가로·세로 같은 길이로 보이게 (방 모양이 찌그러지지 않게)
    ax.set_xlabel("x: 가로 [m]")
    ax.set_ylabel("y: 세로 [m]  (위쪽이 앞벽)")
    # 앞벽 위치를 글자로 알려 준다
    ax.text(width / 2, -pad / 2, "앞벽 (y = 0)", ha="center", va="center", fontsize=8, color="dimgray")


def _text_color_for(rgba):
    """배경색이 어두우면 흰 글씨, 밝으면 검은 글씨를 고른다. (숫자가 잘 보이도록)"""
    r, g, b = rgba[:3]
    brightness = 0.299 * r + 0.587 * g + 0.114 * b   # 사람 눈이 느끼는 밝기 공식
    return "white" if brightness < 0.5 else "black"


# ---------------------------------------------------------------------
# 1) 후보 위치 히트맵
# ---------------------------------------------------------------------
def plot_candidate_heatmap(room_dims, results, baseline, best, forbidden_zones=None, ax=None):
    """
    후보 위치마다 균일도 점수(std_db)를 색으로 칠한 평면도를 그린다.

    매개변수
    --------
    room_dims : (가로, 세로, 높이) [m]
    results   : evaluate_candidates()의 결과 리스트 (각 항목에 "position", "std_db")
    baseline  : 기준 배치 딕셔너리 ("position" 필요)
    best      : 추천 배치 딕셔너리 ("position" 필요)
    forbidden_zones : [(x_min, y_min, x_max, y_max), ...] 설치 금지 구역 (없으면 None)
    ax        : 그릴 축. None이면 새 그림을 만든다.

    반환값: 그림을 그린 ax
            (후보 점들은 gid="candidates"인 scatter로 그려 테스트에서 찾을 수 있게 했다)
    """
    results = list(results)
    if len(results) == 0:
        raise ValueError("후보 평가 결과(results)가 비어 있어 히트맵을 그릴 수 없습니다.")
    if ax is None:
        ax = _new_ax()

    _draw_room(ax, room_dims)

    # (1) 설치 금지 구역: 회색 사각형
    for zone in (forbidden_zones or []):
        x_min, y_min, x_max, y_max = zone
        ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               facecolor="lightgray", edgecolor="gray", hatch="//", zorder=2))

    # (2) 후보 위치: 네모 점, 색 = std_db
    xs = [r["position"][0] for r in results]
    ys = [r["position"][1] for r in results]
    stds = [r["std_db"] for r in results]
    sc = ax.scatter(xs, ys, c=stds, cmap=HEATMAP_CMAP, marker="s", s=260,
                    edgecolors="white", linewidths=0.8, zorder=3)
    sc.set_gid("candidates")
    cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("균일도 점수 std_db (dB, 작을수록 균일)")

    # (3) 기준 배치(별)와 추천 배치(다이아몬드)
    bx, by = baseline["position"][0], baseline["position"][1]
    rx, ry = best["position"][0], best["position"][1]
    ax.scatter([bx], [by], marker="*", s=420, color="gold", edgecolors="black", linewidths=1, zorder=5)
    ax.annotate("기준", (bx, by), xytext=(9, -9), textcoords="offset points", fontsize=9,
                fontweight="bold", zorder=6)
    ax.scatter([rx], [ry], marker="D", s=170, facecolors="none", edgecolors="black", linewidths=2.2, zorder=5)
    ax.annotate("추천", (rx, ry), xytext=(9, -9), textcoords="offset points", fontsize=9,
                fontweight="bold", zorder=6)

    # (4) 범례: 그림에 쓴 모양을 그대로 흉내 낸 "견본"을 만들어 붙인다
    handles = [
        Line2D([], [], marker="s", linestyle="", color="tab:blue", markersize=9, label="후보 위치 (색 = std_db)"),
        Line2D([], [], marker="*", linestyle="", markerfacecolor="gold", markeredgecolor="black",
               markersize=14, label="기준 배치"),
        Line2D([], [], marker="D", linestyle="", markerfacecolor="none", markeredgecolor="black",
               markeredgewidth=2, markersize=9, label="추천 배치"),
    ]
    if forbidden_zones:
        handles.append(Patch(facecolor="lightgray", edgecolor="gray", hatch="//", label="설치 금지 구역"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=2, fontsize=8, frameon=False)
    ax.set_title("후보 위치별 균일도 (스피커 높이 기준)")
    return ax


# ---------------------------------------------------------------------
# 2) 청취 지점별 음향 수준
# ---------------------------------------------------------------------
def plot_listening_levels(room_dims, listening_points, level_result, title, ax=None, vmin=None, vmax=None):
    """
    청취 지점마다 원을 그리고, 그 자리의 음향 수준(levels_db)을 색과 숫자로 보여 준다.

    매개변수
    --------
    listening_points : [(x, y, z), ...] 청취 지점 (보통 12곳)
    level_result     : evaluate_position() 결과 + "position"(스피커 위치)
                       (levels_db 길이는 listening_points 개수와 같아야 한다)
    title            : 패널 제목
    vmin, vmax       : 색 범위 [dB]. 기준/추천 두 패널에 같은 값을 주면 색을 공정하게 비교할 수 있다.
                       None이면 이 패널의 최솟값/최댓값을 쓴다.

    반환값: 그림을 그린 ax (청취 지점 원은 gid="listeners")
    """
    points = list(listening_points)
    levels = list(level_result["levels_db"])
    if len(points) != len(levels):
        raise ValueError(
            f"청취 지점 수({len(points)})와 음향 수준 개수({len(levels)})가 다릅니다."
        )
    if ax is None:
        ax = _new_ax()

    _draw_room(ax, room_dims)

    # 색 범위: Normalize는 "dB 값 → 0~1 사이 비율"로 바꿔 주는 자(ruler)다
    norm = mcolors.Normalize(vmin=min(levels) if vmin is None else vmin,
                             vmax=max(levels) if vmax is None else vmax)
    cmap = matplotlib.colormaps[LEVEL_CMAP]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    sc = ax.scatter(xs, ys, c=levels, cmap=cmap, norm=norm, marker="o", s=650,
                    edgecolors="black", linewidths=0.8, zorder=3)
    sc.set_gid("listeners")

    # 원 안에 dB 숫자 쓰기 (배경색에 따라 글자색을 바꿔 잘 보이게)
    for x, y, v in zip(xs, ys, levels):
        ax.text(x, y, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                color=_text_color_for(cmap(norm(v))), zorder=4)

    # 스피커 위치
    sx, sy = level_result["position"][0], level_result["position"][1]
    ax.scatter([sx], [sy], marker="^", s=220, color="tab:red", edgecolors="black", zorder=5)
    ax.annotate("스피커", (sx, sy), xytext=(9, 4), textcoords="offset points", fontsize=9, zorder=6)

    cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("음향 수준 (dB, 클수록 크게 들림)")
    ax.set_title(title)
    return ax


# ---------------------------------------------------------------------
# 3) 보고서용 3패널 그림
# ---------------------------------------------------------------------
def make_report_figure(room_dims, listening_points, recommend_result, forbidden_zones=None,
                       save_path=None, show=False):
    """
    recommend_placement()의 결과를 3패널 그림 한 장으로 만든다.

    매개변수
    --------
    recommend_result : recommend_placement() / select_best()의 반환 딕셔너리
                       ("results", "baseline", "best", "improvement_percent" 필요)
    save_path        : PNG로 저장할 경로 (None이면 저장하지 않음). 폴더가 없으면 만든다.
    show             : True면 창을 띄운다. False면 창 없이(화면 없이) 그림만 만든다.

    반환값: matplotlib Figure
            (본 패널 3개에는 "panel_heatmap", "panel_baseline", "panel_best" 라벨이 붙어 있다)
    """
    results = recommend_result.get("results")
    if not results:
        raise ValueError(
            "recommend_result에 후보 평가 결과(results)가 없거나 비어 있어 그림을 만들 수 없습니다."
        )
    baseline = recommend_result["baseline"]
    best = recommend_result["best"]

    # 창을 띄울 때만 pyplot을 쓴다. 저장만 할 때는 Figure를 직접 만들어 화면이 필요 없다.
    size = (19, 6.2)
    if show:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=size, layout="constrained")
    else:
        fig = Figure(figsize=size, layout="constrained")
    ax_heat, ax_base, ax_best = fig.subplots(1, 3)
    ax_heat.set_label("panel_heatmap")
    ax_base.set_label("panel_baseline")
    ax_best.set_label("panel_best")

    # 왼쪽: 후보 히트맵
    plot_candidate_heatmap(room_dims, results, baseline, best, forbidden_zones=forbidden_zones, ax=ax_heat)

    # 가운데·오른쪽: 두 배치의 dB를 "같은 색 범위"로 그려야 색을 바로 비교할 수 있다
    all_levels = list(baseline["levels_db"]) + list(best["levels_db"])
    vmin, vmax = min(all_levels), max(all_levels)
    plot_listening_levels(room_dims, listening_points, baseline,
                          f"기준 배치 (std_db {baseline['std_db']:.2f} dB)", ax=ax_base, vmin=vmin, vmax=vmax)
    plot_listening_levels(room_dims, listening_points, best,
                          f"추천 배치 (std_db {best['std_db']:.2f} dB)", ax=ax_best, vmin=vmin, vmax=vmax)

    improvement = float(recommend_result.get("improvement_percent", 0.0))
    word = "개선" if improvement >= 0 else "악화"
    fig.suptitle(
        f"스피커 배치 비교 — 균일도 std_db: 기준 {baseline['std_db']:.2f} dB → "
        f"추천 {best['std_db']:.2f} dB  (개선율 {improvement:.1f} %, {word})",
        fontsize=14, fontweight="bold",
    )

    if save_path is not None:
        folder = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=200, format="png")

    if show:
        plt.show()
    return fig


# 이 파일을 직접 실행했을 때만 아래 예제가 실행된다
if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="스피커 배치 결과 그림(report.png) 만들기")
    parser.add_argument("--show", action="store_true", help="저장한 뒤 그림 창도 띄운다")
    args = parser.parse_args()

    if not args.show:
        # 창을 안 띄울 때는 화면이 필요 없는 Agg 백엔드를 쓴다
        matplotlib.use("Agg")

    from demo_basic import ROOM_WIDTH, ROOM_LENGTH, ROOM_HEIGHT
    from candidates import (
        generate_candidates,
        DEFAULT_STEP, DEFAULT_MIN_WALL_DISTANCE, DEFAULT_HEIGHT, DEFAULT_FORBIDDEN_ZONES,
    )
    from evaluate import make_listening_points
    from recommend import recommend_placement

    room_dims = (ROOM_WIDTH, ROOM_LENGTH, ROOM_HEIGHT)
    cands = generate_candidates(room_dims, DEFAULT_STEP, DEFAULT_MIN_WALL_DISTANCE,
                                DEFAULT_HEIGHT, forbidden_zones=DEFAULT_FORBIDDEN_ZONES)
    listeners = make_listening_points()

    start = time.perf_counter()
    rec = recommend_placement(room_dims, cands, listeners)
    print(f"추천 계산 완료: 후보 {rec['num_candidates']}개, {time.perf_counter() - start:.2f}초")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "report.png")
    make_report_figure(room_dims, listeners, rec, forbidden_zones=DEFAULT_FORBIDDEN_ZONES,
                       save_path=out_path, show=args.show)
    print(f"그림 저장: {out_path}")
