"""
visualize.py 테스트

- 화면(모니터) 없이 돌도록 맨 먼저 Agg 백엔드를 고른다.
- 시뮬레이션은 돌리지 않고, recommend_placement()가 돌려주는 모양을 흉내 낸
  "가짜 추천 결과"를 손으로 만들어 쓴다. (그래서 빠르다)

실행 방법:  pytest soundstage/test_visualize.py
"""

import matplotlib

matplotlib.use("Agg")  # 창을 띄우지 않는 백엔드 (pyplot을 import 하기 전에 정해야 한다)

import pytest

from visualize import make_report_figure, plot_candidate_heatmap, plot_listening_levels

ROOM = (10.0, 8.0, 3.0)
ZONES = ((4.0, 0.0, 6.0, 2.0),)
# 청취 지점 12곳 (3열 x 4행)
LISTENERS = [(x, y, 1.2) for y in (2.0, 3.67, 5.33, 7.0) for x in (1.0, 5.0, 9.0)]


def fake_eval(pos, base_db, spread):
    """가짜 평가 결과: 지점별 dB를 조금씩 다르게 만들어 std_db가 있는 것처럼 보이게 한다."""
    levels = [base_db - spread * (i % 4) for i in range(len(LISTENERS))]
    return {"position": pos, "levels_db": levels, "std_db": spread, "min_db": min(levels)}


def fake_recommend(num_candidates=6):
    """recommend_placement()의 반환 딕셔너리와 같은 모양의 가짜 결과."""
    results = [fake_eval((1.0 + i, 3.0, 2.5), -20.0, 0.5 + 0.3 * i) for i in range(num_candidates)]
    baseline = fake_eval((5.0, 0.5, 2.5), -18.0, 3.0)
    best = results[0]
    return {
        "best": best,
        "top": results[:5],
        "baseline": baseline,
        "threshold_db": baseline["min_db"],
        "num_candidates": len(results),
        "num_passed": len(results),
        "improvement_percent": (3.0 - best["std_db"]) / 3.0 * 100,
        "results": results,
    }


def panels(fig):
    """색 막대(colorbar) 축을 빼고, 본 패널만 골라낸다."""
    return [ax for ax in fig.axes if ax.get_label().startswith("panel_")]


def test_make_report_figure_saves_png(tmp_path):
    out = tmp_path / "sub" / "report.png"   # 없는 폴더도 만들어 주는지 함께 확인
    make_report_figure(ROOM, LISTENERS, fake_recommend(), forbidden_zones=ZONES, save_path=str(out))
    assert out.exists()
    # PNG 파일은 항상 이 8바이트로 시작한다
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_report_has_three_panels():
    fig = make_report_figure(ROOM, LISTENERS, fake_recommend(), forbidden_zones=ZONES)
    labels = [ax.get_label() for ax in panels(fig)]
    assert labels == ["panel_heatmap", "panel_baseline", "panel_best"]
    # 제목에 개선율이 한국어로 들어갔는지
    assert "개선율" in fig._suptitle.get_text()


def test_empty_results_raises_korean_message():
    rec = fake_recommend()
    rec["results"] = []
    with pytest.raises(ValueError, match="비어 있어"):
        make_report_figure(ROOM, LISTENERS, rec)


def test_missing_results_key_raises():
    rec = fake_recommend()
    del rec["results"]
    with pytest.raises(ValueError, match="results"):
        make_report_figure(ROOM, LISTENERS, rec)


def test_heatmap_empty_results_raises():
    rec = fake_recommend()
    with pytest.raises(ValueError, match="비어 있어"):
        plot_candidate_heatmap(ROOM, [], rec["baseline"], rec["best"])


@pytest.mark.parametrize("n", [1, 6, 20])
def test_heatmap_draws_one_point_per_candidate(n):
    rec = fake_recommend(num_candidates=n)
    ax = plot_candidate_heatmap(ROOM, rec["results"], rec["baseline"], rec["best"], forbidden_zones=ZONES)
    cand = [c for c in ax.collections if c.get_gid() == "candidates"]
    assert len(cand) == 1
    assert len(cand[0].get_offsets()) == n


def test_listening_levels_shared_color_range():
    rec = fake_recommend()
    ax = plot_listening_levels(ROOM, LISTENERS, rec["baseline"], "기준", vmin=-40.0, vmax=-10.0)
    dots = [c for c in ax.collections if c.get_gid() == "listeners"][0]
    assert len(dots.get_offsets()) == len(LISTENERS)
    # 넘겨준 vmin/vmax가 그대로 색 범위로 쓰였는지
    assert dots.norm.vmin == -40.0 and dots.norm.vmax == -10.0
    # 지점마다 숫자 글자가 하나씩 있는지
    assert len(ax.texts) >= len(LISTENERS)


def test_listening_levels_length_mismatch_raises():
    rec = fake_recommend()
    with pytest.raises(ValueError, match="청취 지점 수"):
        plot_listening_levels(ROOM, LISTENERS[:5], rec["baseline"], "기준")
