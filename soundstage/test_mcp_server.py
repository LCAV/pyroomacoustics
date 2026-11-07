"""
mcp_server.py 의 도구 함수 테스트 (MCP 통신 없이 파이썬 함수로 직접 호출)

실제 MCP 프로토콜로 동작하는지는 mcp_smoke_test.py 에서 따로 확인한다.

실행 방법:  pytest soundstage/test_mcp_server.py
"""

import json
import os

import pytest

import mcp_server
from candidates import (
    generate_candidates,
    DEFAULT_STEP, DEFAULT_MIN_WALL_DISTANCE, DEFAULT_HEIGHT, DEFAULT_FORBIDDEN_ZONES,
)
from demo_basic import ROOM_WIDTH, ROOM_LENGTH, ROOM_HEIGHT, WALL_ABSORPTION
from evaluate import evaluate_position, make_listening_points
from recommend import recommend_placement
from mcp_server import (
    SoundstageToolError,
    evaluate_speaker_position,
    get_room_info,
    list_candidate_positions,
    recommend_speaker_placement,
)

ROOM_DIMS = (ROOM_WIDTH, ROOM_LENGTH, ROOM_HEIGHT)


@pytest.fixture(scope="module")
def mcp_rec():
    """MCP 도구의 추천 결과 (시뮬레이션이 몇 초 걸리므로 모듈에서 한 번만 계산)"""
    return recommend_speaker_placement()


@pytest.fixture(scope="module")
def direct_rec():
    """recommend.py를 기본 설정으로 직접 쓴 결과 (비교 기준)"""
    cands = generate_candidates(ROOM_DIMS, DEFAULT_STEP, DEFAULT_MIN_WALL_DISTANCE,
                                DEFAULT_HEIGHT, forbidden_zones=DEFAULT_FORBIDDEN_ZONES)
    return recommend_placement(ROOM_DIMS, cands, make_listening_points())


# ---------------------------------------------------------------------
# rooms.json 이 지금 기본 설정(demo_basic/candidates)과 같은지
# ---------------------------------------------------------------------
def test_rooms_json_matches_default_settings():
    """5420 임시값은 demo_basic/candidates의 기본 설정을 그대로 옮긴 것이어야 한다."""
    info = get_room_info("5420")
    assert info["dimensions_m"] == list(ROOM_DIMS)
    assert info["wall_absorption"] == WALL_ABSORPTION
    assert info["forbidden_zones"] == [list(z) for z in DEFAULT_FORBIDDEN_ZONES]
    assert info["speaker_height_m"] == DEFAULT_HEIGHT
    assert info["min_wall_distance_m"] == DEFAULT_MIN_WALL_DISTANCE
    assert info["listening_points"] == [list(p) for p in make_listening_points()]
    assert info["measured"] is False


# ---------------------------------------------------------------------
# 1) get_room_info
# ---------------------------------------------------------------------
def test_get_room_info_has_required_keys():
    info = get_room_info()
    for key in ("room_name", "measured", "dimensions_m", "wall_absorption",
                "forbidden_zones", "listening_points", "num_listening_points"):
        assert key in info, f"{key} 키가 없습니다"
    assert info["room_name"] == "5420"
    assert info["num_listening_points"] == 12
    assert len(info["listening_points"]) == 12


# ---------------------------------------------------------------------
# 2) list_candidate_positions
# ---------------------------------------------------------------------
def test_list_candidate_positions_matches_generate_candidates():
    out = list_candidate_positions()
    expected = generate_candidates(ROOM_DIMS, DEFAULT_STEP, DEFAULT_MIN_WALL_DISTANCE,
                                   DEFAULT_HEIGHT, forbidden_zones=DEFAULT_FORBIDDEN_ZONES)
    assert out["count"] == len(expected) == 76
    assert out["positions"] == [list(c) for c in expected]


def test_list_candidate_positions_step_changes_count():
    assert list_candidate_positions(step=0.5)["count"] > list_candidate_positions(step=1.0)["count"]


@pytest.mark.parametrize("bad_step", [0, -1.0])
def test_list_candidate_positions_bad_step(bad_step):
    with pytest.raises(ValueError, match="격자 간격"):
        list_candidate_positions(step=bad_step)


# ---------------------------------------------------------------------
# 3) evaluate_speaker_position
# ---------------------------------------------------------------------
def test_evaluate_speaker_position_matches_evaluate_py():
    out = evaluate_speaker_position(3.5, 4.5, 2.5)
    direct = evaluate_position(ROOM_DIMS, (3.5, 4.5, 2.5), make_listening_points())
    assert out["std_db"] == pytest.approx(direct["std_db"])
    assert out["mean_db"] == pytest.approx(direct["mean_db"])
    assert out["min_db"] == pytest.approx(direct["min_db"])
    assert [p["level_db"] for p in out["levels_db"]] == pytest.approx(direct["levels_db"])
    assert len(out["levels_db"]) == 12
    assert out["in_forbidden_zone"] is False


def test_evaluate_speaker_position_flags_forbidden_zone():
    # 교탁 자리 [4,0,6,2] 안: 계산은 되지만 금지 구역 표시가 붙어야 한다
    assert evaluate_speaker_position(5.0, 1.0, 2.5)["in_forbidden_zone"] is True


@pytest.mark.parametrize("pos", [(11.0, 4.0, 2.5), (5.0, -1.0, 2.5), (5.0, 4.0, 3.0), (0.0, 4.0, 2.5)])
def test_evaluate_speaker_position_outside_room(pos):
    with pytest.raises(SoundstageToolError, match="방 밖"):
        evaluate_speaker_position(*pos)


# ---------------------------------------------------------------------
# 4) recommend_speaker_placement
# ---------------------------------------------------------------------
def test_recommend_matches_recommend_py(mcp_rec, direct_rec):
    assert mcp_rec["recommended"]["position"] == list(direct_rec["best"]["position"])
    assert mcp_rec["recommended"]["position"] == [3.5, 4.5, 2.5]
    assert mcp_rec["improvement_percent"] == pytest.approx(direct_rec["improvement_percent"])
    assert mcp_rec["improvement_percent"] == pytest.approx(15.8, abs=0.05)
    assert mcp_rec["baseline"]["position"] == [5.0, 0.5, 2.5]
    assert mcp_rec["num_candidates"] == direct_rec["num_candidates"]
    assert mcp_rec["num_passed"] == direct_rec["num_passed"]
    assert mcp_rec["threshold_db"] == pytest.approx(direct_rec["threshold_db"])


def test_recommend_top_candidates(mcp_rec):
    top = mcp_rec["top_candidates"]
    assert len(top) == 5
    assert [t["rank"] for t in top] == [1, 2, 3, 4, 5]
    assert top[0]["position"] == mcp_rec["recommended"]["position"]
    stds = [t["std_db"] for t in top]
    assert stds == sorted(stds)
    assert "figure_path" not in mcp_rec


@pytest.mark.parametrize("bad", [0, -3, True])
def test_recommend_bad_top_n(bad):
    with pytest.raises(SoundstageToolError, match="top_n"):
        recommend_speaker_placement(top_n=bad)


def test_recommend_save_figure(tmp_path, monkeypatch):
    path = str(tmp_path / "mcp_report.png")
    monkeypatch.setattr(mcp_server, "REPORT_PATH", path)
    out = recommend_speaker_placement(top_n=2, save_figure=True)
    assert out["figure_path"] == path
    assert os.path.getsize(path) > 0
    assert len(out["top_candidates"]) == 2


# ---------------------------------------------------------------------
# 공통: 없는 방 이름, JSON 직렬화, stdout 오염 없음
# ---------------------------------------------------------------------
@pytest.mark.parametrize("call", [
    lambda: get_room_info("9999"),
    lambda: list_candidate_positions("9999"),
    lambda: evaluate_speaker_position(3.0, 3.0, 2.0, room_name="9999"),
    lambda: recommend_speaker_placement("9999"),
])
def test_unknown_room(call):
    with pytest.raises(SoundstageToolError, match="등록되어 있지 않습니다"):
        call()


def test_error_is_value_error_and_tool_error():
    """직접 호출하면 ValueError로, MCP SDK에서는 ToolError(메시지 그대로 전달)로 취급되어야 한다."""
    from mcp.server.mcpserver.exceptions import ToolError
    with pytest.raises(ValueError) as info:
        get_room_info("없는방")
    assert isinstance(info.value, ToolError)


def test_results_are_json_serializable(mcp_rec):
    for value in (get_room_info(), list_candidate_positions(),
                  evaluate_speaker_position(3.5, 4.5, 2.5), mcp_rec):
        text = json.dumps(value, ensure_ascii=False)
        assert json.loads(text) == value     # 되돌렸을 때도 같은 값 (numpy 값이 섞이지 않았다는 뜻)


def test_tools_do_not_write_stdout(capfd):
    """stdio MCP 서버에서 stdout은 통신선이므로, 도구가 stdout에 아무것도 찍으면 안 된다."""
    get_room_info()
    list_candidate_positions()
    evaluate_speaker_position(3.5, 4.5, 2.5)
    with pytest.raises(ValueError):
        evaluate_speaker_position(99.0, 1.0, 1.0)
    assert capfd.readouterr().out == ""


def test_four_tools_registered():
    import asyncio
    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert names == {"get_room_info", "list_candidate_positions",
                     "evaluate_speaker_position", "recommend_speaker_placement"}
