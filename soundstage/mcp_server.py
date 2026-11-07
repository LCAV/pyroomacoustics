"""
soundstage 이슈 6: 공간 조회·배치 추천 기능을 MCP 서버의 도구(tool)로 제공

MCP(Model Context Protocol) 서버는 Claude 같은 AI가 우리 기능을 "도구"로 호출할 수 있게 해 준다.
공식 MCP Python SDK(패키지 이름 mcp)를 쓴다.
  - mcp 2.x에서는 예전의 FastMCP 클래스가 MCPServer로 이름만 바뀌었다. (사용법은 같다)

제공 도구 4개
  1) get_room_info               : 강의실 정보 조회 (치수, 흡음률, 금지 구역, 청취 지점, 실측 여부)
  2) list_candidate_positions    : 스피커 후보 위치 목록
  3) evaluate_speaker_position   : 한 위치에 스피커를 뒀을 때 지점별 음향 수준과 균일도
  4) recommend_speaker_placement : 최적 배치 추천 + 기준 배치(전면 중앙 상단)와 비교

계산은 모두 기존 모듈(candidates / evaluate / recommend / visualize)의 함수를 그대로 쓴다.
강의실 설정은 같은 폴더의 rooms.json에서 읽는다. (실측 후에는 rooms.json만 고치면 된다)

[주의] stdio 방식 MCP 서버는 표준 출력(stdout)이 곧 통신선이다.
       stdout에 글자를 찍으면 통신이 깨지므로 print를 쓰지 말고, 로그는 stderr(logging)로만 남긴다.

실행 방법:  python soundstage/mcp_server.py   (보통은 Claude Code 같은 MCP 클라이언트가 대신 실행한다)
"""

import contextlib
import json
import logging
import math
import os
import sys

# 어느 폴더에서 실행하든 같은 폴더의 candidates.py 등을 import할 수 있게 한다
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# 화면(창)이 없는 서버이므로, visualize를 불러오기 전에 창이 필요 없는 Agg 백엔드로 고정한다
import matplotlib  # noqa: E402
matplotlib.use("Agg")

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from candidates import generate_candidates  # noqa: E402
from evaluate import evaluate_position  # noqa: E402
from recommend import default_baseline_position, recommend_placement  # noqa: E402
from visualize import make_report_figure  # noqa: E402

# 로그는 반드시 stderr로 (stdout은 MCP 통신 전용)
# Windows에서 stderr 기본 인코딩(cp949) 때문에 한글 로그가 깨지지 않도록 UTF-8로 맞춘다
if hasattr(sys.stderr, "reconfigure"):
    with contextlib.suppress(Exception):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="[soundstage-mcp] %(levelname)s %(message)s")
log = logging.getLogger("soundstage.mcp")

ROOMS_PATH = os.path.join(HERE, "rooms.json")
OUTPUT_DIR = os.path.join(HERE, "output")
REPORT_PATH = os.path.join(OUTPUT_DIR, "mcp_report.png")


class SoundstageToolError(ToolError, ValueError):
    """
    도구 사용자(AI)에게 그대로 보여 줄 한국어 오류.

    - ToolError를 상속해야 MCP SDK가 메시지를 가리지 않고 AI에게 그대로 전달한다.
      (일반 예외는 "Error executing tool ..."로만 보이고 내용이 숨겨진다)
    - ValueError도 상속해서, 파이썬에서 직접 호출할 때는 평범한 ValueError처럼 잡을 수 있다.
    """


@contextlib.contextmanager
def _tool_guard():
    """
    모든 도구 본문을 감싸는 안전장치.
      1) 혹시 계산 중에 누가 print를 해도 stdout이 아니라 stderr로 가게 한다.
      2) 기존 모듈이 내는 ValueError(한국어 메시지)를 SoundstageToolError로 바꿔 AI에게 전달한다.
    """
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    except SoundstageToolError:
        raise
    except ValueError as exc:
        raise SoundstageToolError(str(exc)) from exc


# ---------------------------------------------------------------------
# 강의실 설정 읽기
# ---------------------------------------------------------------------
def load_rooms(path=ROOMS_PATH):
    """rooms.json을 읽어 {방 이름: 설정 딕셔너리}를 돌려준다."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise SoundstageToolError(f"강의실 설정 파일을 찾을 수 없습니다: {path}")
    except json.JSONDecodeError as exc:
        raise SoundstageToolError(f"강의실 설정 파일 {path} 의 JSON 형식이 잘못되었습니다: {exc}")
    rooms = data.get("rooms")
    if not isinstance(rooms, dict) or not rooms:
        raise SoundstageToolError(f"강의실 설정 파일 {path} 에 \"rooms\" 항목이 없거나 비어 있습니다.")
    return rooms


def get_room(room_name):
    """방 이름으로 설정을 찾는다. 없으면 등록된 방 목록을 알려 주는 오류를 낸다."""
    rooms = load_rooms()
    key = str(room_name).strip()
    if key not in rooms:
        names = ", ".join(sorted(rooms))
        raise SoundstageToolError(
            f"'{room_name}' 강의실은 등록되어 있지 않습니다. 등록된 강의실: {names}"
        )
    room = rooms[key]
    required = ("dimensions_m", "wall_absorption", "forbidden_zones",
                "speaker_height_m", "min_wall_distance_m", "listening_points")
    missing = [k for k in required if k not in room]
    if missing:
        raise SoundstageToolError(f"rooms.json의 '{key}' 강의실 설정에 다음 항목이 없습니다: {missing}")
    return key, room


def _pos(values):
    """좌표 하나를 순수 파이썬 float 리스트로 바꾼다. (numpy 값이 섞이지 않게)"""
    return [float(v) for v in values]


def _room_dims(room):
    return tuple(float(v) for v in room["dimensions_m"])


def _forbidden(room):
    return [tuple(float(v) for v in zone) for zone in room["forbidden_zones"]]


def _listeners(room):
    return [tuple(float(v) for v in p) for p in room["listening_points"]]


def _in_any_zone(x, y, zones):
    """candidates.py의 규칙과 같이, 경계 포함 직사각형 안이면 True."""
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in zones)


def _summary(result):
    """평가 딕셔너리에서 AI에게 보여 줄 핵심 값만 골라 순수 파이썬 값으로 만든다."""
    return {
        "position": _pos(result["position"]),
        "std_db": float(result["std_db"]),
        "mean_db": float(result["mean_db"]),
        "min_db": float(result["min_db"]),
    }


def _candidates_for(room, step):
    return generate_candidates(_room_dims(room), step, room["min_wall_distance_m"],
                               room["speaker_height_m"], forbidden_zones=_forbidden(room))


# ---------------------------------------------------------------------
# MCP 서버와 도구 4개
# ---------------------------------------------------------------------
mcp = MCPServer(
    "soundstage",
    instructions=(
        "강의실 스피커 배치 시뮬레이션 도구입니다. 좌표 단위는 m이고, 원점은 앞벽(y=0, 교탁/스크린 쪽)·"
        "왼쪽 벽(x=0)·바닥(z=0)이 만나는 모서리입니다. 음향 수준은 RIR 에너지를 dB로 바꾼 상대값이며, "
        "균일도 점수 std_db는 청취 지점들 사이 dB 표준편차로 작을수록 모든 자리에서 고르게 들립니다. "
        "먼저 get_room_info로 방 정보를 확인하고, measured가 false면 임시값이라는 점을 사용자에게 알려 주세요."
    ),
)


@mcp.tool()
def get_room_info(room_name: str = "5420") -> dict:
    """
    강의실의 공간 정보를 조회한다. (계산 없이 설정만 읽으므로 빠르다)

    언제 쓰나: 방 크기·벽 흡음률·스피커 설치 금지 구역·청취 지점(관객 자리) 좌표를 알고 싶을 때,
    또는 다른 도구를 쓰기 전에 좌표 범위를 확인할 때 가장 먼저 호출한다.

    Args:
        room_name: 강의실 이름 (예: "5420"). rooms.json에 등록된 이름이어야 한다.

    Returns:
        room_name, description, measured(실측값이면 true, 임시값이면 false), note,
        dimensions_m [가로 x, 세로 y, 높이 z] (m), wall_absorption (0~1, 클수록 소리를 많이 흡수),
        forbidden_zones [[x_min, y_min, x_max, y_max], ...] (스피커 설치 금지 직사각형, 경계 포함),
        speaker_height_m (스피커 후보 설치 높이), min_wall_distance_m (후보가 벽에서 떨어지는 최소 거리),
        listening_points [[x, y, z], ...] (음향 수준을 재는 청취 지점), num_listening_points
    """
    with _tool_guard():
        name, room = get_room(room_name)
        points = [_pos(p) for p in room["listening_points"]]
        return {
            "room_name": name,
            "description": str(room.get("description", "")),
            "measured": bool(room.get("measured", False)),
            "note": str(room.get("note", "")),
            "dimensions_m": _pos(room["dimensions_m"]),
            "wall_absorption": float(room["wall_absorption"]),
            "forbidden_zones": [_pos(z) for z in room["forbidden_zones"]],
            "speaker_height_m": float(room["speaker_height_m"]),
            "min_wall_distance_m": float(room["min_wall_distance_m"]),
            "listening_points": points,
            "num_listening_points": len(points),
        }


@mcp.tool()
def list_candidate_positions(room_name: str = "5420", step: float = 1.0) -> dict:
    """
    스피커를 놓을 수 있는 후보 위치 목록을 만든다. (계산 없이 격자만 만들므로 빠르다)

    언제 쓰나: "스피커를 어디어디에 둘 수 있어?", "후보가 몇 개야?"처럼 설치 가능한 자리를 알고 싶을 때.
    방 바닥을 step 간격 격자로 나눈 뒤, 벽에서 min_wall_distance_m보다 가까운 점과
    설치 금지 구역 안의 점을 뺀다. 모든 후보는 speaker_height_m 높이에 놓인다.

    Args:
        room_name: 강의실 이름 (예: "5420")
        step: 격자 간격 [m]. 0보다 커야 한다. 작을수록 후보가 촘촘하고 많아진다. 기본 1.0

    Returns:
        room_name, step, count(후보 개수), positions [[x, y, z], ...] (앞줄→뒷줄, 왼쪽→오른쪽 순서)
    """
    with _tool_guard():
        name, room = get_room(room_name)
        cands = _candidates_for(room, step)
        return {
            "room_name": name,
            "step": float(step),
            "count": len(cands),
            "positions": [_pos(c) for c in cands],
        }


@mcp.tool()
def evaluate_speaker_position(x: float, y: float, z: float, room_name: str = "5420") -> dict:
    """
    스피커 1개를 (x, y, z)에 두었을 때 각 청취 지점에서 소리가 얼마나 크게, 얼마나 고르게 들리는지 계산한다.
    (pyroomacoustics 시뮬레이션 1회, 보통 1초 이내)

    언제 쓰나: "스피커를 (3, 4, 2.5)에 두면 어때?"처럼 특정 위치 하나를 평가하거나 두 위치를 비교할 때.
    최적 위치를 찾고 싶으면 recommend_speaker_placement를 쓴다.

    Args:
        x: 스피커 가로 좌표 [m]. 0 < x < 방 가로
        y: 스피커 세로 좌표 [m]. 0 < y < 방 세로 (y=0이 앞벽, 교탁/스크린 쪽)
        z: 스피커 높이 [m]. 0 < z < 방 높이
        room_name: 강의실 이름 (예: "5420")

    Returns:
        room_name, position [x, y, z],
        levels_db [{"point": [x, y, z], "level_db": 값}, ...] (청취 지점별 음향 수준, 상대 dB),
        mean_db (평균), std_db (지점 간 표준편차, 작을수록 균일), min_db (가장 작게 들리는 지점),
        in_forbidden_zone (설치 금지 구역 안이면 true. 계산은 하지만 실제로는 설치할 수 없는 자리)
    """
    with _tool_guard():
        name, room = get_room(room_name)
        listeners = _listeners(room)
        # 방 밖 좌표 검사는 evaluate_position()이 한국어 ValueError로 해 준다
        result = evaluate_position(_room_dims(room), (x, y, z), listeners,
                                   absorption=float(room["wall_absorption"]))
        return {
            "room_name": name,
            "position": _pos((x, y, z)),
            "levels_db": [{"point": _pos(p), "level_db": float(v)}
                          for p, v in zip(listeners, result["levels_db"])],
            "mean_db": float(result["mean_db"]),
            "std_db": float(result["std_db"]),
            "min_db": float(result["min_db"]),
            "in_forbidden_zone": _in_any_zone(float(x), float(y), _forbidden(room)),
        }


@mcp.tool()
def recommend_speaker_placement(room_name: str = "5420", top_n: int = 5,
                                save_figure: bool = False) -> dict:
    """
    후보 위치(1 m 격자)를 모두 시뮬레이션해서 가장 고르게 들리는 스피커 위치를 추천하고,
    기준 배치(전면 중앙 상단: x=방 가로 절반, y=0.5 m, z=설치 높이)와 비교한다. (수 초 걸림)

    언제 쓰나: "스피커를 어디에 두는 게 제일 좋아?", "지금 배치보다 얼마나 좋아져?"라고 물을 때.
    고르는 규칙: 가장 작게 들리는 자리의 크기(min_db)가 기준 배치보다 나빠지지 않는 후보 중
    std_db(지점 간 dB 표준편차)가 가장 작은 후보를 추천한다.

    Args:
        room_name: 강의실 이름 (예: "5420")
        top_n: 조건을 통과한 후보 중 상위 몇 개를 함께 돌려줄지 (1 이상 정수, 기본 5)
        save_figure: true면 히트맵·비교 그림을 soundstage/output/mcp_report.png로 저장하고 경로를 돌려준다

    Returns:
        room_name,
        baseline {position, std_db, mean_db, min_db} (기준 배치),
        recommended {position, std_db, mean_db, min_db} (추천 배치),
        improvement_percent (균일도 개선율 % = (기준 std_db - 추천 std_db) / 기준 std_db * 100),
        threshold_db (최소 음향 수준 조건: min_db >= 이 값), num_candidates (전체 후보 수),
        num_passed (조건 통과 후보 수), top_candidates [{rank, position, std_db, mean_db, min_db}, ...],
        figure_path (save_figure가 true일 때만)
    """
    with _tool_guard():
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise SoundstageToolError(f"top_n은 1 이상의 정수여야 합니다. (받은 값: {top_n!r})")
        name, room = get_room(room_name)
        dims = _room_dims(room)
        listeners = _listeners(room)
        cands = _candidates_for(room, 1.0)
        baseline_pos = default_baseline_position(dims, cands)

        log.info("추천 계산 시작: 방 %s, 후보 %d개", name, len(cands))
        rec = recommend_placement(dims, cands, listeners, baseline_pos=baseline_pos,
                                  absorption=float(room["wall_absorption"]), top_n=top_n)

        out = {
            "room_name": name,
            "baseline": _summary(rec["baseline"]),
            "recommended": _summary(rec["best"]),
            "improvement_percent": float(rec["improvement_percent"]),
            "threshold_db": float(rec["threshold_db"]),
            "num_candidates": int(rec["num_candidates"]),
            "num_passed": int(rec["num_passed"]),
            "top_candidates": [{"rank": i, **_summary(r)}
                               for i, r in enumerate(rec["top"], start=1)],
        }
        if not math.isfinite(out["improvement_percent"]):
            raise SoundstageToolError("개선율 계산 결과가 올바른 숫자가 아닙니다.")

        if save_figure:
            make_report_figure(dims, listeners, rec, forbidden_zones=_forbidden(room),
                               save_path=REPORT_PATH)
            out["figure_path"] = REPORT_PATH
            log.info("그림 저장: %s", REPORT_PATH)
        return out


if __name__ == "__main__":
    log.info("soundstage MCP 서버 시작 (stdio)")
    mcp.run()  # 기본 전송 방식이 stdio
