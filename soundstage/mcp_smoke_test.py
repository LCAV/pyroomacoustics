"""
MCP 서버 스모크 테스트: 실제 MCP 프로토콜(stdio)로 서버를 띄워 짧게 확인한다.

  1) 공식 SDK의 stdio 클라이언트로 mcp_server.py를 하위 프로세스로 실행
  2) 초기화(initialize) → 도구 목록(tools/list)에 도구 4개가 보이는지 확인
  3) get_room_info 를 한 번 호출해 결과를 출력
  4) 없는 방 이름으로 호출했을 때 한국어 오류 메시지가 그대로 전달되는지 확인

이 파일은 "클라이언트"이므로 print를 써도 된다. (서버 쪽 stdout과는 별개)
pytest가 자동으로 모으지 않도록 파일 이름을 test_로 시작하지 않게 했다.

실행 방법:  python soundstage/mcp_smoke_test.py
"""

import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters, stdio_client

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "mcp_server.py")
EXPECTED_TOOLS = {"get_room_info", "list_candidate_positions",
                  "evaluate_speaker_position", "recommend_speaker_placement"}


def _text(result):
    """CallToolResult의 텍스트 내용을 하나의 문자열로 합친다."""
    return "".join(getattr(c, "text", "") for c in result.content)


async def main():
    # 지금 실행 중인 파이썬(= .venv의 python)으로 서버를 띄운다
    params = StdioServerParameters(command=sys.executable, args=[SERVER])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"[1] 초기화 성공: 서버 이름={init.server_info.name}, "
                  f"프로토콜={init.protocol_version}")

            tools = (await session.list_tools()).tools
            names = sorted(t.name for t in tools)
            print(f"[2] 도구 {len(names)}개: {names}")
            assert set(names) == EXPECTED_TOOLS, f"도구 목록이 다릅니다: {names}"

            result = await session.call_tool("get_room_info", {"room_name": "5420"})
            assert not result.is_error, _text(result)
            info = json.loads(_text(result))
            print("[3] get_room_info 호출 성공:")
            print("    " + json.dumps(
                {k: info[k] for k in ("room_name", "measured", "note", "dimensions_m",
                                      "wall_absorption", "forbidden_zones",
                                      "num_listening_points")},
                ensure_ascii=False))
            assert info["room_name"] == "5420" and info["num_listening_points"] == 12

            bad = await session.call_tool("get_room_info", {"room_name": "9999"})
            print(f"[4] 없는 방 호출: is_error={bad.is_error}, 메시지={_text(bad)!r}")
            assert bad.is_error and "등록되어 있지 않습니다" in _text(bad)

    print("스모크 테스트 통과")


if __name__ == "__main__":
    asyncio.run(main())
