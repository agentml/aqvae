#!/usr/bin/env python3
"""세션 트랜스크립트(JSONL)에서 대화 원문을 docs/conversation_raw.md 로 추출.

두 가지 모드:
  1) 수동:   python scripts/export_transcript.py <transcript.jsonl> [out.md]
  2) 훅(stdin): Claude Code PostToolUse 훅이 JSON(stdin)으로 호출.
     git commit 명령일 때만 동작(주요 작업 종료 시점). 그 외엔 no-op.

사용자 발화 + 어시스턴트 응답 텍스트를 verbatim 추출. 내부 thinking·tool I/O 제외
(tool 은 이름만 표기). task-notification 등 시스템 주입 라인은 필터.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter


def extract(jsonl_path: str) -> str:
    turns = []
    with open(jsonl_path) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") not in ("user", "assistant"):
                continue
            msg = d.get("message", {})
            role, content = msg.get("role"), msg.get("content")
            texts, tools, is_tr = [], [], False
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    bt = b.get("type")
                    if bt == "text":
                        texts.append(b.get("text", ""))
                    elif bt == "tool_use":
                        tools.append(b.get("name", "?"))
                    elif bt == "tool_result":
                        is_tr = True
            body = "\n".join(t for t in texts if t and t.strip())
            # 시스템 주입(task-notification/system-reminder 단독) 필터
            if role == "user":
                if is_tr and not body.strip():
                    continue
                stripped = re.sub(r"<(task-notification|system-reminder|local-command[^>]*)>.*?</\1>",
                                  "", body, flags=re.S).strip()
                stripped = re.sub(r"\[SYSTEM NOTIFICATION.*", "", stripped, flags=re.S).strip()
                if not stripped:
                    continue
                turns.append(("user", stripped, []))
            else:
                if body.strip() or tools:
                    turns.append(("assistant", body, tools))
    # 연속 동일 role 병합
    merged = []
    for r, b, tl in turns:
        if merged and merged[-1][0] == r:
            merged[-1][1].append(b)
            merged[-1][2].extend(tl)
        else:
            merged.append([r, [b], list(tl)])
    out = ["# 대화 원시 로그 (verbatim, 트랜스크립트 추출)", "",
           "> 세션 JSONL 에서 사용자 발화·어시스턴트 응답을 *있는 그대로* 추출(자동 갱신).",
           "> 제외: 내부 thinking, tool I/O(이름만 표기), 시스템 주입 라인.", "", "---", ""]
    for r, bodies, tools in merged:
        out.append("## 👤 사용자" if r == "user" else "## 🤖 Claude")
        txt = "\n\n".join(b for b in bodies if b.strip())
        if txt.strip():
            out += ["", txt]
        if r == "assistant" and tools:
            c = Counter(tools)
            out += ["", "`[tools: " + ", ".join(f"{k}×{v}" if v > 1 else k
                                                 for k, v in c.items()) + "]`"]
        out += ["", "---", ""]
    return "\n".join(out)


def main():
    # 훅 모드: stdin 에 JSON
    if not sys.stdin.isatty() and len(sys.argv) < 2:
        try:
            payload = json.load(sys.stdin)
        except Exception:
            return
        cmd = (payload.get("tool_input", {}) or {}).get("command", "")
        if "git commit" not in cmd:                       # 커밋일 때만
            return
        tpath = payload.get("transcript_path")
        cwd = payload.get("cwd", ".")
        if not tpath or not os.path.exists(tpath):
            return
        out_path = os.path.join(cwd, "docs", "conversation_raw.md")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        open(out_path, "w").write(extract(tpath))
        return
    # 수동 모드
    if len(sys.argv) < 2:
        print("usage: export_transcript.py <transcript.jsonl> [out.md]", file=sys.stderr)
        sys.exit(1)
    out_path = sys.argv[2] if len(sys.argv) > 2 else "docs/conversation_raw.md"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    open(out_path, "w").write(extract(sys.argv[1]))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
