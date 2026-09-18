#!/usr/bin/env python3
"""사내 vLLM 서버(Qwen3.8-27B) 영상 질의 CLI.

사용:
    python3 vllm_ask.py <영상경로> "질문"
    python3 vllm_ask.py ~/Videos/t00_ep000.mp4 "이 영상 설명해줘"

로컬 파일이면 서버 media 디렉토리로 scp 후 file:// 로 질의한다.
(서버가 --allowed-local-media-path 로 그 디렉토리만 허용하기 때문)
표준 라이브러리만 사용 — 추가 설치 불필요.
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

MODEL = "Qwen3.8-27B"
# 서버 접속 정보는 저장소에 남기지 않고 환경 변수로 받는다.
BASE_URL = os.environ.get("VLM_BASE_URL", "")
SSH_HOST = os.environ.get("VLM_SSH_HOST", "")
REMOTE_MEDIA = os.environ.get("VLM_REMOTE_MEDIA", "")


def require_config():
    missing = [n for n, v in (("VLM_BASE_URL", BASE_URL),
                              ("VLM_SSH_HOST", SSH_HOST),
                              ("VLM_REMOTE_MEDIA", REMOTE_MEDIA)) if not v]
    if missing:
        sys.exit("[설정 필요] 다음 환경 변수를 설정하세요: " + ", ".join(missing) + "\n"
                 '  export VLM_BASE_URL="http://<서버주소>:<포트>/v1"\n'
                 '  export VLM_SSH_HOST="<ssh 호스트 별칭>"\n'
                 '  export VLM_REMOTE_MEDIA="<서버의 allowed-local-media-path 경로>"')


def resolve_video_url(path: str) -> str:
    """로컬 경로면 서버로 업로드하고 file:// URL을 돌려준다."""
    if path.startswith(("http://", "https://", "file://")):
        return path

    local = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(local):
        sys.exit(f"[에러] 파일이 없습니다: {local}")

    remote = f"{REMOTE_MEDIA}/{os.path.basename(local)}"
    print(f"[업로드] {local} -> {SSH_HOST}:{remote}", file=sys.stderr)
    subprocess.run(["scp", "-q", local, f"{SSH_HOST}:{remote}"], check=True)
    return f"file://{remote}"


def ask(video_url: str, text: str, max_tokens: int, think: bool) -> str:
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": text},
            ],
        }],
        "max_tokens": max_tokens,
        # thinking 켜면 영어 추론이 먼저 나와 답변 토큰을 다 잡아먹는다
        "chat_template_kwargs": {"enable_thinking": think},
    }
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            body = json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"[에러] HTTP {e.code}\n{e.read().decode()[:2000]}")
    return body["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser(description="사내 vLLM 서버 영상 질의")
    ap.add_argument("video", help="로컬 영상 경로 또는 http(s)/file:// URL")
    ap.add_argument("text", nargs="?", default="이 영상 설명해줘", help="질문 (기본: 이 영상 설명해줘)")
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--think", action="store_true", help="thinking 모드 켜기 (기본: 꺼짐)")
    args = ap.parse_args()
    require_config()

    print(ask(resolve_video_url(args.video), args.text, args.max_tokens, args.think))


if __name__ == "__main__":
    main()
