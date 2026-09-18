#!/usr/bin/env python3
"""사내 vLLM 서버(Qwen3.8-27B) VLM 한계 프로브.

정답 라벨 없이 VLM의 지각/판단 한계를 재는 도구. 합성 영상을 ffmpeg로 만들기 때문에
정답이 구성상 자명하다 (라벨링 불필요, 스크립트로 완전 재현 가능).

사용:
    python3 vlm_probe.py temporal   # 시간 해상도 임계 (몇 초 이상 지속돼야 보이나)
    python3 vlm_probe.py sampling   # 영상 길이별 샘플 fps
    python3 vlm_probe.py spatial    # 공간 해상도 임계 (몇 픽셀 이상이어야 보이나)
    python3 vlm_probe.py bundling   # 한 프롬프트의 항목 수에 따른 정확도 변화
    python3 vlm_probe.py agreement  # 실제 영상에서 단독 질의 vs 묶음 질의 답변 일치율
    python3 vlm_probe.py prompt     # 프롬프트 길이/위치/전제/언어에 따른 정확도 변화

표준 라이브러리 + ffmpeg만 사용.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from collections import Counter

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
WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")


def ask(name: str, text: str, max_tokens: int = 16):
    """서버 media 디렉토리의 영상 1개에 질문하고 (응답, prompt_tokens) 반환."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": f"file://{REMOTE_MEDIA}/{name}"}},
            {"type": "text", "text": text}]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,                      # 프로브는 결정론적으로
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as f:
        d = json.load(f)
    return d["choices"][0]["message"]["content"].strip(), d["usage"]["prompt_tokens"]


def make(name: str, duration: int, boxes=(), size="640x480", fps=30):
    """검정 배경에 사각형을 그린 합성 영상을 만들고 서버로 올린다.

    boxes: (x, y, w, h, color, start_frame, end_frame) 튜플들. 비면 빈 영상.
    """
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, name)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", f"color=c=black:s={size}:r={fps}:d={duration}"]
    if boxes:
        vf = ",".join(
            f"drawbox=x={x}:y={y}:w={w}:h={h}:color={c}:t=fill:enable='between(n,{s},{e})'"
            for x, y, w, h, c, s, e in boxes)
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, check=True)
    subprocess.run(["scp", "-q", path, f"{SSH_HOST}:{REMOTE_MEDIA}/{name}"], check=True)
    return name


YESNO = "'예' 또는 '아니오' 한 단어로만 답하세요."


def probe_temporal(reps):
    """자극 지속 프레임 수를 줄여가며 탐지 임계를 찾는다. 10초 영상 고정."""
    q = f"이 영상에 빨간색 사각형이 한 번이라도 나타납니까? {YESNO}"
    print(f"{'자극':>10} {'정답':>5} {'응답':>22} {'판정':>5}")
    for n in (0, 1, 2, 5, 15, 30, 90):
        boxes = () if n == 0 else ((280, 200, 80, 80, "red", 150, 150 + n - 1),)
        name = make(f"t_{n}f.mp4", 10, boxes)
        truth = "아니오" if n == 0 else "예"
        outs = [ask(name, q)[0] for _ in range(reps)]
        ok = all(o.startswith(truth) for o in outs)
        print(f"{n:>7}프레임 {truth:>5} {str(dict(Counter(outs))):>24} {'OK' if ok else 'FAIL':>5}")


def probe_sampling(reps):
    """영상 길이별 prompt_tokens로 유효 샘플 fps를 역산한다."""
    print(f"{'길이':>6} {'prompt_tok':>11} {'추정 프레임':>11} {'유효 fps':>9}")
    for d in (5, 10, 20, 60, 180):
        name = make(f"d_{d}s.mp4", d)
        _, tok = ask(name, "한 단어로 답하세요: 검정", 4)
        # 640x480 -> 프레임쌍당 300 tok (patch16 x merge2), temporal_patch_size=2
        frames = (tok - 127) / 300 * 2
        print(f"{d:>5}s {tok:>10} {frames:>10.1f} {frames/d:>9.2f}")


def probe_spatial(reps):
    """자극 크기(픽셀)를 줄여가며 탐지 임계를 찾는다. 지속시간은 충분히 길게 고정."""
    q = f"이 영상에 빨간색 사각형이 한 번이라도 나타납니까? {YESNO}"
    print(f"{'크기':>9} {'정답':>5} {'응답':>22} {'판정':>5}")
    for s in (0, 80, 48, 32, 24, 16, 8, 4):
        # 시간축 영향을 배제하려 전 구간(10초) 표시
        boxes = () if s == 0 else (((640 - s) // 2, (480 - s) // 2, s, s, "red", 0, 299),)
        name = make(f"s_{s}px.mp4", 10, boxes)
        truth = "아니오" if s == 0 else "예"
        outs = [ask(name, q)[0] for _ in range(reps)]
        ok = all(o.startswith(truth) for o in outs)
        print(f"{s:>6}px {truth:>5} {str(dict(Counter(outs))):>24} {'OK' if ok else 'FAIL':>5}")


# 쉬운 항목: 색 존재 여부만 (4개 등장 / 4개 미등장)
ITEMS_EASY = [(f"{c} 사각형이 나타납니까?", t) for c, t in
              [("빨간색", True), ("보라색", False), ("초록색", True), ("주황색", False),
               ("파란색", True), ("분홍색", False), ("노란색", True), ("하늘색", False)]]

# 어려운 항목: 공간/시간/개수 추론 필요 (4 참 / 4 거짓)
ITEMS_HARD = [
    ("빨간 사각형이 파란 사각형보다 먼저 나타납니까?", True),
    ("파란 사각형은 화면 위쪽 절반에 있습니까?", False),
    ("초록 사각형은 화면 오른쪽 절반에 있습니까?", True),
    ("노란 사각형이 네 개 중 가장 먼저 나타납니까?", False),
    ("영상에 나타나는 사각형은 모두 네 개입니까?", True),
    ("빨간 사각형과 초록 사각형이 같은 순간에 함께 보입니까?", False),
    ("빨간 사각형은 화면 왼쪽 위에 있습니까?", True),
    ("보라색 사각형이 나타납니까?", False),
]


def probe_bundling(reps, hard=False):
    """한 프롬프트에 K개 항목을 묶었을 때 항목당 정확도가 떨어지는지 본다.

    떨어지면 -> 작업 분해(multi-agent)가 정당화된다.
    떨어지지 않으면 -> 묶어서 물어도 되므로 분해할 이유가 없다.
    """
    name = make("bundle.mp4", 20, (
        (50, 50, 80, 80, "red", 0, 59),        # 0-2s   좌상
        (510, 50, 80, 80, "green", 150, 209),  # 5-7s   우상
        (50, 350, 80, 80, "blue", 300, 359),   # 10-12s 좌하
        (510, 350, 80, 80, "yellow", 450, 509),# 15-17s 우하
    ))
    items = ITEMS_HARD if hard else ITEMS_EASY
    print(f"[{'어려움' if hard else '쉬움'}] {'묶음크기':>8} {'항목정확도':>11} {'형식오류':>9}")
    for k in (1, 2, 4, 8):
        correct = total = malformed = 0
        for _ in range(reps):
            for i in range(0, len(items), k):
                chunk = items[i:i + k]
                lines = "\n".join(f"{j+1}. {q}" for j, (q, _) in enumerate(chunk))
                q = (f"{lines}\n\n각 항목에 대해 '번호: 예' 또는 '번호: 아니오' 형식으로 "
                     f"{len(chunk)}줄로만 답하세요.")
                out, _ = ask(name, q, max_tokens=16 * len(chunk) + 16)
                got = dict(re.findall(r"(\d+)\s*[:.]?\s*(예|아니오)", out))
                for j, (_, truth) in enumerate(chunk):
                    total += 1
                    a = got.get(str(j + 1))
                    if a is None:
                        malformed += 1
                    elif (a == "예") == truth:
                        correct += 1
        print(f"{k:>6}개 {correct}/{total} ({correct/total:>5.1%}) {malformed:>8}")


# 실제 영상용: 정답을 모르는 항목들. 정답 대신 "따로 물을 때 vs 묶어 물을 때"의
# 답변 일치율을 본다 (라벨 없이 묶음 효과를 재는 방법).
ITEMS_REAL = [
    "로봇 팔이 두 개 이상 보입니까?",
    "테이블 위에 컵이 있습니까?",
    "사람의 손이 보입니까?",
    "화면이 두 개의 시점으로 나뉘어 있습니까?",
    "로봇이 물체를 집어 올립니까?",
    "배경에 창문이 보입니까?",
    "접시 또는 받침이 보입니까?",
    "로봇 그리퍼가 열렸다 닫힙니까?",
]


# 영어 대조군: ITEMS_HARD와 같은 내용, 같은 정답
ITEMS_HARD_EN = [
    ("Does the red square appear before the blue square?", True),
    ("Is the blue square in the upper half of the frame?", False),
    ("Is the green square in the right half of the frame?", True),
    ("Does the yellow square appear first among the four?", False),
    ("Are there exactly four squares in the video?", True),
    ("Are the red and green squares visible at the same moment?", False),
    ("Is the red square in the top-left of the frame?", True),
    ("Does a purple square appear?", False),
]

# 질문과 무관한 지시문. 프롬프트 길이/잡음 내성 측정에 쓴다.
PAD = ("당신은 신중한 조수입니다. 답변은 항상 정중해야 하며, 사용자의 의도를 존중해야 합니다. "
       "확실하지 않은 내용은 추측하지 말고, 근거가 부족하면 그 사실을 밝혀야 합니다. "
       "답변에는 불필요한 수식어를 넣지 말고, 전문 용어를 쓸 때에는 풀어서 설명해야 합니다. "
       "사용자가 여러 질문을 한꺼번에 하면 순서대로 빠짐없이 답해야 합니다. ")


def _parse(out, n):
    """번호별 예/아니오(또는 yes/no)를 뽑아 불리언 딕셔너리로 만든다."""
    got = {}
    for num, ans in re.findall(r"(\d+)\s*[:.]?\s*(예|아니오|yes|no)", out, re.I):
        got[num] = ans.lower() in ("예", "yes")
    return got


def _score(video, items, prefix="", suffix="", en=False):
    """항목 전체를 한 번에 묶어 묻고 (정답수, 총수, 형식오류수, prompt_tokens) 반환."""
    lines = "\n".join(f"{i+1}. {q}" for i, (q, _) in enumerate(items))
    fmt = (f"Answer in exactly {len(items)} lines, each as 'number: yes' or 'number: no'."
           if en else
           f"각 항목에 '번호: 예' 또는 '번호: 아니오' 형식으로 {len(items)}줄로만 답하세요.")
    out, tok = ask(video, f"{prefix}{lines}\n\n{fmt}{suffix}", max_tokens=16 * len(items) + 16)
    got = _parse(out, len(items))
    correct = malformed = 0
    for i, (_, truth) in enumerate(items):
        a = got.get(str(i + 1))
        if a is None:
            malformed += 1
        elif a == truth:
            correct += 1
    return correct, len(items), malformed, tok


def probe_prompt(reps):
    """프롬프트 조건이 정확도에 미치는 영향. 경계 난이도 항목(ITEMS_HARD)을 탐침으로 쓴다."""
    v = make("bundle.mp4", 20, (
        (50, 50, 80, 80, "red", 0, 59),
        (510, 50, 80, 80, "green", 150, 209),
        (50, 350, 80, 80, "blue", 300, 359),
        (510, 350, 80, 80, "yellow", 450, 509),
    ))

    def agg(**kw):
        c = t = m = 0
        tok = 0
        for _ in range(reps):
            a, b, d, tok = _score(v, **kw)
            c += a; t += b; m += d
        return f"{c}/{t} ({c/t:>5.1%})", m, tok

    print("[A] 무관한 지시문을 앞에 붙였을 때")
    print(f"    {'패딩 길이':>10} {'prompt_tok':>11} {'정확도':>16} {'형식오류':>9}")
    for mult in (0, 1, 5, 20, 60):
        pad = (PAD * mult) + ("\n\n" if mult else "")
        acc, mal, tok = agg(items=ITEMS_HARD, prefix=pad)
        print(f"    {len(pad):>8}자 {tok:>10} {acc:>18} {mal:>8}")

    print("\n[B] 긴 프롬프트(약 2,400자) 안에서 질문의 위치")
    print(f"    {'위치':>10} {'정확도':>16} {'형식오류':>9}")
    half = PAD * 10
    for name, pre, suf in (("맨 앞", "", "\n\n" + half + half),
                           ("가운데", half, "\n\n" + half),
                           ("맨 뒤", half + half, "")):
        acc, mal, _ = agg(items=ITEMS_HARD, prefix=pre, suffix=suf)
        print(f"    {name:>8} {acc:>18} {mal:>8}")

    print("\n[C] 전제를 심었을 때 (동조 편향)")
    print(f"    {'전제':>10} {'정확도':>16} {'형식오류':>9}")
    for name, pre in (
            ("없음(중립)", ""),
            ("참인 전제", "참고로 이 영상에는 사각형이 네 개 나타나며, 빨간 사각형이 가장 먼저 나타납니다.\n\n"),
            ("거짓 전제", "참고로 이 영상에는 사각형이 두 개만 나타나며, 노란 사각형이 가장 먼저 나타납니다.\n\n")):
        acc, mal, _ = agg(items=ITEMS_HARD, prefix=pre)
        print(f"    {name:>8} {acc:>18} {mal:>8}")

    print("\n[D] 질문 언어")
    print(f"    {'언어':>10} {'정확도':>16} {'형식오류':>9}")
    for name, items, en in (("한국어", ITEMS_HARD, False), ("영어", ITEMS_HARD_EN, True)):
        acc, mal, _ = agg(items=items, en=en)
        print(f"    {name:>8} {acc:>18} {mal:>8}")


def probe_agreement(reps, video="t00_ep000.mp4"):
    """라벨 없는 실제 영상에서 묶음 효과 측정: K=1 답변과 K=8 답변의 불일치율."""
    def one(k):
        answers = {}
        for i in range(0, len(ITEMS_REAL), k):
            chunk = ITEMS_REAL[i:i + k]
            lines = "\n".join(f"{j+1}. {q}" for j, q in enumerate(chunk))
            q = (f"{lines}\n\n각 항목에 '번호: 예' 또는 '번호: 아니오' 형식으로 "
                 f"{len(chunk)}줄로만 답하세요.")
            out, _ = ask(video, q, max_tokens=16 * len(chunk) + 16)
            got = dict(re.findall(r"(\d+)\s*[:.]?\s*(예|아니오)", out))
            for j in range(len(chunk)):
                answers[i + j] = got.get(str(j + 1))
        return answers

    print(f"대상: {video} (정답 라벨 없음 — 일치율로 측정)")
    singles = [one(1) for _ in range(reps)]
    bundles = [one(8) for _ in range(reps)]
    self_s = sum(singles[0][i] == singles[r][i] for r in range(reps) for i in range(8))
    self_b = sum(bundles[0][i] == bundles[r][i] for r in range(reps) for i in range(8))
    cross = sum(singles[0][i] == bundles[0][i] for i in range(8))
    print(f"  K=1 반복 자기일치  : {self_s}/{reps*8}")
    print(f"  K=8 반복 자기일치  : {self_b}/{reps*8}")
    print(f"  K=1 vs K=8 일치    : {cross}/8")
    for i, q in enumerate(ITEMS_REAL):
        flag = "  <-- 불일치" if singles[0][i] != bundles[0][i] else ""
        print(f"   {i+1}. 단독={singles[0][i]:<4} 묶음={bundles[0][i]:<4} {q}{flag}")


PROBES = {"temporal": probe_temporal, "sampling": probe_sampling,
          "spatial": probe_spatial, "bundling": probe_bundling,
          "agreement": probe_agreement, "prompt": probe_prompt}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe", choices=list(PROBES))
    ap.add_argument("--reps", type=int, default=3, help="반복 횟수 (기본 3)")
    ap.add_argument("--hard", action="store_true", help="bundling: 공간/시간 추론 항목 사용")
    args = ap.parse_args()
    require_config()
    if args.probe == "bundling":
        probe_bundling(args.reps, args.hard)
    else:
        PROBES[args.probe](args.reps)
