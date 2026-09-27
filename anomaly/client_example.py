"""
API 서버 호출 예제 (다른 프로그램에서 검사 기능을 쓰는 방법)

사용법:
  python client_example.py 이미지경로 [--heatmap 저장경로]
예:
  python client_example.py data/sample/test/defect/000.png --heatmap result.png
"""
import argparse
import base64
import time

import httpx

SERVER = "http://127.0.0.1:8000"

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--model", default=None)
parser.add_argument("--threshold", type=float, default=None, help="0~1")
parser.add_argument("--heatmap", default=None, help="히트맵을 저장할 파일 경로")
args = parser.parse_args()

form = {"return_heatmap": str(bool(args.heatmap)).lower()}
if args.model:
    form["model"] = args.model
if args.threshold is not None:
    form["threshold"] = str(args.threshold)

t0 = time.perf_counter()
with open(args.image, "rb") as f:
    resp = httpx.post(f"{SERVER}/inspect", files={"file": f}, data=form, timeout=60)
resp.raise_for_status()
r = resp.json()
total_ms = (time.perf_counter() - t0) * 1000

print(f"판정      : {r['result']}")
print(f"이상 점수 : {r['score']} / 100  (임계값 {r['threshold']})")
print(f"불량 영역 : {r['boxes'] or '없음'}")
print(f"서버 처리 : {r['elapsed_ms']} ms   왕복 전체: {total_ms:.1f} ms")

if args.heatmap and "heatmap_png_base64" in r:
    with open(args.heatmap, "wb") as f:
        f.write(base64.b64decode(r["heatmap_png_base64"]))
    print(f"히트맵 저장: {args.heatmap}")
