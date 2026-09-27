"""
세그멘테이션용 샘플 데이터 생성기
- 이미지와 같은 이름의 마스크(흑백 PNG, 흰색=불량)를 짝으로 만든다.
  data/sample_seg/images/000.png
  data/sample_seg/masks/000.png
정상 이미지도 일부 포함한다 (마스크는 전부 검정). 실제 현장 데이터도 이 구조로 준비하면 된다.
"""
import random
from pathlib import Path

import numpy as np
from PIL import Image

from make_sample_data import add_defect, make_normal

ROOT = Path(__file__).parent / "data" / "sample_seg"
N_DEFECT, N_NORMAL = 80, 20
random.seed(2)
np.random.seed(2)

if __name__ == "__main__":
    (ROOT / "images").mkdir(parents=True, exist_ok=True)
    (ROOT / "masks").mkdir(parents=True, exist_ok=True)
    i = 0
    for _ in range(N_DEFECT):
        img, mask = add_defect(make_normal(), return_mask=True)
        img.save(ROOT / "images" / f"{i:03d}.png")
        mask.save(ROOT / "masks" / f"{i:03d}.png")
        i += 1
    for _ in range(N_NORMAL):
        img = make_normal()
        img.save(ROOT / "images" / f"{i:03d}.png")
        Image.new("L", img.size, 0).save(ROOT / "masks" / f"{i:03d}.png")
        i += 1
    print(f"생성 완료: {ROOT}  (불량 {N_DEFECT}장 + 정상 {N_NORMAL}장, 마스크 포함)")
