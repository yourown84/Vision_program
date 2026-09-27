"""
분류용 샘플 데이터 생성기
- 폴더 이름이 곧 클래스(정답 라벨)가 된다.
  data/sample_cls/good/*.png       정상
  data/sample_cls/scratch/*.png    긁힘
  data/sample_cls/blob/*.png       얼룩
  data/sample_cls/missing/*.png    누락
"""
import random
from pathlib import Path

import numpy as np

from make_sample_data import add_defect, make_normal, save

ROOT = Path(__file__).parent / "data" / "sample_cls"
N_PER_CLASS = 50
random.seed(1)
np.random.seed(1)

if __name__ == "__main__":
    save([make_normal() for _ in range(N_PER_CLASS)], ROOT / "good")
    for kind in ["scratch", "blob", "missing"]:
        save([add_defect(make_normal(), kind) for _ in range(N_PER_CLASS)], ROOT / kind)
    print(f"생성 완료: {ROOT}  (4 클래스 x {N_PER_CLASS}장)")
