"""
샘플 데이터 생성기
- 실제 이미지가 없을 때 파이프라인 동작을 확인하기 위한 합성 이미지
- 정상: 규칙적인 원형 패턴 (부품 표면 느낌) + 약간의 노이즈
- 불량: 정상 이미지에 긁힘(선) 또는 얼룩(점)을 추가

폴더 구조 (anomalib Folder 형식):
  data/sample/train/good/*.png       정상 (학습용)
  data/sample/test/good/*.png        정상 (검증용)
  data/sample/test/defect/*.png      불량 (검증용)
"""
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).parent / "data" / "sample"
SIZE = 256
N_TRAIN_GOOD, N_TEST_GOOD, N_TEST_DEFECT = 60, 15, 20
random.seed(0)
np.random.seed(0)


def make_normal():
    img = Image.new("RGB", (SIZE, SIZE), (150, 150, 155))
    d = ImageDraw.Draw(img)
    # 규칙적인 원 패턴
    for y in range(24, SIZE, 40):
        for x in range(24, SIZE, 40):
            r = 11 + random.uniform(-0.8, 0.8)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(95, 95, 105))
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    # 조명 변화 + 노이즈
    arr = np.asarray(img).astype(np.float32)
    gx = np.linspace(-1, 1, SIZE)[None, :, None] * random.uniform(-8, 8)
    gy = np.linspace(-1, 1, SIZE)[:, None, None] * random.uniform(-8, 8)
    arr = arr + gx + gy + np.random.normal(0, 4, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_defect(img, kind=None, return_mask=False):
    """불량을 그려 넣는다. return_mask=True 면 (이미지, 불량 마스크) 를 돌려준다."""
    d = ImageDraw.Draw(img)
    mask = Image.new("L", img.size, 0)
    md = ImageDraw.Draw(mask)
    kind = kind or random.choice(["scratch", "blob", "missing"])
    if kind == "scratch":
        x0, y0 = random.randint(20, SIZE - 20), random.randint(20, SIZE - 20)
        x1, y1 = x0 + random.randint(-60, 60), y0 + random.randint(-60, 60)
        w = random.randint(2, 4)
        d.line([x0, y0, x1, y1], fill=(230, 230, 235), width=w)
        md.line([x0, y0, x1, y1], fill=255, width=w + 2)
    elif kind == "blob":
        x, y = random.randint(20, SIZE - 20), random.randint(20, SIZE - 20)
        r = random.randint(6, 14)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(40, 30, 30))
        md.ellipse([x - r, y - r, x + r, y + r], fill=255)
    else:  # 원 하나가 빠진 경우
        x, y = random.choice(range(24, SIZE, 40)), random.choice(range(24, SIZE, 40))
        d.ellipse([x - 13, y - 13, x + 13, y + 13], fill=(150, 150, 155))
        md.ellipse([x - 13, y - 13, x + 13, y + 13], fill=255)
    return (img, mask) if return_mask else img


def save(images, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(images):
        im.save(folder / f"{i:03d}.png")


if __name__ == "__main__":
    save([make_normal() for _ in range(N_TRAIN_GOOD)], ROOT / "train" / "good")
    save([make_normal() for _ in range(N_TEST_GOOD)], ROOT / "test" / "good")
    save([add_defect(make_normal()) for _ in range(N_TEST_DEFECT)], ROOT / "test" / "defect")
    print(f"생성 완료: {ROOT}")
    print(f"  학습 정상 {N_TRAIN_GOOD}장 / 검증 정상 {N_TEST_GOOD}장 / 검증 불량 {N_TEST_DEFECT}장")
