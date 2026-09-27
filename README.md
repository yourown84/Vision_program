# 딥러닝 비전 검사 프로젝트

- `anomaly/` : 비전 검사 시스템 (어노멀리 · 분류 · 세그멘테이션, UI, API 서버, 관리 서버, 에이전트). 자세한 사용법은 [anomaly/README.md](anomaly/README.md)
- `mnist/` : 첫 학습 연습용 예제 (손글씨 분류)

## 새 PC에 설치하기

1. Python 3.12 설치: https://www.python.org/downloads/windows/ (설치 화면에서 **Add python.exe to PATH** 체크)
2. 이 저장소를 내려받아 원하는 폴더에 둡니다 (GitHub 에서 `Code > Download ZIP` 또는 `git clone`).
3. `setup.bat` 더블클릭. 가상환경을 만들고 PyTorch 와 라이브러리를 설치합니다 (약 3GB 다운로드, 10~20분).
4. `anomaly\start_ui.bat` 더블클릭으로 화면 실행.

GPU 가 없는 PC 에서도 동작합니다 (`anomaly\config.bat` 에서 `INSPECT_DEVICE=cpu`).

## 저장소에 포함되지 않는 것

- 학습된 모델 (`anomaly/models*`): 크기 때문에 제외. 관리 서버의 배포 기능으로 보내거나 폴더를 따로 복사하세요.
- 학습 데이터 (`anomaly/data`): 샘플 데이터는 `setup.bat` 이 다시 생성합니다.
- 가상환경 (`.venv`), 캐시 (`.cache`): PC 마다 `setup.bat` 으로 새로 만듭니다.
