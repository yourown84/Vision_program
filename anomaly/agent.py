"""
라인 PC 에이전트 (Agent) - 각 라인 PC 에서 API 서버(server.py)와 함께 실행
- 몇 초마다 관리 서버에 상태를 보고한다 (heartbeat).
- 관리 서버가 지정한 모델이 이 PC 에 없거나 버전이 다르면 내려받아 설치하고, API 서버에 교체를 지시한다.

실행: start_agent.bat
환경 변수:
  MANAGER_URL  관리 서버 주소 (예: http://192.168.0.10:9000)  [필수]
  AGENT_ID     이 PC 의 이름 (기본: 컴퓨터 이름)
  LOCAL_API    이 PC 의 API 서버 주소 (기본: http://127.0.0.1:8000)
  POLL_SEC     보고 주기 초 (기본 5)
"""
import os
import socket
import sys
import time

import httpx

import deploy

MANAGER_URL = os.environ.get("MANAGER_URL", "").rstrip("/")
AGENT_ID = os.environ.get("AGENT_ID") or socket.gethostname()
LOCAL_API = os.environ.get("LOCAL_API", "http://127.0.0.1:8000").rstrip("/")
POLL_SEC = float(os.environ.get("POLL_SEC", "5"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def local_status(c: httpx.Client) -> dict:
    try:
        return c.get(f"{LOCAL_API}/status", timeout=5).json() | {"api_ok": True}
    except Exception as e:  # noqa: BLE001
        return {"api_ok": False, "api_error": str(e)[:100], "loaded": {}, "available": {}}


def apply_assignment(c: httpx.Client, kind: str, want: dict, status: dict, last_error: dict) -> None:
    name, version = want["name"], want["version"]
    if deploy.local_version(kind, name) != version:
        log(f"[{kind}] 새 모델 내려받는 중: {name} v{version}")
        data = deploy.download_model(MANAGER_URL, kind, name)
        deploy.install_model(kind, name, data, version)
        log(f"[{kind}] 설치 완료 ({len(data) / 2**20:.1f} MB)")
    if status.get("api_ok") and status["loaded"].get(kind) != name:
        r = c.post(f"{LOCAL_API}/load_model", params={"kind": kind, "model": name}, timeout=300)
        r.raise_for_status()
        log(f"[{kind}] API 서버에 모델 교체 완료: {name} ({r.json().get('load_ms')} ms)")
    last_error.pop(kind, None)


def main() -> None:
    if not MANAGER_URL:
        print("MANAGER_URL 환경 변수가 필요합니다. 예: set MANAGER_URL=http://192.168.0.10:9000")
        sys.exit(1)
    log(f"에이전트 시작: id={AGENT_ID}, manager={MANAGER_URL}, local_api={LOCAL_API}")
    last_error: dict[str, str] = {}
    with httpx.Client() as c:
        while True:
            try:
                status = local_status(c)
                body = {
                    "agent_id": AGENT_ID, "host": socket.gethostname(),
                    "device": status.get("device"), "api_ok": status["api_ok"],
                    "loaded": status.get("loaded", {}), "available": status.get("available", {}),
                    "errors": last_error,
                }
                r = c.post(f"{MANAGER_URL}/api/agents/heartbeat", json=body, timeout=10)
                r.raise_for_status()
                for kind, want in r.json().get("assignments", {}).items():
                    try:
                        apply_assignment(c, kind, want, status, last_error)
                    except Exception as e:  # noqa: BLE001
                        msg = str(e)[:200]
                        if last_error.get(kind) != msg:
                            log(f"[{kind}] 배포 실패: {msg}")
                        last_error[kind] = msg
            except Exception as e:  # noqa: BLE001
                log(f"관리 서버 연결 실패: {str(e)[:120]}")
            time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
