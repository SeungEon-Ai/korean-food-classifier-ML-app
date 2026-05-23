"""
config.yaml 파일을 로드해서 dict로 반환.
다른 모듈에서 from src.config import load_config 로 사용.
"""
from pathlib import Path
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """설정 파일을 로드해서 dict로 반환.

    Args:
        config_path: config.yaml 경로

    Returns:
        설정 dict
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없음: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


if __name__ == "__main__":
    # 단독 실행시 설정 확인용
    config = load_config()
    import json
    print(json.dumps(config, indent=2, ensure_ascii=False))
