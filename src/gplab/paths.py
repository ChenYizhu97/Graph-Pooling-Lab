from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def project_root() -> Path:
    if PACKAGE_ROOT.parent.name == "src":
        return PACKAGE_ROOT.parents[1]
    return Path.cwd()


def default_config_path(name: str) -> str:
    return str(project_root() / "config" / name)

