"""
Documentation utilities for EdgeVoiceAI framework.
Allows mounting framework documentation directly into FastAPI applications
or launching local documentation servers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


def get_site_directory() -> Path:
    """
    Locates the pre-built MkDocs static site directory.
    Checks the local project root, parent directories, and package distribution.
    """
    # 1. Check relative to this module in development repository
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "site"
    if candidate.is_dir() and (candidate / "index.html").is_file():
        return candidate

    # 2. Check bundled package directory if installed via wheel
    pkg_site = Path(__file__).resolve().parent / "site"
    if pkg_site.is_dir() and (pkg_site / "index.html").is_file():
        return pkg_site

    # 3. Fallback to candidate even if not yet built
    return candidate


def mount_docs(
    app: Any,
    path: str = "/docs",
    name: str = "edgevoiceai_docs",
    docs_dir: Optional[str | Path] = None,
) -> bool:
    """
    Mounts the EdgeVoiceAI framework documentation directly into a FastAPI/Starlette app.

    Usage:
        from fastapi import FastAPI
        from edgevoiceai.docs import mount_docs

        app = FastAPI(docs_url="/api/docs")  # Move Swagger to /api/docs
        mount_docs(app, path="/docs")

    Args:
        app: The FastAPI or Starlette application instance.
        path: URL path where the documentation will be mounted (default: "/docs").
        name: Internal mount name.
        docs_dir: Optional custom path to the built documentation directory.

    Returns:
        bool: True if documentation was mounted successfully, False otherwise.
    """
    try:
        from starlette.staticfiles import StaticFiles
    except ImportError:
        print("⚠️ [EdgeVoiceAI Docs] starlette/fastapi is required to mount documentation.")
        return False

    site_path = Path(docs_dir) if docs_dir else get_site_directory()

    if not site_path.is_dir() or not (site_path / "index.html").is_file():
        # Try auto-building if mkdocs is installed
        try:
            import subprocess
            repo_root = Path(__file__).resolve().parent.parent
            if (repo_root / "mkdocs.yml").is_file():
                print("📦 [EdgeVoiceAI Docs] Building framework documentation...")
                subprocess.run(["mkdocs", "build"], cwd=str(repo_root), check=True, capture_output=True)
        except Exception:
            pass

    if site_path.is_dir() and (site_path / "index.html").is_file():
        app.mount(path, StaticFiles(directory=str(site_path), html=True), name=name)
        print(f"📚 [EdgeVoiceAI Docs] Framework documentation mounted at '{path}'")
        return True
    else:
        print(f"⚠️ [EdgeVoiceAI Docs] Documentation site directory not found at '{site_path}'. Run 'mkdocs build' first.")
        return False


def serve_docs(port: int = 8000, host: str = "127.0.0.1") -> None:
    """
    Launches a local live-reloading or static documentation server.
    """
    import subprocess
    repo_root = Path(__file__).resolve().parent.parent
    if (repo_root / "mkdocs.yml").is_file():
        cmd = ["mkdocs", "serve", "-a", f"{host}:{port}"]
        subprocess.run(cmd, cwd=str(repo_root))
    else:
        site_path = get_site_directory()
        if site_path.is_dir():
            import http.server
            import socketserver
            os.chdir(str(site_path))
            handler = http.server.SimpleHTTPRequestHandler
            with socketserver.TCPServer((host, port), handler) as httpd:
                print(f"📚 Serving EdgeVoiceAI docs at http://{host}:{port}/")
                httpd.serve_forever()


if __name__ == "__main__":
    serve_docs()

