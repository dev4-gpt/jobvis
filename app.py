"""Hugging Face Spaces root entrypoint for Job Scout Gradio UI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src/ to python path so job_scout package is discoverable
ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from job_scout.app import CSS, THEME, build_app  # noqa: E402

# Build the Gradio Blocks application
demo = build_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, theme=THEME, css=CSS)  # noqa: S104
