from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


APP_ROOT = Path(__file__).resolve().parent


def read_text(relative_path: str) -> str:
    return (APP_ROOT / relative_path).read_text(encoding="utf-8")


def build_embedded_html() -> str:
    html_template = read_text("index.html")
    css = read_text("style.css")
    js = read_text("script.js")
    model_payload = json.loads(read_text("model/sentinel_model.json"))

    html_template = html_template.replace(
        '<link rel="stylesheet" href="style.css" />',
        f"<style>\n{css}\n</style>",
    )

    html_template = html_template.replace(
        '<script src="script.js"></script>',
        "<script>\n"
        f"window.__SENTINEL_MODEL__ = {json.dumps(model_payload)};\n"
        "</script>\n"
        f"<script>\n{js}\n</script>",
    )

    return html_template


def main() -> None:
    st.set_page_config(page_title="Project Sentinel", page_icon=":microscope:", layout="wide")
    components.html(build_embedded_html(), height=200, scrolling=True)

    


if __name__ == "__main__":
    main()

