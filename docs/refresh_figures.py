"""Embed fresh PDF renders in the standalone paper storyboard."""
import base64
import html
from pathlib import Path
import re
import struct
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
STORYBOARD = HERE / "E-I Paper Storyboard.html"


def render(match):
    tag = match.group()
    pdf = HERE / html.unescape(re.search(r'data-pdf="([^"]+)"', tag)[1])
    with tempfile.TemporaryDirectory() as directory:
        prefix = Path(directory) / "preview"
        subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "155", str(pdf), str(prefix)],
                       check=True)
        png = prefix.with_suffix(".png").read_bytes()
    width, height = struct.unpack(">II", png[16:24])
    source = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    tag = re.sub(r' (?:src|width|height)="[^"]*"', "", tag)
    return tag[:-1] + f' width="{width}" height="{height}" src="{source}">'


if __name__ == "__main__":
    page = STORYBOARD.read_text()
    updated, count = re.subn(r'<img\b[^>]*\bdata-pdf="[^"]+"[^>]*>', render, page)
    if count == 0:
        raise ValueError("No storyboard images reference a figure PDF.")
    STORYBOARD.write_text(updated)
    print(f"Embedded {count} paper figures in {STORYBOARD.name}")
