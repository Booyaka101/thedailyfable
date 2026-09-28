"""Render a written page to a 600 dpi PNG and a PDF.

    python page_export.py ../out/letter/trace.npz ../out/letter/letter
"""
import sys
import numpy as np
from PIL import Image, JpegImagePlugin  # noqa: F401  PDF export encodes as JPEG
import ink


def main(trace, stem):
    d = np.load(trace)
    page = ink.Page()
    page.add(d["tip"] - d["offset"], d["p"], 1.0 / float(d["rate"]))
    im = Image.fromarray(page.render_u8())
    im.save(stem + ".png", dpi=(ink.DPI, ink.DPI))
    im.save(stem + ".pdf", resolution=ink.DPI)
    print(f"wrote {stem}.png {im.size} and {stem}.pdf")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
