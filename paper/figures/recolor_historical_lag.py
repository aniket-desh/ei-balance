"""Apply the paper palette to the archived lag PDF and remove the three repeated slice titles.

The archived heatmaps and colorbar use indexed images. Only their RGB lookup
tables change. The pixel indices, line paths, contour levels and bands come
directly from the archived figure.
"""

import hashlib
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import matplotlib as mpl
from matplotlib.colors import to_rgb
import numpy as np
from pypdf import PdfWriter
from pypdf.generic import ByteStringObject

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'experiments' / 'figures'))
from paper_style import JOINT, LAG_CMAP, MUTED, SLATE


def palette_translation():
    levels = np.arange(256)
    old = mpl.colormaps['RdBu_r'](levels)[:, :3] * 255
    new = np.rint(LAG_CMAP(levels)[:, :3] * 255).astype(np.uint8)
    translation = {}
    # Matplotlib truncates colors in the nine data images and rounds in its
    # rasterized colorbar. Both encodings map to the identical new RGB value.
    for encoded in (np.floor(old), np.rint(old)):
        for level, rgb in enumerate(encoded.astype(np.uint8)):
            slot = rgb.tobytes()
            if slot in translation and translation[slot] != new[level].tobytes():
                raise ValueError('The archived colormap has ambiguous indices.')
            translation[slot] = new[level].tobytes()
    return translation


def recolor_commands(data):
    # The source hash below fixes these paint commands. Keep the original
    # paths and typesetting byte for byte, including the sample spread bands.
    for old, color in [('0 0.4470588235 0.6980392157', MUTED),  # N = 1000 dashed
                       ('0.8352941176 0.368627451 0', SLATE),   # N = 2000 squares
                       ('0.3', SLATE), ('0', JOINT)]:
        operands = rb'\s+'.join(re.escape(part.encode()) for part in old.split())
        operator = rb'(rg|RG)' if ' ' in old else rb'(g|G)'
        pattern = rb'(?<!\S)' + operands + rb'\s+' + operator + rb'(?=\s)'
        rgb = ' '.join(f'{v:.10f}' for v in to_rgb(color)).encode()
        data, count = re.subn(pattern, lambda m: rgb + (b' RG' if m[1].isupper() else b' rg'), data)
        if not count:
            raise ValueError(f'Missing archived paint command for {old}')
    for letter in 'def':
        old = b'(' + f'{letter}   Equal-lag slice'.encode('utf-16-be') + b')'
        new = b'(' + letter.encode('utf-16-be') + b')'
        if data.count(old) != 1:
            raise ValueError(f'Missing archived title for panel {letter}')
        data = data.replace(old, new)
    return data


def main():
    source = HERE / 'historical_lag_validation.pdf'
    provenance = json.loads((HERE / 'historical_lag_provenance.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != provenance['sha256']:
        raise ValueError('The historical PDF differs from its provenance record.')
    writer = PdfWriter(source, incremental=True)
    page = writer.pages[0]
    translation = palette_translation()
    for ref in page['/Resources']['/XObject'].values():
        image = ref.get_object()
        if image['/Subtype'] != '/Image':
            continue
        space = image['/ColorSpace']
        if list(space[:2]) != ['/Indexed', '/DeviceRGB']:
            raise ValueError('Expected an indexed RGB image.')
        colors = space[3].original_bytes
        space[3] = ByteStringObject(b''.join(translation[colors[i:i + 3]] for i in range(0, len(colors), 3)))
    contents = page['/Contents']
    contents.set_data(recolor_commands(contents.get_data()))
    target = HERE / 'historical_lag_validation_nord.pdf'
    writer.add_metadata({'/Subject': f"Nord display revision of {source.name}, SHA-256 {provenance['sha256']}"})
    with TemporaryDirectory(dir=HERE, prefix='.lag-display-') as directory:
        pending = Path(directory) / target.name
        writer.write(pending)
        pending.replace(target)
    print(target)


if __name__ == '__main__':
    main()
