"""Shared print-first style for the paper; does not transform result data."""
from pathlib import Path
import os
import shutil
import subprocess

from matplotlib import font_manager
import matplotlib.pyplot as plt

WIDTH = 6.5  # letter paper minus the template's two one-inch margins
INK = '#252525'
M0 = '#555555'
M4 = '#405F73'
GRAY = '#777777'
LIGHT_GRAY = '#D8D8D8'
FILL = '#C8D1D7'


def set_style():
    """Use the same installed TeX font as the manuscript, with no silent fallback."""
    # The PGF PDF backend embeds the same OpenType/CFF fonts correctly through
    # XeLaTeX, avoiding Matplotlib's Type-42/CFF subtype mismatch.
    texbin = Path(os.environ.get('PAPER_TEX_BIN', ''))
    if os.environ.get('PAPER_TEX_BIN') and texbin.is_dir():
        os.environ['PATH'] = str(texbin) + os.pathsep + os.environ.get('PATH', '')
    if not shutil.which('xelatex'):
        raise RuntimeError('XeLaTeX is required for correctly embedded publication PDF fonts.')
    candidates = [Path(os.environ['PAPER_FONT_DIR'])] if os.environ.get('PAPER_FONT_DIR') else []
    kpsewhich = shutil.which('kpsewhich')
    if kpsewhich:
        found = subprocess.run([kpsewhich, 'texgyretermes-regular.otf'],
                               capture_output=True, text=True, check=True).stdout.strip()
        if found:
            candidates.insert(0, Path(found).parent)
    for directory in candidates:
        if (directory / 'texgyretermes-regular.otf').is_file():
            for style in ('regular', 'italic', 'bold', 'bolditalic'):
                font_manager.fontManager.addfont(str(directory / f'texgyretermes-{style}.otf'))
            break
    font_manager.findfont('TeX Gyre Termes', fallback_to_default=False)
    plt.rcParams.update({
        'font.family': 'TeX Gyre Termes', 'font.size': 10,
        'axes.labelsize': 10, 'axes.titlesize': 10,
        'axes.titleweight': 'normal', 'axes.linewidth': .6,
        'axes.edgecolor': INK, 'axes.labelcolor': INK,
        'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'legend.frameon': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'lines.linewidth': 1.0, 'lines.markersize': 4.2,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'axes.grid': False,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'TeX Gyre Termes',
        'mathtext.it': 'TeX Gyre Termes:italic',
        'mathtext.bf': 'TeX Gyre Termes:bold',
        'mathtext.cal': 'STIXGeneral:italic',
        'savefig.dpi': 400,
        'pgf.texsystem': 'xelatex', 'pgf.rcfonts': False,
        'pgf.preamble': (r'\usepackage{fontspec}\setmainfont{TeX Gyre Termes}'
                         r'\usepackage{unicode-math}\setmathfont{TeX Gyre Termes Math}'),
    })
