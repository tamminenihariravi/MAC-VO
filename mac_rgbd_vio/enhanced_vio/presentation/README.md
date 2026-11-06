# Enhanced MAC RGB-D VIO presentation

Twenty 16:9 slides in LaTeX Beamer, using simple English, mathematical symbol definitions and physical explanations. The deck covers the implemented method and the walking capture recorded on 29 October 2026 at 20:11.

## Open and present

Open `mac_rgbd_vio_20_slides.pdf` in a PDF viewer and use full-screen presentation mode.

## Edit and compile

The main document is `main.tex`. All figures and chart data use relative paths.

- **Overleaf:** upload the source ZIP as a new project and set `main.tex` as the main document. Compile with pdfLaTeX.
- **Local TeX Live / MiKTeX:** run `pdflatex -interaction=nonstopmode -halt-on-error main.tex` twice from this folder.
- **Tectonic:** run `tectonic main.tex`. The bundled private build tool/cache used during creation is not included in the source ZIP.

The PDF contains exactly 20 slides, including the cover. Equations and data plots are native LaTeX. PGFPlots reads the measured values from `data/*.csv`, so graph labels, colors and axes remain editable. Concept diagrams are PNGs, and their full generation prompts are in `figures/illustration_prompts.md`.

The `\src{...}` commands hold source references and explanatory speaker notes for each slide. They are hidden from the normal slide PDF. `speaker_notes.md` provides a readable copy. To produce Beamer note pages separately, add `\setbeameroption{show notes}` before `\begin{document}` and compile a separate output.

## Meaning of the results

The recorded and CPU-revised replay scores use the **same 902 valid matched poses** and separate rigid SE(3) alignments with scale fixed at one. Ground-truth gaps limit coverage to 63.7% of recorded poses. The revised replay has lower ATE and estimator time, but worse path inflation and more tracking failures. Replay speed does not prove live walking performance. The historical 135-second take is a different experiment and is not used as the comparison baseline here.

No new GT-assisted trajectory correction, fixed-height projection or endpoint closure is included. The figures are drawn from existing measured outputs, not invented data. The two concept illustrations are generated explanatory graphics and are labeled as such.

## Contents

- `main.tex`: editable slide source and per-slide notes.
- `figures/hardware.jpg`: existing project hardware photo.
- `figures/pipeline.png`, `figures/keyframes.png`: generated concept illustrations.
- `data/*.csv`: measured data for native plots.
- `data/source_metrics.json`: exact comparison metrics and provenance paths.
- `export_data.py`: exporter used with the original local recording and analysis outputs.
- `speaker_notes.md`: explanations and sources by slide.
- `SOURCES.md`: data and reference summary.

The source ZIP excludes private compiler binaries, package caches, temporary logs and preview renders.
