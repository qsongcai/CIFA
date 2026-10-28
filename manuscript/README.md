# Manuscript — Electronics major revision (electronics-4654208)

This folder contains the complete first-round major-revision submission.

| File | Description |
|---|---|
| `CIFA_Electronics_clean.tex` / `.pdf` | Clean revised manuscript (all changes accepted). |
| `CIFA_Electronics_tracked.tex` / `.pdf` | Tracked-changes manuscript; additions in blue, deletions in red strikethrough. Line numbers cited in the response letter refer to this PDF. |
| `Response_Letter.docx` | Point-by-point response to all three reviewers and the editorial comment. |
| `fig_*.pdf` | The seven figure files included by the manuscripts. |
| `Definitions/` | Official MDPI class, bibliography styles, and logo files. |
| `bbm.sty`, `mathcomp.sty`, `verse.sty` | Small support packages required by the MDPI class; included so the project compiles on a minimal TeX install. |

## Compile

```bash
pdflatex CIFA_Electronics_clean.tex
pdflatex CIFA_Electronics_clean.tex   # second pass for cross-references
```

The clean and tracked manuscripts compile with no undefined references and no
overfull boxes in the clean version.
