# Manuscript — Electronics minor revision (electronics-4654208)

This folder contains the complete third-round (Round 3) revision submission.

| File | Description |
|---|---|
| `CIFA_Electronics_clean.tex` / `.pdf` | Clean revised manuscript, 39 pages (all changes accepted). |
| `CIFA_Electronics_tracked.tex` / `.pdf` | Tracked-changes manuscript, 41 pages; additions in blue, deletions in red strikethrough. Line numbers cited in the response letters refer to this PDF. |
| `Letters/` | Round 3 documents `Response_to_Editors_and_Reviewers_R3` and `Cover_Letter_Reviewer2_R3`; the Round 2 point-by-point `Response_to_Reviewers` and three `Cover_Letter_Reviewer{1,2,3}` documents are retained. |
| `fig_*.pdf` | The seven figure files included by the manuscripts; redrawn as vector files for print legibility with the underlying data unchanged. |
| `Definitions/` | Official MDPI class, bibliography styles, and logo files. |
| `bbm.sty`, `mathcomp.sty`, `verse.sty` | Small support packages required by the MDPI class; included so the project compiles on a minimal TeX install. |

## Compile

```bash
pdflatex CIFA_Electronics_clean.tex
pdflatex CIFA_Electronics_clean.tex   # second pass for cross-references
```

The clean and tracked manuscripts compile with no undefined references and no
overfull boxes.
