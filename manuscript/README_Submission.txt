================================================================================
 SUBMISSION PACKAGE — CONTENTS AND FILE GUIDE
================================================================================
 Manuscript : "Content-Conditioned Invariant Feature-Statistic Augmentation:
               Unifying Robustness and Weight Averaging for Domain Generalization"
 Short title: CIFA
 Journal    : Electronics (MDPI)
 Manuscript ID: electronics-4654208
 Stage      : Minor revision (Round 2)
 Date       : 3 November 2026
================================================================================


1. DIRECTORY TREE
--------------------------------------------------------------------------------
CIFA_Submission_Minor_20261103/
|
|-- CIFA_Electronics_clean.tex        Clean manuscript, LaTeX source
|-- CIFA_Electronics_clean.pdf        Clean manuscript, compiled PDF (39 pages)
|-- CIFA_Electronics_tracked.tex      Tracked-changes manuscript, LaTeX source
|-- CIFA_Electronics_tracked.pdf      Tracked-changes manuscript, compiled PDF (41 pages)
|
|-- bbm.sty                            LaTeX support package (blackboard-bold symbols)
|-- mathcomp.sty                      LaTeX support package (extra math symbols)
|-- verse.sty                         LaTeX support package (required by mdpi.cls)
|
|-- Definitions/                      MDPI class, BST styles and logo files
|
|-- Letters/
|     |-- Response_to_Reviewers.pdf   Full point-by-point response (all reviewers)
|     |-- Response_to_Reviewers.tex   LaTeX source
|     |-- Cover_Letter_Reviewer1.pdf  Single-reviewer cover letter (1 page)
|     |-- Cover_Letter_Reviewer1.tex
|     |-- Cover_Letter_Reviewer2.pdf  Single-reviewer cover letter (3 pages)
|     |-- Cover_Letter_Reviewer2.tex
|     |-- Cover_Letter_Reviewer3.pdf  Single-reviewer cover letter (2 pages)
|     `-- Cover_Letter_Reviewer3.tex
|
|-- fig_decomp.pdf                    Figure 1
|-- fig_coverage.pdf                  Figure 2
|-- fig_pairwise_delta.pdf            Figure 3
|-- fig_sensitivity.pdf               Figure 4
|-- fig_separability.pdf              Figure 5
|-- fig_gatepenalty_delta.pdf         Figure 6
`-- fig_ablation_delta.pdf            Figure 7


2. MANUSCRIPT FILES
--------------------------------------------------------------------------------
* CIFA_Electronics_clean.tex / .pdf
    The clean version with all Round 2 revisions incorporated and no visible
    revision marks (LaTeX source and compiled PDF). The PDF is 39 pages.

* CIFA_Electronics_tracked.tex / .pdf
    The same content with every Round 2 revision marked, for reviewer
    inspection. Additions are shown in BLUE and deletions in RED with a
    strikethrough. Continuous line numbers are printed in the right margin;
    the line numbers cited in the response letters all refer to THIS file.
    The PDF is 41 pages.

The two .tex files differ only in the revision-mark macros; the scientific
content is identical. The clean PDF is the unmarked rendering of the tracked PDF.


3. RESPONSE AND COVER LETTERS (Letters/)
--------------------------------------------------------------------------------
* Response_to_Reviewers.pdf
    The full point-by-point reply, organised by reviewer:
    Reviewer 1 (1 point), Reviewer 2 (6 points), Reviewer 3 (3 points).
    Each response opens with a blue location tag giving the exact page and
    line range in the tracked manuscript.

* Cover_Letter_Reviewer1.pdf / Reviewer2.pdf / Reviewer3.pdf
    Three separate single-reviewer letters, each addressed to one reviewer
    and containing only that reviewer's points, with the same tracked-PDF
    location tags. LaTeX sources are included alongside the PDFs.


4. FIGURES (vector PDF, embedded in the manuscript)
--------------------------------------------------------------------------------
* Figure 1  fig_decomp.pdf
    Feature-moment decomposition m = b + d on PACS (ResNet-50, layer 1).
* Figure 2  fig_coverage.pdf
    Coverage-gap terms in Theorem 1 (synthetic illustration in moment space).
* Figure 3  fig_pairwise_delta.pdf
    Per-domain paired accuracy difference CIFA minus SWAD (ten seeds).
* Figure 4  fig_sensitivity.pdf
    Hyperparameter sensitivity on PACS.
* Figure 5  fig_separability.pdf
    Style-deviation separability, PACS versus OfficeHome.
* Figure 6  fig_gatepenalty_delta.pdf
    Gate x penalty 2x2 factorial on PACS, paired difference vs full CIFA.
* Figure 7  fig_ablation_delta.pdf
    Component ablation on PACS, paired difference vs full CIFA.

All seven figures were redrawn as vector files for print legibility (legends
moved outside the axes, larger fonts/markers/line widths); the underlying data
and every reported number are unchanged.


5. SUPPORT FILES
--------------------------------------------------------------------------------
* bbm.sty, mathcomp.sty, verse.sty
    Standard LaTeX packages required by the manuscript and by mdpi.cls.
* Definitions/
    MDPI document class (mdpi.cls), bibliography styles (.bst/.sty) and logo
    files. This folder must stay in place relative to the .tex files.


6. HOW TO COMPILE
--------------------------------------------------------------------------------
The reference list is embedded in the manuscript (thebibliography environment),
so no BibTeX run is needed. From the folder that contains the .tex files:

    pdflatex CIFA_Electronics_clean.tex
    pdflatex CIFA_Electronics_clean.tex        (second pass resolves references)

and likewise for CIFA_Electronics_tracked.tex. Both files compile without
errors or overfull boxes. The letters in Letters/ are standard article-class
documents and also compile with two pdflatex passes.


7. QUICK READING GUIDE
--------------------------------------------------------------------------------
* Editors / production : CIFA_Electronics_clean.pdf
* Reviewers            : CIFA_Electronics_tracked.pdf (line numbers in margins)
* Full response        : Letters/Response_to_Reviewers.pdf
* Per-reviewer letters : Letters/Cover_Letter_Reviewer{1,2,3}.pdf
* Source code and data : https://github.com/qsongcai/CIFA
================================================================================
