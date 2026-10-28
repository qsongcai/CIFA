================================================================================
 SUBMISSION PACKAGE — CONTENTS AND FILE GUIDE
================================================================================
 Manuscript : "Content-Conditioned Invariant Feature-Statistic Augmentation:
               Unifying Robustness and Weight Averaging for Domain Generalization"
 Short title: CIFA
 Journal    : Electronics (MDPI)
 Manuscript ID: electronics-4654208
 Stage      : Major revision (Round 1)
 Date       : 28 October 2026
================================================================================


1. DIRECTORY TREE
--------------------------------------------------------------------------------
CIFA_Submission_Final/
|
|-- CIFA_Electronics_clean.tex        Clean manuscript, LaTeX source
|-- CIFA_Electronics_clean.pdf        Clean manuscript, compiled PDF (39 pages)
|-- CIFA_Electronics_tracked.tex      Tracked-changes manuscript, LaTeX source
|-- CIFA_Electronics_tracked.pdf      Tracked-changes manuscript, compiled PDF (44 pages)
|-- Response_Letter.docx              Point-by-point response to all reviewers
|
|-- bbm.sty                           LaTeX support package (blackboard-bold symbols)
|-- mathcomp.sty                      LaTeX support package (extra math symbols)
|-- verse.sty                         LaTeX support package (required by mdpi.cls)
|
|-- fig_decomp.pdf                    Figure 1
|-- fig_coverage.pdf                 Figure 2
|-- fig_pairwise_delta.pdf           Figure 3
|-- fig_sensitivity.pdf              Figure 4
|-- fig_separability.pdf             Figure 5
|-- fig_gatepenalty_delta.pdf         Figure 6
`-- fig_ablation_delta.pdf            Figure 7


2. MANUSCRIPT FILES
--------------------------------------------------------------------------------
* CIFA_Electronics_clean.tex / .pdf
    The clean version of the manuscript, with all Round 1 revisions incorporated
    and no visible revision marks (provided as both LaTeX source and compiled
    PDF). The PDF is 39 pages.

* CIFA_Electronics_tracked.tex / .pdf
    The same content with every revision marked, for reviewer inspection.
    Additions are shown in BLUE and deletions in RED with a strikethrough.
    Continuous line numbers are printed in the right margin; the line numbers
    cited in the Response Letter all refer to THIS file. The PDF is 44 pages.

* Response_Letter.docx
    The point-by-point reply. It reproduces each reviewer comment in full and
    answers it, with the corresponding line numbers in the tracked manuscript.
    Structure: Editorial Comment, Reviewer 1 (10 points), Reviewer 2 (8 points
    plus an overall assessment), Reviewer 3 (11 main points plus 4 minor items).

Note: the two .tex files differ only in the revision-mark macros; the scientific
content is identical. The clean PDF is the unmarked rendering of the tracked PDF.


3. FIGURES (vector PDF, embedded in the manuscript)
--------------------------------------------------------------------------------
* Figure 1  fig_decomp.pdf
    Feature-moment decomposition m = b + d on PACS (ResNet-50, layer 1):
    instance points, class prototypes and style-deviation arrows for selected
    channels.

* Figure 2  fig_coverage.pdf
    Coverage-gap terms in Theorem 1 (synthetic illustration in moment space):
    the target-dependent gap is unidentifiable without target access, while the
    source-side term h_s is observable and converges empirically.

* Figure 3  fig_pairwise_delta.pdf
    Per-domain paired accuracy difference CIFA minus SWAD (percentage points,
    ten seeds per benchmark). All four PACS domains stay near zero or slightly
    negative.

* Figure 4  fig_sensitivity.pdf
    Hyperparameter sensitivity on PACS: each panel changes one choice while
    holding the others at the base configuration.

* Figure 5  fig_separability.pdf
    Style-deviation separability, PACS versus OfficeHome: style-to-content
    energy ratio, domain-prediction accuracy and cross-domain class leakage.

* Figure 6  fig_gatepenalty_delta.pdf
    Gate x penalty 2x2 factorial on PACS, paired difference of each cell
    relative to full CIFA.

* Figure 7  fig_ablation_delta.pdf
    Remaining component ablation on PACS, paired difference of each variant
    relative to full CIFA (weight averaging is the dominant component).


4. SUPPORT FILES
--------------------------------------------------------------------------------
* bbm.sty, mathcomp.sty, verse.sty
    Standard LaTeX packages required by the manuscript and by mdpi.cls.
    They must stay in the same folder as the .tex files when compiling.


5. HOW TO COMPILE
--------------------------------------------------------------------------------
The reference list is embedded in the manuscript (thebibliography environment),
so no BibTeX run is needed. From the folder that contains the .tex files:

    pdflatex CIFA_Electronics_clean.tex
    pdflatex CIFA_Electronics_clean.tex        (second pass resolves references)

and likewise for CIFA_Electronics_tracked.tex. Both files compile without errors.


6. QUICK READING GUIDE
--------------------------------------------------------------------------------
* Editors / production : CIFA_Electronics_clean.pdf
* Reviewers            : CIFA_Electronics_tracked.pdf (line numbers in margins)
* Revision summary     : Response_Letter.docx
* Source code and data : https://github.com/qsongcai/CIFA
================================================================================
