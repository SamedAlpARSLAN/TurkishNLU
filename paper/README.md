# Paper (AIST 2026, Springer LNCS — double-blind)

`main.tex` + `references.bib` are a ready-to-fill LNCS skeleton matching the
plan's section structure. The methodology already contains real numbers (the
alignment round-trip table); `% TODO` markers show where to paste results from
the experiment matrix.

## Compile (Overleaf, recommended)

1. New Project → **Upload Project**, or start from the Overleaf **Springer LNCS**
   template and replace `main.tex` / `references.bib`.
2. Overleaf already ships `llncs.cls` and `splncs04.bst`. If compiling locally,
   download them from Springer's author pages.
3. Set the compiler to **pdfLaTeX**.

## Fill-in checklist

- [ ] Abstract: add the one-sentence headline result.
- [ ] Table 2 (`tab:main`): paste mean±std from `outputs/results_table.md`
      (produced by `scripts/aggregate.py`).
- [ ] §7 Error Analysis: paste the three bucket tables from
      `scripts/error_report.py` (affix count, surface rarity, segmentation shift).
- [ ] §8 Conclusion: instantiate the narrative tied to H1–H3.
- [ ] Reproducibility: replace `<ANONYMIZED-REPO-URL>` with an
      `anonymous.4open.science` mirror.
- [ ] Verify every `references.bib` entry marked **VERIFY**.

## Double-blind reminders

- No author names, affiliations, ORCID, acknowledgments.
- Cite the **anonymous** repo, not your real GitHub.
- Refer to your own prior work in the third person.
- ≤ 15 pages of content **including** references.

De-anonymise only in the **camera-ready** (after acceptance).
