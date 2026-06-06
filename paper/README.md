# Paper (AIST 2026, Springer LNCS — double-blind)

`main.tex` is a **full prose draft** (Abstract, Introduction, Related Work, Data,
Method, Setup, Results framing, Error Analysis, Conclusion) in LNCS format. The
Method already contains real numbers (the tokenizer-alignment and BIO round-trip
tables); only the main results table (`tab:main`), the error-analysis tables, and
one abstract sentence remain — all marked with `% TODO` and auto-fillable by
`scripts/make_tables.py` once the matrix has run.

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
- [ ] §7 Error Analysis: paste the bucket tables from `scripts/error_report.py`
      (7 axes), and **upload `results/figures/affix_error.pdf`** next to `main.tex`
      on Overleaf (referenced by `\includegraphics`, Fig.~\ref{fig:affix}).
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
