# AIST 2026 — Submission guide

> Details below are from the official call (https://aistconf.org/calls/papers/),
> fetched 2026-06-05. **Re-confirm on the site before you submit** — conference
> dates and systems can change.

## The conference

- **AIST 2026** — 13th Int. Conf. on Analysis of Images, Social Networks and
  Texts, **Astana, Kazakhstan** (Nazarbayev University), **16–18 October 2026**.
- **Proceedings:** Springer **LNCS** (companion volume in **CCIS**),
  peer-reviewed.
- Our track: **Natural language processing and applications / Computational
  linguistics**.

## ⚠️ Deadlines (AoE = "anywhere on Earth", UTC−12)

| Milestone | Date | Note |
|---|---|---|
| **Abstract** | **July 1, 2026** | **Separate, earlier deadline — register the title + abstract first.** |
| **Full paper** | **July 10, 2026** | the PDF upload |
| Notification | August 10, 2026 | |
| Camera-ready | August 18, 2026 | |
| Conference | October 16–18, 2026 | ≥1 author must attend |

> **Catch the plan missed:** there is an **abstract deadline on July 1**, nine
> days *before* the paper deadline. Most OpenReview venues require the abstract/
> title to be registered by the abstract deadline to be allowed to upload the PDF
> on the paper deadline. Plan to have the **abstract + title final by ~June 28**.

## Format

- **Springer LNCS style.** LaTeX template on Overleaf (preferred) or the Word
  template. Up to **15 pages of content *including* references** (more room than
  our 12-page assumption — use it for the error-analysis section).
- Grab the LNCS Overleaf template: search "Springer LNCS" in Overleaf → *Use as
  Template*, or download `llncs2e.zip` from Springer's author pages.

## 🔒 Double-blind — anonymize everything

Review is **double-blind with ≥3 PC members**. The PDF must be anonymous:

- Remove author names, affiliations (Devr-i Robotik / Teknopark, university),
  acknowledgments, and ORCID.
- **Do not link your real GitHub repo.** Instead cite an **anonymized** mirror:
  upload the code to <https://anonymous.4open.science/> and cite that URL. Reveal
  the real repo only in the **camera-ready**.
- Refer to your own prior work in the third person ("X et al. show…", not "we
  previously showed…").

## Submission system: OpenReview

- Group: <https://openreview.net/group?id=aistconf.org/AIST/2026/Conference>
- Steps:
  1. Create/sign in to an **OpenReview** account (profile activation can take a
     day or two if your account is new — do this **now**, not on July 9).
  2. Open the AIST 2026 venue group → **New Submission**.
  3. Enter **title, anonymous author placeholders, abstract, topics** by **July 1**.
  4. Upload the **anonymized LNCS PDF** by **July 10**.
  5. Add the anonymous code URL in the paper's reproducibility statement.

## Pre-submission checklist

- [ ] OpenReview account active; AIST 2026 venue located.
- [ ] Abstract + title submitted by **July 1**.
- [ ] LNCS PDF ≤ 15 pages incl. references.
- [ ] **Anonymized**: no names, affiliations, acknowledgments, real repo links.
- [ ] Anonymous code mirror (anonymous.4open.science) live and cited.
- [ ] All claims scoped to *slot filling / task-oriented NLU* (not "general
      Turkish tokenization") — the novelty framing from the plan (§4, §14).
- [ ] Results tables = mean ± std over 3 seeds; headline comparison has a
      significance test (`scripts/aggregate.py --compare`).
- [ ] §8 alignment validation reported as 100% (it is — see `validate_alignment.py`).
- [ ] Reproducibility statement: data (MASSIVE, CC BY 4.0), seeds, configs, code.
- [ ] arXiv preprint optional and **allowed** — but post it *anonymously-safe*
      (i.e., only after you're comfortable; preprints are not required).

## After acceptance (camera-ready, by Aug 18)

- De-anonymize: real authors, affiliations, acknowledgments, **real GitHub repo**.
- Apply reviewer feedback; sign the Springer copyright form.
- Register at least one author for the conference.

## If you only do three things this week

1. **Create the OpenReview account today** and find the AIST 2026 group.
2. Lock the **title + abstract** (you have the contribution statement in the plan
   §5) so July 1 is trivial.
3. Get the **anonymized repo** workflow ready (anonymous.4open.science).
