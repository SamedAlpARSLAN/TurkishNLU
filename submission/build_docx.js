const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
} = require("docx");

// ---- helpers ----------------------------------------------------------------
const FONT = "Times New Roman";
function p(text, opts = {}) {
  const runs = Array.isArray(text) ? text : [text];
  return new Paragraph({
    alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { after: opts.after == null ? 140 : opts.after, line: 276 },
    children: runs.map((r) =>
      typeof r === "string"
        ? new TextRun({ text: r, font: FONT, size: 22 })
        : new TextRun({ text: r.t, bold: !!r.b, italics: !!r.i, font: FONT, size: r.size || 22 })
    ),
  });
}
function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true, font: FONT, size: 26 })] });
}
function h2(text) {
  return new Paragraph({ spacing: { before: 160, after: 80 },
    children: [new TextRun({ text, bold: true, italics: true, font: FONT, size: 23 })] });
}
function caption(text) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 160 },
    children: [new TextRun({ text, font: FONT, size: 20, italics: true })] });
}

const B = { style: BorderStyle.SINGLE, size: 2, color: "000000" };
const cellBorders = { top: B, bottom: B, left: B, right: B };
function cell(text, w, opts = {}) {
  return new TableCell({
    borders: cellBorders, width: { size: w, type: WidthType.DXA },
    margins: { top: 40, bottom: 40, left: 100, right: 100 },
    shading: opts.head ? { fill: "E6E6E6", type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({ alignment: opts.align || AlignmentType.CENTER, spacing: { after: 0 },
      children: [new TextRun({ text, bold: !!opts.head || !!opts.b, font: FONT, size: 20 })] })],
  });
}
function table(widths, rows) {
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, ri) =>
      new TableRow({ children: r.map((c, ci) =>
        cell(c, widths[ci], { head: ri === 0, align: ci === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })) })),
  });
}

// ---- content ----------------------------------------------------------------
const children = [];

// Title
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
  children: [new TextRun({ text:
    "Does Morphology Matter for Turkish Slot Filling? A Systematic Study of Segmentation Strategies for Joint Intent Detection and Slot Filling",
    bold: true, font: FONT, size: 30 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "Anonymous Author(s)", font: FONT, size: 22 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({ text: "Anonymous Institution  (double-blind submission)", font: FONT, size: 20, italics: true })] }));

// Abstract
children.push(new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: "Abstract.", bold: true, font: FONT, size: 22 })] }));
children.push(p(
  "Turkish is agglutinative: a single surface word may carry a root and several inflectional suffixes, and slot values are frequently embedded inside this morphological structure. Because slot filling is a token-level sequence-labelling task whose labels are tied to token boundaries, the choice of segmentation should matter most precisely for slot filling — yet this intersection has not been studied systematically for task-oriented natural language understanding. We present a controlled comparison of segmentation strategies (model-native subword, whitespace, unsupervised morphological, and character) as an independent variable for joint intent detection and slot filling in Turkish, on the open MASSIVE-tr benchmark, across a monolingual encoder (BERTurk) and two multilingual encoders (mBERT and XLM-R). To make the comparison fair, we introduce a cross-scheme BIO label re-alignment procedure with a mandatory round-trip validation that recovers gold word labels exactly for every encoder and scheme, and we complement the experiments with a training-free analysis of how each tokenizer aligns with Turkish morphology and a seven-axis, morpheme-level slot error analysis. Our central finding is a careful negative one: native subword tokenization matches or outperforms morphological pre-segmentation for every encoder (BERTurk slot F1 0.78 over three seeds), and unsupervised pre-segmentation hurts the weakest multilingual model the most. Yet slot errors rise monotonically with affix count, showing that morphology is where slot filling struggles even though boundary-aligned pre-segmentation is not the remedy — pointing instead at the representations a pretrained model learns for sub-word fragments."
));
children.push(p([{ b: true, t: "Keywords: " }, "Turkish NLP, slot filling, intent detection, subword tokenization, morphological segmentation, task-oriented dialogue, agglutinative languages."], { after: 200 }));

// 1 Introduction
children.push(h1("1  Introduction"));
children.push(p("Task-oriented dialogue understanding couples two tasks. Intent detection is sentence-level classification — mapping an utterance to a user goal such as alarm_set or weather_query. Slot filling is token-level sequence labelling: each token is tagged in the BIO scheme to mark the spans that carry slot values (a date, a city, a person). The two are usually learned jointly, since the intent constrains which slots are plausible and vice versa (Chen et al., 2019). Critically, a slot label is attached to a token boundary: which spans exist, and where they begin and end, is determined by how the input is tokenized."));
children.push(p("This makes slot filling acutely sensitive to segmentation in an agglutinative language. In Turkish the orthographic word boundary and the meaning-unit boundary do not coincide: a single surface word packs a root and a sequence of suffixes, and the slot value is frequently embedded inside that structure. The word faturalarımdan (“from my bills”) decomposes as fatura + lar + ım + dan (bill + plural + first-person-singular-possessive + ablative); the lexical content a slot tagger must recover sits in the root and the early morphemes, wrapped in inflection. The segmentation decision therefore determines where a slot label can fall, which makes slot filling the natural locus at which to ask whether segmentation matters."));
children.push(p("Standard transformer tokenizers (WordPiece, BPE, SentencePiece) are frequency-driven and learned without morphological supervision; they routinely place subword boundaries that violate morpheme boundaries. We confirm this quantitatively (Section 4.2): on MASSIVE-tr slot words, the multilingual mBERT produces 2.67 subwords per word and respects morpheme boundaries in only 16% of fragmented words, against 2.05 and 41% for the monolingual BERTurk. The central question follows: does a morphology-aware pre-segmentation yield a measurable gain for Turkish slot filling — and for which words and which encoders — or is the pretrained model’s own subwording already sufficient?"));
children.push(p("Two literatures bear on this question but leave its intersection open. Studies of tokenization in Turkish transformers measure intrinsic or downstream effects on tasks such as language modelling, NLI, NER and parsing, but not task-oriented NLU. Conversely, Turkish intent+slot studies adopt a single, fixed tokenizer and never treat segmentation as a variable. Neither addresses the methodological crux that any such comparison must solve: BIO labels defined over surface words must be re-projected onto each scheme’s units and then folded back to a common reference unit, or the F1 scores are not comparable. We scope our claims narrowly to slot filling and task-oriented NLU; we make no general “Turkish tokenization” claim."));
children.push(p([{ b: true, t: "Contributions. " }, "(i) The first systematic comparison that treats segmentation as a controlled independent variable for Turkish joint intent+slot filling on MASSIVE-tr, across one monolingual and two multilingual encoders. (ii) A fair cross-scheme BIO label re-alignment procedure with a mandatory round-trip validation; we report that it recovers gold word labels exactly (100%) for every encoder and scheme, and we argue this guard is a reusable methodological contribution and a pitfall warning. (iii) A fine-grained slot error analysis along seven axes — affix count, surface rarity, model fragmentation, morpheme-boundary violation, root|suffix boundary, domain, and slot type. (iv) A full reproducibility package: code, segmentation scripts, the alignment validation, run configurations, and the seed list."]));

// 2 Related Work
children.push(h1("2  Related Work"));
children.push(p([{ b: true, t: "Turkish intent detection and slot filling. " }, "Joint models in the style of Joint BERT (Chen et al., 2019) have been applied to Turkish by translating and adapting the ATIS benchmark and reporting intent accuracy and slot F1 with a fixed, standard tokenizer (Büyük, 2023; IEEE, 2020). These studies establish baselines but do not vary the tokenizer; segmentation is held constant and its effect on slot filling is never isolated."]));
children.push(p([{ b: true, t: "Tokenization and segmentation for Turkish. " }, "A complementary line studies tokenization itself. Toraman et al. (2023) analyse the impact of tokenizer choice on Turkish language models; Altınok (2026) evaluates Turkish subword strategies at scale, jointly varying vocabulary and corpus size with a morphology-aware diagnostic toolkit (boundary-level F1 against gold morphemes, over-/under-segmentation indices); and hybrid morphology-aware tokenizers have been proposed for Turkish (Bayram et al., 2025). The evaluation toolkit for morphologically rich languages now includes intrinsic metrics such as fertility and boundary alignment against morpheme references (Arnett et al., 2025). This literature, however, targets language modelling and tasks such as NLI, STS, NER, POS and dependency parsing — not task-oriented NLU, and not slot filling, where the token boundary is the label boundary."]));
children.push(p([{ b: true, t: "Gap. " }, "The intersection is open: a controlled segmentation comparison for Turkish joint intent+slot filling, with a fair cross-scheme BIO alignment methodology and a morpheme-boundary-level slot error analysis. Tokenization studies do not cover slot filling; slot-filling studies do not vary segmentation. We position our contribution at this intersection and at the alignment methodology, rather than as a general claim about Turkish tokenization."]));

// 3 Data
children.push(h1("3  Data"));
children.push(p("We use MASSIVE-tr (FitzGerald et al., 2023), the Turkish split of Amazon MASSIVE (CC BY 4.0): 11,514 train / 2,033 dev / 2,974 test utterances over 18 domains, with 60 intents and 55 slot types (109 BIO labels) and official splits, ensuring comparability. Utterances average 5.5 tokens. We parse the bracketed annotations into word-level BIO tags deterministically. Of the 17,668 slot-bearing words in the training split, 18% carry at least one affix (Morfessor estimate). This figure frames our expectations: the agglutinative pressure the study targets is concentrated in a minority of slot words, so an overall F1 gap may be modest even where the effect on affixed words is large — which is precisely the question the error analysis (Section 7) is designed to answer. As an optional second domain, our code also loads MultiATIS++-tr (flight domain; Xu et al., 2020)."));

// 4 Method
children.push(h1("4  Method"));
children.push(h2("4.1  Segmentation strategies (independent variable)"));
children.push(p("We adopt a pre-segmentation paradigm: the text is first split by a strategy, and the resulting pre-tokens are then sub-tokenized by the model’s own tokenizer. We compare four strategies. Native feeds the raw string to the tokenizer (the current practice and our control), with labels aligned by character offsets. Whitespace feeds one pre-token per surface word. Morphological splits each word into morphemes with an unsupervised Morfessor model (Virpioja et al., 2013) trained only on the training-split word types (a rule-based Zeyrek backend, a port of Zemberek (Akın & Akın, 2007), is also provided). Character splits each word into characters, a lower-bound reference. We verified empirically that native and whitespace produce identical tokenization for all three encoders on MASSIVE-tr (0% divergence over 11,514 utterances), including SentencePiece XLM-R; the meaningful contrast is therefore model-default subword vs. morphological vs. character. Morfessor genuinely segments Turkish (it splits ~60% of word types); we use it as the primary backend because the rule-based Zeyrek is highly conservative on slot words (it segments only 3.7% of slot word types, many of which are proper nouns), and we report this divergence as a limitation of the unsupervised proxy."));
children.push(h2("4.2  Tokenizer morphological alignment (motivation)"));
children.push(p("Before any training we quantify how each encoder’s native tokenizer relates to Turkish morphology, using fertility (subwords per word) and the boundary precision/recall/F1 of subword boundaries against morpheme boundaries from an unsupervised reference (Morfessor), following tokenizer-evaluation practice for morphologically rich languages (Arnett et al., 2025; Toraman et al., 2023). Table 1 shows that the multilingual mBERT fragments Turkish most and aligns with morphology least, whereas the monolingual BERTurk is the most morphologically coherent — predicting, before training, that mBERT should be the model most perturbed by an imposed segmentation."));
children.push(table([2200, 1500, 1300, 1300, 1300, 1300], [
  ["Encoder", "Fertility", "Bound. P", "Bound. R", "Bound. F1", "Respect"],
  ["BERTurk", "2.05", "0.46", "0.54", "0.50", "0.41"],
  ["mBERT", "2.67", "0.31", "0.54", "0.39", "0.16"],
  ["XLM-R", "2.21", "0.45", "0.61", "0.52", "0.36"],
]));
children.push(caption("Table 1. Native-tokenizer morphological alignment on MASSIVE-tr slot words (type-level; Morfessor reference, 51.9% coverage). Lower fertility and higher boundary-F1/respect mean more morphologically coherent subwording."));
children.push(h2("4.3  Cross-scheme BIO re-alignment"));
children.push(p("The independent variable changes the token units, so word-level BIO tags must be re-projected onto each scheme’s pre-tokens. Splitting a word labelled B-x into k pieces yields B-x, I-x, ... (k−1 copies of I-x); an I-x word yields I-x...; an O word yields O.... This preserves the span semantics exactly. Each pre-token is then sub-tokenized; the first sub-token of each pre-token carries the label and the rest are masked in the loss. Crucially, evaluation always folds predictions back to the surface word, by reading the prediction at each word’s head sub-token, so that F1 is computed at a common reference unit and is comparable across schemes. Because mis-alignment silently distorts F1, we make correctness a hard precondition: projecting gold tags through a scheme and folding them back must recover the original word labels exactly. Table 2 reports this round-trip on dev for every encoder and scheme; all reach 100% with zero mismatches, and only character segmentation loses a small number of words to the length cap."));
children.push(table([2600, 3000, 1900, 1300], [
  ["Encoder", "Scheme", "Recovery", "Trunc."],
  ["BERTurk", "native / whitespace / morph.", "100.00%", "0"],
  ["BERTurk", "character", "100.00%", "114"],
  ["mBERT", "native / whitespace / morph.", "100.00%", "0"],
  ["mBERT", "character", "100.00%", "114"],
  ["XLM-R", "native / whitespace / morph.", "100.00%", "0"],
  ["XLM-R", "character", "100.00%", "154"],
]));
children.push(caption("Table 2. BIO re-alignment round-trip on MASSIVE-tr dev (2,033 utterances; 11,033 slot-bearing words). All schemes recover gold word labels exactly; only character segmentation overflows max_length=64."));
children.push(h2("4.4  Joint model and ablations"));
children.push(p("A shared encoder feeds two heads: an intent classifier off the [CLS] state and a token slot tagger off the sequence states; the loss is intent_loss + λ·slot_loss (Chen et al., 2019). Our released code additionally provides four optional components that interact with segmentation — subword pooling (first/mean/max of a fragmented word’s sub-tokens), a linear-chain CRF over the supervised head positions, a focal slot loss, and constrained (BIO-repair) decoding — which we leave to future exploration."));

// 5 Setup
children.push(h1("5  Experimental Setup"));
children.push(p("Encoders: BERTurk (dbmdz/bert-base-turkish-cased; Schweter, 2020), mBERT (bert-base-multilingual-cased; Devlin et al., 2019), and XLM-R (xlm-roberta-base; Conneau et al., 2020). The matrix crosses three encoders, the segmentation strategies, and seeds {42, 123, 2024}; whitespace is omitted from the default sweep because it is identical to native here. Optimisation uses AdamW (lr 5e-5, batch 32, 6 epochs, linear warmup 0.1, max_length 64), with model selection on dev frame accuracy. Metrics are intent accuracy, span-level slot F1 (seqeval, strict IOB2), and frame accuracy (intent and all slots correct), the strictest measure. We report mean±std over seeds and test paired differences with both a bootstrap on frame accuracy and an exact McNemar test."));

// 6 Results
children.push(h1("6  Results"));
children.push(p("Table 3 reports test performance. The picture is consistent and, for the morphology hypotheses, negative. Native subword tokenization matches or beats morphological pre-segmentation for every encoder in slot F1 (BERTurk 0.777 vs. 0.746; mBERT 0.721 vs. 0.681; XLM-R 0.749 vs. 0.720). Unsupervised morphological pre-segmentation thus does not help, and it hurts the weaker multilingual mBERT most (−0.040 slot F1) — contrary to the naive expectation that the most fragmenting tokenizer would benefit most, and consistent with mBERT’s poorer Turkish subword embeddings being further disrupted by unfamiliar morpheme fragments. H1 holds: intent accuracy moves little across segmentations (≤0.013) while slot F1 moves up to 0.040, confirming slot filling is the segmentation-sensitive task. BERTurk’s intent accuracy reaches 0.887 and its slot F1 0.777, at the level of published MASSIVE-tr systems; the native > morphological gap is about eight times the seed standard deviation. Character segmentation is a clear lower bound (BERTurk slot F1 0.226). The monolingual BERTurk leads throughout, matching its status as the most morphologically coherent tokenizer (Table 1)."));
children.push(table([1700, 2400, 1500, 1400, 1400], [
  ["Encoder", "Segmentation", "Intent Acc", "Slot F1", "Frame Acc"],
  ["BERTurk", "native", "0.887", "0.777", "0.681"],
  ["BERTurk", "morphological", "0.874", "0.746", "0.652"],
  ["BERTurk", "character", "0.720", "0.226", "0.257"],
  ["mBERT", "native", "0.852", "0.721", "0.613"],
  ["mBERT", "morphological", "0.840", "0.681", "0.576"],
  ["XLM-R", "native", "0.871", "0.749", "0.645"],
  ["XLM-R", "morphological", "0.863", "0.720", "0.623"],
]));
children.push(caption("Table 3. Test results on MASSIVE-tr. BERTurk is averaged over 3 seeds (native 0.777±0.002, morphological 0.746±0.004 slot F1) and mBERT-native over 2; remaining cells are a single seed (variance comparable, ≤0.005). Character is run for BERTurk as the lower bound."));

// 7 Error Analysis
children.push(h1("7  Error Analysis"));
children.push(p("We stratify per-word slot errors along seven axes: affix count, surface-form frequency, model fragmentation, segmentation shift (whether subword boundaries violate morpheme boundaries), root|suffix boundary respected vs. violated, domain, and slot type. Slot error rate rises monotonically with affix count for every encoder (native scheme): 0.18→0.25→0.27 (BERTurk), 0.22→0.33→0.34 (mBERT), and 0.20→0.28→0.31 (XLM-R) for 0, 1, 2 affixes (Figure 1); the weakest encoder is hit hardest. Errors rise similarly with surface rarity and concentrate where the model’s subwording violates the root|suffix boundary. That affix-heavy words are the hard cases, yet morpheme-aligned pre-segmentation does not help them, indicates the bottleneck is the representation a pretrained model has learned for fragments, not the boundary placement alone — a sharper target for future work than naive pre-segmentation."));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 },
  children: [new ImageRun({ type: "png", data: fs.readFileSync("results/figures/affix_error.png"),
    transformation: { width: 360, height: 234 },
    altText: { title: "Affix error", description: "Slot error rate vs affix count", name: "affix_error" } })] }));
children.push(caption("Figure 1. Slot-word error rate rises with morphological complexity (affix count). Error rate is on the y-axis; affix count on the x-axis (BERTurk, native)."));

// 8 Conclusion
children.push(h1("8  Conclusion and Limitations"));
children.push(p("We framed segmentation as a controlled variable for Turkish joint intent and slot filling, contributed a validated cross-scheme BIO re-alignment procedure, a training-free tokenizer-morphology analysis, and a seven-axis morphology-aware error analysis. The central empirical finding is a careful negative one: across a monolingual and two multilingual encoders, native subword tokenization matches or outperforms unsupervised morphological pre-segmentation for Turkish slot filling, which hurts the weakest multilingual model most. Yet slot errors concentrate monotonically on morphologically complex words, so morphology is where the task is hard — the remedy is better fragment representations, not boundary-aligned pre-segmentation. Slot filling, not intent detection, is the segmentation-sensitive task, and the most morphologically coherent tokenizer (BERTurk) leads. Limitations: a single primary benchmark/domain (MASSIVE-tr; MultiATIS++-tr is supported but optional); an unsupervised morphological proxy (Morfessor) which we show diverges from rule-based analysis; morphological ambiguity resolved by the most-probable analysis; and character segmentation under-trained on CPU. All code, configurations, and the seed list are released for reproducibility (anonymized repository)."));

// References
children.push(h1("References"));
const refs = [
  "Akın, A. A., & Akın, M. D. (2007). Zemberek, an open source NLP framework for Turkic languages.",
  "Altınok, D. (2026). Optimal Turkish Subword Strategies at Scale: Systematic Evaluation of Data–Vocabulary–Morphology Interplay. arXiv:2602.06942.",
  "Arnett, C., Hudspeth, M., & O’Connor, B. (2025). Evaluating Morphological Alignment of Tokenizers in 70 Languages. Tokenization Workshop at ICML. arXiv:2507.06378.",
  "Bayram, M. A., Fincan, A. A., Gümüş, A. S., Karakaş, S., Diri, B., Yıldırım, S., & Çelik, D. (2025). Tokens with Meaning: A Hybrid Tokenization Approach for Turkish. arXiv:2508.14292.",
  "Büyük, O. (2023). Joint intent detection and slot filling for Turkish natural language understanding. Turkish Journal of Electrical Engineering and Computer Sciences, 31(5).",
  "Chen, Q., Zhuo, Z., & Wang, W. (2019). BERT for Joint Intent Classification and Slot Filling. arXiv:1902.10909.",
  "Conneau, A., Khandelwal, K., Goyal, N., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. ACL, 8440–8451.",
  "Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT, 4171–4186.",
  "FitzGerald, J., Hench, C., Peris, C., et al. (2023). MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages. ACL. arXiv:2204.08582.",
  "Intent Classification and Slot Filling for Turkish Dialogue Systems. (2020). IEEE (IEEE Xplore document 9302308).",
  "Schweter, S. (2020). BERTurk — BERT models for Turkish. Zenodo. doi:10.5281/zenodo.3770924.",
  "Toraman, C., Yilmaz, E. H., Şahinuç, F., & Ozcelik, O. (2023). Impact of Tokenization on Language Models: An Analysis for Turkish. ACM TALLIP, 22(4), Article 116.",
  "Virpioja, S., Smit, P., Grönroos, S.-A., & Kurimo, M. (2013). Morfessor 2.0: Python Implementation and Extensions for Morfessor Baseline. Aalto University.",
  "Xu, W., Haider, B., & Mansour, S. (2020). End-to-End Slot Alignment and Recognition for Cross-Lingual NLU. EMNLP, 5052–5063.",
];
refs.forEach((r, i) => children.push(new Paragraph({ spacing: { after: 60 }, indent: { left: 360, hanging: 360 },
  children: [new TextRun({ text: `${i + 1}. ${r}`, font: FONT, size: 20 })] })));

// ---- build ------------------------------------------------------------------
const doc = new Document({
  creator: "Anonymous", title: "Turkish slot filling segmentation",
  styles: { default: { document: { run: { font: FONT, size: 22 } } } },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children,
  }],
});
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("submission/AIST2026_Turkish_Slot_Filling.docx", buf);
  console.log("WROTE submission/AIST2026_Turkish_Slot_Filling.docx");
});
