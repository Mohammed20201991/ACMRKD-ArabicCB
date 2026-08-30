# ACMRKD-ArabicCB

## Adaptive Contextual Modeling + Risk-Aware Knowledge Distillation for Arabic Cyberbullying Detection

[![Paper](https://img.shields.io/badge/Paper-Al--Noor%20Journal-blue)](https://njemcs.edu.iq/index.php/njemcs/article/view/144)
[![DOI](https://img.shields.io/badge/DOI-10.71229%2Fer3kch93-green)](https://doi.org/10.71229/er3kch93)
[![Language](https://img.shields.io/badge/Language-Arabic-orange)]()
[![Task](https://img.shields.io/badge/Task-Cyberbullying%20Detection-red)]()
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

> **ACMRKD-ArabicCB** is a research framework for Arabic cyberbullying detection that combines contextual modeling, risk-aware decision making, and selective prediction with human deferral.
>
> The framework is associated with the published study:
>
> **M. A. S. Al-Hitawi, "Selective Arabic Cyberbullying Detection with Multi-View Modeling, Risk Triage, and Human Deferral," Al-Noor Journal of Engineering Management and Computer Science, vol. 2, no. 4, pp. 25–36, 2026.**

---

## Overview
Arabic cyberbullying detection is challenging because harmful content can be expressed through direct insults, indirect language, contextual expressions, dialectal variations, sarcasm, and ambiguous statements.
Traditional binary classifiers are generally forced to make a decision for every input. However, in real-world moderation systems, an uncertain prediction should not necessarily be treated as an autonomous decision.
The research behind **ACMRKD-ArabicCB** therefore focuses on a more practical setting:

```text
                         Arabic Text
                              │
                              ▼
                  ┌──────────────────────┐
                  │ Text Normalization   │
                  │ & Preprocessing      │
                  └──────────┬───────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Multi-View Representation   │
              │                             │
              │ • Word n-grams              │
              │ • Character n-grams         │
              │ • Contextual representations│
              └─────────────┬───────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Cyberbullying Model │
                 └──────────┬──────────┘
                            │
                   ┌────────┴─────────┐
                   ▼                  ▼
              High Confidence     Uncertain
                   │                  │
                   ▼                  ▼
              Automated          Human Review
                Decision              │
                   │                  │
                   └────────┬─────────┘
                            ▼
                    Risk Prioritization
```

The central principle is:

> **Do not force the model to make an autonomous decision when the evidence is insufficient.**
Instead, uncertain cases can be deferred to human moderators and potentially prioritized according to risk.

---

# 🎯 Research Objectives

The framework targets five main objectives:

1. Detect cyberbullying in Arabic text.
2. Exploit complementary lexical and character-level information.
3. Reduce errors through confidence-aware selective prediction.
4. Prioritize potentially harmful cases for human review.
5. Evaluate whether strong within-corpus performance generalizes to external datasets.

---

# Published Research

The scientific foundation of this repository is the following published article:

### Selective Arabic Cyberbullying Detection with Multi-View Modeling, Risk Triage, and Human Deferral

**Author:** Mohammed A.S. Al-Hitawi
**Affiliation:** Department of Artificial Intelligence, College of Information Technology, University of Fallujah, Iraq

**Journal:** Al-Noor Journal of Engineering Management and Computer Science

**Volume:** 2
**Issue:** 4
**Pages:** 25–36
**Year:** 2026

**DOI:** `10.71229/er3kch93`

Paper:

https://njemcs.edu.iq/index.php/njemcs/article/view/144

DOI:

https://doi.org/10.71229/er3kch93

The published article introduces selective Arabic cyberbullying detection with an uncertainty-based reject option and human-in-the-loop moderation.

---

# Methodology

## 1. Arabic Text Processing
The original corpus was transformed into logical records and normalized before modeling.
The published experiment started with:
* **13,230 logical records**
* **269 duplicate records removed**
* **12,961 unique normalized texts**

The final dataset contained:

| Class             |    Samples |
| ----------------- | ---------: |
| Cyberbullying     |      6,739 |
| Non-cyberbullying |      6,222 |
| **Total**         | **12,961** |

This provides a relatively balanced binary classification setting.

---

## 2. Multi-View Text Representation
The published model uses complementary textual representations.

### Word-level features

Word:

* Unigrams
* Bigrams

are represented using TF-IDF.

### Character-level features

Character n-grams from:

```text
3–5 characters
```

are also represented using TF-IDF.

The word- and character-level representations are concatenated before classification.

```text
Arabic Text
    │
    ├──────────────► Word Unigrams
    │
    ├──────────────► Word Bigrams
    │
    └──────────────► Character n-grams (3–5)
                         │
                         ▼
                  TF-IDF Encoding
                         │
                         ▼
                 Feature Concatenation
                         │
                         ▼
                 Class-Balanced LinearSVC
```

This multi-view representation allows the classifier to capture both lexical patterns and subword/orthographic patterns.

---
# Classification

The published baseline uses a **class-balanced LinearSVC** classifier.

The classifier operates on the concatenated TF-IDF representation:

```text
X = [Word TF-IDF || Character TF-IDF]
```

where:

* `Word TF-IDF` captures lexical information.
* `Character TF-IDF` captures character-level patterns.
* `||` denotes feature concatenation.

Class balancing is used to reduce the influence of class-frequency differences.

---

# 🛡️ Selective Classification

A major contribution of the research is the use of a **reject/deferral option**.

Instead of:

```text
Input → Cyberbullying / Non-Cyberbullying
```

the system can operate as:

```text
                     ┌──► Cyberbullying
                     │
Input → Model ───────┤
                     │
                     └──► Human Review
```

Cases with insufficient confidence are not automatically accepted as final decisions.

This approach is particularly important for moderation systems because an incorrect automated decision can have substantially different consequences depending on the content.

---

# Risk Triage

The system introduces a prioritization concept for deferred cases.

Rather than sending all uncertain cases to human reviewers with equal priority, potentially high-risk abusive content can be flagged for earlier review.

Conceptually:

```text
                 Model Prediction
                        │
                        ▼
                 Confidence Check
                        │
              ┌─────────┴─────────┐
              │                   │
        High confidence       Low confidence
              │                   │
              ▼                   ▼
       Automated result      Human deferral
                                  │
                                  ▼
                           Risk prioritization
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
                High priority              Normal review
```

The important distinction is that **risk triage supports human moderation rather than replacing human judgment**.

---

# Published Results

The main experiment used **five-fold stratified cross-validation**.

Reported results:

| Metric   |              Result |
| -------- | ------------------: |
| F1       | **0.9914 ± 0.0013** |
| Macro-F1 | **0.9911 ± 0.0014** |
| ROC-AUC  | **0.9996 ± 0.0001** |

These results demonstrate very strong performance under the evaluated within-corpus setting.

However, the research deliberately avoids interpreting these scores as proof of universal real-world robustness.

---

# Selective Prediction Results

The deferral experiments demonstrate the value of allowing the model to abstain from uncertain decisions.

### 5% Deferral

When approximately **5% of cases** were deferred:

* **85.7% of the errors** were captured.
* F1 on retained cases increased to **0.9987**.

### 10% Deferral

When approximately **10% of cases** were deferred:

* **94.6% of the errors** were captured.
* F1 on retained cases increased to **0.9995**.

This demonstrates that a relatively small human-review workload can potentially remove a large proportion of model errors from the automatically accepted cases.

---

# External Generalization Audit

An important part of the research is that the model was not evaluated only on the original corpus.

The authors conducted external audits using offensive-language and hate-speech-related data.

The results showed a substantial decrease in performance:

| Evaluation                               |         F1 |    ROC-AUC |
| ---------------------------------------- | ---------: | ---------: |
| Within-corpus                            | **0.9914** | **0.9996** |
| External offensive-language audit        | **0.3882** | **0.7318** |
| External subtype/hate-speech proxy audit | **0.5771** | **0.7996** |

These findings highlight an important limitation:

> **Very high in-domain performance does not necessarily imply reliable cross-domain cyberbullying detection.**

The external audits therefore support the need for human review, uncertainty handling, and risk-aware moderation.

---

# Experimental Design

The research follows the general workflow:

```text
                 ┌─────────────────────┐
                 │ Arabic Corpus       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Cleaning &          │
                 │ Normalization       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Duplicate Removal   │
                 └──────────┬──────────┘
                            │
                            ▼
             ┌─────────────────────────────┐
             │ Multi-View TF-IDF           │
             │                             │
             │ Word 1–2 grams              │
             │ Character 3–5 grams         │
             └─────────────┬───────────────┘
                           │
                           ▼
                 ┌─────────────────────┐
                 │ Feature Fusion      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Class-Balanced      │
                 │ LinearSVC            │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Confidence /        │
                 │ Selective Decision   │
                 └──────────┬──────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
          Automated Decision       Human Deferral
                                       │
                                       ▼
                                Risk Prioritization
```

---

# ACMRKD Extension

The name **ACMRKD-ArabicCB** denotes an extended research direction:

> **Adaptive Contextual Modeling + Risk-Aware Knowledge Distillation**

The published paper establishes the selective-detection and human-deferral foundation. The ACMRKD implementation is intended to extend this foundation toward contextual neural modeling and knowledge distillation.
Conceptually:

```text
                       Arabic Input
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Contextual Encoder   │
                 └──────────┬───────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
          Binary        Expression       Risk
        Detection       Modeling       Modeling
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                    Teacher Knowledge
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Risk-Aware Knowledge │
                 │ Distillation         │
                 └──────────┬───────────┘
                            ▼
                     Student Model
                            │
                            ▼
                 Confidence / Uncertainty
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
             Accept                Defer
                                      │
                                      ▼
                                Human Review
```
The purpose of the extension is to investigate whether a compact student model can preserve useful information learned by richer teacher models while maintaining selective and risk-aware decision behavior.
**Important:** the knowledge-distillation component should be considered an extension of the published research rather than a component claimed by the published article itself.

---

---

# Evaluation Metrics

The project should report both conventional classification metrics and selective-prediction metrics.

### Classification

* Accuracy
* Precision
* Recall
* F1
* Macro-F1
* ROC-AUC
* Confusion Matrix

### Selective Prediction

* Coverage
* Selective Risk
* Error Capture Rate
* Deferral Rate
* Retained-set F1
* Risk/coverage relationship

### External Evaluation

Models should also be tested on datasets or domains that differ from the training corpus whenever possible.

This is especially important because the published research demonstrated a significant gap between within-corpus and external performance.

---

# Human-in-the-Loop Moderation

ACMRKD-ArabicCB is designed around the principle that automated moderation should assist human reviewers rather than blindly replace them.

A production-oriented system can therefore operate as:

```text
                 ┌──────────────────┐
                 │ Incoming Content  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ AI Detection     │
                 └────────┬─────────┘
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
             Confident          Uncertain
                 │                 │
                 ▼                 ▼
             Automatic          Deferred
              Result              │
                                   ▼
                             Risk Scoring
                                   │
                                   ▼
                           Human Moderator
                                   │
                                   ▼
                              Final Action
```

The model therefore becomes a **decision-support system** rather than an unconditional autonomous moderator.

---

# Limitations

Several limitations should be considered.

### 1. Domain Shift

Performance can decrease substantially when the model is evaluated on content from a different dataset or task domain.

### 2. Arabic Linguistic Diversity

Arabic contains:
* Modern Standard Arabic
* Dialects
* Code-switching
* Arabizi
* Informal spelling
* Morphological variation
* Social-media abbreviations

A model trained on one corpus may not generalize to all Arabic varieties.

### 3. Context Dependence

Some cyberbullying cases cannot be reliably interpreted from an isolated comment.

Conversation history, previous comments, user interaction, and platform context may be required.

### 4. Ambiguous Content

Sarcasm, humor, quotations, and implicit aggression can be difficult to classify automatically.

### 5. Human Review Remains Important

Selective prediction reduces the number of cases that require human intervention, but it does not eliminate the need for human judgment.

---

# Ethical Considerations

Cyberbullying detection systems can affect users' safety, privacy, and freedom of expression.

Therefore:

* Avoid unnecessary collection of personal information.
* Anonymize user identifiers where possible.
* Do not expose private user information in released datasets.
* Treat model predictions as probabilistic decisions.
* Avoid using automated predictions as the sole basis for punitive actions.
* Provide human review for uncertain or high-impact cases.
* Evaluate false positives and false negatives separately.
* Consider cultural and dialectal differences in Arabic content.

---

# Citation

If you use this project or the associated methodology in academic research, please cite:

```bibtex
@article{alhitawi2026selective,
  title     = {Selective Arabic Cyberbullying Detection with Multi-View Modeling, Risk Triage, and Human Deferral},
  author    = {Al-Hitawi, Mohammed A.S.},
  journal   = {Al-Noor Journal of Engineering Management and Computer Science},
  volume    = {2},
  number    = {4},
  pages     = {25--36},
  year      = {2026},
  doi       = {10.71229/er3kch93}
}
```

---

# Related Work

The published study builds upon research in:

* Arabic cyberbullying detection
* Arabic offensive-language detection
* Arabic transformer models
* Selective classification
* Human-in-the-loop moderation
* Risk-aware content moderation

Important related models and resources include **AraBERT**, **ARBERT**, and **MARBERT**, as well as previous Arabic cyberbullying corpora and detection systems.

---

# Publication

**Selective Arabic Cyberbullying Detection with Multi-View Modeling, Risk Triage, and Human Deferral**

Published in:

**Al-Noor Journal of Engineering Management and Computer Science**

Volume 2, Issue 4, 2026, pp. 25–36.

[Read the published article](https://njemcs.edu.iq/index.php/njemcs/article/view/144)

[DOI: 10.71229/er3kch93](https://doi.org/10.71229/er3kch93)

---

# 📜 License

The associated published work is distributed under the:

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).**

See:

https://creativecommons.org/licenses/by-nc-sa/4.0/

---

# 👨‍🔬 Author

**Mohammed A.S. Al-Hitawi**

Department of Artificial Intelligence
College of Information Technology
University of Fallujah
Anbar, Iraq

ORCID:

https://orcid.org/0009-0009-7905-0978

---

# Research Vision

ACMRKD-ArabicCB aims to move Arabic cyberbullying detection from:

```text
"Classify every message"
```

toward:

```text
"Make reliable decisions,
identify uncertainty,
prioritize risk,
and involve humans when necessary."
```

The long-term objective is a robust Arabic content-moderation framework capable of handling linguistic variation, domain shift, uncertainty, and real-world moderation requirements.

---

## Acknowledgment

This project is based on the research presented in:

> **Al-Hitawi, M. A. S. (2026). Selective Arabic Cyberbullying Detection with Multi-View Modeling, Risk Triage, and Human Deferral. Al-Noor Journal of Engineering Management and Computer Science, 2(4), 25–36.**

The published study provides the empirical foundation for the selective classification, risk triage, and human-deferral components of this research direction.
