# 11. zzzDavid/ICDAR-2019-SROIE

## Overview

| Field | Value |
|---|---|
| **Repository** | [zzzDavid/ICDAR-2019-SROIE](https://github.com/zzzDavid/ICDAR-2019-SROIE) |
| **Stars** | 413 |
| **Forks** | 141 |
| **Language** | Python (100%) |
| **License** | MIT |
| **Created** | 2019-03-31 |
| **Last Push** | 2020-07-20 (archived/inactive) |
| **Approach** | Deep learning (CTPN + CRNN + Bi-LSTM) on SROIE competition dataset |

This is a competition solution for the ICDAR 2019 SROIE (Scanned Receipts OCR and Information Extraction) challenge. It is **not a reusable library** -- it is a benchmark dataset + model implementation targeting 4 fixed fields (company, address, date, total) from English-language scanned receipts.

## Architecture & How It Works

The project tackles three sub-tasks with separate model architectures:

### Task 1: Text Localization
- **Model**: CTPN (Connectionist Text Proposal Network) & SSD (Single Shot Detector)
- **Output**: 4-vertex bounding boxes around text regions
- **Performance**: 86.94% H-mean (85.23% recall, 88.73% precision)

### Task 2: OCR Recognition
- **Model**: CRNN (Convolutional Recurrent Neural Network)
- **Output**: Recognized text strings from detected regions
- **Performance**: 38.63% H-mean (26.33% recall, 72.53% precision) -- notably weak

### Task 3: Key Information Extraction (most relevant to us)
- **Model**: 2-layer stacked bidirectional LSTM
- **Approach**: Character-wise classification -- each character in the receipt text is classified into one of 5 labels (company, address, date, total, other)
- **Input**: All OCR text from a receipt concatenated into a single sequence with spatial ordering
- **Framework**: PyTorch
- **Performance**: 75.58% precision and recall

**Data flow**:
1. Receipt images -> text localization (bounding boxes)
2. Bounding boxes -> OCR text recognition
3. OCR text -> character-level Bi-LSTM -> per-character label prediction
4. Group consecutive same-label characters -> extract fields

### Key source files:
- `task3/src/my_classes.py` -- `TextBox` and `TextLine` classes that preserve spatial layout (x/y coordinates, y-span overlaps for line merging)
- `task3/src/my_data.py` -- Dataset loader that encodes characters from a vocabulary of `ASCII_UPPERCASE + digits + punctuation + whitespace`, creates character-level tensors with label tensors
- `task3/src/train.py` -- Bi-LSTM training loop
- `task3/src/test.py` -- Inference and evaluation

### Dataset format:
- **1000 scanned receipt images** (600 train, 400 test)
- **Annotations**: Bounding box coordinates in `x1,y1,x2,y2,x3,y3,x4,y4,transcript` format
- **Ground truth**: JSON with exactly 4 fields: `company`, `address`, `date`, `total`
- **Corrected data**: The `data/` directory contains cleaned annotations fixing errors in the original ICDAR dataset

## Key Features

- **Spatial-aware text ordering**: `TextBox`/`TextLine` classes merge text boxes that overlap vertically into logical lines, then sort horizontally -- this preserves reading order from 2D OCR output
- **Character-level NER**: Treats information extraction as character-level sequence labeling rather than line-level or word-level, which handles fields that span multiple words
- **Corrected dataset**: Includes fixes for annotation errors in the original ICDAR 2019 SROIE dataset
- **Minimalist approach**: Deliberately simple stacked Bi-LSTM (no transformers, no attention) that still achieves competitive results

## Japanese Support

**None**. The vocabulary is explicitly `ascii_uppercase + digits + punctuation + " \t\n"` -- no Unicode, no CJK characters. The dataset contains only English-language receipts from Southeast Asia. The character encoding would need complete replacement for Japanese support.

## Strengths vs Our Project

1. **Standardized benchmark**: Provides a well-known evaluation dataset (SROIE) with 1000 annotated receipts, enabling reproducible comparison across different approaches
2. **Spatial layout preservation**: The `TextBox`/`TextLine` spatial ordering system is clever -- it uses y-span overlap to merge text boxes into logical lines, then sorts by x-coordinate. This is more principled than naive line-by-line concatenation
3. **End-to-end pipeline**: Covers the full pipeline from raw image to structured data, whereas we rely on Google Cloud Vision as a black box for OCR
4. **No API dependency**: Runs entirely locally with PyTorch -- no cloud API costs or rate limits

## Weaknesses vs Our Project

1. **Fixed 4-field schema**: Can only extract company, address, date, total. Our Pydantic schema handles dozens of fields (items, tax categories, payment methods, merchant details, etc.)
2. **English-only**: Zero Japanese support. Our project is purpose-built for Japanese receipts
3. **No LLM reasoning**: Character-level LSTM has no semantic understanding -- it learns positional patterns rather than understanding what text means. Our DeepSeek pipeline can reason about ambiguous cases
4. **75.58% accuracy is mediocre**: Our pipeline targets much higher accuracy with multi-pass verification and confidence routing
5. **Archived/inactive**: Last pushed in 2020. No maintenance, no modern model support
6. **No validation layer**: No Pydantic schema, no post-processing, no confidence scoring
7. **No retry/confidence gating**: No mechanism to detect and recover from extraction errors

## What We Can Learn

1. **Spatial text ordering algorithm**: The `TextBox`/`TextLine` approach of merging text boxes by y-span overlap and then sorting by x-coordinate could improve our text normalization step. When Google Cloud Vision returns bounding boxes, we could use this technique to reconstruct reading order more accurately than relying on the API's native ordering.

2. **Character-level labeling as a concept**: While we wouldn't use a Bi-LSTM, the idea of treating extraction as sequence labeling (where each token gets a field label) could be useful for designing prompts or fine-tuning. Instead of asking the LLM "what is the date?", we could ask it to label each line of OCR text with its field type.

3. **SROIE as a benchmark**: The corrected SROIE dataset could be used to benchmark our pipeline against published baselines on English receipts, giving us a standardized comparison point.

4. **Data quality corrections**: The project demonstrates the importance of auditing and correcting ground truth data -- something relevant to our own fixture management.

## Recommendation

**Do not adopt as a tool or dependency.** This is a 2019 competition solution that is architecturally outdated (pre-transformer, pre-LLM). However, it has two practical uses:

1. **Benchmark dataset**: Use the corrected SROIE dataset (1000 receipts with ground truth) to evaluate our pipeline on English receipts if we ever expand beyond Japanese
2. **Spatial ordering technique**: Consider adapting the `TextBox`/`TextLine` spatial merge algorithm for our text normalization step when processing Google Cloud Vision bounding box output
