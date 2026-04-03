# 20. Asprise/receipt-ocr

**Repository:** https://github.com/Asprise/receipt-ocr
**Stars:** ~99 | **Language:** C# (62.7%), Java (15.2%), PHP (8.6%), JS (7.7%), Python (5.8%)
**Last Updated:** Low activity (9 commits total) | **License:** Not specified (commercial product)

## Overview

A multi-language SDK/client library for Asprise's commercial cloud-based Receipt OCR API. Unlike the other projects in this research batch, this is not an independent pipeline -- it is a thin API client that sends receipt images to Asprise's proprietary cloud service and receives structured JSON back. The repository provides sample code in 6+ programming languages (C#, Java, JavaScript/Node.js, PHP, Python, Ruby) demonstrating how to call the API. The actual OCR and extraction logic is entirely server-side and proprietary.

The Python client is literally 14 lines of code -- a `requests.post()` call with three parameters.

## Architecture & How It Works

```
Client Application (your code)
    |
    v  HTTP POST (multipart form: image file + params)
    |
Asprise Cloud API (https://ocr.asprise.com/api/v1/receipt)
    |  Parameters:
    |    client_id: 'TEST' (free tier) or paid key
    |    recognizer: 'auto' | 'US' | 'CA' | 'JP' | 'SG' | ...
    |    ref_no: optional tracking reference
    |
    v  (proprietary OCR + extraction engine)
    |
    v  JSON Response
    |
    +-- request metadata (request_id, timestamps)
    +-- receipts[] array:
         +-- merchant (name, address, phone, tax_reg_no, company_reg_no)
         +-- receipt_no, date, time
         +-- items[] (description, quantity, unitPrice, amount)
         +-- currency, subtotal, total, tax, tip, service_charge
         +-- payment_method, card_type, card_number (masked)
         +-- ocr_text (raw OCR output)
         +-- ocr_confidence (e.g., 96.82)
         +-- image width/height
```

**Python implementation (complete):**
```python
import requests
r = requests.post('https://ocr.asprise.com/api/v1/receipt',
    data={'client_id': 'TEST', 'recognizer': 'auto', 'ref_no': 'ref_123'},
    files={"file": open("receipt.jpg", "rb")})
print(r.text)
```

**Language-specific directories:**
- `csharp-vb-net-receipt-ocr/` -- C# and VB.NET samples
- `java-receipt-ocr/` -- Java with Maven (pom.xml)
- `javascript-nodejs-receipt-ocr/` -- Node.js sample
- `php-receipt-ocr/` -- PHP sample
- `python-receipt-ocr/` -- Python sample (14 lines)

## Key Features

1. **Multi-receipt detection** -- Can detect and extract data from images containing multiple receipts.
2. **Comprehensive field extraction** -- Extracts a rich set of fields: merchant info (name, address, phone, tax registration, company registration), transaction details (receipt number, date, time), line items (description, quantity, unit price, amount), financial totals (subtotal, total, tax, service charge, tip), payment info (method, card type, masked card number), and image metadata.
3. **Confidence scoring** -- Returns `ocr_confidence` as a percentage (demonstrated 94-96% in samples).
4. **Country-specific recognizers** -- Supports `recognizer` parameter with options including 'auto', 'US', 'CA', 'JP', 'SG', allowing country-specific optimization.
5. **Raw OCR text included** -- Returns the raw OCR text alongside structured extraction, useful for debugging and custom post-processing.
6. **Multi-language SDK** -- Client libraries in 6+ programming languages.
7. **Free test tier** -- `client_id: 'TEST'` allows testing without payment.
8. **Card detection** -- Identifies payment card types (American Express, Visa, etc.) and masked card numbers.

## Japanese Support

**Confirmed JP recognizer exists.** The Python client code reveals `recognizer` accepts 'JP' as a value, indicating Japanese receipt support. However:
- No Japanese sample output is provided in the README
- All documentation and examples use English-language receipts (McDonald's Singapore)
- No information about how it handles Japanese-specific features (era dates, yen formatting, tax categories, reduced tax rate items)
- The quality of Japanese extraction is unknown -- the 'JP' mode may just set the OCR language rather than providing deep Japanese receipt understanding

## Strengths vs Our Project

1. **Zero-infrastructure setup** -- No need to configure Google Cloud, manage API keys for multiple services, or run an LLM. One HTTP call does everything.
2. **Comprehensive field extraction in one call** -- The API returns merchant info, line items, totals, tax, payment method, and card detection all at once. Our pipeline requires OCR + LLM + post-processing to achieve similar coverage.
3. **Multi-receipt detection** -- Handles images with multiple receipts, which our pipeline does not.
4. **Built-in confidence scoring** -- Returns OCR confidence as part of the response without needing our confidence routing infrastructure.
5. **Card type detection** -- Identifies payment card types (AmEx, Visa, etc.) and masked numbers. Our pipeline extracts payment method but not card type.
6. **Raw OCR text in response** -- Including the raw text alongside structured output is useful for debugging. We could consider including this in our output.
7. **Multi-language SDK** -- If we ever need to integrate receipt parsing into a non-Python application, having C#/Java/JS/PHP clients is convenient.

## Weaknesses vs Our Project

1. **Black box** -- The OCR and extraction logic is entirely proprietary. No way to debug, customize, or improve extraction for specific receipt types. When it fails, you're stuck.
2. **Cloud dependency** -- Every receipt must be sent to Asprise's servers. No offline mode, no self-hosting, no data sovereignty.
3. **Unknown Japanese quality** -- While 'JP' recognizer exists, there's no evidence of how well it handles Japanese-specific challenges (era dates, kanji merchant names, Japanese tax categories, reduced rate items). Our pipeline is specifically tuned for these.
4. **No customization** -- Cannot add merchant rules, custom post-processing, field validation, or domain-specific logic. Our field registry, merchant rules, and subset-sum matching have no equivalent.
5. **No LLM reasoning** -- The API uses traditional OCR + extraction, not LLM-based understanding. For ambiguous or damaged text, an LLM can reason about context in ways a rule-based system cannot.
6. **Commercial pricing** -- Free tier is for testing only. Production use requires paid subscription with unknown pricing.
7. **Latency** -- Every request requires a network round-trip to Asprise's cloud. Our pipeline can be optimized for batch processing with local LLM fallback.
8. **No confidence routing** -- While it reports confidence, there's no mechanism to retry with different settings or route to alternative extraction methods based on confidence.
9. **Minimal repository** -- 9 commits, no tests, no CI. The repo is just sample code, not a maintained library.
10. **Privacy concerns** -- Receipt images containing personal financial data are sent to a third-party cloud service.

## What We Can Learn

1. **API response design** -- Asprise's JSON response structure is well-designed. Including `ocr_text` (raw OCR) alongside structured fields, plus `ocr_confidence` as a top-level metric, is a pattern we should consider for our output format. Having the raw OCR text available makes debugging extraction failures much easier.
2. **Country-specific recognizer concept** -- The idea of a `recognizer` parameter that switches behavior by country could inform how we parameterize our pipeline. We currently have Japanese-specific logic hardcoded throughout; making it configurable per country/locale would be more extensible.
3. **Multi-receipt detection as a feature** -- If users need to process images containing multiple receipts (e.g., photographing a pile of receipts), we should consider adding OpenCV-based receipt boundary detection as a pre-processing step, similar to what household_accounts does.
4. **Comprehensive field taxonomy** -- Asprise extracts fields we don't: card type, card number (masked), service charge, tip, company registration number, tax registration number. Some of these (especially tax registration number / インボイス番号) could be valuable additions to our schema.
5. **Free test tier pattern** -- Offering a free test mode (like their `client_id: 'TEST'`) is a good developer experience pattern, similar to AIReceiptParser's mock mode.

## Recommendation

**Do not adopt Asprise as a replacement for our pipeline.** The loss of control (black box extraction, no customization, cloud-only, unknown Japanese quality) makes it unsuitable for our use case. Our pipeline's Japanese-specific tuning, confidence routing, merchant rules, and validation layers provide capabilities that a generic commercial API cannot match.

However, Asprise could serve as a **benchmarking reference** -- we could send our test fixtures through their JP recognizer and compare extraction accuracy against our pipeline. This would give us a commercial-grade baseline to measure against.

Specific takeaways:
1. **Add raw OCR text to our output** -- Include the Cloud Vision raw text in our result object for debugging
2. **Consider tax registration number extraction** -- インボイス番号 (invoice registration number) is increasingly important in Japan's new invoice system
3. **Benchmark against Asprise** -- Use their free test tier to compare accuracy on our fixtures
