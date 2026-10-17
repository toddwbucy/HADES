---
title: State Notes
doc_id: state-notes
version: '0.10'
date: 2026-10-17
author: Example Author
kind: graphable-document
budget_tokens: 10000
estimated_tokens: 250
source: Synthetic test fixture
source_status: authored
keywords: [buffer, glossary]
---
# State Notes
<!-- node: state-notes | keywords: buffer -->
This document defines a buffer.

## The buffer
<!-- node: state-notes/buffer | keywords: buffer, unresolved | xref: other/document -->
Standing: candidate. This is source text, not an accepted assertion.

The buffer holds entries. See Illustration 1.

### A boundary
<!-- node: state-notes/buffer/boundary | keywords: buffer -->
#### Internal detail
This heading stays inside its H3.

```markdown
## Not a node
<!-- node: fake | keywords: fake -->
```

> **Illustration 1.** A shelf
> <!-- illustration: state-notes/ill-1 | type: text | subject: buffer/boundary | keywords: buffer -->
> A shelf holds books.

## Glossary
<!-- node: state-notes/glossary | keywords: glossary -->
- **buffer (B)**: The entries retained between calls.
  Rejected synonyms: cache.
- **glossary**: The definitions used by this document.
