# Round-6 Direct-Answer Integrity Report

- Overall: **PASS**
- Primary samples: 2
- Episode records: 8
- Four arms share the same frozen prefix, question, assistant history, role sequence, and direct-answer instruction.
- F retains six prefix crops; O/R/X retain exactly two and receive original plus two retained images.
- Each arm performs exactly one generation and zero tool executions.
- Grounding output is recorded as a protocol violation and is never executed.
- GPU workers never load GT answers, boxes, categories, coverage, or annotations.
- Oracle selection uses GT coverage only during controller-side preparation and offline audit.
- Baseline files are fingerprint checked and read only.

## Problems
- None.
