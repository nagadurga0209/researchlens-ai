# Sample questions & demo script

Use these to demonstrate ResearchLens AI (e.g. in an interview or portfolio
walkthrough). Upload 2–3 papers first.

## Single-paper question answering (Chat tab)
- "What problem does this paper try to solve?"
- "What methodology or algorithm is proposed?"
- "Which datasets and evaluation metrics are used?"
- "What are the reported results?"
- "What limitations do the authors acknowledge?"
- "What future work is suggested?"

## Grounding / honesty demo
Ask something the papers can't answer, to show it refuses instead of inventing:
- "What is the current stock price of the company?"
- "Who won the football match last night?"
→ Expect: "I could not find enough supporting evidence…".

## Summarization (Summary tab)
- Select a long paper and generate a structured summary. Point out the
  hierarchical note ("Summarized from N sections") and the STATED vs
  INTERPRETATION labels.

## Comparison (Compare tab)
- Select two papers and compare. Point out the structured table and the
  "Not found in the provided document" cells where a paper is silent.

## Research gap analysis (Gaps tab)
- Select one or more papers. Point out the disclaimer that gaps are
  **potential directions inferred from the literature**, not verified findings,
  and the page-level source hints.

## Source citations
- After any answer, expand the "Sources" panel to show filename, page number,
  similarity score, and the exact passage the answer was grounded in.

## Suggested 3-minute demo order
1. Upload two papers (show indexing status: pages + chunks).
2. Ask one factual question → show grounded answer + sources.
3. Ask an unanswerable question → show the honest "not found".
4. Generate a summary of one paper.
5. Compare the two papers (table).
6. Open the **Gaps** tab for the research-gap analysis.
7. Run `python eval/run_eval.py` in a terminal → show real retrieval metrics.
