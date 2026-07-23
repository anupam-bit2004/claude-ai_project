# Project 1: LLM Evaluation Platform

## What This Does
Data-driven LLM evaluation platform with automated 
MLflow audit trail. Tests any chatbot domain without 
code changes — just add entries to golden_dataset.json.

## Architecture
golden_dataset.json → DeepEval evaluation → 
MLflow logging → Pass/Fail quality gate

## Governance Relevance
- Satisfies EU AI Act Article 12 (automatic audit trail)
- Every evaluation run logged with score, timestamp, 
  model version, pass/fail status
- Failed runs block deployment automatically

## Results
- MED_001: Paracetamol relevancy ✅ PASSED (1.0)
- MED_002: Diabetes relevancy ✅ PASSED (1.0)
- MED_003: Hallucination detection ❌ FAILED (0.5)
  ← correctly caught fabricated medical information

## Tools
DeepEval | MLflow | Ollama (Mistral) | pytest | Python
