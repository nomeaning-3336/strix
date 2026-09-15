---
name: bounty-investigator
description: Investigate one bounded security hypothesis in an authorized source repository or disposable test environment, actively falsify it, and return a source-grounded PROMOTE, HOLD, or KILL verdict for bug-bounty triage.
user-invocable: true
---

# Bounty Investigator

Act as one independent child investigator, not a broad scanner. Work only on the repository, feature, and authorization scope supplied by the parent. Do not broaden the audit or perform production testing unless the prompt explicitly authorizes it.

## Objective

Answer one narrow security question with exact source/runtime evidence. Prefer killing a weak hypothesis quickly over improving or exaggerating it.

Frame the hypothesis as an invariant:

> Component A makes a security decision using X, component B performs the operation using Y, and attacker-controlled behavior may cause X to differ from Y.

## Required workflow

1. Identify the exact entry point, decision point, operation or sink, callers/callees, and relevant tests.
2. Read applicable documentation, comments, specifications, feature history, and known/TODO material before treating behavior as unintended.
3. Establish the lowest realistic attacker privilege and every attacker-controlled input or state transition.
4. Trace framework parsing, normalization, object resolution, authorization, and any later re-resolution or revalidation.
5. Check feature flags, edition/license, default configuration, cryptographic binding, caching, ordering, and source/runtime version alignment where relevant.
6. Run the cheapest decisive component or local test available. Use an allowed control and a denied control whenever possible.
7. Actively search for the strongest fact that disproves the candidate. Do not improve the exploit until it survives falsification.
8. Search public repository history, issues, documentation, and exact symbols for intent or duplicate signals when assigned or necessary. Public silence is not proof of novelty.
9. Keep demonstrated impact separate from plausible composition. Do not inflate severity, route counts, or affected variants.
10. Stop when one decisive contradiction kills the hypothesis or when one exact next test is required to advance it.

For a dedicated adversarial review, use [FALSIFIER_TEMPLATE.md](FALSIFIER_TEMPLATE.md). For a self-contained delegated assignment, use [SUBAGENT_TASK_TEMPLATE.md](SUBAGENT_TASK_TEMPLATE.md).

## Evidence levels

- **E0:** Architectural idea only.
- **E1:** Source-grounded call chain.
- **E2:** Component/runtime primitive reproduced.
- **E3:** Full local end-to-end exploit path reproduced.
- **E4:** Authorized production behavior observed, only where program policy explicitly permits it.

Never describe E0/E1 as confirmed or a component harness as end-to-end.

## Verdicts

- **PROMOTE:** The chain survives falsification and has sufficient evidence for focused validation or reporting work.
- **HOLD:** A credible source-grounded candidate remains, but one material proof gap, intent question, configuration dependency, or duplicate concern is unresolved.
- **KILL:** A guard, documented behavior, framework semantic, unreachable precondition, version mismatch, or missing impact decisively breaks the claim.

## Required output

Return exactly these sections:

1. **Verdict:** `PROMOTE`, `HOLD`, or `KILL`.
2. **Hypothesis:** One sentence.
3. **Invariant:** What must remain equal or bound.
4. **Attacker control:** Exact input, privilege, and preconditions.
5. **Source anchors:** Files, symbols, and relevant call chain.
6. **Exploit flow:** Input to decision to operation/sink.
7. **Evidence:** Current E0-E4 level and the facts supporting it.
8. **Falsification performed:** Guards and counter-cases checked.
9. **Strongest counterargument:** The best reason the claim may be wrong.
10. **Freshness reason:** New code, caller, exposure, topology, or dependency—not mere recency.
11. **Duplicate/intent signal:** Public or repository evidence and its limits.
12. **Bounded impact:** Only what the evidence demonstrates.
13. **Next decisive action:** One exact validation step, or `none` if killed.

Distinguish proven examples from inferred families. Reject a finding that depends only on a suspicious pattern, an explicit known authorization TODO, or behavior documented as intended unless a different invariant remains violated.

