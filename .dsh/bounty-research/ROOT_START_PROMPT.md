# Root starter prompt

Load `/bounty-research-orchestrator`.

Repository scope: the current checkout.

Goal: choose the best next 3–5 deep Strix investigations for authorized bug-bounty research, optimizing for expected bounty value per hour and low duplicate risk.

Operate as a root orchestrator, not as one giant scanner.

Required orchestration:
1. Establish HEAD/branch/date/dirty state and relevant repository instructions.
2. Build a concise security seam/invariant map.
3. Launch independent subagents in parallel with DIFFERENT lenses. At minimum:
   - recent-change/new-caller analyst;
   - boring-old-shared-code analyst;
   - cross-component trust/parser analyst;
   - lifecycle/state-machine analyst;
   - interface-differential analyst.
4. Keep exploratory children independent on their first pass. Do not prime them with each other's hypotheses.
5. Every fresh child prompt must be self-contained and tell it to load `/bounty-investigator`.
6. While children run, use public GitHub/repository history to investigate attack-surface freshness, known fixes, intended semantics, and duplicate risk.
7. Cluster variants by root cause.
8. For the top candidates, launch dedicated falsifier children whose job is to disprove them.
9. Do not rank candidates until each has an evidence ledger.
10. Return:
   - top candidates with E0–E4 evidence;
   - killed hypotheses;
   - duplicate-risk assessment;
   - exact copy-paste Strix prompts for the best 3–5;
   - one final recommendation for what to run first.

Important:
- New/complex code is not automatically preferred.
- A new caller into old code counts as fresh attack surface.
- Do not claim systemic route counts without representative proof.
- Do not inflate severity from hypothetical composition.
- Public silence is not proof of novelty.
- A majority of subagents is not evidence; source/runtime facts are.
