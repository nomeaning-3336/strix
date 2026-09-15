# Subagent delegation template

Load `/bounty-investigator`.

You are one independent child investigator. You do not have the parent conversation, so everything you need is below.

Repository: <CURRENT REPOSITORY / SCOPE>

Assignment:
<ONE NARROW QUESTION OR LENS>

Known context:
<ONLY FACTS REQUIRED TO DO THIS TASK>

Explicitly out of scope:
<AREAS / KNOWN BUGS / OTHER CHILD ASSIGNMENTS TO AVOID>

Security invariant to test:
<X must remain bound/equal to Y>

Attacker model:
<LOWEST REALISTIC PRIVILEGE / INPUT CONTROL>

Required work:
1. Find exact source entry point, decision point, operation/sink, callers/callees, and relevant tests/docs.
2. Establish exactly which values are attacker-controlled and how the framework parses/resolves them.
3. Search for guards or later revalidation that would kill the chain.
4. Run the cheapest decisive local/component test available.
5. Actively try to disprove the hypothesis.
6. Do not broaden into a generic audit.
7. Do not claim severity or affected counts beyond evidence.

Return exactly the `/bounty-investigator` output format with:
PROMOTE / HOLD / KILL,
source anchors,
E0–E4,
falsification performed,
strongest counterargument,
freshness reason,
duplicate/intent signal,
bounded impact,
and one next decisive action.

