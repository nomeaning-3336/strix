# Candidate falsifier template

Load `/bounty-investigator`.

Role: adversarial verifier. Your goal is to KILL this candidate if possible. Do not improve the exploit until it survives falsification.

Candidate hypothesis:
<HYPOTHESIS>

Claimed invariant violation:
<X != Y>

Claimed source anchors:
<FILES / SYMBOLS>

Claimed attacker control:
<INPUT / PRIVILEGE>

Claimed exploit flow:
<FLOW>

Evidence already available:
<E1/E2 DETAILS>

Try to disprove it by checking:
- framework parameter/parser semantics;
- route constraints / declared params;
- before filters / middleware;
- policy/service-level authorization;
- later object re-resolution;
- signed metadata / cryptographic binding;
- feature flags / license / edition;
- default configuration;
- ordering / overwrite behavior;
- source/runtime version mismatch;
- documented intended semantics;
- public known/duplicate signals.

Return PROMOTE only if the chain still holds after these checks. Otherwise HOLD or KILL with the decisive contradiction.

