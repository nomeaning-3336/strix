---
name: developer_intent
description: Intent and known-issue discipline before filing — how to tell a vulnerability from the project's documented and spec-tested design, which evidence outranks which, and when a report must be a follow-up instead
---

# Developer Intent and Known Issues

Proving the code does what you observed is only half of a finding. The other half
is proving the behaviour is *not what the project intended* — and that half is
where most source-aware false positives come from.

A behaviour that looks looser than you would have designed is not automatically a
vulnerability. Repositories are full of deliberate decisions: an authorization
endpoint that accepts either of two boundaries, a flag that is off by default on
purpose, a parser that is permissive because its format is permissive. Filing
those as vulnerabilities costs triage time and credibility, and it is worse when
the repository's own document — sometimes one you quoted yourself — says the
behaviour is intended.

This skill governs a **source-aware finding**: one whose evidence is the target's
code, or one filed during a white-box scan. The reporting tool enforces it: a
source-aware report that arrives without an intent record is rejected before it is
filed.

## The pass, in order

1. **Pin the behaviour first.** One sentence: the exact input, the exact path, the
   exact outcome. No claim yet. Every piece of evidence below is compared against
   *this* sentence — an entry only supports or contradicts intent when it is about
   the behaviour you actually observed.
2. **Search the project's own documentation.** `doc/`, `docs/`, `README*`, guides,
   `CHANGELOG`, ADRs, API references, per-feature docs. Grep the endpoint name,
   the parameter, the permission or scope name, the feature flag, and synonyms.
   **Read the surrounding section, not just the matching line** — the sentence
   that decides your finding is usually the next one.
3. **Read the specs and tests that cover that path.** A request, integration or
   unit spec that asserts the observed behaviour is strong evidence of intent.
   Look for the ones that exercise the *specific* boundary you think is broken.
4. **Check the history.** `git log -S <symbol>` / `git -G <pattern>` / `git blame`
   on the construct. A behaviour introduced deliberately, together with its docs
   or its specs, is design rather than an accident. A behaviour that has always
   conflicted with its own docs is your finding.
5. **Look for the known issue.** Has this been reported, rejected, documented as
   accepted risk, or listed as a known limitation? Search the tracker references
   you have, the changelog, and `SECURITY.md` / threat-model docs.
6. **Re-read the semantics for an OR/alternative.** If access is granted when *any
   one* of several boundaries, roles, conditions or flags is satisfied, then
   satisfying one of them is not a bypass of the others. Confirm you are not
   reading an OR as an AND. A boundary you did not expect to be sufficient is
   only a finding if the *set* of alternatives is wrong, not if your token matched
   one of them.
7. **Reconcile your own counterevidence before filing.** If you already gathered
   or cited something that contradicts the finding — including a document you
   quoted in the argument against your own claim — you must resolve it explicitly.
   Never file something your own cited evidence refutes.

Record what you searched, not just what you found: `known_issue_or_duplicate_search`
and `intent_search_scope` are part of the report, and "searched and found nothing"
is a legitimate, useful outcome that must be labelled as such.

## Authority hierarchy

Not every sentence of documentation is equal. When two sources disagree, the
higher one governs, and `authority` on each evidence entry says which you are
citing:

| authority | what it is | example |
| --- | --- | --- |
| `security_contract` | a published security policy, threat model, or an explicit security promise the product makes | tenant-isolation policy, `SECURITY.md` guarantees |
| `implementation_spec` | the exact semantics of this route, API, permission or flag: authorization guides, API references, request/integration specs, schemas | "import endpoints may resolve a group or user boundary" |
| `operator_doc` | how to run or configure the feature; describes deployment, not the contract | admin guide, config reference |
| `history` | changelog, ADR, the commit that introduced the behaviour with its docs or tests | "added deliberately in <commit>, with specs" |
| `comment` | code comments, TODOs, review discussion | "this is intentionally permissive" in a comment |

A `security_contract` or `implementation_spec` entry that describes your observed
behaviour **as intended** is decisive. An `operator_doc`, `history` or `comment`
that does the same is evidence, not a verdict — it is raised as a warning and your
`assumptions` must say why it does not settle the question. Never let a comment
outrank a contract, and never let "the docs mention it" stand in for "the docs
describe exactly this behaviour".

## The rebuttable presumption

When authoritative evidence describes exactly the behaviour you observed, the
default disposition is **intended behaviour / not applicable**. Filing requires a
`security_contract_conflict` — a concrete claim a reviewer can check:

- `conflicting_security_invariant` — another documented invariant that this
  behaviour breaks.
- `external_security_promise` — the product promised something to users or
  customers that this contradicts.
- `cross_tenant_harm` — the behaviour lets one tenant reach another's data,
  namespace or resources.
- `exploitable_boundary_not_waivable` — an actual security boundary is crossed,
  and no amount of documentation can waive it.

The conflict must name the invariant, cite **where it comes from** (a source other
than the document you are overriding), and state the concrete harm. "A stricter
model would be better", "defence in depth", and "the intent might be stale" are
not conflicts — they are preferences. If that is all you have, this is not a
finding.

Conversely, **documentation never sanitizes genuinely insecure behaviour**. If the
docs and the code disagree, that disagreement *is* the finding: report the
conflict — ambiguous or undocumented intent with a demonstrable security
consequence — rather than dressing it up as a bypass. If two authoritative sources
contradict each other, or the only supporting evidence is stale, the outcome is a
follow-up (`record_coverage(outcome="needs_follow_up")`), not a filed
vulnerability. Surface the ambiguity; do not resolve it in your own favour.

## Independent review before filing

A source-aware finding must be reviewed by a **different agent** before it is
filed — the agent that built the case is the worst judge of whether it is really
a case. Use `create_agent` (or `send_message_to_agent` to an existing worker) and
ask for the intent review explicitly:

- the reviewer re-reads the same evidence and the same searches;
- the reviewer states a verdict: `intent_confirmed`, `conflict_confirmed`, or
  `unresolved`;
- `intent_review` records `reviewer_kind: "independent_agent"`, the reviewer, the
  verdict and what they checked.

Routing:

- `intent_confirmed` → do not file unless you also carry a real
  `security_contract_conflict`.
- `conflict_confirmed` → file, with the conflict.
- `unresolved` → `record_coverage(outcome="needs_follow_up", evidence=<what is
  unresolved>)` and move on. **Unresolved intent is a follow-up, not a
  vulnerability.** A finding cannot review itself, and "no reviewer available" is
  not an exemption — it is a follow-up.

## Filing checklist

Before `create_vulnerability_report` on a source-aware finding:

- [ ] The behaviour is pinned in one sentence.
- [ ] `intent_check_status` is `completed` (the search ran, even if it found
      nothing) or `unavailable` (the source or its docs could not be searched).
- [ ] `intent_search_scope` lists where you looked: docs, specs/tests, and the git
      commands used.
- [ ] `intent_evidence` carries what you found, each entry with its `authority`
      and what it supports or contradicts.
- [ ] `known_issue_or_duplicate_search` says where you looked and what came back.
- [ ] `alternative_semantics_check` states whether an any-one-of condition
      explains the observation.
- [ ] If authoritative evidence supports the behaviour: `security_contract_conflict`
      with a kind, the invariant, its source, and the harm.
- [ ] `intent_review` from a different agent, with its verdict.
- [ ] `assumptions` labels every presumption, including "no intent evidence was
      found".

If you cannot tick these, the honest output is a coverage entry — `not_applicable`
with the document that defines the behaviour, or `needs_follow_up` with the gap —
not a report.
