Phase 1 Locked Contracts — MVP Invariants

This project is now operating beyond initial build and into controlled hardening. The following Phase 1 contracts are frozen and must remain unchanged unless an explicit future phase authorizes a deliberate redesign.
1. Scoring formulas are locked
The scoring engine logic in scoring.py is considered baseline-stable. Core score construction, component calculations, and final score assembly must not be altered during Phase 2. Any work in this phase must preserve the existing mathematical behavior exactly.
2. Contribution modifier is locked
The contribution modifier structure, including its dependence on the existing structured flag system, is fixed. Phase 2 may validate, stress-test, and verify this logic under broader inputs, but must not change how the modifier is derived.
3. Attribution logic is locked
The diagnostic attribution layer is fixed in structure and intent. Attribution must remain deterministic, additive, interpretable, and traceable to underlying records. Phase 2 may verify reconciliation and robustness across broader datasets, but may not redesign attribution methodology.
4. CSV is authoritative
CSV files remain the source of truth for the MVP. Configuration, task data, and flag data must continue to originate from persisted CSV inputs. No hidden state, parallel data source, database-backed fallback, or inferred persistence layer may be introduced in Phase 2.
5. app.py is presentation only
The Streamlit application must remain a thin UI layer over the scoring and validation engine. Business logic, scoring logic, attribution logic, and input validation should remain outside the UI wherever they currently reside. Phase 2 may extend display behavior, but must not turn app.py into a logic-heavy controller.
6. No silent coercion is allowed
Malformed, missing, inconsistent, or invalid input data must never be silently corrected, ignored, defaulted, truncated, or auto-normalized in a way that hides the issue. The system must remain explicit and auditable in all input handling.
7. Validation failure blocks scoring
If required inputs fail validation, scoring must not proceed. The system must fail clearly, descriptively, and non-silently. Partial scoring on invalid data is not permitted. Input integrity is a precondition for score generation.
Implementation rule for Phase 2
All Phase 2 enhancements must be additive, verifiable, and reversible. They may harden the system, expand controlled input coverage, and improve validation robustness, but they may not violate these locked Phase 1 contracts.


Phase 2 Validation Contract

All persisted CSV inputs must pass strict validation before scoring is permitted. Validation applies to tasks.csv, flags.csv, and all referenced config files. The system must verify required schema presence, non-blank identifiers, valid config references, parseable required numerics, valid timing and error domains, recognized flag types, valid task-linked flag references, and intern-task consistency across files. Validation is a blocking precondition for scoring. Malformed inputs must fail explicitly with descriptive error output. No row dropping, fallback defaults, or silent coercion are permitted.
