# Codex Collaboration Log

This project was developed with assistance from AI tools (ChatGPT / Codex) as a collaborative development partner.

# AI was used for:

Structuring the scoring model and system architecture
Generating boilerplate code and refining logic
Debugging inconsistencies in dataset alignment
Designing validation rules and system constraints
Drafting documentation and test coverage

This app was implemented in 2 Phases with a Phase 3 plan

# Phase 1
Phase 1 involved building initial scoring algorithm and initial app logic.
No AI was used to build initial mathematic layer

# Phase 2
Phase 2 was conducted primarily in (ChatGPT/Codex) with the overarching strategy involved using ChatGPT as the prompt generator and verification layer and Codex acting as the executor. Prompt generation was safeguarded using initial maintainance contracts within project context, created post Phase 1 completion.

In conclusion, (ChatGPT/Codex) assisted in building, in order:

locked scoring
blocking validation
attribution
cross-intern comparison
diagnostic interpretation
diagnostic validation
normalized insights
cross-intern patterns
manager actions
manager-facing UI

# Phase 3

Phase 3 will involve connection an external database and LLM insights to complete the project into a presentable demo.

# Where AI required correction:

Initial suggestions included overly flexible data handling, which violated the strict validation requirement
Early implementations attempted silent normalization of task IDs, which conflicted with the “no silent coercion” rule
Some generated scoring logic required manual tightening to ensure deterministic and auditable behavior

# Key learnings:

AI accelerates iteration but must be constrained by clear system contracts
Strict validation and deterministic logic must be enforced manually
AI is most effective when used for scaffolding, not decision-making authority

All final system behavior, validation rules, and scoring integrity decisions were explicitly reviewed and enforced by the developer.

