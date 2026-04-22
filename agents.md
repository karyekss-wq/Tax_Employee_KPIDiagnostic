# Codex Collaboration Log

This project was developed with assistance from AI tools (ChatGPT / Codex) as a collaborative development partner.

# AI was used for:

Structuring the scoring model and system architecture
Generating boilerplate code and refining logic
Debugging inconsistencies in dataset alignment
Designing validation rules and system constraints
Drafting documentation and test coverage

# Where AI required correction:

Initial suggestions included overly flexible data handling, which violated the strict validation requirement
Early implementations attempted silent normalization of task IDs, which conflicted with the “no silent coercion” rule
Some generated scoring logic required manual tightening to ensure deterministic and auditable behavior

# Key learnings:

AI accelerates iteration but must be constrained by clear system contracts
Strict validation and deterministic logic must be enforced manually
AI is most effective when used for scaffolding, not decision-making authority

All final system behavior, validation rules, and scoring integrity decisions were explicitly reviewed and enforced by the developer.

