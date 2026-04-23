# Tax Employee KPI Diagnostic Dashboard

# What it does
The Tax Employee KPI Diagnostic Dashboard is a deterministic performance analytics system designed to evaluate and diagnose the performance of first-year tax interns during busy season. It converts raw task-level data such as time spent, errors, and workload characteristics into structured performance metrics, interpretable diagnostics, and actionable management insights.
Unlike traditional dashboards that stop at reporting metrics, this tool builds a full pipeline from scoring to explanation to decision support. It calculates performance using a strictly validated, configuration-driven scoring engine, then layers on attribution, cross-intern comparison, structured diagnostics, system-wide pattern detection, and finally manager-facing action recommendations. The result is a transparent, auditable system that helps identify not just who is underperforming, but why, and what should be done about it.

# Installation
Follow these steps to set up the project locally.

1. Clone the repository

git clone https://github.com/karyekss-wq/Tax_Employee_KPIDiagnostic.git
cd Tax_Employee_KPIDiagnostic

2. Create a virtual environment

python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

3. Install dependencies

pip install -r requirements.txt

If requirements.txt is missing, install core dependencies manually:

pip install streamlit pandas numpy

# Usage
Run scoring from terminal:
This executes the full scoring pipeline and writes outputs.

python project_output.py
Expected output:

Wrote outputs/scoring_output.json
Wrote outputs/task_metrics_output.csv

# Launch the dashboard UI
streamlit run app.py
Then open the local URL shown in terminal (typically):
http://localhost:8501

# Run tests
python3 -m unittest discover -s tests -p 'test_*.py'

Expected output:
Ran 30 tests in X.XXXs
OK

# Examples
Example 1 — Analyze a single intern
Launch the app
Select an intern from the sidebar
View:
Final Score
Efficiency, Accuracy, Output, Contribution
Diagnostic explanations
Example insight:
“Accuracy is the primary weakness driven by major error concentration.”

Example 2 — Cross-intern comparison
Navigate to:
Cross-Intern Insights
You will see:
Ranked leaderboard
Metric distributions
Outliers (lowest efficiency, highest error burden)
Variance detection
Example output:
Intern INT003 has the lowest efficiency score
Intern INT001 is the top performer by final score

Example 3 — Manager View (decision layer)
Navigate to:
Manager View
This page shows:
Executive summary
Total interns
Top performer
High-priority actions
Priority action queue
“Increase review focus for INT003 due to accuracy drag”
“Prioritize training for Class B efficiency overruns”
System patterns
“Accuracy weakness recurring across 60% of interns”
“Efficiency overruns concentrated in Class B tasks”
Intern snapshot
Each intern’s:
strength
weakness
action priority

# Known limitations
Current limitations
Data is CSV-based (no database or persistence layer)
No historical tracking of performance over time
No external integrations (e.g., tax software, time tracking systems)
UI is functional but not fully optimized for enterprise workflows
No simulation capability yet (what-if analysis)
Future improvements

# Phase 3 roadmap:
Scenario simulation (adjust assumptions and see score impact)
Historical tracking and trend analysis
Database-backed storage
Multi-team / multi-office aggregation
Product enhancements:
Manager-specific dashboards with filtering and prioritization
Exportable reports (PDF / CSV summaries)
Role-based access control
Advanced features:
Controlled AI summarization layer (non-decision-making)
Integration with tax software (UltraTax, etc.)
Provenance/audit layer for workflow integrity (original concept)

# Summary
This project is not just a KPI dashboard. It is a full performance intelligence system that:
scores work deterministically
explains outcomes transparently
identifies system-wide issues
generates actionable management decisions
All while maintaining strict validation, no silent assumptions, and full auditability.

