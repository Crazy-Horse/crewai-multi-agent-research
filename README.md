# Multi-Agent Financial Research (CrewAI)

A **CrewAI-based multi-agent research system** for financial and macroeconomic analysis.
This project demonstrates how **autonomous agents**, coordinated via **CrewAI**, can collaboratively extract market data, enrich it with macroeconomic and weather signals, and produce **ML-ready datasets** for quantitative research.

Designed for **financial research, data science, and applied AI engineering** use cases.

---

## One-Sentence Audience Framing

**For data scientists, quantitative analysts, and AI architects who want to build production-grade financial research agents using modern multi-agent orchestration.**

---

## Business Impact

This system enables:

* Faster macro + market research cycles
* Automated feature engineering pipelines
* Reproducible, auditable datasets for modeling
* Reduced analyst toil through agent specialization

It is especially valuable for **commodities, macro strategy, risk analysis, and alternative data research**.

---

## Architecture Overview (CrewAI)

In this implementation:

* Agents are **role-driven** (not graph-driven)
* Tasks are executed **sequentially**
* Tools encapsulate deterministic data operations
* State flows implicitly via task outputs

**Agent Roles**

* **Researcher** — frames the problem and research scope
* **Reporting Analyst** — executes data extraction and dataset construction

**Tools**

* Yahoo Finance (OHLCV)
* FRED + World Bank (macro & FX)
* Open-Meteo (multi-region weather)
* Dataset builder (feature engineering + targets)

---

## Key Features

* CrewAI multi-agent orchestration
* Role-based agent design
* Deterministic tool-driven data extraction
* Multi-region weather enrichment (Open-Meteo)
* Macro + FX integration (FRED + World Bank)
* ML-ready feature engineering
* Leakage-safe multi-horizon targets
* Fully reproducible execution

---

## Technologies Used

* **CrewAI**
* **Python 3.11+**
* **uv** (dependency & environment management)
* **Yahoo Finance**
* **FRED API**
* **World Bank API**
* **Open-Meteo API**
* **Pandas / NumPy**

---

## Quick Start (Using `uv` – Recommended)

This project **does not use pip**.
All dependencies are managed via **`uv`**.

---

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your shell, then verify:

```bash
uv --version
```

---

### 2. Clone the Repository

```bash
git clone <YOUR_CREWAI_REPO_URL>
cd multi-agent-research-crewai
```

---

### 3. Create and Sync the Virtual Environment

```bash
uv venv
uv sync
```

Activate the environment:

```bash
source .venv/bin/activate
```

---

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
FRED_API_KEY=your_fred_api_key
OPENAI_API_KEY=your_openai_api_key
```

Optional:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key
```

---

### 5. Run the Crew

```bash
uv run python main.py
```

Or, if exposed as a CLI:

```bash
uv run crewai run
```

The crew executes tasks sequentially:

1. Research framing
2. Market data extraction
3. Macro + FX enrichment
4. Weather enrichment (6 coffee regions)
5. Feature engineering
6. Dataset generation

---

## Output Artifacts

Generated outputs include:

```
data/raw/
data/processed/features_daily.csv
data/processed/features_daily_data_dictionary.md
```

These datasets are:

* Time-aligned
* Leakage-safe
* Suitable for linear or nonlinear models
* Reproducible across runs

---

## How This Differs from the LangGraph Version

| Dimension            | CrewAI               | LangGraph           |
| -------------------- | -------------------- | ------------------- |
| Control Flow         | Task-based           | Graph-based         |
| State                | Implicit             | Explicit            |
| Determinism          | Medium               | High                |
| Debuggability        | Moderate             | Strong              |
| Production Readiness | Fast prototyping     | Strong pipelines    |
| Best For             | Exploratory research | Regulated workflows |

This repo exists specifically to **compare CrewAI vs LangGraph using the same financial research problem**.

---

## Use Cases

* Commodity price modeling
* Macro-driven return prediction
* Weather-impact analysis
* Research automation
* Feature engineering pipelines
* AI-assisted financial research teams

---

## Why `uv`

This project uses **`uv`** instead of `pip` because:

* Faster dependency resolution
* Deterministic installs
* Built-in virtual environments
* Cleaner CI/CD
* Modern Python best practice

---

## Repository Structure

```
.
├── crew.py
├── main.py
├── tasks.yaml
├── tools/
│   ├── yahoo_finance_tools.py
│   ├── fred_tools.py
│   ├── open_meteo_tools.py
│   └── dataset_build_tools.py
├── data/
│   ├── raw/
│   └── processed/
├── pyproject.toml
└── README.md
```

---

## Related Projects

* **LangGraph version:** [LANGGRAPH_REPO_URL](https://github.com/Crazy-Horse/multi-agent-research)
* **Blog post:** `<BLOG_URL>`
* **Architecture diagrams:** `<DIAGRAMS_URL>`

---

## License

MIT License
