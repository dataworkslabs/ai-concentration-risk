# AI Concentration Risk

This project maps foundation model dependencies across 199 enterprise AI vendors spanning 10 sectors. The goal is to measure how concentrated the enterprise AI ecosystem is around a small number of foundation model providers.

## Findings

- 79% of vendors publish no public information about what their product runs on
- Among the 21% that do disclose, OpenAI and Anthropic account for 85% of confirmed dependencies
- The Herfindahl-Hirschman Index (HHI) for the visible market is 3,950 — well above the 2,500 threshold used to define high concentration
- Concentration is cross-sectoral, appearing across productivity, legal, marketing, cybersecurity, and developer tools

## Method

1. Built a Python scraper (`extract_dependencies.py`) that fetches each vendor's website, engineering blog, job postings, press releases, and GitHub
2. Sent scraped text to the Anthropic Batch API to extract model dependencies with confidence scores and source excerpts
3. Validated results in two passes — checking model name appears in excerpt, and excerpt appears in source text
4. Manually reviewed and classified remaining results

## Files

- `extract_dependencies.py` — main data collection script
- `vendors.csv` — list of 199 vendors across 10 sectors
- `deps.sqlite` — SQLite database with all results
- `final_project.Rmd` — R Markdown analysis and visualizations

## Requirements

```
pip install -r requirements.txt
```

Set your Anthropic API key:
```
export ANTHROPIC_API_KEY=your_key_here
```

Run on a vendor CSV:
```
python extract_dependencies.py vendors.csv --db deps.sqlite
```

## Course

DATA 607 — Final Project  
CUNY School of Professional Studies  
May 2026
