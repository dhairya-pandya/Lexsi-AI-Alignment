# Project site

A single static page summarising the cross-lingual tamper study: the finding, every experiment we ran, the low-rank repair, and the conclusions. No build step, no framework, one `index.html` with inline styles and SVG charts. Same Apple-style design system as the repo figures (see [`../assets/DESIGN.md`](../assets/DESIGN.md)).

## Deploy to Vercel

This folder is the site root. Two ways:

**From the CLI**

```bash
cd site
npx vercel        # preview
npx vercel --prod # production
```

**From the dashboard**

Import the GitHub repo, then set **Root Directory** to `site`. Framework preset: *Other*. No build or install command needed.

## Preview locally

```bash
cd site
python3 -m http.server 8000   # then open http://localhost:8000
```

## Edit

Everything lives in `index.html`. The numbers come straight from the study (English 0–4% vs Bengali 59–61% harm; repair 50% to 14% at k=1; MMLU 0.55 to 0.56; anchoring 0.22/0.22/0.33). The two charts are hand-written SVG; colours are CSS variables that also carry the dark-mode palette.
