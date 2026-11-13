# Design system — "Signal readout"

The visual language for this project's banner and figures. Built to look like a bilingual safety **instrument**, not a generated research-paper hero. The whole system exists to carry one idea: *the same model, measured in two languages, gives two verdicts.*

## Principle: color is the variable

Hue is never decoration here. It encodes the one thing the study is about — **language and its verdict**:

- **jade** = English / contained / refused
- **coral** = Bengali / exposed / answered
- **amber** = Hindi, and chart annotations

A reader who learns the key in the banner reads every later figure without a legend. That is the point.

## Tokens

```
--ground    #0C1922   deep teal-navy, the instrument ground (a choice, not tinted-black)
--panel     #102531   raised chart surface
--line      #244150   hairline / frame
--line-soft #1B3341   interior gridlines
--safe      #46D6A4   jade   — English / contained
--breach    #FF5C72   coral  — Bengali / exposed
--third     #F0B24A   amber  — Hindi / annotation
--text      #EAF2F4
--muted     #88A4B0
--faint     #5A7482
```

## Type

- **Archivo** (one Latin family, 400–900). Heavy tabular numerals are the data hero; regular/medium for UI. Using one family for display and body is deliberate.
- **Noto Sans Bengali / Devanagari** for native script, shown as *real content* (the same request in another language), never as ornament.
- Tabular figures (`font-variant-numeric: tabular-nums`) on every number so columns and percentages line up.

## What this system refuses

Written against the common tells of generated design, so none of these appear:

- no cream-and-terracotta serif hero; no black background with one neon accent
- no SaaS card kit — one shadow, one radius, on everything
- no ALL-CAPS tracked eyebrow labels; no `A · B · C` middle-dot meta; no `→` appended to text; no monospace "data" labels as chrome
- no decorative motion, no gradient washes used as filler

Instead: an instrument frame with corner crop ticks, a vertical seam splitting the two verdicts, hairline gridlines, and honesty as a design value — confidence intervals are drawn, and the **negative** result (prevention fails) gets equal, unhidden space.

## Boldness, spent once

The giant coral `59–61%` against the quiet jade `0–4%` is the one loud element. Everything around it stays disciplined.

## Regenerating the assets

The banner and results figure are authored as HTML (`banner.html`, `results.html`) and rasterized with headless Chrome so Google Fonts and the Bengali script render:

```bash
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"   # or any Chrome/Chromium
"$CHROME" --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
  --virtual-time-budget=5000 --window-size=1280,480 --screenshot=banner.png banner.html
"$CHROME" --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
  --virtual-time-budget=5000 --window-size=1120,1740 --screenshot=results.png results.html
```

Edit the HTML, re-run, done. The PNGs are committed so the README renders without a build step.
