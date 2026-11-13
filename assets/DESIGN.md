# Design system: ink on paper

The visual language for this project's site and figures. Ink on warm paper, a tall high-contrast serif for voice, a monospace for data, hairline rules and square structure, with one 3x3 grid mark as the motif. Colour carries meaning, never decoration, and there is exactly one loud moment per view.

Adapted from the *Udte-Bhawre* design language (its paper chrome and "bold in one place" rule), retuned for a safety-evaluation study.

## Principle: colour is the verdict

Two system colours do all the semantic work, so a reader never needs a legend:

- **forest green `#1F7A4D`** = English / contained / safe
- **rust `#B23B2A`** = Bengali / exposed / harmful
- amber `#B07A1E` = a null or partial result; faint gray for a control

Everything else is ink on paper. The single boldest element in any view (the Bengali `59–61%` serif numeral) is the one place the eye is pulled.

## Tokens

```
--paper      #FAFAF8   warm paper canvas
--surface    #FFFFFF   card
--sink       #F3F2EE   track / code well
--ink        #111111   primary text
--muted      #5F5F5A   secondary text
--faint      #9A9A94   ticks, captions
--rule       #E4E4DE   hairline borders and gridlines
--up         #1F7A4D   forest green  (safe / English)
--down       #B23B2A   rust          (harmful / Bengali)
--accent     #1A3C2A   deep forest   (one bold moment, button hover)
```

A full dark theme mirrors every token (paper `#121210`, ink `#ECEBE5`, up `#5CC98F`, down `#EF7D6B`), applied by `prefers-color-scheme` and a `data-theme` override.

Cards are **square-ish**: 6px radius, **1px solid `--rule`** border, no shadows, no gradients. Whitespace and hairlines do the separating.

## Type

- **Fraunces** (optical-sizing on) for every headline and every large numeral: tall, high-contrast, italic for the second half of a title. The free stand-in for *Awesome Serif*.
- **Geist** for body and sub-text.
- **Geist Mono** for eyebrows, labels, ticks, tags, and small data, the "counter voice". Eyebrows are mono, uppercase, letter-spaced, in rust.

All data uses tabular figures. A range like `0–4%` wraps the dash in a small raised `.sep` span so it reads as "0 to 4", never as a strikethrough.

## Motif

A **3x3 grid mark** (nine cells, one lit rust) sits by the wordmark and each eyebrow. Data markers are **squares**, not dots, echoing the grid. Charts are minimal: hairline gridlines, 2.5px rust/forest lines, a faint rust CI band, mono labels. No frames stacked on frames.

## Regenerating the assets

Authored as HTML, rasterised with headless Chrome at 2x (Fraunces, Geist and Geist Mono load from Google Fonts):

```bash
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
  --virtual-time-budget=3000 --window-size=1280,500  --screenshot=banner.png  banner.html
"$CHROME" --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
  --virtual-time-budget=3000 --window-size=1120,1560 --screenshot=results.png results.html
```

The live site (`site/index.html`) and the animation (`anim/repair_scene.py`) use the same tokens. Edit the HTML, re-run, done. The PNGs are committed so the README renders without a build step.
