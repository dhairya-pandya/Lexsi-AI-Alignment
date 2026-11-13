# Design system: Apple HIG

The visual language for this project's banner and figures, built on Apple's Human Interface Guidelines: **San Francisco** type, generous whitespace, a centered hero, **bento** cards, and Apple's own **system colors** carrying the safety semantics.

## Principle: Apple's system colors are the semantics

The finding is a safety verdict, so it uses Apple's safety palette directly, with no invented accent:

- **systemGreen** = English / contained / safe
- **systemRed** = Bengali / exposed / harmful
- **systemBlue** = section labels and links only

A reader fluent in Apple software already knows green-good / red-bad; the figures lean on that, not on a legend.

## Tokens

```
--page      #F5F5F7   Apple light-gray canvas
--card      #FFFFFF   bento card
--hair      #E8E8ED   dividers
--ink       #1D1D1F   label / primary text (Apple's near-black)
--gray      #86868B   secondary text
--blue      #0071E3   systemBlue, eyebrows and links
--green     #34C759   systemGreen, bars/lines/dots
--green-txt #248A3D   accessible green for text on white
--red       #FF3B30   systemRed, bars/lines/dots
--red-txt   #D70015   accessible red for text on white
```

Card radius **22px**. Grid gap **16px**. Whitespace is a material, not a leftover.

## Type

- **San Francisco** via `-apple-system, "SF Pro Display", "SF Pro Text"`. On macOS this is the real system font; rasterized with headless Chrome so it renders authentically.
- Scale follows Apple marketing: a large tight headline (weight 700, tracking −0.025em), a calm gray sub-headline (weight 400), SF for every label and number.
- Tabular figures on all data. A range like `0–4%` wraps the dash in a small, raised `.sep` span so it reads as "0 to 4", never as a strikethrough through the digits.

## Layout

- **Banner:** a centered Apple product-page hero: eyebrow, one short two-sentence headline (`One model. Two verdicts.`), a gray sub, then the two numbers split by a hairline divider. Everything centered, lots of air.
- **Results:** an Apple **bento grid**: white rounded cards on the gray canvas, mixed spans (a full-width hero, a full-width chart, a two-up row). Each card carries a blue eyebrow, a bold title, and one chart or one big number.

Charts are minimal: hairline gridlines (`#EDEDF0`), 3px rounded system-color lines, a faint red CI band, SF labels. No frames, no shadows stacked on everything, no decoration, the Apple restraint.

## Regenerating the assets

Authored as HTML (`banner.html`, `results.html`), rasterized with headless Chrome at 2× so SF Pro renders:

```bash
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"   # any Chrome/Chromium on macOS
"$CHROME" --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
  --virtual-time-budget=2500 --window-size=1280,500  --screenshot=banner.png  banner.html
"$CHROME" --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
  --virtual-time-budget=2500 --window-size=1120,1560 --screenshot=results.png results.html
```

Edit the HTML, re-run, done. The PNGs are committed so the README renders without a build step.
