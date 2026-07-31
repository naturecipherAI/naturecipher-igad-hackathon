# Assets

## Logo

| File | Use |
|---|---|
| `logo.svg` | `fill="currentColor"` — inherits the surrounding text colour. Use in the app, where a theme is in scope |
| `logo-teal.svg` | Fixed `#0F6E8C`. Use in the README and anywhere `currentColor` is stripped, which GitHub does inside `<img>` |
| `favicon.png` | Browser tab icon, already wired in `dashboard/index.html` |

The mark is also inlined directly in `dashboard/index.html` rather than linked, so
it paints before any network round-trip and picks up the active theme.

## GIFs — not yet recorded

Three slots are referenced by the root README. Drop the files in with these exact
names and they appear with no edit:

| File | Shows | Length |
|---|---|---|
| `demo-months.gif` | Scrubbing Sep → Oct → Nov; ASAL North turns rust on October | ~5s |
| `demo-threshold.gif` | Dragging the threshold 0.25 → 0.35; the drought signal disappears | ~6s |
| `demo-bulletin.gif` | Asking a question, then generating the bulletin | ~8s |

These have to be captured from a real browser session — they cannot be generated
from the repo.

**Capture settings.** 1280×800 window, browser zoom 100%, dark theme (the accent
reads stronger against it). Record with ScreenToGif on Windows, or record video
and convert:

```bash
ffmpeg -i capture.mp4 -vf "fps=12,scale=960:-1:flags=lanczos,split[a][b];\
[a]palettegen[p];[b][p]paletteuse" -loop 0 demo-months.gif
```

12 fps at 960 px keeps a 6-second clip near 3 MB, comfortably under GitHub's
limit and Devpost's 5 MB cap.

**What to actually record.** Let each state settle for a beat before moving — a
GIF that changes too fast reads as noise. On the threshold clip especially, hold
long enough at both 0.25 and 0.35 that the "N of 9 flagged" counter is readable.
That counter changing is the point of the shot.
