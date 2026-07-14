# Timeline deck

`timeline.pptx` is a PowerPoint rebuild of the claude.ai Design "Timeline Component"
(`Timeline Component.dc.html`, project `# Animated Timeline Component`), which shows the
9 `latex/tmp/timeline.md` milestones as an interactive, click-through card carousel with a
CSS-animated progress bar. PowerPoint can't run that component directly, so this deck
reconstructs it as 10 slides (1 title + 9 milestones) styled to match, using
`Timeline_Component_Deck.pptx` (a 1:1 px->EMU export of the live component, at 9525 EMU/px)
as the layout baseline.

Regenerate with `build_timeline_pptx.py` (needs `python-pptx`; not a `pyproject.toml`
dependency, install into `.venvD` or a separate venv):

```
.venvD/bin/pip install python-pptx
.venvD/bin/python latex/final_presentation/build_timeline_pptx.py
```

It writes `timeline.pptx` next to itself.

## Layout

Every position (title, date pill, headline, bullets, image-placeholder box, timeline track,
dots, date labels) is copied from `Timeline_Component_Deck.pptx`'s own EMU coordinates rather
than eyeballed, so the timeline track sits at the same relative height as the live
component (~598px of 1080px = 55.4% down the slide - the middle, not the bottom).

One deliberate simplification: the live component bottom-aligns top-anchored milestones
(CSS `flex-end`, content hugs the track, growing upward - exact height depends on the
browser's text wrapping for however many bullets that milestone has). pptx has no flexbox
equivalent, so every top-anchored block here is top-aligned instead, at the same fixed y
regardless of bullet count. Simpler and overlap-free, at the cost of not being pixel-identical
to the component for the top-anchored slides specifically.

Image-type milestones (Interim Presentation 1/2, Compute Unlock, Final Presentation) keep a
dashed placeholder box on the left for a real screenshot/chart. Its text column is 1050px
wide, past the 572px the baseline used for its short single-line placeholder headlines
("Product Launch") - long real headlines like "Unlock of Computational Resources" measure
~689px at 30pt bold (Liberation Sans, metric-compatible with Arial) and were wrapping into
the bullets below it at the baseline's original width.

## Scoped Morph transition

The ask: animate the timeline track/dots moving between slides (like the live component's
CSS `transition`), without PowerPoint's Morph also sliding/resizing the headline and bullet
text around, which is what a plain "apply Morph as the slide transition" does.

Morph has no per-shape opt-out in the UI or the file format - `<p:transition>` is one setting
for the whole slide. But Morph decides *what counts as the same object* across two slides
by matching shape identity, primarily by name, and PowerPoint has a documented escape hatch
for controlling that: give two shapes on consecutive slides an identical name starting with
`!!`, and Morph force-matches them (interpolating position/size/color) regardless of anything
else; a shape with no `!!`-name match on the adjacent slide is left unmatched and simply
cross-fades instead (Morph's default treatment of anything it can't match).

So only the timeline track, the progress fill bar, and each of the 9 dots get stable,
identical names across all 9 milestone slides:

- `!!tl_track`, `!!tl_fill`
- `!!tl_dot_1` .. `!!tl_dot_9` (one per milestone position, always drawn, only size/color
  change between active/inactive)

Every other shape (date pill, headline, bullets, image placeholder, date labels under the
dots) gets a unique name per slide (e.g. `content_headline_m6`), so Morph can never
accidentally match, say, slide 4's headline to slide 5's and slide it around - they always
just fade.

Result: advancing through the 9 milestone slides with Morph applied smoothly slides the
active dot along the track, grows/shrinks it, and animates the fill bar's width - while all
text content cuts/fades per slide, matching the ask.

The transition itself is written as raw OOXML (`build_timeline_pptx.py::add_morph_transition`)
since python-pptx has no API for it: `mc:AlternateContent` / `mc:Choice Requires="p159"` /
`p159:morph option="byObject"` (namespace `http://schemas.microsoft.com/office/powerpoint/2015/09/main`,
per the MS-PPTX open spec), falling back to a plain `p:fade` for readers that don't support
it. The title slide just uses plain Fade (nothing to morph into on slide 1).

Sources: [MS-PPTX: morph](https://learn.microsoft.com/en-us/openspecs/office_standards/ms-pptx/68d26d78-f7f5-47ab-835d-4e6c82ff39f0),
[MS-PPTX: Slide Transition Extensions](https://learn.microsoft.com/en-us/openspecs/office_standards/ms-pptx/22ebe6b5-2ade-43d9-977a-98fa194725c2),
[Morph transition: tips and tricks (Microsoft Support)](https://support.microsoft.com/en-us/office/morph-transition-tips-and-tricks-bc7f48ff-f152-4ee8-9081-d3121788024f),
[Selective morph transitions in PowerPoint (Office Watch)](https://office-watch.com/2023/selective-morph-transitions-in-powerpoint/),
[Exclamation-named objects (Indezine)](https://www.indezine.com/products/powerpoint/learn/animationsandtransitions/transitions/morph-exclamation-named-objects.html).

## What's verified vs. not

Checked without a PowerPoint install available in this environment: every XML part is
well-formed, `<p:sld>`'s child order (`cSld` -> `clrMapOvr` -> `transition`/`AlternateContent`)
is schema-correct on every slide, the file reloads cleanly through `python-pptx`, all 11
`!!`-names are present identically on all 9 milestone slides with no accidental collisions
among the per-slide content names, and no shape falls outside the slide bounds.

Not verified, since that needs an actual PowerPoint render: that Morph's effect-options
actually resolve to "Objects" and play as smooth dot/bar motion rather than falling back to
something else, and that text layout (line wraps, vertical spacing) looks right on screen.
Open it in real PowerPoint once before presenting from it.
