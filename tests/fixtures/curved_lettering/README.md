These tiny TrueType fonts contain original geometric rectangles made solely for
the curved-lettering regression tests. They contain no outlines or other data
copied from a commercial or system font and are covered by the repository's
license. They are not intended as readable typefaces.

`CurveTestPrimary.ttf` covers ASCII and includes native `ffi`, `fi`, and `fl`
ligatures. It deliberately lacks accented letters. `CurveTestFallback.ttf`
covers É, é, and a combining acute accent. Setting both families on a Qt font
forces same-script fallback without depending on fonts installed on the host.

Qt 6.11.2 can return zero source indexes for the fallback run and subsequent
primary-font run. The tests verify correct curved placement without depending
on that Qt bug remaining present in future versions. Ligature and combining-mark
tests also ensure that per-grapheme queries retain native shaping and paint each
glyph exactly once.

To regenerate the binaries, run `generate_fonts.py` with a Python environment
containing fontTools. fontTools is only needed for regeneration, not for running
the tests or the application. Timestamps are fixed for reproducible output.
