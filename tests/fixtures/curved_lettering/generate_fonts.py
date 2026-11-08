"""Regenerate original geometric font fixtures (fontTools is generation-only)."""
from pathlib import Path

from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

DEST = Path(__file__).resolve().parent


def build(name, codepoints, filename, ligatures=False):
    builder = FontBuilder(1000, isTTF=True)
    names = ['.notdef'] + [f'uni{cp:04X}' for cp in codepoints]
    if ligatures:
        names += ['f_f_i', 'f_i', 'f_l']
    builder.setupGlyphOrder(names)
    builder.setupCharacterMap({cp: f'uni{cp:04X}' for cp in codepoints})
    glyphs, metrics = {}, {}
    for index, glyphname in enumerate(names):
        pen = TTGlyphPen(None)
        if glyphname != 'uni0020':
            # Original rectangles, with varying widths; no existing font data.
            right = 300 + index % 5 * 40
            if glyphname.startswith('f_'):
                right = 1000
            bottom, top = (720, 820) if glyphname == 'uni0301' else (0, 700)
            pen.moveTo((50, bottom))
            pen.lineTo((right, bottom))
            pen.lineTo((right, top))
            pen.lineTo((50, top))
            pen.closePath()
        glyphs[glyphname] = pen.glyph()
        advance = 1200 if glyphname.startswith('f_') else 600
        if glyphname == 'uni0020':
            advance = 300
        elif glyphname == 'uni0301':
            advance = 0
        metrics[glyphname] = (advance, 50)
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics(metrics)
    builder.setupHorizontalHeader(ascent=900, descent=-200)
    builder.setupNameTable({
        'familyName': name, 'styleName': 'Regular',
        'uniqueFontIdentifier': name + '-1',
        'fullName': name + ' Regular',
        'psName': name.replace(' ', '') + '-Regular',
        'version': 'Version 1.0',
        'copyright': 'Original geometric test outlines for Local Comic Translate.',
    })
    builder.setupOS2(sTypoAscender=900, sTypoDescender=-200,
                     usWinAscent=900, usWinDescent=200)
    builder.setupPost()
    if ligatures:
        addOpenTypeFeaturesFromString(builder.font, '''
            languagesystem DFLT dflt;
            languagesystem latn dflt;
            feature liga {
                sub uni0066 uni0066 uni0069 by f_f_i;
                sub uni0066 uni0069 by f_i;
                sub uni0066 uni006C by f_l;
            } liga;
        ''')
    builder.font.recalcTimestamp = False
    builder.font['head'].created = builder.font['head'].modified = 2082844800
    builder.save(DEST / filename)


if __name__ == '__main__':
    build('Curve Test Primary', list(range(32, 127)), 'CurveTestPrimary.ttf', ligatures=True)
    build('Curve Test Fallback', [0x00C9, 0x00E9, 0x0301], 'CurveTestFallback.ttf')
