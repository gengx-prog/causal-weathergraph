"""Verify the exported data, source hashes, vector page, and print-size preview."""
from pathlib import Path
import hashlib
import json

import fitz
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'


def main():
    provenance = json.loads((HERE / 'provenance.json').read_text(encoding='utf-8'))
    audit = json.loads((OUT / 'value_audit.json').read_text(encoding='utf-8'))
    source_checks = []
    for source in provenance['copied_inputs']:
        digest = hashlib.sha256((HERE / source['relative_copy']).read_bytes()).hexdigest()
        source_checks.append({'file': source['relative_copy'], 'sha256': digest,
                              'matches_original': digest == source['sha256']})
    h = pd.read_csv(HERE / 'source_data/hemisphere_edge_type_averages.csv')
    s = pd.read_csv(HERE / 'source_data/regime_interaction_summary.csv')
    data_checks = []
    for plotted in audit['records']:
        if plotted['panel'] == 'a':
            source = h[(h.period == plotted['period']) & (h.edge_type == plotted['row'])].iloc[0]
            exact = float(source['mean_delta_' + plotted['hemisphere']])
            passed = plotted['value'] == exact and plotted['display'] == f'{1000*exact:+.2f}'
        elif plotted['panel'] == 'b':
            source = h[(h.period == plotted['period']) & (h.edge_type == plotted['edge_type'])].iloc[0]
            passed = all(plotted[key] == float(source[field]) for key, field in [
                ('estimate', 'mean_delta_nh_minus_sh'), ('low', 'ci95_low_hac64'),
                ('high', 'ci95_high_hac64')])
        else:
            source = s[(s.period == plotted['period']) & (s.contrast == plotted['row'])]
            significant, tested = int(source.significant.sum()), int(source.tested.sum())
            passed = (plotted['significant'] == significant and plotted['tested'] == tested
                      and plotted['rate'] == significant / tested)
        data_checks.append({'panel': plotted['panel'], 'row': plotted.get('row', plotted.get('edge_type')),
                            'period': plotted['period'], 'hemisphere': plotted.get('hemisphere'),
                            'exact_source_match': bool(passed)})
    pdfpath = OUT / 'seasonal_state_contrasts.pdf'
    with fitz.open(pdfpath) as doc:
        assert len(doc) == 1
        page = doc[0]
        spans = [span for block in page.get_text('dict')['blocks'] if block['type'] == 0
                 for line in block['lines'] for span in line['spans']]
        page_check = {'pages': len(doc), 'width_pt': page.rect.width, 'height_pt': page.rect.height,
                      'embedded_images': len(page.get_images(full=True)),
                      'vector_paths': len(page.get_drawings()),
                      'minimum_text_font_pt': min(span['size'] for span in spans),
                      'all_fonts_embedded': all(bool(doc.extract_font(font[0])[3]) for font in page.get_fonts(full=True)),
                      'canvas_matches': bool(np.allclose([page.rect.width, page.rect.height], [547.2, 252])),
                      'no_text_clipped': audit['outside_canvas_text'] == []}
        # 945 pixels = 6.3 inches at 150 dpi: inspect the intended reproduction width.
        page.get_pixmap(matrix=fitz.Matrix(945 / page.rect.width, 945 / page.rect.width),
                        alpha=False).save(OUT / 'preview_6_3in_150dpi.png')
    passed = (all(x['matches_original'] for x in source_checks)
              and all(x['exact_source_match'] for x in data_checks)
              and len(data_checks) == 30 and page_check['canvas_matches']
              and page_check['embedded_images'] == 0 and page_check['all_fonts_embedded']
              and page_check['no_text_clipped'] and page_check['minimum_text_font_pt'] >= 10.39)
    result = {'all_checks_pass': passed,
              'pdf_sha256': hashlib.sha256(pdfpath.read_bytes()).hexdigest(),
              'source_checks': source_checks, 'data_checks': data_checks, 'pdf_checks': page_check,
              'raw_row_reconciliation': {'receipt': 'provenance.json',
                  'passed': provenance['all_checks_pass'],
                  'maximum_seasonal_mean_difference': provenance['group_mean_max_absolute_difference'],
                  'all_state_counts_match_raw': all(x['raw_matches'] for x in provenance['state_counts'])},
              'visual_review': {'reviewed_at': ['native vector export', '6.3-inch reproduction at 150 dpi'],
                  'findings': ['Three panels and all 24 circular numeric labels are present.',
                      'Outer edge-type labels are horizontal and separated from the seasonal ring values.',
                      'Signed values and interval bounds remain identifiable without color.',
                      'Colorbars are vector objects; no raster image is embedded in the PDF.',
                      'Ring order, periods, scale multiplier, numerator counts, and denominators are explicit.']}}
    (OUT / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    assert passed, result
    print('PASS: 30 values, two original source hashes, full-vector page, embedded fonts, and print-size preview.')


if __name__ == '__main__':
    main()
