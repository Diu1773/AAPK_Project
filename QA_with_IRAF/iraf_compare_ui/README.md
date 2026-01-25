# IRAF vs AAPKI Photometry QA

A small PyQt5 app to compare IRAF photometry output (`*.txt` from `txdump`)
with AAPKI photometry output (`*_photometry.tsv`).

## Run

```bash
python main.py
```

## Expected inputs

- **AAPKI result dir**: a folder that contains `*_photometry.tsv` files
  (example: `.../data/M38/result`).
- **IRAF result dir**: a folder that contains `*.txt` files produced by
  `iraf.txdump` (example: `.../QA_with_IRAF/PyRAF/M38/result`).

## Notes

- Matching is done by XY position using a KDTree (tolerance in pixels).
- AAPKI positions can be matched using `xcenter/ycenter` or `x_init/y_init`.
- Export writes `iraf_compare_all.csv` to the chosen output directory.
