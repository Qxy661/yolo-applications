# Quality Assessment - Wind Turbine Blade Defect Dataset

## Quality Metrics

### Image Quality

| Metric | Criterion | Action |
|--------|-----------|--------|
| Min size | 100x100 px | Remove if smaller |
| Format | RGB/RGBA/L | Convert or remove other modes |
| Corruption | PIL verify fails | Remove |
| Blur | Laplacian variance < threshold | Flag for review |

### Annotation Quality

| Metric | Criterion | Action |
|--------|-----------|--------|
| Format | YOLO txt (5+ values per line) | Fix or remove |
| Coordinate range | 0-1 normalized | Clamp or remove |
| Box area | > 0.1% of image | Flag tiny boxes |
| Box area | < 50% of image | Flag oversized boxes |
| Class ID | 0-4 valid range | Remap or remove |

## Known Issues

### Class Imbalance

The dataset has significant class imbalance:
- **Crack**: Over-represented (from QQ6 dataset)
- **Lightning**: Under-represented (only from Blade30)
- **Hole**: Under-represented

**Mitigation**: Use weighted loss functions (EIoU + class weights) during training.

### Category Mapping Challenges

Different datasets use different category names:
- Blade30: `contamination` → mapped to `erosion`
- QQ6: `dirt`, `oil_leakage` → mapped to `erosion`
- QQ5: `paint` → mapped to `peeling`

Some information loss during mapping. For example, "oil leakage" and "dirt" are different defects but mapped to same class.

### Data Quality Pipeline

```
Raw Data → clean_data.py → convert_format.py → merge_datasets.py → split_dataset.py
   ↓              ↓               ↓                    ↓                  ↓
 Download    Remove bad     Standardize          Deduplicate        Stratified
 from web    images         format               & remap            split
```

## Recommended Actions

1. **Run data cleaning first**: `python clean_data.py --input raw --output cleaned`
2. **Review flagged images**: Check images with quality issues
3. **Verify class mapping**: Ensure merged classes make sense
4. **Monitor training**: Watch for class-specific mAP differences
5. **Consider augmentation**: Use CopyPaste for under-represented classes

## Quality Report Format

The `clean_data.py` script generates `clean_report.json`:

```json
{
  "total_images": 9051,
  "valid_images": 8800,
  "invalid_images": 251,
  "duplicate_groups": 45,
  "annotation_issues": 120,
  "issues": ["image1.jpg: size too small", "..."]
}
```
