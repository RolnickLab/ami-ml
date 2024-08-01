# Global Moth Model
Research related to the development of a global moth species classification model for automated moth monitoring.

## Process
The below steps are carrried out to train a global model.  

1. **Fetch Leps Checklist**: Download the Lepidoptera taxonomy from GBIF ([DOI](https://www.gbif.org/occurrence/download/)).
2. **Fetch DwC-A**: Fetch the Darwin Core Archive from GBIF for the order Lepidoptera ([DOI](https://doi.org/10.15468/dl.6j5bzj)). 
3. **Curate Moth Checklist** (`prepare_gbif_checklist.py`): Clean and curate the Lepidoptera checklist to have only moth species. Remove all non-species taxa and butterfly families. A curated list is [here](https://docs.google.com/spreadsheets/d/1E6Zn2hXbHGMMAiPhtDXFO9_hDtl68lG5fx2vg0jyBvg/edit?usp=sharing).

