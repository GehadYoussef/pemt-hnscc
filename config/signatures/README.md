# Gene signatures

One file per signature. One gene symbol per line. Lines starting with `#` are
treated as comments and skipped.

Filenames become signature names. For example, `puram_pemt.txt` becomes the
`puram_pemt` key in `load_signatures()`.

These six Puram et al. 2017 NMF meta-programs need to be populated from the
paper's Supplementary Table S5 before the pipeline can score cells:

- puram_pemt.txt
- puram_epi_dif_1.txt
- puram_epi_dif_2.txt
- puram_cell_cycle.txt
- puram_hypoxia.txt
- puram_stress.txt

Lineage markers (fibroblast_stromal, immune, endothelial) live in
`config/signatures.yaml` until you decide to override them here.
