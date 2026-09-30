# Gene signatures

One file per signature, with one gene symbol per line. Blank lines and lines
starting with `#` are skipped.

The file name becomes the signature name. For example, `puram_pemt.txt` is the
`puram_pemt` key returned by `pemt.load_signatures()`.

The six files hold 100 genes each from the Puram et al. 2017 NMF
meta-programs. The header of each file names the supplementary table it was
taken from.

- puram_pemt.txt
- puram_epi_dif_1.txt
- puram_epi_dif_2.txt
- puram_cell_cycle.txt
- puram_hypoxia.txt
- puram_stress.txt

The lineage and canonical marker sets (epithelial, fibroblast_stromal, immune,
endothelial, hypoxia_canonical, cell_cycle_canonical) are defined in
`config/signatures.yaml`. `load_signatures()` reads that file first and then
the `.txt` files here, so a `.txt` file with the same name replaces the YAML
entry.
