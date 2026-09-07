## Essential Commands

```Shell
python3 -u scripts/fetch_cbioportal_data.py --file-list data/tcga_blca_file_names.txt --outdir data --limit 5
python3 -u scripts/fetch_cbioportal_data.py --file-list data/tcga_blca_slides.tsv --limit 10 --batch-size 5
# or short
python3 -u scripts/fetch_cbioportal_data.py -n 3 --file-list data/tcga_blca_file_names.txt
# from local images
python3 -u scripts/fetch_cbioportal_data.py --image-dir image --limit 2
```
