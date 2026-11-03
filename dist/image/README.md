# Put whole-slide images here

Place the TCGA BLCA `.svs` files you want to process in this folder:

```bash
cp /path/to/TCGA-*.svs .
```

Or download the 5 demo slides used by this package:

```bash
./scripts/utils/download_svs.sh --demo
```

Training needs at least 2 slides containing both FGFR3 mutant (label `1`) and
wild-type (label `0`) cases. The demo set is 3 MUT + 2 WT.
