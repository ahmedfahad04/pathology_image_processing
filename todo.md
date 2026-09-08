1. Find what type of metdata is used in the Nature paper
2. Then map those metadata that we have download from the following url

```Shell
curl --request GET   --url "https://api.gdc.cancer.gov/files/3caa2d31-55fc-4117-9c53-c15624e8927a?expand=cases,cases.samples,cases.samples.portions,cases.diagnoses,cases.demographic&pretty=true" >> data.txt
```

## Findings

1. **Downloaded** Mutant images
2. Extracted almost all **metadata** for each of the `.svs` image
3. When we call a `.svs` image as mutant, it means the WSI is completely a mutant. We have to consider the whole pattern of the WSI (`.svs`). Initially the considered the WSI as mutant from the patient DNA sequencing values.
4. chr:1806089 G→T is NOT an x,y on the SVS.
   FGFR3_mutation_details: chr 4 start 1806089 end 1806089 is a DNA address on human chromosome 4 (GRCh37), not a pixel.
   Think: book page 4, line 1806089, letter G changed to T → protein G370C. You cannot see that letter change under H&E pink/purple stain.
   Your example TCGA-ZF-AA4U  G370C 91/85 VAF 0.52 means: bulk tumor DNA from the patient's FFPE tissue block had that typo in ~52% of reads. No x,y box is stored.
