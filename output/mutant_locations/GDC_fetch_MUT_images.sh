#!/usr/bin/env bash
# Download the MUT SVS files that are NOT currently in image/
# Requires: pip install gdc-client OR use curl with GDC API token-free for open data
# Example uses gdc-client: gdc-client download -m manifest.txt
# Below: per-file curl fallback via GDC data endpoint (file_id needed)
set -e

# MUT TCGA-CF-A5U8-01  G370C/G372C  VAF 0.45  RSEM 12326
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-2F-A9KQ-01Z-00-DX1.1C8CB2DD-5CC6-4E99-A0F9-32A0F598F5F9.svs
# gdc-client: gdc-client download $(grep TCGA-2F-A9KQ-01Z-00-DX1.1C8CB2DD-5CC6-4E99-A0F9-32A0F598F5F9.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-2F-A9KQ-01Z-00-DX1.1C8CB2DD-5CC6-4E99-A0F9-32A0F598F5F9.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-CF-A5U8-01  G370C/G372C  VAF 0.45  RSEM 12326
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-CF-A5U8-01Z-00-DX1.D0385BD3-3128-41C6-889D-4EC916B2B228.svs
# gdc-client: gdc-client download $(grep TCGA-CF-A5U8-01Z-00-DX1.D0385BD3-3128-41C6-889D-4EC916B2B228.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-CF-A5U8-01Z-00-DX1.D0385BD3-3128-41C6-889D-4EC916B2B228.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-DK-AA6P-01  G370C/G372C  VAF 0.60  RSEM 2737
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-DK-AA6P-01A-01-TSA.E2677252-A0BA-4FB9-8F17-F53B14CABBA5.svs
# gdc-client: gdc-client download $(grep TCGA-DK-AA6P-01A-01-TSA.E2677252-A0BA-4FB9-8F17-F53B14CABBA5.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-DK-AA6P-01A-01-TSA.E2677252-A0BA-4FB9-8F17-F53B14CABBA5.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-E7-A7DU-01  G370C/G372C  VAF 0.58  RSEM 19293
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-E7-A7DU-01A-01-TSA.6C57A327-FD34-446C-B528-17AB52827881.svs
# gdc-client: gdc-client download $(grep TCGA-E7-A7DU-01A-01-TSA.6C57A327-FD34-446C-B528-17AB52827881.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-E7-A7DU-01A-01-TSA.6C57A327-FD34-446C-B528-17AB52827881.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-E7-A7DU-01  G370C/G372C  VAF 0.58  RSEM 19293
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-E7-A7DU-01Z-00-DX1.25C7F59E-1F03-47EF-9F17-DB9ADB16276E.svs
# gdc-client: gdc-client download $(grep TCGA-E7-A7DU-01Z-00-DX1.25C7F59E-1F03-47EF-9F17-DB9ADB16276E.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-E7-A7DU-01Z-00-DX1.25C7F59E-1F03-47EF-9F17-DB9ADB16276E.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-ZF-AA4U-01  G370C/G372C  VAF 0.52  RSEM 19663
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-ZF-AA4U-01A-01-TS1.831DB569-494E-45A9-A8D1-4748BF3ED5F6.svs
# gdc-client: gdc-client download $(grep TCGA-ZF-AA4U-01A-01-TS1.831DB569-494E-45A9-A8D1-4748BF3ED5F6.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-ZF-AA4U-01A-01-TS1.831DB569-494E-45A9-A8D1-4748BF3ED5F6.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-ZF-AA4U-01  G370C/G372C  VAF 0.52  RSEM 19663
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-ZF-AA4U-01Z-00-DX1.E7FBFD1C-E974-4B9C-8DA8-D14FEC22D117.svs
# gdc-client: gdc-client download $(grep TCGA-ZF-AA4U-01Z-00-DX1.E7FBFD1C-E974-4B9C-8DA8-D14FEC22D117.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-ZF-AA4U-01Z-00-DX1.E7FBFD1C-E974-4B9C-8DA8-D14FEC22D117.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-BT-A20T-01  G380R/G382R  VAF 0.42  RSEM 21598
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-BT-A20T-01A-01-TSA.aa2eb218-f0b5-49fc-b812-83c09ce3a2fc.svs
# gdc-client: gdc-client download $(grep TCGA-BT-A20T-01A-01-TSA.aa2eb218-f0b5-49fc-b812-83c09ce3a2fc.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-BT-A20T-01A-01-TSA.aa2eb218-f0b5-49fc-b812-83c09ce3a2fc.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-BT-A20T-01  G380R/G382R  VAF 0.42  RSEM 21598
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-BT-A20T-01Z-00-DX1.96460E53-65E0-425F-B079-939D7AA537BE.svs
# gdc-client: gdc-client download $(grep TCGA-BT-A20T-01Z-00-DX1.96460E53-65E0-425F-B079-939D7AA537BE.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-BT-A20T-01Z-00-DX1.96460E53-65E0-425F-B079-939D7AA537BE.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-G2-A2EO-01  G380R/G382R  VAF 0.52  RSEM 12480
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-G2-A2EO-01A-01-TSA.96ADE22A-FF0E-4D79-9FD9-E6860BE4D576.svs
# gdc-client: gdc-client download $(grep TCGA-G2-A2EO-01A-01-TSA.96ADE22A-FF0E-4D79-9FD9-E6860BE4D576.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-G2-A2EO-01A-01-TSA.96ADE22A-FF0E-4D79-9FD9-E6860BE4D576.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-G2-A2EO-01  G380R/G382R  VAF 0.52  RSEM 12480
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-G2-A2EO-01Z-00-DX1.E2FEF97D-F947-4497-8605-BBBFC3EC665B.svs
# gdc-client: gdc-client download $(grep TCGA-G2-A2EO-01Z-00-DX1.E2FEF97D-F947-4497-8605-BBBFC3EC665B.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-G2-A2EO-01Z-00-DX1.E2FEF97D-F947-4497-8605-BBBFC3EC665B.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-G2-A2EO-01  G380R/G382R  VAF 0.52  RSEM 12480
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-G2-A2EO-01Z-00-DX2.94CC403C-1986-4906-AF11-92E349C8A114.svs
# gdc-client: gdc-client download $(grep TCGA-G2-A2EO-01Z-00-DX2.94CC403C-1986-4906-AF11-92E349C8A114.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-G2-A2EO-01Z-00-DX2.94CC403C-1986-4906-AF11-92E349C8A114.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-G2-A2EO-01  G380R/G382R  VAF 0.52  RSEM 12480
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-G2-A2EO-01Z-00-DX3.F64AE021-27AB-4B3C-9DD2-9B8A358326C3.svs
# gdc-client: gdc-client download $(grep TCGA-G2-A2EO-01Z-00-DX3.F64AE021-27AB-4B3C-9DD2-9B8A358326C3.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-G2-A2EO-01Z-00-DX3.F64AE021-27AB-4B3C-9DD2-9B8A358326C3.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-G2-A2EO-01  G380R/G382R  VAF 0.52  RSEM 12480
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-G2-A2EO-01Z-00-DX4.BE1980E2-83B2-4E04-A6A4-9B71B15793B5.svs
# gdc-client: gdc-client download $(grep TCGA-G2-A2EO-01Z-00-DX4.BE1980E2-83B2-4E04-A6A4-9B71B15793B5.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-G2-A2EO-01Z-00-DX4.BE1980E2-83B2-4E04-A6A4-9B71B15793B5.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-G2-A2EO-01  G380R/G382R  VAF 0.52  RSEM 12480
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-G2-A2EO-01Z-00-DX5.85F4C6CF-E10B-4EF7-B477-D42E9F78247F.svs
# gdc-client: gdc-client download $(grep TCGA-G2-A2EO-01Z-00-DX5.85F4C6CF-E10B-4EF7-B477-D42E9F78247F.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-G2-A2EO-01Z-00-DX5.85F4C6CF-E10B-4EF7-B477-D42E9F78247F.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-G2-A2EO-01  G380R/G382R  VAF 0.52  RSEM 12480
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-G2-A2EO-01Z-00-DX6.8FBEA23E-DC4C-4420-A72A-FDDC0CC1F80A.svs
# gdc-client: gdc-client download $(grep TCGA-G2-A2EO-01Z-00-DX6.8FBEA23E-DC4C-4420-A72A-FDDC0CC1F80A.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-G2-A2EO-01Z-00-DX6.8FBEA23E-DC4C-4420-A72A-FDDC0CC1F80A.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-G2-A2EO-01  G380R/G382R  VAF 0.52  RSEM 12480
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-G2-A2EO-01Z-00-DX7.5784EBED-47AE-4ADA-84F2-3E58965EE2BC.svs
# gdc-client: gdc-client download $(grep TCGA-G2-A2EO-01Z-00-DX7.5784EBED-47AE-4ADA-84F2-3E58965EE2BC.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-G2-A2EO-01Z-00-DX7.5784EBED-47AE-4ADA-84F2-3E58965EE2BC.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-G2-A2EO-01  G380R/G382R  VAF 0.52  RSEM 12480
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-G2-A2EO-01Z-00-DX8.04D3920F-D1FD-4B37-8229-877FC64C3492.svs
# gdc-client: gdc-client download $(grep TCGA-G2-A2EO-01Z-00-DX8.04D3920F-D1FD-4B37-8229-877FC64C3492.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-G2-A2EO-01Z-00-DX8.04D3920F-D1FD-4B37-8229-877FC64C3492.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-ZF-A9RG-01  K652E (K650E)  VAF 0.50  RSEM nan
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-ZF-A9RG-01A-02-TSB.B2FD6615-0F9B-4BAD-BED6-3850872A9E05.svs
# gdc-client: gdc-client download $(grep TCGA-ZF-A9RG-01A-02-TSB.B2FD6615-0F9B-4BAD-BED6-3850872A9E05.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-ZF-A9RG-01A-02-TSB.B2FD6615-0F9B-4BAD-BED6-3850872A9E05.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# MUT TCGA-ZF-A9RG-01  K652E (K650E)  VAF 0.50  RSEM nan
# GDC portal search: https://portal.gdc.cancer.gov/files/TCGA-ZF-A9RG-01Z-00-DX1.E9C92201-31AD-4D7E-87C0-64842D705380.svs
# gdc-client: gdc-client download $(grep TCGA-ZF-A9RG-01Z-00-DX1.E9C92201-31AD-4D7E-87C0-64842D705380.svs gdc_manifest.txt | cut -f1)
echo "To fetch TCGA-ZF-A9RG-01Z-00-DX1.E9C92201-31AD-4D7E-87C0-64842D705380.svs: go to https://portal.gdc.cancer.gov/files and search file_name"

# For bulk: use output/tcga_blca_file_names.txt and scripts/analysis/fetch_cbioportal_data.py already does metadata; for SVS binary use GDC API:
# curl 'https://api.gdc.cancer.gov/files/<FILE_UUID>/data' -o image/<FILE_NAME>.svs