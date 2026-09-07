1. Find what type of metdata is used in the Nature paper
2. Then map those metadata that we have download from the following url

```Shell
curl --request GET   --url "https://api.gdc.cancer.gov/files/3caa2d31-55fc-4117-9c53-c15624e8927a?expand=cases,cases.samples,cases.samples.portions,cases.diagnoses,cases.demographic&pretty=true" >> data.txt
```
