**Note**: If you are referred from ICE-NODE paper, please follow the relevant instructions on the following snapshot of the codebase: [MLHC 2022 version](https://github.com/barahona-research-group/ICE-NODE/tree/mlhc2022).

## Roadmap

- [ ] Pipeline validators.
- [ ] Integrate consort diagramming in the pipeline.
- [ ] `lib.ehr.tvx*` test.
- [ ] `lib.ehr.coding_scheme.CodeMap` test.
- [ ] `lib.ehr.*` documentation / document edge cases tested.
- [ ] `lib.ehr` custom exceptions / adapt tests.
- [ ] FHIR resources adaptation.
- [ ] Support for SNOMED-CT.
- [ ] CLI for running pipelines.
- [ ] GUI for configuring the dataset and the tvx_ehr.
  - Pipeline 10 + 10 steps.
  - Selection of dataset CodingScheme space.


##### Coverage

| Branch | Coverage                                                                                                                                                   |
|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| main   | ![main_cov_ehr](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/A-Alaa/7c4939ecfd6b99a7b77dd1c4f789fd1b/raw/covbadge_main_ehr.json) |
| dev    | ![dev_cov_ehr](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/A-Alaa/f15bea7fb1837fba360e742b10244429/raw/covbadge_dev_ehr.json)   |


## Citation

To cite this work, please use the following BibTex entry:

```
@article{Alaa2022ICENODEIO,
  title={ICE-NODE: Integration of Clinical Embeddings with Neural Ordinary Differential Equations},
  author={Asem Alaa and Erik Mayer and Mauricio Barahona},
  journal={ArXiv},
  year={2022},
  volume={abs/2207.01873}
}
```
