# Third-party code

The CNN architecture and preprocessing are derived from Greg Olmschenk and collaborators' RAMjET implementation accompanying [Olmschenk et al. (2021)](https://arxiv.org/abs/2101.10919), at [commit afe6f6cf8f0b96106f3eec48f68a1d7f1f0aefb9](https://github.com/golmschenk/qusi/tree/afe6f6cf8f0b96106f3eec48f68a1d7f1f0aefb9), also archived at [Zenodo](https://zenodo.org/records/4460662).

The Apache 2.0 license is reproduced in [licenses/RAMjET-Apache-2.0.txt](licenses/RAMjET-Apache-2.0.txt). Derived files are `model.py`, `preprocessing.py`, and `ordered_preprocessing.py` under `src/mdwarf_transits/`. Changes include explicit framework defaults, input validation, local random streams, provenance indices, raw-logit access, and explicit percentile arithmetic. The published architecture and training augmentation order are retained.

Other scientific libraries are installed as dependencies and retain their respective licenses. This notice does not grant a new license to the remaining project code.
