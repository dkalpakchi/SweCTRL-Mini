# SweCTRL-Mini
The official repository for the resources connected to the Swedish language model SweCTRL-Mini

**Paper (preprint)**: https://arxiv.org/abs/2304.13994

**Technical note**: https://doi.org/10.5281/zenodo.7868205

**Website (you can partially mine training data there)**: ~~https://swectrl.dev/~~ We are looking for a new hosting provider. I will release the code to host the website locally around January 2024. If you need it more urgently, please create an issue in this repo, and we'll se what can be done.

**The model on the Huggingface Hub (BigScience Open RAIL-M license)**: https://huggingface.co/dkalpakchi/SweCTRL-Mini

## The roadmap for 2024
- [ ] Jan 2024 -- release of the code to host your own copy of the website
- [ ] Feb 2024 -- release of visualized annotations (preliminary)

## Use for token classification
SweCTRL-Mini was originally trained for text generation (for which it includes the pre-trained LM head). We haven't tested it for any other use case. That said, if you remove the LM head, the remainder of the network can be used as a learned feature encoder, for instance, for token classification. We have provided the starting point for such an experiment in `hf_addons/token_classification.py`, which can be initialized by invoking the following (for the 4-way classification):
```py
from hf_addons.token_classification import CTRLForTokenClassification
model = CTRLForTokenClassification.from_pretrained("dkalpakchi/SweCTRL-Mini", num_labels=4)
```

**NOTE:** We have not tested such an approach ourselves and do not know what kind of performance to expect. If you did so yourself, please reach out and tell us how it went! :)
