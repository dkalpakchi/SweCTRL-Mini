# SweCTRL-Mini
The official repository for the resources connected to the Swedish language model SweCTRL-Mini

**Paper (preprint)**: https://arxiv.org/abs/2304.13994

**Technical note**: https://doi.org/10.5281/zenodo.7868205

**Website (you can partially mine training data there)**: https://swectrl.dev/  (temporarily down due to hardware failure)

**The model on the Huggingface Hub (BigScience Open RAIL-M license)**: https://huggingface.co/dkalpakchi/SweCTRL-Mini

## The roadmap for 2023
- [x] Apr 28th -- release of the associated paper and the technical note
- [x] May ~~3rd~~ 1st -- publishing the website for the model with the interface to search in its training data
- [x] May 8th -- release of the model with an accompanying license
- [x] May ~~1st~~ 11th -- release of the full version of the code for training and evaluating the model
- [x] May ~~25th~~11th -- release of raw annotations for human evaluation
- [ ] August/September -- release of visualized annotations (preliminary)

## Use for token classification
SweCTRL-Mini was originally trained for text generation (for which it includes the pre-trained LM head). We haven't tested it for any other use case. That said, if you remove the LM head, the remainder of the network can be used as a learned feature encoder, for instance, for token classification. We have provided the starting point for such an experiment in `hf_addons/token_classification.py`, which can be initialized by invoking the following (for the 4-way classification):
```py
from hf_addons.token_classification import CTRLForTokenClassification
model = CTRLForTokenClassification.from_pretrained("dkalpakchi/SweCTRL-Mini", num_labels=4)
```

**NOTE:** We have not tested such an approach ourselves and do not know what kind of performance to expect. If you did so yourself, please reach out and tell us how it went! :)
