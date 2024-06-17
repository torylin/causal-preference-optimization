
# (DR-)CPO

*(Doubly robust) causal preference optimization* is a method for optimizing language models to generate texts consistent with human preferences using direct outcome datasets, or datasets consisting of texts associated with numerical outcomes. Importantly, (DR-)CPO reframes language model optimzation as a causal inference problem and introduces two optimization approaches that solve unbiased surrogates for this problem. For more information, please see our UAI 2024 paper, [Optimizing Language Models for Human Preferences is a Causal Inference Problem](https://arxiv.org/pdf/2402.14979).

## Usage
 
To run the methods we introduce in the paper—CPO, DR-CPO, and OO-RLHF—please clone this repository and use the template found in `run_template.sh`.
