# Beyond Soft Prompts: Domain-Specific Layers as Parametric Implicit Prompts for Cross-Domain Few-Shot Learning
The code for Beyond Soft Prompts.
## quick start

### create ENV
```
conda create -n NEW
conda activate NEW
pip install -r requirements.txt
```

### run
**Noting:** before you start, you should download bert-base-uncased from https://huggingface.co/google-bert/bert-base-uncased, and change the path in the run_cross_domain_cross_intent_ours.sh file to your own file path.
The specific parameters per dataset in the paper are consistent with run_cross_domain_cross_intent_ours.sh.
```
sh run_cross_domain_cross_intent_ours.sh
```
