# Beyond Soft Prompts: Domain-Specific Layers as Parametric Implicit Prompts for Cross-Domain Few-Shot Learning
The code for Beyond Soft Prompts.
## quick start

### create ENV
```
conda create -n LAQDA python=3.7
source activate LAQDA
pip install -r requirements.txt
```

### run
**Noting:** before you start, you should download bert-base-uncased from https://huggingface.co/google-bert/bert-base-uncased, and change the path in the run.sh file to your own file path.
The specific parameters per dataset in the paper are consistent with run.sh.
```
sh run.sh
```
