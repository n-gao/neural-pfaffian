# Neural Pfaffians: Solving Many Many-Electron Schrödinger Equations

![Title](figures/title.png)

Reference implementation of Neural Pfaffians from <be>

<b>[Neural Pfaffians: Solving Many Many-Electron Schrödinger Equations](https://arxiv.org/abs/2405.14762)</b><br>
by Nicholas Gao, Stephan Günnemann<br/>
published as Oral at NeurIPS 2024.

## Installation
1. Install [`uv`](https://docs.astral.sh/uv/):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. Create a virtual environment and install dependencies
    ```sh
    uv sync
    source .venv/bin/activate
    ```

## Models
The code supports various models, FermiNet, PsiFormer, and Moon. In addition to classical Slater determinants and Pfaffian wave functions.
You can also freely configure your desired wave function by editing the modular configuration files.
Note that having no MetaGNN only permits single structure calculations.
Pfaffians as antisymmetrizer are required for running molecules with different nuclei and/or number of electrons.

For instance, to perform a single-structure calculation with PsiFormer run
```sh
neural_pfaffian with configs/models/psiformer.yaml configs/systems/single/lih.yaml
```
To run PESNet (MetaGNN + FermiNet) on the N2 potential energy surface run
```sh
neural_pfaffian with configs/models/pesnet.yaml configs/systems/pes/n2.yaml
```
By default, the code uses the Neural Pfaffian (MetaGNN + Moon + Pfaffian) which works for all molecular systems.

## Running the code
We encourage the use of `seml` to manage all experiments, but we also supply commands to run the experiments directly.
With `seml`:
```bash
seml n2_ablation add configs/seml/train_n2.yaml start
```
Without `seml`:
```bash
neural_pfaffian with configs/systems/n2.yaml
```

## Contact
Please contact [n.gao@tum.de](mailto:n.gao@tum.de) if you have any questions.

## Cite
Please cite our paper if you use our method or code in your own works:
```
@inproceedings{gao_pfaffian_2024,
    title = {Neural Pfaffians: Solving Many Many-Electron Schr\"odinger Equations},
    author = {Gao, Nicholas and G{\"u}nnemann, Stephan},
    booktitle = {Neural Information Processing Systems (NeurIPS)},
    year = {2024}
}
```