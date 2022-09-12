#Graph Extended Filtration Learning

## Summary
It is difficult for a standard message passing GNN with finite receptive field to keep track of arbitrary cycles of varied length for graph classification. We address this problem by using extended persistence with explicit cycle representatives in the readout function. Extended persistence is a quantifiable multiscale method to capture the relative prominence of cycles and connected components. We also improve the computation of extended perisistence by using a link-cut tree data structure to dynamically maintain cycle information and introduce parallelism. 

## Installation
* python==3.9.1
* torch==1.10.1
* CUDA==11.2
* GCC==7.5.0
* torch-geometric==2.0.4
* torch-scatter==2.0.9
* torch-sparse==0.6.15

For full requirements, see `requirements.txt`

```
pip install -r requirements.txt
```

## Running Experiments
Replace "proteins" with any of dd/mutag/imdb-multi/molbace/molbbbp/2cycles/pinwheels below to run experiment on any dataset

To run the proteins readout ablation experiment run the following commands (in parallel):

```
source proteins_extpers.sh

source proteins_sum.sh

source proteins_max.sh

source proteins_avg.sh
```

In general, change the --readout flag to the readout function of choice in the shell file.

To run the proteins filtration convolution length experiment, run the following commands (in parallel):
```
source proteinslen1_sup.sh

source proteinslen2_sup.sh

source proteinslen3_sup.sh

source proteinslen4_sup.sh

source proteinslen5_sup.sh
```

To run the standard message passing graph neural network baseline, run the command:

```
cd external_experiments/sup_baseline

source run_sup_baseline.sh
```

