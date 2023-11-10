# ORRC-SynA-natural-pruning
Implement spike-drive using OR residual connection and propose SynA attention for natural pruning.

### Run DVSGesture on OR-Spiking ResNet
```python
CUDA_VISIBLE_DEVICE="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 OR_Spiking_ResNet_DvsGesture.py
```
### Run CIFAR10DVS on OR-Spiking ResNet
```python
CUDA_VISIBLE_DEVICE="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 OR_Spiking_ResNet_CIFAR10DVS.py
```
### Run MNIST on OR-Spiking ResNet
```python
CUDA_VISIBLE_DEVICE="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 OR_Spiking_ResNet_MNIST.py
```
### Run Fashion-MNIST on OR-Spiking ResNet
```python
CUDA_VISIBLE_DEVICE="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 OR_Spiking_ResNet_FashionMNIST.py
```
### Run CIFAR10 on OR-Spiking ResNet
```python
CUDA_VISIBLE_DEVICE="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 OR_Spiking_ResNet_CIFAR10.py
```

After a file completes its execution (completing the entire training and testing process), a folder named "result_data" will be generated to store the results. Within this folder, there will be .pt files containing the firing rate data for each neuron in every model of the network. You can utilize this data in the evaluation code located in the "evaluation" folder to calculate energy consumption and spike counts.
