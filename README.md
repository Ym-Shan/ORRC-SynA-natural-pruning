# ORRC-SynA-natural-pruning
Implement spike-drive using OR-residual connection and propose SynA attention for natural pruning.

**Experimental Environment Configuration**

In fact, we utilized only PyTorch and SpikingJelly==0.0.0.14 for all our experiments. These experiments were conducted on a server personally configured by me, with the following hardware specifications:

- **CPU:** Intel Xeon(R) Gold 6133 * 2
- **GPU:** NVIDIA 3090 * 4
- **GPU Driver:** 530.41.03
- **CUDA Version:** 12.1
- **Operating System:** Ubuntu 18.04

Additional environment configurations can be found in the requirements.txt file (which may contain some unnecessary items).

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

### Citation
@article{shan2023or,
  title={OR Residual Connection Achieving Comparable Accuracy to ADD Residual Connection in Deep Residual Spiking Neural Networks},
  author={Shan, Yimeng and Qiu, Xuerui and Zhu, Rui-jie and Li, Ruike and Wang, Meng and Qu, Haicheng},
  journal={arXiv preprint arXiv:2311.06570},
  year={2023}
}
