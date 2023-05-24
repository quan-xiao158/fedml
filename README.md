# pFedML: A Personalized Federated Learning Repository Based on FedML

***Our project is based on the FedMl platform and expands on the algorithms of the FedMl platform,To use our expanded algorithm, please first learn how to use it on the FedML website,Click [here](https://doc.fedml.ai/) to jump to the user manual document for the FedMl platform.The following is an expanded algorithm classification.***

## Methods

### SP

#### FedBABU

* [[2106.06042\] FedBABU: Towards Enhanced Representation for Federated Image Classification (arxiv.org)](https://arxiv.org/abs/2106.06042)
* More personalized federated methods in more scenarios will be updated...

## Usage

* Please refer to [FedML](https://fedml.ai/ ).

## The Code Tree

* **core**: The FedML low-level API package. This package implements distributed computing by communication backend like MPI, NCCL, MQTT, gRPC, PyTorch RPC, and also supports topology management. 
  Other low-level APIs related to security and privacy are also supported. All algorithms and Scenarios are built based on the "core" package.

* **data**: FedML will provide some default datasets for users to get started. Customization templates are also provided.

* **model**: FedML model zoo.

* **device**: FedML computing resource management.

* **simulation**: FedML parrot can support (1) simulating FL using a single process (2) MPI-based FL Simulator (3) NCCL-based FL Simulator (fastest)

* **cross_silo**: Cross-silo Federated Learning for cross-organization/account training

* **cross_device**: Cross-device Federated Learning for Smartphones and IoTs

* **distributed**: Distributed Training: Accelerate Model Training with Lightweight Cheetah

* **serve**: Model serving, tailored for edge inference

* **mlops**: APIs related to machine learning operation platform (open.fedml.ai)

* **centralized**: Some centralized trainer code examples for benchmarking purposes.

* **utils**: Common utilities shared by other modules.
