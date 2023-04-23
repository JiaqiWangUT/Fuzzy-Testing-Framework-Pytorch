# Fuzzy Testing Framework
This repository contains the PyTorch implementation of FuzzyTesting Framework of DNN, especially for image classification tasks.

### The structure of the repository

Our original framework is implemented based on several Fuzzy Testing Framework such as DeepHunter, DeepGauge, DeepTest. Besides, we implemented decision oracle to replace classification oracle in order to generate more meaningful corner cases/uncommon cases

├── README.md

├── captum/

├── gen_input/

├── data_loader.py

├── image_transforms.py

├── mutate.py

├── Mutation_Strategy.py

├── NeuronCoverage.py

├── style_operator.py

├── utils.py

└── requirements.txt

#### Mutation_Strategy.py
Mutation_Strategy.py contains the core implementation of fuzzy testing framework

#### captum/
This directory contains several XAI algorithms, we use one of them to generate visual concept.

#### gen_input/
This directory contains new generated test set.

#### data
data_loader.py: load local data such as imageNet.

#### Mutation
image_transforms.py contains pixel-level mutation and affine transformation.
style_operator.py contains style transfer.
torchattacks contains several adversarial attack methods.
imgaug can be used to generate weather filter.

#### Coverage
NeuronCoverage.py contains several NeuronCoverage Metrics

