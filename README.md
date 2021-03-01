# flood-pedestrian-simulator

The flood-pedestrian simulator is an agent-based model that can be used to model evacuation of large crowds in small urban areas. It can simulate the dynamic interactions that may occure between individuals and floodwater at a microscopic level.

The simulator consists of source codes that is used to dynamically couple a hydrodynamic model to a pedestrian model within the [FLAMEGPU](http://www.flamegpu.com/) framework. The pedestrian model is represented by formulation of a standard social force model with pedestrian agents representing evacuees that can move continuously in space and time. The hydrodynamic model includes a grid of fixed agents, called flood agent, that represents the computational grid for a flood model based on solution of shallow water equations. More information about the development stages and evaluation of the simulator can be found in [Shirvani et al. (2020)](https://iwaponline.com/jh/article/22/5/1078/75432/Agent-based-modelling-of-pedestrian-responses) and [Shirvani et al. (2021)](https://onlinelibrary.wiley.com/doi/abs/10.1111/jfr3.12695). 


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
By clonnig/downloading this repo's content you will have access to the flood-pedestrian source code, functions.c, and descriptions of agents, XMLModelFile.xml required to build and run a model on FLAMEGPU framework.

### Prerequisites
Before clone/download, consider the following things that you need. 

#### Required software
+ **FLAMEGPU v1.5** - you can get it from [FLAMEGPU master branch on GitHub](https://github.com/FLAMEGPU/FLAMEGPU) on your local machine - For more information about FLAMEGPU and download the latest Technical Report and User Guide go to http://www.flamegpu.com/.
+ **Visual Studio 2015 or earlier** - you can download the latest version of the Visual Studia from [Microsoft website](https://visualstudio.microsoft.com/downloads/). 
+ **CUDA Toolkit 10.1** - You can download it from [Nvidia's developer download archive](https://developer.nvidia.com/cuda-10.1-download-archive-base) - alternatively you may use later versions, but it needs extra manual modification to the solution file (see note 1).

#### Required hardware
+ **Nvidia Graphics card** - the simulator should be able to run on any Nvidia Graphics Card with a minimum 2GB memory installed on a normal machine.

### Installing
Installation steps goes here. 

## Running the tests

Explain how to run the already created tests with initial conditions 

### Break down

Explain what these tests test and why
```
```

```
Give examples
```

### To know more about FLAMEGPU reade FLAMEGPU.md file and visit http://www.flamegpu.com
