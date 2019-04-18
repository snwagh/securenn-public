# SecureNN: 3-Party Secure Computation for Neural Network Training

#### Sameer Wagh, Divya Gupta, and Nishanth Chandran

Secure multi-party computation (SMC/MPC) provides a cryptographically secure framework for computations where the privacy of data is a requirement. MPC protocols enable computations over this shared data while providing strong privacy guarantees – the parties only learn output of the computation while learning nothing about the individual inputs. Here we develop a framework for efficient 3-party protocols tailored for state-of-the-art neural networks. SecureNN builds on novel modular arithmetic to implement exact non-linear functions while avoiding the use of interconversion protocols as well as general purpose number theoretic libraries. 

<p align="center">
<img align="middle" src="https://snwagh.github.io/public/Images/flow.png" width="500" >
</p>

We develop and implement efficient protocols for the above set of functionalities. This work is published in [Privacy Enhancing Technologies Symposium (PETS) 2019](https://petsymposium.org/2019/). Paper available [here](http://snwagh.github.io/publications/).

### Table of Contents

- [Requirements](#requirements)
- [SecureNN Source Code](#securenn-source-code)
    - [Repository Structure](#repository-structure)
    - [Building SecureNN](#building-securenn)
    - [Running SecureNN](#running-securenn)
- [Additional Resources](#additional-resources)
    - [Neural Networks](#neural-networks)
    - [Debugging](#debugging)
    - [Citation](#citation)



### Requirements
---
* The code should work on any Linux distribution of your choice (It has been developed and tested with [Ubuntu](http://www.ubuntu.com/) 16.04 and 18.04).

* **Required packages for SecureNN:**
  * [`g++`](https://packages.debian.org/testing/g++)
  * [`make`](https://packages.debian.org/testing/make)
  * [`libssl-dev`](https://packages.debian.org/testing/libssl-dev)

  Install these packages with your favorite package manager, e.g, `sudo apt-get install <package-name>`.


### SecureNN Source Code
---

#### Repository Structure

* `files/`    - Shared keys, IP addresses and data files.
* `lib_eigen/`    - [Eigen library](http://eigen.tuxfamily.org/) for faster matrix multiplication.
* `mnist/`    - Parsing code for converting MNIST data into SecureNN format data.
* `src/`    - Source code for SecureNN.
* `utils/` - Dependencies for AES randomness.

#### Building SecureNN

To build SecureNN, run the following commands:

```
git clone https://github.com/snwagh/SecureNN.git
cd SecureNN
make
```

#### Running SecureNN

SecureNN can be run either as a single party (to verify correctness) or as a 3 (or 4) party protocol. It can be run on a single machine (localhost) or over a network. Finally, the output can be written to the terminal or to a file (from Party *P_0*). The makefile contains the promts for each. To run SecureNN, run the appropriate command after building (a few examples given below). 

```
make standalone
make abcTerminal
make abcFile
```



### Additional Resources
---
#### Neural Networks

SecureNN currently supports three types of layers, fully connected, convolutional (without padding), and convolutional layers (with zero padding). The network can be specified in `src/main.cpp`. The core protocols from SecureNN are implemented in `src/Functionalities.cpp`. The code supports both training and testing. 

#### Debugging

A number of debugging friendly functions are implemented in the library. For memory bugs, use [valgrind](http://www.valgrind.org), install using `
sudo apt-get install valgrind`. Then run a single party in debug mode:

```
* Set makefile flags to -g -O0 (instead of -O3)
* make clean; make
* valgrind --tool=memcheck --leak-check=full --track-origins=yes --dsymutil=yes <executable-file-command>
```

libmiracl.a is compiled locally, if it throws errors, download the source files from [https://github.com/miracl/MIRACL.git](https://github.com/miracl/MIRACL.git) and compile miracl.a yourself and copy into this repo.

Matrix multiplication assembly code only works for Intel C/C++ compiler. Use the non-assembly code from `src/tools.cpp` if needed (might have correctness issues).

#### Citation
You can cite the paper using the following bibtex entry:
```
@article{wagh2019securenn,
  title={{S}ecure{NN}: 3-{P}arty {S}ecure {C}omputation for {N}eural {N}etwork {T}raining},
  author={Wagh, Sameer and Gupta, Divya and Chandran, Nishanth},
  journal={Proceedings on Privacy Enhancing Technologies},
  year={2019}
}
```

---
Report any bugs to [swagh@princeton.edu](swagh@princeton.edu)
