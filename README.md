# gwcosmo

A package to estimate cosmological parameters using gravitational-wave observations. 

If you use gwcosmo in a scientific publication, please cite [R. Gray et al. Phys. Rev. D 101, 122001](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.122001) and [R. Gray et al. arXiv:2111.04629](https://arxiv.org/abs/2111.04629), and include the following statement in your manuscript: "This work makes use of gwcosmo which is available at https://git.ligo.org/lscsoft/gwcosmo".

## How-to install

* Clone the `gwcosmo` repository with 
    ```
    git clone <repository>
    ```
    The name of the repository can be copied from the git interface (top right button). If you do not have ssh key on git, please use the `https` protocol
* Complete the install by following one of the options below. Note that `gwcosmo` requires Python version 3.7-3.9 to run.

### Installing with Anaconda

You will need an [Anaconda distribution](https://www.anaconda.com/). The conda distribution is correctly initialized when, if you open your terminal, you will see the name of the python environment used. The default name is `(base)`.

Once the conda distribution is installed and activated on your machine, please follow these steps:

* Enter the cloned gwcosmo directory.

* Create a conda virtual environment to host gwcosmo. Use
```
conda create -n gwcosmo
```
* When the virtual environment is ready, activate it with (your python distribution will change to `gwcosmo`)
```
conda activate gwcosmo
```
* Install `gwcosmo` by running 
```
python setup.py install
```
* You are ready to use `gwcosmo`. Note that, if you modify the code, you can easily reinstall it by using
```
python setup.py install --force
```

### Installing with pip and venv

`venv` is included in Python for versions >=3.3.

* Create a virtual environment to host gwcosmo. Use
```
python -m venv env
```
* When the virtual environment is ready, activate it with
```
source env/bin/activate
```
* Enter the cloned gwcosmo directory.
* Install `gwcosmo` by running 
```
pip install .
```
* Alternatively, if you are planning to modify `gwcosmo` run the following instead:
```
pip install -e .
```
The `-e` stands for "editable" and means that your installation will automatically update when you make changes to the code.
