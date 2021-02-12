# GWcosmo

A package to estimate cosmological parameters using gravitational-wave observations. If you use GWcosmo in a scientific publication, please cite 

```
@article{Gray:2019ksv,
    author = "Gray, Rachel and others",
    title = "{Cosmological inference using gravitational wave standard sirens: A mock data analysis}",
    eprint = "1908.06050",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    reportNumber = "LIGO-P1900017",
    doi = "10.1103/PhysRevD.101.122001",
    journal = "Phys. Rev. D",
    volume = "101",
    number = "12",
    pages = "122001",
    year = "2020"
}
```

## How-to install

You will need an [Anaconda distribution](https://www.anaconda.com/). The conda distribution is correctly initialized when, if you open your terminal, you will see the name of the python environment used. The default name is `(base)`.

Once the conda distribution is installed and activated on your machine, please follow these steps:

* Clone the gwcosmo repository with 
    ```
    git clone <repository>
    ```
    the name of the repository can be copied from the git interface (top right button). If you do not have ssh key on git, please use the `https` protocol

* Enter in the cloned directory

* Create a conda virtual environment to host gwcosmo. Use
```
conda create -n gwcosmo python=3.6
```
* When the virtual environment is ready, activate it with (your python distribution will change to `gwcosmo`)
```
conda activate gwcosmo
```
* Run the following line to install all the python packages required by `gwcosmo`
```
pip install -r requirements
```
* Install `gwcosmo` by running 
```
python setup.py install
```
* You are ready to use `gwcosmo`. Note that, if you modify the code, you can easily reinstall it by using
```
python setup.py install --force
```


