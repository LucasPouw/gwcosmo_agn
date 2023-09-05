
Installation
==============

The recommended method for installing gwcosmo is using pip. We recommend [setting up a virtual environment first](#setting-up-a-virtual-environment). Once you have done this and activated your virtual environment, simply run

```
pip install gwcosmo
```
    
Note that gwcosmo requires Python version 3.8 or higher.

## Installing from source

You may choose to install gwcosmo from source instead. Gwcosmo can be found at [https://git.ligo.org/lscsoft/gwcosmo/](https://git.ligo.org/lscsoft/gwcosmo/).

* Clone the gwcosmo repository with 
    ```
    git clone <repository>
    ```
    The name of the repository can be copied from the git interface (top right button). If you do not have ssh key on git, please use the `https` protocol.
* Enter the cloned gwcosmo directory.
* If installing a branch of gwcosmo which is not the master branch, checkout the desired branch using
    ```
    git checkout <branch_name>
    ```
* Install gwcosmo by running 
    ```
    pip install .
    ```
* Alternatively, if you are planning to modify gwcosmo run the following instead:
    ```
    pip install -e .
    ```
    The `-e` stands for "editable" and means that your installation will automatically update when you make changes to the code, including checking out a different branch.
    If you have not installed gwcosmo in this fashion and later need to update your installation, simply rerun
    ```
    pip install .
    ```
    in the relevant directory.

## Setting up a virtual environment

### Using venv

`venv` is included in Python for versions >=3.3.

* Create a virtual environment to host gwcosmo. Use
    ```
    python -m venv env
    ```
* When the virtual environment is ready, activate it with
    ```
    source env/bin/activate
    ```
* To deactivate, run 
    ```
    deactivate
    ```

### Using Anaconda

You will need an [Anaconda distribution](https://www.anaconda.com/). The conda distribution is correctly initialized when, if you open your terminal, you will see the name of the python environment used. The default name is `(base)`.

Once the conda distribution is installed and activated on your machine, please follow these steps:

* Create a conda virtual environment to host gwcosmo. Use
    ```
    conda create -n gwcosmo
    ```
    To specify a specific version of gwcosmo, you can run, e.g.
    ```
    conda create -n gwcosmo python=3.9
    ```
* When the virtual environment is ready, activate it with (your python distribution will change to `gwcosmo`)
    ```
    conda activate gwcosmo
    ```
* To deactivate, run 
    ```
    conda deactivate
    ```



