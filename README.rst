************
Introduction
************

:Author: James R. McIntosh
:Contact: j.mcintosh@columbia.edu

Purpose
========
- Code base to test machine learning approaches for EEG rhythm phase prediction.
- Related to arxiv manuscript: `McIntosh, J. R., & Sajda, P. (2019). Estimation of phase in EEG rhythms for real-time applications. arXiv preprint arXiv:1910.08784 <https://arxiv.org/abs/1910.08784>`_.

Installation and getting started
================================
- After cloning the repository, the following commands will install the necessary libraries:

.. code-block:: bash

        conda env create -f py_eegepe.yml

- To get started, an across subject analysis as described in the manuscript can be run.

.. code-block:: bash

        conda activate py_eegepe
        cd examples/
        python gen_figure_supp_as.py

- Before the gen_figure_supp_as.py python scrip will work, the data has to be correctly configured in two locations:

    - gen_figure_supp_as.py : datadir variable (the root of the dataset folder), dataset variable (the name of the dataset folder)
    - data_specific.py : specifier, specifier_proc and _sublist which act to configure the location of data within the project directory. As well as preprocessor which handles the data specific pre-processing to be carried out (see examples for existing datasets within data_specific.py)

- Examples directory has been configured to operate with the `Child Mind Institute - healthy brain network data <http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/>`_. After downloading this data must be processed by a matlab script (convert_mat_to_eeglab.m) to enable loading with MNE.

    - The path structure should look like this:

    - .. code-block:: bash

        [datadir]/data/[subject_ID]/EEG/raw/mat_format/

    - And where resting data is present, the following will be added after running the matlab script:

    - .. code-block:: bash

        [datadir]/data/[subject_ID]/EEG/raw/eeglab_format/

    - data_specific.py is currently setup to deal with this configuration, but this can be changed as is required.

- gen_figure_supp_as.py as well as the other high-level functions in examples/ and manuscripts/ are configured to cycle through a series of experiments as configured in paradigm.py

- X
