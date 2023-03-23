# ImmuneApp v1.0
## ImmuneApp for HLA-I epitope prediction and immunopeptidome analysis
The selection of human leukocyte antigen (HLA) epitopes is critical for the development of cancer immunotherapy strategies and vaccines.
Advances in liquid chromatography and mass spectrometry facilitated a new era of large-scale immunopeptidomics and improved HLA-peptide binding prediction.
However, current prediction algorithms still lack interpretability and have limited predictive capabilities.
Moreover, many clinical multiallelic datasets are undetected or barely interpreted for clinical application.
Thus, the development of powerful predictors of antigen presentation and tools for interpreting clinical immunopeptidomics data is urgently needed.
Here, we developed an automated tool, namely ImmuneApp, for accurate antigen presentation prediction and personalized downstream analysis for immunopeptidomics cohorts.
We demonstrated ImmuneApp could yield superior predictive performance and more efficient feature representation for peptides presented to HLA when compared to the 
existing methods. Moreover, ImmuneApp extracted interpretable patterns and unraveled biophysical determinants of antigen binding preference.
Importantly, ImmuneApp improved the prioritization of immunogenic neo-epitopes through our comparison with many other endogenous factors that drive immunogenicity.
By implementing the state-of-the-art of models as the engine, ImmuneApp allows one-stop analysis, statistical evaluation, 
and visualization for clinical immunopeptidomics cohorts, which includes quality control, binding annotations, HLA assignment, motif discovery and decomposition, 
and antigen presentation prediction via two functional modules. Through the application of ImmuneApp to multiple disease-related immunopeptidomics datasets, 
including melanoma tumor tissues, lung, and gastric cancer biopsies, we demonstrated its utility in a clinical setting.
ImmuneApp is freely available at https://bioinfo.uth.edu/iapp/.

![image](https://github.com/BioDataStudy/NetBCE/blob/main/data/github_1.jpg)

# Installation
Download ImmuneApp by
```
git clone https://github.com/bsml320/ImmuneApp
```
Installation has been tested in Linux server, CentOS Linux release 7.8.2003 (Core), with Python 3.7. Since the package is written in python 3x, python3x with the pip tool must be installed. ImmuneApp uses the following dependencies: numpy, scipy, pandas, h5py, keras version=2.3.1, tensorflow=1.15, seaborn, logomaker, and shutil, pathlib. We highly recommend that users leave a message under the ImmuneApp issue interface (https://github.com/bsml320/ImmuneApp/issues) when encountering any installation and running problems. We will deal with it in time. You can install these packages by the following commands:
```
conda create -n ImmuneApp python=3.7
conda activate ImmuneApp
pip install pandas
pip install numpy
pip install scipy
pip install -v keras==2.3.1
pip install -v tensorflow==1.15
pip install seaborn
pip install logomaker
pip install shutil
pip install pathlib
pip install protobuf==3.20
pip install h5py==2.10.0
```
