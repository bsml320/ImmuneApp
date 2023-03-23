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

<div align=center><img src="https://bioinfo.uth.edu/iapp/github/Figure1.jpg" width="800px"></div>

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
__Note: please unzip the gibbscluster.zip in the folder first, and modify the "FULL PATH" in the gibbscluster execution file__

# Usage

### ImmuneApp provides two services: precise antigen presentation prediction and clinical immunopeptidomic cohorts analysis.

__1__. For antigen presentation prediction, this module accept two different types of input; FASTA and Peptide. In addition, candidate HLA molecules should be specified in the predictions. For FASTA input, the peptide length(s) should be specified.

### Example of antigen presentation prediction: 
For __peptides input__, please uses:
```
python ImmuneApp_prediction.py -f 'testdata/test_peplist.txt' -a 'HLA-A*01:01' 'HLA-A*02:01' 'HLA-A*03:01' 'HLA-B*07:02' -ap 'yes' -o 'results'
```

For __FASTA input__, please uses:
```
python ImmuneApp_prediction.py -fa 'testdata/test.fasta' -a 'HLA-A*01:01' 'HLA-A*02:01' 'HLA-A*03:01' 'HLA-B*07:02' -ap 'yes' -o 'results'
```
__2__. For immunopeptidome analysis, this module accept (clinical) immunopeptidomic samples as input, together with HLA molecule(s) by HLA tying tool.

### Example of immunopeptidome analysis: 
For __single sample__, please uses:
```
python ImmuneApp_immunopeptidomics_analysis.py -f testdata/Melanoma_tissue_sample_of_patient_5.txt -a HLA-A*01:01,HLA-A*25:01,HLA-B*08:01,HLA-B*18:01 -o results
```

For __multiple samples__, separate the different sample names or HLA alleles with spaces, uses:
```
python ImmuneApp_immunopeptidomics_analysis.py -f testdata/Melanoma_tissue_sample_of_patient_5.txt testdata/Melanoma_tissue_sample_of_patient_8.txt -a HLA-A*01:01,HLA-A*25:01,HLA-B*08:01,HLA-B*18:01 HLA-A*01:01,HLA-A*03:01,HLA-B*07:02,HLA-B*08:01,HLA-C*07:02,HLA-C*07:01 -o results
```

For details of other parameters, run:
```
python ImmuneApp_immunopeptidomics_analysis.py --help

python ImmuneApp_prediction.py --help
```
# Web Server
Researchers can run ImmuneApp online at https://bioinfo.uth.edu/iapp/. For commercial usage inquiries, please contact the authors. 

ImmuneApp implements four main modules: “Discovery”, “Analysis”, “Results” and “Controller”. In the backend, three well-trained deep learning models (ImmuneApp_BA, ImmuneApp_EL and ImmuneApp_AP) are used for the predictions of binding affinities, ligand probabilities, and overall antigen presentation as well as immunopeptidomic cohorts analysis, respectively. The “Controller” module checks the input data format, sends the data from frontend interfaces to the backend, creates the results using models, and then provides the results on the “Results” page. We implemented both pages in a responsive manner by using the HTML5, CSS, Bootstrap3, and JavaScript. Additionally, the "Controller" is called through Ajax technology to submit jobs, retrieve data, and show results. There is no limit to the number of tasks submitted by each user. ImmuneApp can automatically handle the jobs in a queue, which allows up to 5 jobs to execute concurrently.

<div align=center><img src="https://bioinfo.uth.edu/iapp/github/Supplementary figure 7.jpg" width="1200px"></div>

The “Discovery” module accepts two input types: “FASTA” and “Peptide”. Users can directly copy the input data to an online submission text box. Moreover, MHC molecules and the peptide length (only FASTA input) need to be specified for running prediction. The “Analysis” module accepts clinical immunopeptidomic samples as input, together with MHC molecules. The input sample(s) can be directly copied to an online submission text box or uploaded from the users local disk. Sample identity should be specified. This module provides intuitive report for personalized analysis, statistical reports, and visualization of results for clinical immunopeptidomic cohorts.

<div align=center><img src="https://bioinfo.uth.edu/iapp/github/Supplementary figure 8.jpg" width="1000px"></div>



