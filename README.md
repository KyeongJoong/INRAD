# INRAD



### Time-Series Anomaly Detection with Implicit Neural Representation

INRAD is a novel implicit neural representation-based method for multivariate time-series anomaly detection, along with a temporal encoding technique. You may refer to You may refer to [our paper](https://arxiv.org/pdf/2201.11950.pdf) for more details.

Since size of datasets we use in our experiments is too large, we show you the simplified example on a subset of SMD dataset. 
For the clear reproducibility check on our proposed method, we describe the specific procedure as below.

#### Clone the repo

```
git clone https://github.com/KyeongJoong/INRAD && cd INRAD
```


#### Implementation

For the reproducibility check of our results on five datasets we use, please refer to 'main.py' or 'main.ipynb' file after 
downloading each dataset (Please refer to 'Dataset Details' below).
If you are interested in running your own dataset, please refer to 'example_INRAD.ipynb' or 'example_INRAD.html' file. 

#### Packages (python == 3.7)

numpy==1.20

pandas==1.2.4

pytorch==1.8.1

scikit-learn==0.24.2

#### Dataset details

1. SMD
Source (download and preprocess introduction) link: https://github.com/NetManAIOps/OmniAnomaly
Note that preprocess code on the link should be applied only on SMD dataset.
Save the downloaded data inside the 'data/SMD' folder along with preprocessed data inside 'data/SMD/processed' folder.


2. SMAP & MSL 
Source (download introduction) link: https://github.com/khundman/telemanom
Save the contents inside 'data/SMAP_MSL' folder along with 'labeled_anomalies.csv' file.
Refer to the remained folder tree inside 'data/SMAP_MSL' folder.


3. SWAT
Source link: https://itrust.sutd.edu.sg/itrust-labs_datasets/ (Request for the dataset is needed on this link.)
Following previous studies, we use

SWaT_Dataset_Normal_v0.xlsx
SWaT_Dataset_Attack_v0.xlsx
in 'SWaT.A1 _ A2_Dec 2015\Physical' folder ('SWaT.A1 & A2_Dec 2015 dataset' zip file).

Basic preprocessing:
- Erase empty row in each file.
- Transform the xlsx format into csv format. (.xlsx --> .csv)

Save the files ('SWaT_Dataset_Normal_v0.csv' and 'SWaT_Dataset_Attack_v0.xlsx') inside the 'data/SWaT' folder.


4. WADI 
Source link: https://itrust.sutd.edu.sg/itrust-labs_datasets/ (Request for the dataset is needed on this link).
Following previous studies, we use

'WADI_14days.csv'
'WADI_attackdata.csv'
in 'WADI.A1_9 Oct 2017' folder
and
'WADI_attackdataLABLE.csv'
in 'WADI.A2_19 Nov 2019' folder ('WADI.A1_9 Oct 2017' zip file).

Basic preprocessing:
- Erase first row in 'WADI_attackdataLABLE.csv' file.
- Erase first 4 rows in 'WADI_14days.csv' file
