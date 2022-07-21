# BiVFM-Fleet: A Bi-Model Driven Transient Multiphase Virtual Flow Metering Paradigm for Arbitrary Well Patterns Based on the Mechanism Model and Data-Driven Model

    feature_engineering.py
    func.py
    generate_dataset.py
    model.py
    olga_func.py
    train.py


## demo_work1: Case
**AutoOLGAFleet**
- mech_case: 
   1. base
   2. config
   3. auto_generate
   4. _other: info_

**BiVFM**
- dataset
   1. dataset_0_tpl
   2. dataset_1_csv
   3. dataset_2_xy
   4. dataset_3_pkl
- logs
   1. logs_**_Model_**
      1. log
      2. save_dir
      3. log_imgs
      4. imgs_save
   2. ...
- feature_extractor.xlsx
- conf_base.yaml
- conf_**_info_**.yaml


Flow measurement is the basis of production monitoring and productivity allocation. Virtual flowmeter (VFM) provides a multiphase flow prediction method for petroleum assets as its advantages of flexibility and low cost compared with the multiphase flowmeter (MPFM). Mechanism model and data-driven model are two important methods used in VFM, but the previous studies on VFM can not fully combine the advantages of them two, and are with harsh requirements for operating conditions and measurements. So it is difficult to meet the production requirements in the oil fields. In this study, a bi-model-driven distributed virtual flowmeter framework (BiVFM-Fleet) is proposed, in which the mechanism-driven model is coupled with the data-driven model. In this framework, the domain knowledge is obtained with the large-scale distributed multiphase flow mechanism calculation method, combined with the grid search to realize the study of the flow law for arbitrary well patterns under complex conditions. Furthermore, the multiphase flow convolution network (MPFNet) based on Temporal Convolutional Network (TCN) Layer is proposed as the core component of the framework to optimize the problem of multiphase flow. The BiVFM-Fleet in the case of 30 single well flow cycle evaluations has achieved state-of-the-art performance with a MAPE of 0.15%, a 140% improvement over the mainstream method LSTM and 33.33% higher than the following method TCN. In the case of 1,024 large-scale multi-well evaluations, a MAPE of 5.05% is obtained, which is 0.34% higher than TCN, indicating a 6.73% performance improvement. The transient flow prediction case of multi-well flow is calculated with real-time data source monitoring. In addition, the Python package of BiVFM-Fleet framework and the largest data set of transient flow patterns for multiphase flow at present are released, providing the basement for subsequent multiphase flow studies.


Paper: A Bi-Model Driven Transient Multiphase Virtual Flow Metering Paradigm for Arbitrary Well Patterns Based on the Mechanism Model and Data-Driven Model

Pip Package: https://pypi.org/project/BiVFM-Fleet

Github Repository: https://github.com/mmmahhhhe/BiVFM-Fleet


