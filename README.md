# An uncertainty-aware and sex-prior guided biological age estimation from orthopantomogram images (UASP-BAE)
<center>
<img src='.\demo\UASP-BAE-0.png', width='300'>
</center>

[Paper: 2023_JBHI_UASP-OPGage.pdf](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6221020) 

Before implementing the code, you need to configure and setup some libraries required by the following [Requirements](requirements.txt):

* [Inplace-abn-1.1.1](https://github.com/mapillary/inplace_abn)
* pytorch-1.11 (CUDA.11.2)
* OpenCV-4.6, ...

## Database Setting
Before using the UASP-BAE codes to train the model, you need to prepare the training dataset, validation dataset and testing  dataset in the following format:
```
datasets
   └──train
      └──xxxx_OPGs.jpg
   └──val
      └──xxxx_OPGs.jpg
   └──test
      └──xxxx_OPGs.jpg
   |──train.5-25.csv
   |──val.5-25.csv
   |──test.5-25.csv
```
The information in the CSV file includes: name, gender, img_id, check_id, check_date, birth_date, age, t1, t2, t3, t4, t5, t6, t7, t8 
* gender: 1-->male, 2-->female
* check_id: xxxx -->  xxxx_OPGs.jpg
* age = (check_date - birth_date) / 365days  --> eg. 15.25634356 year old
* t1 to t8 : Manual stage of 8 teeth on the right maxilla.

## Experimental results
We provide [Some experimental results ]()
<center>
<img src='.\demo\result-1.png', width='500'>
<img src='.\demo\result-2.png', width='500'>
<img src='.\demo\result-3.png', width='500'>
</center>

## [Training and Validation](UASP-BAE.train.py)
We provide [UASP-BAE.train code](UASP-BAE.train.py), and you can perform it by the following command:
```
CUDA_VISIBLE_DEVICES=0 python UASP-BAE.train.py \
--model_name "UASP_BAE_net1" \
--batch-size 16
```

## [Inference](UASP-BAE.test.py)
We provide [inference code](UASP-BAE.test.py), and you can perform it by the following command:
```
CUDA_VISIBLE_DEVICES=0 python UASP-BAE.test.py \
--model_name "UASP_BAE_net1" \
--model_path "./models/UASP_BAE_net1/9_uasp_bae_net1_mae-0.8001.ckpt" \
--batch-size 16
```

## Citation
If you use UASP-BAE in your research, please cite:
```bibtex
@ARTICLE{Dong2023,
  author  = {Dong, Zhang and Jing, Yang and Shaoyi, Du and Wenqing, Bu and Yu-cheng Guo},
  title   = {An uncertainty-aware and sex-prior guided biological age estimation from orthopantomogram images},
  journal = {IEEE Journal of Biomedical and Health Informatics (Under Review R1)}, 
  volume  = {X},
  year    = {2023},
  pages   = {XXXXX}
}
```