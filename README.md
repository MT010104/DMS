# DMS
### Introduction
DMS(Data-Mutation based Selection) is a test case selection method. We conduct experiments on five pairs of widely-used deep learning test sets and models. The results show that DMS significantly outperforms existing test case selection methods in terms of both bug-revealing ability and diversity of bug-revealing direction. Specifically, taking the original test set as the candidate set, DMS can filter out 53.85% to 99.22% of all bug-revealing test cases when selecting 10% of the test cases. Moreover, when selecting 5% of the test cases, the selected cases can cover almost all bug-revealing directions across all subjects. Overall, DMS outperforms baseline approaches with an average improvement of 12.38% to 71.81% in terms of the bug-revealing test cases selected, which further demonstrates the effectiveness of DMS.
![overview2](https://user-images.githubusercontent.com/65756145/197214602-4ca4e0f4-651e-4d4b-b413-d5fc7b7af90c.png)

### Prerequisites
* numpy 1.16.4
* tensorflow 1.14.0
* tqdm 4.36.1
* h5py 2.10.0
* pandas 0.23.4 

### Usage
Please use the following command to generate mutated models.
```Python
python finetune.py -d mnist -n lenet5 -i 40 -e 25
```

Please use the folling command to obtain prediction results of mutated models.
```Python
python finetune_predict.py -d mnist -n lenet5
```

Please use the folling command to extract features of test cases.
```Python
python extract_info.py -d mnist -n lenet5
```

Please use the folling command to obtain results of all test case selection methods.
```Python
python demo_general.py -d mnist -n lenet5
```

### Results
![v2](https://user-images.githubusercontent.com/65756145/197213036-13386f89-d7e9-4510-bb58-5df3d7bd0095.png)
![v1](https://user-images.githubusercontent.com/65756145/197213004-d2019050-dd54-4a56-ac2e-47a36efcf6f9.png)
