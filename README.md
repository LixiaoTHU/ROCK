# ROCK

The official implementation of ROCK ([Recognizing Object by Components with Human Prior Knowledge Enhances Adversarial Robustness of Deep Neural Networks](https://ieeexplore.ieee.org/abstract/document/10019576), TPAMI 2023).

### 1. Dependencies

Please see requirements.txt.

Some noticeable dependencies:
```
pip install torchattacks
cd run_length_encoding
python setup.py install
```
### 2. Datasets
Here we provide the links to download the datasets used in our paper: PartImageNet-C and PascalPart-C:

[Google Drive](https://drive.google.com/drive/folders/1LUVx_ObmIcc-GgVZcyCoSe7U27YhO4z2?usp=sharing)

### 3. Command for training and evaluation:
An example using ROCK(ResNeXt-101) for training and evaluation on PartImageNet dataset with TRADES-AWP+EMA training under $ \epsilon = 8 $.
Training:
```
python trainseg_awp.py --config_path configs/PartImageNet/ --version PartImageNet_part_trades_101_awp_ema_8 --data_parallel --train
```

Evaluation:
```
python trainseg.py --config_path configs/PartImageNet/ --version PartImageNet_part_trades_101_awp_ema_8 --data_parallel --load_best_model --epsilon 8 --attack_type random --attack_choice PGD
```


### 4. If you find this useful in your research, please cite this work:

```
@article{li2023recognizing,
  title={Recognizing Object by Components with Human Prior Knowledge Enhances Adversarial Robustness of Deep Neural Networks},
  author={Li, Xiao and Wang, Ziqi and Zhang, Bo and Sun, Fuchun and Hu, Xiaolin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
```