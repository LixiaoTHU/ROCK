# ROCK

The official implementation of ROCK ([Recognizing Object by Components with Human Prior Knowledge Enhances Adversarial Robustness of Deep Neural Networks](https://ieeexplore.ieee.org/abstract/document/10019576), TPAMI 2023).

#### Abstract:
Adversarial attacks can easily fool object recognition systems based on deep neural networks (DNNs). Although many defense methods have been proposed in recent years, most of them can still be adaptively evaded. One reason for the weak adversarial robustness may be that DNNs are only supervised by category labels and do not have part-based inductive bias like the recognition process of humans. Inspired by a well-known theory in cognitive psychology -- recognition-by-components, we propose a novel object recognition model ROCK (Recognizing Object by Components with human prior Knowledge). It first segments parts of objects from images, then scores part segmentation results with predefined human prior knowledge, and finally outputs prediction based on the scores. The first stage of ROCK corresponds to the process of decomposing objects into parts in human vision. The second stage corresponds to the decision process of the human brain. ROCK shows better robustness than classical recognition models across various attack settings. These results encourage researchers to rethink the rationality of currently widely-used DNN-based object recognition models and explore the potential of part-based models, once important but recently ignored, for improving robustness.

### 1. Dependencies

Please see requirements.txt.

Some noticeable dependencies:
```
pip install torchattacks
unzip run_length_encoding.zip
cd run_length_encoding
python setup.py install
```
### 2. Datasets
Here we provide the link to download the datasets used in our paper: PartImageNet-C and PascalPart-C:

[Google Drive](https://drive.google.com/drive/folders/1LUVx_ObmIcc-GgVZcyCoSe7U27YhO4z2?usp=sharing)

### 3. Commands for training and evaluation:
An example using ROCK(ResNeXt-101) for training and evaluation on PartImageNet dataset with TRADES-AWP+EMA training under epsilon = 8.

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
