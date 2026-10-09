## Joint Selective State Space Model and Detrending for Robust Time Series Anomaly Detection

This respository is the official implementation of "[Joint Selective State Space Model and Detrending for Robust Time Series Anomaly Detection](https://ieeexplore.ieee.org/document/10623192/)" for SPL 2024. 


## 🚀 New Work

Our new work **AnoMamba** has been accepted to **IJCAI 2026**!

AnoMamba aligns reconstruction with time series anomaly detection through selective global dependency modeling.

📄 [Paper](https://www.ijcai.org/proceedings/2026/276) · 💻 [Code](https://github.com/aaaceo890/AnoMamba_ijcai)

### Installation

Run

```
pip install -r requirements.txt
```

to install the required packages.



### Data Preparation

The datasets are provided in `datasets.zip` in the root directory of this repository.

Unzip the datasets into `./datasets` directory:

```
unzip -d ./ datasets.zip
```



### Train model and detect

Run

```
bash run.sh
```

Results are stored in `./result`.



### Results

![image-20240820193425419](https://github.com/aaaceo890/mamba_tsad/blob/master/README/image-20240820193425419.png)



### Citing

If you found this code helpful, please consider citing it as follows:

```
@ARTICLE{10623192,
  author={Chen, Junqi and Tan, Xu and Rahardja, Sylwan and Yang, Jiawei and Rahardja, Susanto},
  journal={IEEE Signal Processing Letters}, 
  title={Joint Selective State Space Model and Detrending for Robust Time Series Anomaly Detection}, 
  year={2024},
  volume={31},
  number={},
  pages={2050-2054},
  doi={10.1109/LSP.2024.3438078}}
```



