## Requirements
You can run pip install -r requirements.txt to deploy the environment.
## Data Preparation
1.  **Data Splitting:** In the experiments, we maintain the same data splitting scheme as the benchmarks.
2.  **Dataset:** We follow the same data preprocessing pipeline of Bootstrapping Your Own Representations for Fake News Detection. Please use the data prepare scripts provided in preprocessing or the preprocessing scripts in prior work to prepare data for each datasets. For all datasets (Weibo/Weibo-21/GossipCop), please download from the official source.
3.  **Data preparation**: Use `clip_data_pre`, `data_pre`, `weibo21_data_pre`, `gossipcop_clip_data_pre` and `gossipcop_clip_data_pre` to preprocess the data of Weibo, Weibo21 and GossipCop, respectively, in order to save time during the data loading phase.
4.  **Data Storage**
    - Place the processed Weibo data in the `./data` directory.
    - Place the Weibo21 data in the `./Weibo_21` directory.
    - Place the  GossipCop data in the `./gossipcop` directory.

## Pretrained Models

1.  **Roberta:** You can download the pretrained Roberta model from "Roberta" and move all files into the `./pretrained_model` directory.
2.  **MAE:** Download the pretrained MAE model from "Masked Autoencoders: A PyTorch Implementation" and move all files into the root directory.
3.  **CLIP:** Download the pretrained CLIP model from "Chinese-CLIP" and move all files into the root directory.

## Training
* **Preparation:** Download the Weibo21, Weibo, and GossipCop datasets, along with their respective api. 
* **Start Training:** After processing the data, train the model by running `python main.py --dataset gossipcop` or `python main.py --dataset weibo21` or `python main.py --dataset weibo`.

