import os
import numpy as np
import pandas as pd
from config import Confiq
import torch
from pathlib import Path
import matplotlib.pyplot as plt

cfg = Confiq()

class Tools:
    def __init__(self):
        pass

    def create_dict(self, codes):
        self.char_dict = {}
        for index, val in enumerate(codes):
            self.char_dict[val] = index+1

        return self.char_dict

    def read_data(self, data_path ,partition):
        self.data = []
        for fn in os.listdir(os.path.join(data_path, partition)):
            with open(os.path.join(data_path, partition, fn)) as f:
                self.data.append(pd.read_csv(f, index_col=None))
        return pd.concat(self.data)

    def find_common_family(self, thres=250):
        self.df_train = self.read_data(cfg.ROOT_PATH, 'train')
        self.commonLables = self.df_train.family_accession.value_counts()[:thres]

        return self.commonLables

    def save_model(model,
                target_dir: str,
                model_name: str):

        target_dir_path = Path(target_dir)
        target_dir_path.mkdir(parents=True,
                                exist_ok=True)

        assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model harus dalam '.pt' or '.pth'"
        model_save_path = target_dir_path / model_name

        print(f"[INFO] Saving model to: {model_save_path}")
        torch.save(obj=model.state_dict(),
                    f=model_save_path)

    def viz_result(acc, loss, epoch, phase='train'):

        x_axis = np.arange(1,epoch,1)

        if(phase == "train"):
            plt.subplot(1,2,1)
            plt.title("Train Acc")
            plt.xlabel('epoch')
            plt.ylabel('acc (%)')
            plt.plot(acc, x_axis)

            plt.subplot(2,2,1)
            plt.title("Train loss")
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.plot(loss, x_axis)

        if(phase == "test"):
            plt.subplot(1,2,1)
            plt.title("Test Acc")
            plt.xlabel('epoch')
            plt.ylabel('acc (%)')
            plt.plot(acc, x_axis)

            plt.subplot(2,2,1)
            plt.title("Test loss")
            plt.xlabel('epoch')
            plt.ylabel('loss')

        plt.show()


    