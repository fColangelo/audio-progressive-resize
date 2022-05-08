from statistics import mean
import torch, torchaudio
from torch.utils.data import Dataset
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from glob import glob
import os, pickle

class General_Dataset(Dataset):
    
    def __init__(self, cfg: DictConfig, mode: str ="training") -> None:
        super().__init__()
        assert mode in ("train", "val", "test"), f"mode should be one train, val or test"
        self.cfg = cfg
        self.dataset_name = "general" #Each dataset overrides this
        self.spectrogram = instantiate(self.cfg.Audio.transform)
        # Iterable containing all audio file in this split of the dataset
        # Individual dataset have their own method to generate it
        self.audio_list = glob("../../../data/test_audio_files/*.wav") 

        ap  = cfg.audio.transform
        statfile_name = f"{self.dataset_name}_{ap.sample_rate}_{ap.n_fft}_{ap.hop_length}_{ap.n_mels}" 
        statfile_path = f"{cfg.original_work_dir}/data/common_files/{statfile_name}"
        if os.path.isfile(statfile_path):
            with open(statfile_path, "rb") as f:
                data_dict = pickle.load(f)
                self.mean = data_dict["mean"] 
                self.std  = data_dict["std"]
        else:    
            data_dict = self.get_stats()
            with open(statfile_path, "wb") as f:
                pickle.dump(data_dict, f)
                
    def get_stats(self) -> dict:
        """Calculates mean and standard deviation for the dataset.
           A single set of stats is returned, as the spectrogram is assumed to be mono-channel. 

        Returns:
            dict: dict containing "mean": meanval, "std": stdval
        """
        mean    = torch.tensor(0)
        sq_mean = torch.tensor(0)
        idx = 0
        for ap in self.audio_list:
            wf, _   = torchaudio.load(ap)
            mean    += torch.mean(wf) 
            sq_mean += torch.mean(torch.square(wf))
            idx     += 1
        mean    /= idx
        sq_mean /= idx
        std      = np.sqrt(sq_mean - mean**2)
        return {"mean": mean, "std": std}
    
    
    def pre_process(self, audio: torch.tensor) -> torch.tensor:
        """Transform an audio waveform into a melspectrogram.
           Parameters are defined in cfg.Audio

        Args:
            audio (torch.tensor): audio waveform as read by torchaudio with shape (n_samples)

        Returns:
            torch.tensor: log-melspectrogram of the audio waveform, with shape (1, n_mel, t_steps)
        """
        return torch.log10(self.spectrogram(audio))
    
    
        
        
            
        