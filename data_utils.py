import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class GrackleDataset(Dataset):
    """
    Dataset para datos de Grackle PINN.
    Lee archivos .dat y proporciona datos normalizados.
    """
    
    def __init__(self, dat_file, use_log_scale=True, device='cpu'):
        """
        Args:
            dat_file (str): Ruta al archivo .dat
            use_log_scale (bool): Si True, aplica escala logarítmica. Si False, normaliza a [0,1]
            device (str): 'cpu' o 'cuda'
        """
        self.device = device
        self.use_log_scale = use_log_scale
        
        # Leer archivo .dat
        self.data = pd.read_csv(dat_file, sep='\s+', comment='#')
        
        # Extraer columnas relevantes
        self.time = self.data['Time'].values.astype(np.float32)
        self.temperature = self.data['Temperature'].values.astype(np.float32)
        self.hi_density = self.data['HI_density'].values.astype(np.float32)
        self.hii_density = self.data['HII_density'].values.astype(np.float32)
        
        # Normalizar datos
        self._normalize()
    
    def _normalize(self):
        """Normaliza los datos usando log scale o MinMaxScaler"""
        
        if self.use_log_scale:
            # Escala logarítmica (evitar log(0))
            self.time = np.log10(np.maximum(self.time, 1e-10))
            self.temperature = np.log10(np.maximum(self.temperature, 1e-10))
            self.hi_density = np.log10(np.maximum(self.hi_density, 1e-10))
            self.hii_density = np.log10(np.maximum(self.hii_density, 1e-10))
        else:
            # Normalización MinMax a [0, 1]
            scaler_time = MinMaxScaler()
            scaler_temp = MinMaxScaler()
            scaler_hi = MinMaxScaler()
            scaler_hii = MinMaxScaler()
            
            self.time = scaler_time.fit_transform(self.time.reshape(-1, 1)).flatten().astype(np.float32)
            self.temperature = scaler_temp.fit_transform(self.temperature.reshape(-1, 1)).flatten().astype(np.float32)
            self.hi_density = scaler_hi.fit_transform(self.hi_density.reshape(-1, 1)).flatten().astype(np.float32)
            self.hii_density = scaler_hii.fit_transform(self.hii_density.reshape(-1, 1)).flatten().astype(np.float32)
    
    def __len__(self):
        return len(self.time)
    
    def __getitem__(self, idx):
        """
        Retorna una muestra como tensores de PyTorch
        """
        sample = {
            'time': torch.tensor(self.time[idx], dtype=torch.float32, device=self.device),
            'temperature': torch.tensor(self.temperature[idx], dtype=torch.float32, device=self.device),
            'hi_density': torch.tensor(self.hi_density[idx], dtype=torch.float32, device=self.device),
            'hii_density': torch.tensor(self.hii_density[idx], dtype=torch.float32, device=self.device)
        }
        return sample