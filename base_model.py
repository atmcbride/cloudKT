import numpy as np

class BaseModel():
    """
    Base class for models. Not for use on its own.
    """
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
    
    @staticmethod
    def select_near_point(tab, l, b, radius=1):
        cond = np.sqrt((tab['GLON'] - l)**2 + (tab['GLAT'] - b)**2) < radius
        return np.where(cond)[0]