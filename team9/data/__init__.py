# data.__init__.py 
from .base import (load_csv,
    load_npz,
    load_pkl
	)

# from .base import TfidfVectorizer as 

__all__ = ['load_csv', 'load_npz', 'load_pkl']