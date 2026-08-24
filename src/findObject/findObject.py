from pathlib import Path
from typing import Optional

def findObject(
        reference:Path, 
        scene:Path,
        output:Optional[Path]=None,
        rectified_output:Optional[Path]=None,
        max_dim:int=2000):
    raise NotImplementedError