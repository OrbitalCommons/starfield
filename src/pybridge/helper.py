import numpy as np
import json, base64

def _bytes_to_str(data: bytes) -> str:
    return base64.b64encode(data).decode('utf-8')

class ResultCollector:
    def __init__(self):

        self.result = None

    def __call__(self, data):
        if isinstance(data, bytes):
            self.collect_bytes(data)
        elif isinstance(data, str):
            self.collect_string(data)
        elif isinstance(data, np.ndarray):
            self.collect_array(data)
        else:
            raise TypeError(f"Unsupported type: {type(data)}")

    def collect_bytes(self, data: bytes):
        self._set_result({
            "type": 'bytes',
            "data": _bytes_to_str(data)
        })

    def collect_string(self, data: str):
        self._set_result({
            "type": 'string',
            "data": data
        })

    def collect_array(self, array: np.ndarray):
        self._set_result({
            "type": 'array',
            "dtype": str(array.dtype),
            "shape": array.shape,
            "data": _bytes_to_str(array.tobytes())
        })

    def _set_result(self, result: dict):
        if self.result is not None:
            raise Exception("Result already collected")
        self.result = json.dumps(result)
    
    def get_result(self) -> str:
        if self.result is None:
            raise Exception("No result collected")
        return self.result
    
rust = ResultCollector()