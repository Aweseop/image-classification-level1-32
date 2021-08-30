from typing import Any

class SWSample():
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print('hello im sw')