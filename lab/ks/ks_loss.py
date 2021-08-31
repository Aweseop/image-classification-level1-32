from typing import Any

class KSLoss():
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print('hello im ks')

ks_criterion_entrypoints = {
    'ksloss': KSLoss
}