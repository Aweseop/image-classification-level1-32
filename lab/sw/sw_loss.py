from typing import Any

class SWLoss():
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print('hello im sw')

sw_criterion_entrypoints = {
    'swloss': SWLoss
}