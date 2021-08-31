from typing import Any

class THLoss():
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print('hello im th')

th_criterion_entrypoints = {
    'thloss': THLoss
}