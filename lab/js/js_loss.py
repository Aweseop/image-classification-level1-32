from typing import Any

class JSLoss():
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print('hello im js')

js_criterion_entrypoints = {
    'jsloss': JSLoss
}