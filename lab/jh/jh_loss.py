from typing import Any

class JHLoss():
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print('hello im jh')

jh_criterion_entrypoints = {
    'jhloss': JHLoss
}