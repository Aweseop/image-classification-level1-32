from typing import Any

class YYLoss():
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print('hello im yy')

yy_criterion_entrypoints = {
    'yyloss': YYLoss
}