class NotFittedError(RuntimeError):
    def __init__(self, item_name: str, before_action: str):
        super().__init__(f'The {item_name} is not yet fitted. Please call `.fit` before {before_action}.')
