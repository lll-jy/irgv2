"""Errors specific to the IRG model."""


class NotFittedError(RuntimeError):
    """Report if some methods requiring fitting is called before fitting."""
    def __init__(self, item_name: str, before_action: str):
        """
        **Args**:

        - `item_name` (`str`): The item need to be fitted.
        - `before_acton` (`str`): The action that requires fitting beforehand.
        """
        super().__init__(f'The {item_name} is not yet fitted. Please call `.fit` before {before_action}.')


class NoPartiallyKnownError(ValueError):
    """Report wrong use of independent table as dependent."""
    def __init__(self, table_name: str):
        """
        **Args**:

        - `table_name`: The table that is independent but executed some partial-only methods.
        """
        super().__init__(f'The table {table_name} is independent. No partially known data can be extracted.')
