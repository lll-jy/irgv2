"""Errors specific to the IRG model."""


class TableNotFoundError(KeyError):
    """Report if some queried table is not in the database."""
    def __init__(self, table_name: str):
        """
        ***Args**:

        - `table_name` (`str`): Name of the table.
        """
        super().__init__(f'Table {table_name} is not found in database.')


class ColumnNotFoundError(KeyError):
    """Report if some queried column is not in the table."""
    def __init__(self, table_name: str, col_name: str):
        """
        ***Args**:

        - `table_name` (`str`): Name of the table.
        - `col_name` (`str`): Name of the column.
        """
        super().__init__(f'Column {col_name} is not found in table {table_name}.')


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
