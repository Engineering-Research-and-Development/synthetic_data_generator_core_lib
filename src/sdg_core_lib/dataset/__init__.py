from dataset.columns import NumericColumn, CategoricalColumn, PrimaryKeyColumn, Column


class ColumnRegistry:
    registry = {
        "continuous": NumericColumn,
        "categorical": CategoricalColumn,
        "primary_key": PrimaryKeyColumn,
    }

    @staticmethod
    def get_column(value_type: str) -> type[Column]:
        return ColumnRegistry.registry[value_type]

