from sdg_core_lib.dataset.datatypes.NumericColumn import NumericColumn


class TypeSimpleFactory:
    @staticmethod
    def create_column_type(typ):
        if type == "numeric":
            return NumericColumn()
        else:
            raise NotImplementedError()
