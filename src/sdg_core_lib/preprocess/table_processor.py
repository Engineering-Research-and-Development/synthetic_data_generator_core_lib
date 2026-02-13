from sdg_core_lib.dataset.columns import Column
from sdg_core_lib.preprocess.base_processor import Processor


class TableProcessor(Processor):
    def __init__(self, dir_path: str):
        super().__init__(dir_path)

    # TODO: External config?
    def _init_steps(self, columns: list[Column]):
        if len(self.steps.keys()) == len(columns):
            pass

        if len(columns) == 0:
            raise ValueError("No columns provided for processing")
        for idx, col in enumerate(columns):
            step_list = self.strategy.get_steps_per_feature(col)

            self.add_steps(step_list, col_position=col.position, data_position=idx)

    def process(self, columns: list[Column]) -> list[Column]:
        self._init_steps(columns)
        col_data = [col.get_data() for col in columns]
        results = super().process(col_data)
        preprocessed_columns = []
        for col in columns:
            preprocessed_columns.append(
                type(col)(
                    col.name,
                    col.value_type,
                    col.position,
                    results.get(col.position),
                    col.column_type,
                )
            )
        return preprocessed_columns

    def inverse_process(self, preprocessed_columns: list[Column]) -> list[Column]:
        self._init_steps(preprocessed_columns)
        col_data = [col.get_data() for col in preprocessed_columns]
        results = super().inverse_process(col_data)
        post_processed_columns = []
        for col in preprocessed_columns:
            post_processed_columns.append(
                type(col)(
                    col.name,
                    col.value_type,
                    col.position,
                    results.get(col.position),
                    col.column_type,
                )
            )
        return post_processed_columns
