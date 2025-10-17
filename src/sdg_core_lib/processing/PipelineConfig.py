from enum import Enum


class PipelineStepConfig(Enum):
    pass

    @classmethod
    def get_possible_values(cls):
        return [e.value for e in cls if e.value is not None]


class ScalerConfig(PipelineStepConfig):
    MINMAX = "minmax"
    STANDARD = "standard"
    NONE = None


class PipelineConfig:
    """
    Configuration class for processing pipelines.

    This class holds configuration parameters for different processing pipeline_steps
    in the pipeline, such as scaling, imputation, and encoding strategies.
    """

    def __init__(self, scaler: str = None):
        """
        Initialize the PipelineConfig with optional scaler configuration.

        Args:
            scaler: The type of scaler to use. Supported values are 'minmax' for MinMaxScaler
                   and 'standard' for StandardScaler. If None, no scaling will be applied.
        """
        self.scaler = ScalerConfig(scaler)

    def get_full_config(self) -> dict[str, PipelineStepConfig]:
        """
        Get a dictionary of all non-None configuration parameters.

        Returns:
            A dictionary where keys are configuration parameter names and values
            are their corresponding values. Only includes parameters that are not None
            and don't start with an underscore.

        Example:
            config = PipelineConfig(scaler="minmax")
            config.get_full_config()
            {'scaler': 'minmax'}
        """
        return {
            name: getattr(self, name)
            for name in vars(self)
            if not name.startswith("_") and getattr(self, name).value is not None
        }
