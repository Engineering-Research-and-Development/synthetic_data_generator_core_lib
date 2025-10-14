class PipelineConfig:
    """
    Configuration class for processing pipelines.

    This class holds configuration parameters for different processing pipeline_steps
    in the pipeline, such as scaling, imputation, and encoding strategies.
    """

    def __init__(self, scaler: str = None) -> None:
        """
        Initialize the PipelineConfig with optional scaler configuration.

        Args:
            scaler: The type of scaler to use. Supported values are 'minmax' for MinMaxScaler
                   and 'standard' for StandardScaler. If None, no scaling will be applied.
        """
        self.scaler = self._set_scaler_config(scaler)

    @staticmethod
    def _set_scaler_config(scaler: str = None) -> str | None:
        """
        Validate and return the scaler configuration.

        Args:
            scaler: The scaler type to validate.

        Returns:
            The validated scaler type string if passed, None otherwise

        Raises:
            ValueError: If an invalid scaler type is provided.
        """
        if scaler is None:
            return None

        if scaler == "minmax":
            return "minmax"
        elif scaler == "standard":
            return "standard"
        else:
            raise ValueError(
                "Invalid scaler configuration. Must be either 'minmax' or 'standard'."
            )

    def get_full_config(self) -> dict:
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
            if not name.startswith("_") and getattr(self, name) is not None
        }
