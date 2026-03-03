from pydantic import BaseModel, PositiveInt, ConfigDict
from sdg_core_lib.commons import DataType
from enum import Enum
from typing import List

class SupportedFeatureTypes(str, Enum):
    continuous = "continuous"
    categorical = "categorical"
    primary_key = "primary_key"
    group_index = "group_index"


class BaseFeature(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    column_name: str
    column_datatype: DataType
    column_type: SupportedFeatureTypes


class FeatureData(BaseFeature):
    column_data: List[float | int | str] | List

class DataSkeleton(BaseFeature):
    column_position: int
    column_size: PositiveInt

class SkeletonOut(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    feature_name: str
    feature_position: int
    is_categorical: bool
    type: DataType
    feature_type: SupportedFeatureTypes
    feature_size: str
