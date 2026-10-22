from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


# camelCase on the wire for vestry; snake_case in python and in stored json
class WireModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=False,
    )
