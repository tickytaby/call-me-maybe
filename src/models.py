from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ParameterSchema(BaseModel):
    type: Literal["number", "string", "boolean"]


class ToolDefinition(BaseModel):
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict[str, ParameterSchema] = Field(default_factory=dict)
    returns: dict[str, Any] | None = None


class PromptItem(BaseModel):
    prompt: str = Field(min_length=1)


class ParameterValue(BaseModel):
    type: Literal["number", "string", "boolean"]
    value: Any

    @model_validator(mode="after")
    def check_value_matches_type(self) -> "ParameterValue":
        checks = {
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "string": lambda v: isinstance(v, str),
            "boolean": lambda v: isinstance(v, bool),
        }
        if not checks[self.type](self.value):
            raise ValueError(
                f"value {self.value!r} does not match declared type {self.type!r}"
            )
        return self


class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
