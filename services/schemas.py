from typing import Optional, Union, Literal, Dict
from langchain_core.pydantic_v1 import BaseModel, Field

class DateRange(BaseModel):
    gte: str = Field(alias="$gte")
    lte: str = Field(alias="$lte")

class MongoDBArgsSchema(BaseModel):
    Date: Optional[DateRange] = None
    Brand: Optional[Literal['BNK48', 'Nissan', 'QRRA', 'PIXXIE', 'Sunsilk', 'Samsung', 'VIIS', 'MXFRUIT', 'Wizzle', 'PheuThai-Party', 'CGM48', 'ALALA', 'Breeze', 'Mindy', 'Tresemme', 'Clear', 'THX', 'Comfort', '4EVE', 'BYD', 'People-Party', 'EMPRESS', 'Sunlight']] = None

    
    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True

class PythonCodeArgs(BaseModel):
    code: str = Field(
        description=(
            "Python code to execute. Use plotting libraries like matplotlib. "
            "Save any plots to /home/ec2-user/mind_agent/image and set the font to "
            "'/home/ec2-user/mind_agent/THSarabunNew.ttf' if generating Thai text."
        )
    )
