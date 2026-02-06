from typing import Literal

from pydantic import AnyUrl, BaseModel, model_validator
from pydantic import BaseModel, Field
from typing import Optional
from pydantic import BaseModel, Field, model_validator, AnyUrl
from typing import Optional

class NodeConstraintSHACL(BaseModel):
    id_: Optional[str] = Field(alias='@id')
    sh_path: Optional[AnyUrl] = Field(alias='sh:path')
    sh_minCount: Optional[int] = Field(alias='sh:minCount')
    sh_maxCount: Optional[int] = Field(alias='sh:maxCount')
    sh_class: Optional[str] = Field(default=None, alias='sh:class')
    sh_datatype: Optional[str] = Field(default=None, alias='sh:datatype')

    @model_validator(mode='after')
    def node_constraint_types_validator(self):
        # 1️⃣ Both are None → invalid
        if self.sh_datatype is None and self.sh_class is None:
            raise ValueError("You must provide either 'sh:datatype' or 'sh:class'.")

        # 2️⃣ Both set → auto-fix by preferring sh:datatype
        if self.sh_datatype is not None and self.sh_class is not None:
            # log or warn if you wish
            print("[NodeConstraintSHACL] Warning: Both 'sh:datatype' and 'sh:class' set; keeping 'sh:datatype'.")
            self.sh_class = None

        return self  # ✅ important: always return self

    class Config:
        populate_by_name = True
        validate_assignment = True
