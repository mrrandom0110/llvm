from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EditorScheme = Literal["cursor", "vscode"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(populate_by_name=True)

    kernel_repo: Path = Path("/home/erickurbanov/linux")
    editor_scheme: EditorScheme = "cursor"
    wsl_distro: str = Field(default="Ubuntu", validation_alias="WSL_DISTRO_NAME")
    test_symbol_path: Path | None = Field(default=None, validation_alias="TEST_SYMBOL_PATH")
    test_symbol_line: int = Field(default=1, validation_alias="TEST_SYMBOL_LINE")
    test_symbol_column: int = Field(default=1, validation_alias="TEST_SYMBOL_COLUMN")
