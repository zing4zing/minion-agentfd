"""
Utilities for Minion-Manus.

This package contains utility functions and classes for the Minion-Manus framework.
"""

from minion_agent.utils.logging import setup_logging
from minion_agent.config import Settings

__all__ = ["setup_logging"]

# 在应用启动时
settings = Settings.from_env()  # 或传入自定义设置
setup_logging(settings) 
