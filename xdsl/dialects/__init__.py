"""
MQTOpt xDSL dialects module.
"""

from .MQTOpt.MQTOptOps import MQTOpt


def get_all_dialects():
    """Return a dictionary of all available dialects."""
    
    def get_mqtopt():
        return MQTOpt
    
    return {
        "mqtopt": get_mqtopt,
    }
