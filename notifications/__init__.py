"""
Notification and messaging integrations for Voice AI Framework.
"""

from edgevoiceai.notifications.whatsapp import (
    WhatsAppClient,
    WhatsAppConfig,
    WhatsAppNotificationTool,
)

__all__ = [
    "WhatsAppConfig",
    "WhatsAppClient",
    "WhatsAppNotificationTool",
]
