from __future__ import annotations

from typing import Any

from defusedxml import ElementTree as ET


def parse_motion_xml(xml_text: str) -> dict[str, Any]:
    """
    Parse Colton's TCP XML payload into fields.

    Expected example:
      <Action>
        <Camera Id="1" Name="Demo Cam" />
        <Policy Id="21" Name="Alarm Demo" />
        <UserString>Alarm</UserString>
      </Action>
    ...
    """
    xml_text = xml_text.strip()
    if not xml_text:
        raise ValueError("Empty XML payload")

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError("Invalid XML payload") from exc

    # Handle if root isn't Action but contains Action inside
    if root.tag != "Action":
        action = root.find(".//Action")
        if action is None:
            raise ValueError(f"Unexpected root tag: {root.tag}")
        root = action

    camera = root.find("Camera")
    policy = root.find("Policy")
    user = root.findtext("UserString")

    out: dict[str, Any] = {
        "camera_id": camera.get("Id") if camera is not None else None,
        "camera_name": camera.get("Name") if camera is not None else None,
        "policy_id": policy.get("Id") if policy is not None else None,
        "policy_name": policy.get("Name") if policy is not None else None,
        "user_string": user.strip() if isinstance(user, str) else None,
    }
    return out
