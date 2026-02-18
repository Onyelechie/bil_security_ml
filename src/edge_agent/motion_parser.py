from __future__ import annotations

from datetime import datetime, timezone
from xml.etree import ElementTree as ET

from .motion_event import MotionEvent


def _get_attr_any(el: ET.Element, *keys: str) -> str | None:
    """
    Helper: some systems may vary attribute casing. Try multiple keys.
    Example: "Id" vs "ID" vs "id".
    """
    for k in keys:
        if k in el.attrib:
            return el.attrib[k]
    return None


def parse_action_xml(xml_text: str, site_id: str) -> MotionEvent:
    """
    Parse one <Action>...</Action> message into MotionEvent.

    This function is PURE (no sockets, no logging) so it's easy to unit test.
    Raises ValueError on malformed/unexpected input.
    """
    # Strip whitespace that can appear if packets contain newlines
    xml_text = xml_text.strip()

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML: {e}") from e

    if root.tag != "Action":
        raise ValueError(f"Unexpected root tag '{root.tag}' (expected 'Action')")

    camera_el = root.find("Camera")
    if camera_el is None:
        raise ValueError("Missing <Camera ... /> tag")

    cam_id_str = _get_attr_any(camera_el, "Id", "ID", "id")
    if cam_id_str is None:
        raise ValueError("Missing Camera Id attribute")

    camera_id = int(cam_id_str)
    camera_name = _get_attr_any(camera_el, "Name", "NAME", "name")

    policy_el = root.find("Policy")
    policy_id = None
    policy_name = None
    if policy_el is not None:
        pol_id_str = _get_attr_any(policy_el, "Id", "ID", "id")
        policy_id = int(pol_id_str) if pol_id_str is not None else None
        policy_name = _get_attr_any(policy_el, "Name", "NAME", "name")

    user_el = root.find("UserString")
    user_string = (user_el.text or "").strip() if user_el is not None else None

    return MotionEvent(
        site_id=site_id,
        camera_id=camera_id,
        camera_name=camera_name,
        policy_id=policy_id,
        policy_name=policy_name,
        user_string=user_string,
        received_at=datetime.now(timezone.utc),
        raw_xml=xml_text,
    )
