import pytest
from edge_agent.triggers.tcp_parse import parse_motion_xml


def test_parse_motion_xml_happy_path():
    xml = (
        '<Action><Camera Id="1" Name="Demo Cam" />'
        '<Policy Id="21" Name="Alarm Demo" />'
        "<UserString>Alarm</UserString></Action>"
    )
    out = parse_motion_xml(xml)
    assert out["camera_id"] == "1"
    assert out["camera_name"] == "Demo Cam"
    assert out["policy_id"] == "21"
    assert out["policy_name"] == "Alarm Demo"
    assert out["user_string"] == "Alarm"


def test_parse_motion_xml_invalid_raises():
    with pytest.raises(ValueError):
        parse_motion_xml("not xml")
