import asyncio
import pytest

from edge_agent.config import EdgeSettings
from edge_agent.triggers.tcp_trigger import TcpMotionTrigger


@pytest.mark.asyncio
async def test_tcp_trigger_emits_event():
    cfg = EdgeSettings(
        tcp_host="127.0.0.1",
        tcp_port=0,  # OS picks a free port
        site_id="site_test",
        edge_pc_id="edge_test",
        server_base_url="http://127.0.0.1:8000",
    )

    trigger = TcpMotionTrigger(cfg)
    await trigger.start()
    try:
        port = trigger.bound_port
        assert port is not None and port > 0

        _r, w = await asyncio.open_connection("127.0.0.1", port)
        xml = (
            b'<Action><Camera Id="1" Name="Demo Cam" />'
            b'<Policy Id="21" Name="Alarm Demo" />'
            b"<UserString>Alarm</UserString></Action>"
        )
        w.write(xml)
        await w.drain()
        w.close()
        await w.wait_closed()

        evt = await asyncio.wait_for(trigger.queue.get(), timeout=2.0)
        assert evt.source == "tcp"
        assert evt.camera_id == "1"
        assert evt.camera_name == "Demo Cam"
        assert evt.policy_name == "Alarm Demo"
        assert evt.user_string == "Alarm"
    finally:
        await trigger.stop()
