"""
Area B - Edge Agent package.

This package contains the on-site edge service that:
- listens for motion events over TCP
- pulls camera frames over RTSP
- runs detection + decision logic
- sends alerts + heartbeats to the central server
"""
