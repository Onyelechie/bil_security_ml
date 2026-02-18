from __future__ import annotations

END_TAG = b"</Action>"

def extract_action_messages(buffer: bytearray) -> list[str]:
    """
    Extract as many complete <Action>...</Action> messages as possible from buffer.

    Why this exists:
    - TCP is a stream, not message-based.
    - One read can contain half an XML message (partial)
    - OR multiple XML messages back-to-back (coalesced)
    """
    messages: list[str] = []

    while True:
        # Find start of next Action
        start = buffer.find(b"<Action")
        if start == -1:
            # No start tag in buffer; drop junk to avoid unbounded growth
            buffer.clear()
            return messages

        # Discard any leading junk before <Action
        if start > 0:
            del buffer[:start]

        # Find end tag
        end = buffer.find(END_TAG)
        if end == -1:
            # Not complete yet; keep buffering
            return messages

        # Include the end tag bytes in the message
        msg_bytes = buffer[: end + len(END_TAG)]
        del buffer[: end + len(END_TAG)]

        # Decode to text (XML is ASCII/UTF-8 safe here)
        messages.append(msg_bytes.decode("utf-8", errors="replace"))
