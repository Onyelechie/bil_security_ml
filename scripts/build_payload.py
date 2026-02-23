#!/usr/bin/env python3
"""Read stdin and emit a JSON object with a 'content' field.

This helper is used by the GitHub Actions workflow to avoid fragile
YAML heredocs and quoting issues when building JSON payloads for curl.
"""
import json
import sys


def main() -> int:
    payload = sys.stdin.read()
    # Ensure payload is a string; json.dumps will quote appropriately
    output = json.dumps({"content": payload})
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
