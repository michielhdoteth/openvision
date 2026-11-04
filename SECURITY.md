# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Open Vision, please report it responsibly.

**Do not open a public issue.**

Instead, email: michiel@openvision.dev

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You should receive a response within 7 days. We will work with you to understand and address the issue before any public disclosure.

## Scope

Open Vision processes video and audio files locally. It does not send data to external services unless you explicitly configure a cloud VLM provider (OpenAI, Groq, etc.). The security surface is:

- **Video/audio processing**: FFmpeg and OpenCV handle file parsing. Vulnerabilities in these dependencies are upstream issues.
- **Local VLM inference**: Models run locally via LM Studio or llama.cpp. No data leaves your machine.
- **Cloud providers**: If you configure `--provider openai` or similar, video frames are sent to that provider's API. This is opt-in and documented.
- **yt-dlp**: Used for downloading videos from URLs. No telemetry is sent.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes       |
| < 1.0   | No        |
