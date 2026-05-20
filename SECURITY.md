# Security policy

## Supported versions

This project is in early development. Only the `main` branch is supported.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.**

Report security issues by emailing the maintainer at the address on the
[GitHub profile](https://github.com/RahulModugula). Include:

- A description of the issue
- A minimal reproducer (or steps to reproduce)
- The version / commit you tested against
- Your assessment of the impact

You should expect a first response within 5 business days. If the issue
is confirmed, a fix will be coordinated and disclosed publicly only
after the fix is available.

## Out of scope

- Bugs in third-party dependencies (please report those upstream).
- Issues that require physical access or admin privileges on the host
  running the service.
- Denial of service via large uploads — the service has a configurable
  `max_upload_bytes` limit; deployers are expected to set this
  appropriately for their environment and put a load balancer in front.
