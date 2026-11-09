# Security Policy

rebot-companion sends commands to a physical robot arm. Bugs that let a client bypass the
lease, safety or motion limits of rebot-armd, or that let the MCP server do something its
flags do not allow (for example clear a fault or take control without `--allow-clear-fault` /
`--allow-preempt`), are treated as security issues.

## Reporting a vulnerability

Please do not open a public issue for a vulnerability.

- Use GitHub private vulnerability reporting: the **Security** tab of
  https://github.com/Seeed-Solution/rebot-companion, then **Report a vulnerability**.
- If that form is not available, open an issue that only asks for a private contact,
  without technical details.

Include the affected version or commit, the setup (rebot-armd version, how the SDK or MCP
server was started), steps to reproduce, and the effect you observed.

The report is kept private until a fix is available on the default branch.

## Supported versions

Only the latest commit on the default branch is supported.

## Scope

Issues in rebot-armd itself belong to https://github.com/Seeed-Solution/rebot-armd.
