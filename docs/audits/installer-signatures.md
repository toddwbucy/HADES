# Installer repository authentication (#60)

## Finding and scope

At `98c214d`, the fresh-VPS Dockerfile uses `trusted=yes` for the remote
ArangoDB repository. The README recommends the bypass after an expired-key
error and downloads a keyring without selecting it in the source entry.
Severity: P2. This weakens authentication before privileged package installation;
it is not evidence of a compromised package or running installation.

[APT's source options](https://manpages.debian.org/bookworm/apt/sources.list.5.en.html)
define `trusted=yes` as bypassing authentication decisions and `signed-by` as
restricting repository verification to selected keys. The
[ArangoDB Linux instructions](https://docs.arango.ai/arangodb/3.12/operations/installation/linux/)
expect package-manager verification. No current vendor-key expiry is asserted.

## Change

Both paths fetch the vendor key over HTTPS, require successful conversion,
make the keyring readable by APT and reference it with `signed-by`. Repository
update uses `APT::Update::Error-Mode=any` so an update error stops installation.
The documentation directs signature failures to current vendor instructions;
no authentication bypass is offered.

## Verification and limits

`python3 -m unittest discover -s scripts/tests -p 'test_installer_signatures.py' -v`
extracts the source options from both maintained installation paths and uses
those options with a disposable, file-only APT repository and ephemeral signing
key. It requires acceptance of valid metadata and rejection of unsigned,
tampered and missing-key metadata. It uses private APT state/configuration,
no host sources or trusted keys, no package installation and no network.
The fixture is included by the existing operational-contract CI discovery;
missing tools fail in CI and explicitly skip on non-APT developer hosts.

The active audit server lacks `apt-get`, so local execution reported a skip;
Ubuntu CI must supply the behavioral evidence before this finding is closed.
Shell syntax and whitespace were checked locally. The full fresh-VPS container
build, live vendor availability and real systemd readiness are separate checks
and have not been performed by this change. No host packages, configuration,
services or production data were changed.
