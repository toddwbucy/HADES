# Training artifact publication

## Reproduction and fix

P2 #81 follows the cancellation audit in #77. At source
`74f1558d63ca1940095a8b5acec016bd79b88a37`, private CPU RPC fixtures replaced
checkpoint and embedding serializers with a writer that emits a prefix and then
raises OSError. Both destination-preservation assertions failed: the previous
complete artifact contained only `partial`. A successful-format control passed.
This injected failure is not a disk-full or crash benchmark.

Both writers now serialize to a private temporary file in the destination
directory, flush and fsync its contents, close it, then atomically replace the
destination with `os.replace`. Ordinary exceptions remove the owned staging file.
Checkpoint parent creation retains existing behavior; embedding output still
requires its parent directory to exist. Successful formats remain a torch
checkpoint and raw little-endian float32 embeddings.

## Publication contract and limits

The replacement is the commit point. Serialization, file-sync or replacement
failure before that point leaves an existing destination unchanged, or leaves a
previously absent destination absent. Readers opening the destination see the old
or new complete file; already-open descriptors can continue reading the old inode.
A completed publication is not rolled back because a response is lost or a client
cancels. The operation-ownership contract in #80 governs running work separately.

Published files use the staging file's private 0600 mode. Existing mode, ownership
and extended attributes are not copied. A destination symlink is replaced as a
directory entry; its former target is not overwritten. Parent-directory traversal
is not an untrusted-path sandbox. Concurrent writers are last-successful-replace
wins; the helper is not a cross-process transaction or lock.

File contents are fsynced, but the directory is not. No power-loss durability
guarantee is claimed. A forced kill can leave an unpublished staging file; ordinary
Python exceptions clean it up. This change does not implement orphan retention
management or certify an arbitrary existing checkpoint as recoverable.

## Verification

Ten new private-file tests cover both serializers failing with/without an existing
artifact, both RPCs failing at file-sync/replacement, unchanged successful formats,
checkpoint tensor contents, destination-symlink semantics and private mode.
All 94 training tests pass on this branch. No live service, database, GPU, backup
or production artifact was changed. Tests run with the existing CPU environment:

```sh
python -m pytest -p no:cacheprovider services/tests/test_training*.py -q
```

Any subsequent maintenance risking production data loss requires the owner's
verified active-data snapshot and arranged downtime policy.
