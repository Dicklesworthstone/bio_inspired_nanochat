# Persistent Synaptic Memory: Implemented Semantics and Limits

This note describes `PersistentLifelongMemoryManager`. It is not a claim of cryptographic
multi-tenant isolation.

## Implemented lifecycle

One manager can have one active `(user, model)` pair. Mounting loads that user's saved fast
weights and slow-weight deltas into the model. Switching users first unmounts the model that
actually owns the live state. An implicit same-user switch to another model is rejected so an
incompatible destination cannot destroy the active session. Unmounting can transfer a bounded fraction of fast weights into
the user's durable slow deltas, persists the partition, clears fast weights, and subtracts the
mounted slow deltas so the model returns to its base state.

Partition data is loaded with PyTorch's restricted `weights_only=True` mode. Layer indices,
tensor shapes, floating dtypes, and finite values are validated before a different active user
is disturbed. Partitions are written to a sibling temporary file and atomically replaced only
after serialization succeeds. A storage failure during unmount leaves both the prior durable
copy and the live session available for retry.

## Identifier and storage limits

Filenames use the first 24 hexadecimal characters of SHA-256 over a fixed project string and
the supplied user ID. This avoids putting a raw user ID in the filename; it is not encryption,
keyed pseudonymization, or protection against offline guessing of low-entropy identifiers. The
`.pt` partition contents are not encrypted by this module. Storage permissions, encryption at
rest, key management, authentication, and authorization are deployment responsibilities that
remain unimplemented here.

The configured `max_delta_norm` bounds each stored slow-delta tensor. The implementation does
not impose a five-megabyte per-user quota, daily decay, low-rank-factor cap, or per-user replay
buffer. Those features must not be inferred from this component.

## Erasure behavior

`forget_user` unmounts and clears an active user's live slow/fast state before unlinking that
user's partition; callers must supply the exact active model. For an inactive user it unlinks
the matching partition when present. The module does not overwrite filesystem blocks, erase
backups, verify deletion across replicas, or run a post-erasure behavioral privacy audit.

The tests cover state separation across normal mount/unmount sequences and fail-closed behavior
for wrong-user, wrong-model, corrupt-partition, and storage-failure paths. They do not prove zero
cross-talk against every failure mode or resistance to memory extraction attacks.
