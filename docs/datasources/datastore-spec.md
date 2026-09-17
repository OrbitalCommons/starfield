# `starfield-datastore` — a pull-through artifact cache

> **`starfield-datastore` lives in its own repository:
> <https://github.com/OrbitalCommons/starfield-datastore>.** The authoritative
> spec is `docs/spec.md` there, alongside issue
> [starfield-datastore#1](https://github.com/OrbitalCommons/starfield-datastore/issues/1)
> (transferred from `starfield#189` with its comment trail).
>
> This document is retained only as the design rationale — why the layers are
> arranged as they are, what mirror and proxy modes trade off, and which archive
> failures drove the fail-closed validation rules. **Where the two disagree,
> the spec in that repository wins.**
>
> Consumers, including this repo, depend on it **by crates.io version, never by
> git rev**.

A single layer through which every crate in this workspace fetches every byte
it does not generate itself: ephemeris kernels, catalog shards, PDS tables,
multi-gigabyte mosaics. It resolves a request through a chain — local disk,
then an organisation mirror, then the upstream archive — and populates the
nearer layers as it goes.

The goal is that a working checkout depends on **one** service you control,
not on a dozen archives you do not.

## 1. Why this, and why now

Nine call sites across this workspace fetch data today, plus `starfield`'s own
`data::downloader`, and both write into `~/.cache/starfield` with no
coordination. Each rolled its own retry, staleness and error handling. That was
fine at two crates and is not at eleven.

More importantly, three failure modes hit in a single afternoon of adding data
sources — none of which a naive URL-keyed cache survives:

| Archive | Failure | What a naive cache stores |
|---|---|---|
| `astrogeology.usgs.gov` | **HTTP 200 + generic catalogue page** for an unknown product slug | An HTML page under a `.tif` key |
| LAADS DAAC | **HTTP 200 + HTML login page** for an unauthenticated download | An HTML page under a `.hdf` key |
| LP DAAC `e4ftl01` | Paths silently relocated to a cloud host; old URLs 404 | Nothing, but every pinned URL rots |

A cache that stores a login page under a `.hdf` key is **worse than no cache**:
it converts a loud, immediate auth failure into a silent, persistent data
corruption that survives restarts and gets copied to the org mirror. This is the
single strongest design constraint in this document, and §4 is built around it.

Ephemeris is the flagship case. `de440.bsp` is ~114 MB, immutable, required for
any resolved-body work, and served by a NAIF host outside anyone's control.
There is no reason for every developer and every CI run to fetch it from JPL.

## 2. Model

### 2.1 Artifact

The unit of caching. Not a URL — a URL is one way of *obtaining* an artifact,
and the whole point is to have several.

```rust
pub struct Artifact {
    /// Stable logical key, and the mirror's object path. Never a URL.
    /// e.g. "naif/spk/de440.bsp", "pds/gbat_0001/1995low.tab"
    pub key: ArtifactKey,
    /// Where to get it upstream, in preference order.
    pub sources: Vec<Source>,
    /// What a correct response looks like. See §4.
    pub check: ContentCheck,
    /// When a cached copy may be reused. See §5.
    pub freshness: Freshness,
    /// Provenance, carried into the mirror manifest.
    pub provenance: Provenance,
}
```

The key is deliberately archive-shaped rather than URL-shaped, so that a
relocated upstream (LP DAAC, twice this year) changes a `Source` and not the
cache layout, the mirror, or anyone's pinned digest.

### 2.2 Resolution chain

```
    request(key)
        │
        ├─► 1. local disk        ~/.cache/starfield/blobs/<digest>
        │      hit → verify → return
        │
        ├─► 2. org mirror        s3://org-bucket/<key>   or   https://mirror/<key>
        │      hit → verify → write layer 1 → return
        │
        └─► 3. upstream          Source[0], Source[1], …  (credentialed)
               hit → verify → write layer 1 (+ 2 if writable) → return
```

Each layer is optional and independently configurable. A CI job runs with
layers 1–2 and **layer 3 disabled**, so an accidental upstream fetch is a hard
error rather than a slow test. A laptop offline runs with layer 1 only.

`ResolveOutcome` reports which layer served the request, so a consumer can log
or assert on it — the mirror is not doing its job if everything comes from
layer 3.

### 2.3 Content addressing

Blobs are stored by SHA-256 under `blobs/<first-two-hex>/<digest>`, with the
logical key a symlink or index entry pointing at the digest. Consequences worth
having:

- **Deduplication.** The same PDS table reachable by two URLs is stored once.
- **Atomic publication.** A blob is written to a temp path, verified, then
  renamed — a reader never observes a partial file.
- **Free integrity.** Verifying the cache is walking it and rehashing.
- **Safe concurrency.** Two processes fetching the same artifact race to the
  same final path; the loser's rename is a no-op.

## 3. Credentials

The immediate driver: MODIS land cover needs NASA Earthdata, and getting there
took three attempts against two different auth schemes.

```rust
pub trait CredentialProvider: Send + Sync {
    /// Credential for a host, or None if this provider has none.
    fn credential_for(&self, host: &str) -> Result<Option<Credential>>;
}

pub enum Credential {
    Basic { user: String, secret: Secret },
    Bearer(Secret),
    AwsSigV4 { access_key: String, secret: Secret, region: String },
}
```

Built-in providers, tried in order and merged by host:

| Provider | Source | Notes |
|---|---|---|
| `NetrcProvider` | `~/.netrc` | What Earthdata URS wants |
| `EnvProvider` | `STARFIELD_TOKEN_<HOST>` | CI |
| `OnePasswordProvider` | `op read` | Developer laptops; never writes the secret to disk |
| `AwsProvider` | standard AWS chain | For an S3 mirror |
| `StaticProvider` | constructed in code | Tests |

Rules, all enforceable and all testable:

1. **`Secret` has no `Display` or `Debug` that reveals the value**, does not
   serialise, and zeroes on drop. A credential must be impossible to log by
   accident, because the usual way secrets leak is a `{:?}` in an error path.
2. **Credentials never enter the cache.** Not in the blob, not in the sidecar
   metadata, not in the key. The mirror manifest records *which provider and
   identity* were used, never the secret.
3. **Redirects drop credentials by default.** `--location-trusted` is required
   for Earthdata's URS redirect and is opt-in per source, because the default
   behaviour of forwarding a bearer token to any redirect target is a
   credential-exfiltration primitive.
4. **A missing credential is a distinct error type** from an auth rejection,
   which is distinct from a soft-auth-wall (§4). "You have no Earthdata login"
   and "your Earthdata login was refused" are different problems for the user.

## 4. Validation — the part that matters

Every artifact declares what a correct response looks like. **A response that
fails its check is never written to any cache layer.**

```rust
pub enum ContentCheck {
    /// Best: exact SHA-256. Immutable archive products should all have one.
    Sha256(&'static str),
    /// Leading bytes, for formats with a signature.
    /// HDF4 = 0e031301, PNG = 89504e47, gzip = 1f8b, FITS = "SIMPLE  ="
    Magic(&'static [u8]),
    /// Reject a response smaller than this. Catches truncation and error pages.
    MinBytes(u64),
    /// Reject an HTML body where a binary was expected. Catches BOTH soft-404s
    /// and soft-auth-walls, which are the same shape of failure.
    NotHtml,
    /// Arbitrary predicate, e.g. "parses as a PDS3 label".
    Custom(fn(&[u8]) -> bool),
    All(Vec<ContentCheck>),
}
```

The default for any binary artifact is `All([NotHtml, MinBytes(1024)])`, so the
soft-404 and soft-auth-wall cases fail closed *without anyone remembering to
think about them*. That default is the main reason to have this crate at all.

A failed check produces `DatastoreError::ContentRejected { key, check, got }`
where `got` summarises the response — status, content-type, first bytes as
text if printable. The most common real cause is an unauthenticated download,
and the error should say so when the body looks like HTML containing a login
form.

**Digests are aspirational, not mandatory.** Most products here do not have a
published checksum, and computing one means downloading ~25 GB. The design is
therefore: ship `Magic`/`NotHtml`/`MinBytes` immediately for everything, and
promote an artifact to `Sha256` once it has been fetched and mirrored once —
at which point the mirror *is* the source of the digest. The manifest (§6)
records digests as they are established.

## 5. Freshness

```rust
pub enum Freshness {
    /// Never re-fetch. Almost everything here: a specific PDS volume file, a
    /// numbered SPICE kernel, a published mosaic.
    Immutable,
    /// Re-fetch after a duration. MPCORB, broker alerts.
    Ttl(Duration),
    /// Conditional GET with ETag / Last-Modified.
    Revalidate,
}
```

`Immutable` is the common case and is what makes mirroring safe: an org mirror
of immutable artifacts can never serve something stale. Marking a mutable
product `Immutable` is the one way to get a silently wrong answer out of this
design, so the type should be chosen deliberately per artifact and reviewed.

## 6. The mirror

### 6.1 Manifest

A checked-in TOML file per artifact set, which is simultaneously the pin, the
mirror build input, and the offline allow-list:

```toml
[[artifact]]
key    = "naif/spk/de440.bsp"
sha256 = "…"
bytes  = 119_857_024
freshness = "immutable"
sources = ["https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp"]
provenance = "NAIF generic kernels, DE440"
```

### 6.2 Two serving modes

**Mirror mode — recommended, and what "not relying on services being up"
actually means.** A batch job walks the manifest, fetches everything upstream
with credentials, verifies, and uploads to S3 under the artifact key. Clients
are configured with the bucket as layer 2 and *no layer 3 at all*. There is no
service to keep running: S3 either serves the object or it does not.

```
starfield-datastore mirror --manifest manifests/*.toml --to s3://org-bucket
starfield-datastore verify --manifest manifests/*.toml --at s3://org-bucket
```

**Proxy mode — optional.** A small HTTP service that does layer-3 pull-through
on demand: client requests a key, service checks S3, fetches upstream if
missing, stores, and **302s the client to a presigned S3 URL** so the bytes
never transit the service. This is convenient for artifacts nobody predicted,
but it reintroduces a service dependency, so it should be a supplement to a
mirrored manifest rather than the primary path.

Either way the client sees one interface, and which mode is in use is
configuration, not code.

### 6.3 Access

The mirror bucket may be public-read (simplest, fine for public-domain NASA and
ESA data — which is nearly everything here) or private with SigV4 via
`AwsProvider`. Note that mirroring *is* redistribution: the licence of each
artifact should be recorded in `Provenance` and checked before a set is
mirrored publicly. Everything currently vendored or downloaded by this
workspace is public domain or equivalent, but that is a property to assert per
artifact rather than assume.

## 7. Configuration

One config resolved from, in order: explicit builder call, environment,
`~/.config/starfield/datastore.toml`, defaults.

| Setting | Env | Default |
|---|---|---|
| Cache root | `STARFIELD_CACHE_DIR` | `~/.cache/starfield` |
| Mirror | `STARFIELD_MIRROR` | none |
| Offline | `STARFIELD_OFFLINE` | false |
| Max cache bytes | `STARFIELD_CACHE_MAX` | unbounded |

`STARFIELD_OFFLINE=1` disables layers 2 and 3 and is what CI should set once
the mirror is warm, so that a test which silently started depending on the
network fails loudly.

## 8. Migration

`starfield-datasource-utils` keeps `download_to_file`, `cache_dir`,
`ensure_cache_subdir` and `file_exists_and_not_empty` as thin shims delegating
to the datastore, so none of the nine existing call sites churn. New code uses
the datastore API directly, and the shims are removed when the last caller is
converted.

Ordering, each step independently useful:

1. **Crate skeleton**: `Artifact`, `ContentCheck`, `Freshness`, local layer,
   content-addressed store. No credentials, no mirror. Immediately fixes the
   soft-404/soft-auth-wall class of bug for existing crates via the shims.
2. **Credentials**: providers, `Secret`, redirect policy. Unblocks MODIS (#70)
   properly rather than through a hand-written `~/.netrc`.
3. **Manifests + `verify`**: pins and digests for what is already downloaded.
4. **Mirror mode**: `mirror` subcommand, S3 layer, `STARFIELD_OFFLINE`.
5. **Ephemeris set**: manifest for the SPICE kernels `starfield` needs, and a
   PR upstream so `starfield::Loader` resolves through the datastore too.
6. **Proxy mode**, only if wanted after 4.

## 9. What this is not

- **Not a general HTTP cache.** It caches *named artifacts*, not arbitrary
  requests. API responses from HORIZONS, SBDB and the alert brokers are
  query-shaped and mostly not worth caching; leave them alone.
- **Not a package manager.** No dependency resolution, no versions beyond what
  the key encodes.
- **Not a substitute for the archives.** It is a mirror, and the manifest
  always records where each artifact came from so it can be re-derived.

## 10. Open questions

1. **Crate name.** `starfield-datastore` reads well and does not collide.
   `starfield-cache` undersells the mirror and credential halves.
2. **Where does it live?** This workspace is the only consumer today, but
   `starfield` itself should use it for kernels, which would make it a
   dependency of `starfield` rather than a sibling — an inversion of the
   current direction. Publishing it standalone and having both depend on it is
   probably right, and is a decision to take before step 5, not after.
3. **S3 SDK weight.** `aws-sdk-s3` is a heavy dependency for a crate most users
   will use only for the local layer. Suggest the S3 layer behind a `mirror-s3`
   feature, with plain HTTPS `GET` against the bucket as the default mirror
   transport — which needs no SDK at all and works with any static file host.
4. **Should `verify` be able to repair?** Re-fetching a blob whose digest no
   longer matches is obviously right; silently doing so on every read is not.
   Suggest `verify --repair` as an explicit operation.
