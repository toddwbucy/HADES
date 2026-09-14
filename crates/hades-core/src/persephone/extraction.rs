//! Persephone Extraction Client — gRPC client for document content extraction.
//!
//! Connects to the Persephone extraction service over a Unix domain socket
//! or TCP endpoint.  Wraps the generated tonic client with connection
//! management, health checking, and ergonomic Rust types.

use std::path::{Path, PathBuf};
use std::time::Duration;

use hades_proto::extraction::extraction_service_client::ExtractionServiceClient;
use hades_proto::extraction::{
    CapabilitiesRequest, ExtractRequest, ExtractResponse, ExtractorInfo, SourceType,
};
use hyper_util::rt::TokioIo;
use tonic::transport::{Channel, Endpoint, Uri};
use tower::service_fn;
use tracing::{debug, info, instrument, warn};

/// Default timeout for extraction requests (large PDFs can be slow).
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600); // 10 min
/// Default connection timeout.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Environment variable naming the extraction service endpoint.
///
/// `services/extraction/config.py` has honoured this variable since it was
/// written, while this client hardcoded the socket path, so the two halves of
/// one contract disagreed. The only reachable endpoint was the root-created
/// `/run/hades/extractor.sock`, which a user-level deployment cannot write, so
/// `hades ingest` had no way to reach an extractor running as the operator.
const ENDPOINT_ENV: &str = "HADES_EXTRACTOR_SOCKET";

/// Compiled-in socket, used when [`ENDPOINT_ENV`] is unset.
///
/// Matches the system install, where `tmpfiles.d` creates `/run/hades` and the
/// service unit binds inside it.
const DEFAULT_SOCKET: &str = "/run/hades/extractor.sock";

/// Configuration for the extraction client.
#[derive(Debug, Clone)]
pub struct ExtractionClientConfig {
    /// Unix socket path or TCP address for the extraction service.
    pub endpoint: ExtractionEndpoint,
    /// Request timeout.
    pub timeout: Duration,
    /// Connection timeout.
    pub connect_timeout: Duration,
}

/// Endpoint for the extraction service.
#[derive(Debug, Clone)]
pub enum ExtractionEndpoint {
    /// Unix domain socket path.
    Unix(PathBuf),
    /// TCP address (e.g. "http://localhost:50052").
    Tcp(String),
}

impl ExtractionClientConfig {
    /// Build a config from the environment, falling back to [`DEFAULT_SOCKET`].
    ///
    /// Accepts the same endpoint spellings as the embedding client:
    /// `http://`, `https://`, `unix:///path`, or a bare absolute path.
    ///
    /// A malformed value is an error rather than a silent fall back to the
    /// default. Falling back would connect somewhere the operator did not ask
    /// for and report success, which is the failure shape this tree has paid
    /// for repeatedly: the message names the value that was rejected.
    pub fn from_env() -> Result<Self, ExtractionError> {
        let endpoint = match std::env::var(ENDPOINT_ENV) {
            Ok(raw) if !raw.trim().is_empty() => parse_endpoint(raw.trim())?,
            _ => ExtractionEndpoint::Unix(PathBuf::from(DEFAULT_SOCKET)),
        };
        Ok(Self {
            endpoint,
            ..Default::default()
        })
    }
}

/// Parse an endpoint string into a [`ExtractionEndpoint`].
fn parse_endpoint(raw: &str) -> Result<ExtractionEndpoint, ExtractionError> {
    if raw.starts_with("http://") || raw.starts_with("https://") {
        Ok(ExtractionEndpoint::Tcp(raw.to_string()))
    } else if let Some(path) = raw.strip_prefix("unix://") {
        // Absolute after the scheme too. `unix://run/extractor.sock` resolves
        // from the process working directory, so the daemon and the CLI would
        // reach different sockets from the same configuration, and a relative
        // socket path means something different from every directory it is run
        // in. The bare-path branch below already requires this.
        if !path.starts_with('/') {
            return Err(ExtractionError::InvalidResponse(format!(
                "{ENDPOINT_ENV} must name an absolute path after unix://; got '{raw}'"
            )));
        }
        Ok(ExtractionEndpoint::Unix(PathBuf::from(path)))
    } else if raw.starts_with('/') {
        Ok(ExtractionEndpoint::Unix(PathBuf::from(raw)))
    } else {
        Err(ExtractionError::InvalidResponse(format!(
            "{ENDPOINT_ENV} must start with http://, https://, unix://, or be an \
             absolute path; got '{raw}'"
        )))
    }
}

impl Default for ExtractionClientConfig {
    fn default() -> Self {
        Self {
            endpoint: ExtractionEndpoint::Unix(PathBuf::from(DEFAULT_SOCKET)),
            timeout: DEFAULT_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }
}

/// Error type for extraction client operations.
#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    /// gRPC transport/connection error.
    #[error("connection error: {0}")]
    Connection(#[from] tonic::transport::Error),

    /// gRPC status error from the service.
    ///
    /// Boxed because `tonic::Status` is 176 bytes and this variant would
    /// otherwise set the size of every `Result` in the module, which is what
    /// `clippy::result_large_err` objects to.
    #[error("service error: {0}")]
    Status(#[source] Box<tonic::Status>),

    /// Invalid response from the service.
    #[error("invalid response: {0}")]
    InvalidResponse(String),
}

impl From<tonic::Status> for ExtractionError {
    fn from(status: tonic::Status) -> Self {
        Self::Status(Box::new(status))
    }
}

/// Client for the Persephone extraction service.
///
/// Provides ergonomic methods for extracting structured content
/// from documents and querying extractor capabilities.
#[derive(Clone)]
pub struct ExtractionClient {
    inner: ExtractionServiceClient<Channel>,
    config: ExtractionClientConfig,
}

impl ExtractionClient {
    /// Connect to the extraction service.
    #[instrument(skip_all)]
    pub async fn connect(config: ExtractionClientConfig) -> Result<Self, ExtractionError> {
        let channel = match &config.endpoint {
            ExtractionEndpoint::Unix(path) => {
                debug!(socket = %path.display(), "connecting to extraction service via UDS");
                Self::connect_unix(path, &config).await?
            }
            ExtractionEndpoint::Tcp(addr) => {
                debug!(addr, "connecting to extraction service via TCP");
                Self::connect_tcp(addr, &config).await?
            }
        };

        let inner = ExtractionServiceClient::new(channel);
        info!("connected to extraction service");

        Ok(Self { inner, config })
    }

    /// Connect to the extraction service with default configuration.
    ///
    /// Honours `HADES_EXTRACTOR_SOCKET`; see
    /// [`ExtractionClientConfig::from_env`]. Prefer [`Self::connect_at`] where a
    /// [`crate::config::HadesConfig`] is in hand, so the YAML key is honoured
    /// too.
    pub async fn connect_default() -> Result<Self, ExtractionError> {
        Self::connect(ExtractionClientConfig::from_env()?).await
    }

    /// Connect to the extraction service named by an endpoint string.
    ///
    /// Takes the same spellings as the environment variable, and is what a
    /// caller holding `config.extraction.service.socket` should use: env
    /// overrides are already folded into that value by `apply_env_overrides`.
    pub async fn connect_at(endpoint: &str) -> Result<Self, ExtractionError> {
        let config = ExtractionClientConfig {
            endpoint: parse_endpoint(endpoint)?,
            ..Default::default()
        };
        Self::connect(config).await
    }

    /// Connect to an extraction service at the given Unix socket path.
    pub async fn connect_unix_at(path: impl Into<PathBuf>) -> Result<Self, ExtractionError> {
        let config = ExtractionClientConfig {
            endpoint: ExtractionEndpoint::Unix(path.into()),
            ..Default::default()
        };
        Self::connect(config).await
    }

    /// Extract structured content from a file on the service's filesystem.
    ///
    /// The extraction service must have read access to `file_path`.
    #[instrument(skip(self, file_path), fields(path = %file_path.as_ref().display()))]
    pub async fn extract_file(
        &self,
        file_path: impl AsRef<Path>,
        options: ExtractOptions,
    ) -> Result<ExtractResult, ExtractionError> {
        let request = ExtractRequest {
            file_path: file_path.as_ref().to_string_lossy().to_string(),
            content: Vec::new(),
            source_type: options.source_type.unwrap_or(SourceType::Unknown).into(),
            extract_tables: options.extract_tables,
            extract_equations: options.extract_equations,
            extract_images: options.extract_images,
            use_ocr: options.use_ocr,
        };

        self.do_extract(request).await
    }

    /// Extract structured content from in-memory bytes.
    ///
    /// `file_name` is used for source-type detection when `source_type` is not
    /// explicitly set.
    #[instrument(skip(self, content), fields(file_name, content_len = content.len()))]
    pub async fn extract_bytes(
        &self,
        file_name: &str,
        content: Vec<u8>,
        options: ExtractOptions,
    ) -> Result<ExtractResult, ExtractionError> {
        let request = ExtractRequest {
            file_path: file_name.to_string(),
            content,
            source_type: options.source_type.unwrap_or(SourceType::Unknown).into(),
            extract_tables: options.extract_tables,
            extract_equations: options.extract_equations,
            extract_images: options.extract_images,
            use_ocr: options.use_ocr,
        };

        self.do_extract(request).await
    }

    /// Query the extractor's capabilities and supported formats.
    #[instrument(skip(self))]
    pub async fn capabilities(&self) -> Result<ExtractorInfo, ExtractionError> {
        let response = self
            .inner
            .clone()
            .capabilities(CapabilitiesRequest {})
            .await?
            .into_inner();

        debug!(
            extensions = ?response.supported_extensions,
            features = ?response.features,
            gpu = response.gpu_available,
            "extractor capabilities retrieved"
        );

        Ok(response)
    }

    /// Check if the service is reachable by calling Capabilities.
    ///
    /// Uses a short timeout so a stalled service returns `false` quickly
    /// rather than blocking for the full request timeout.
    pub async fn health_check(&self) -> bool {
        match tokio::time::timeout(Duration::from_secs(5), self.capabilities()).await {
            Ok(Ok(_)) => true,
            Ok(Err(e)) => {
                warn!(error = %e, "extraction service health check failed");
                false
            }
            Err(_) => {
                warn!("extraction service health check timed out");
                false
            }
        }
    }

    /// Get the configured endpoint.
    pub fn endpoint(&self) -> &ExtractionEndpoint {
        &self.config.endpoint
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    async fn do_extract(&self, request: ExtractRequest) -> Result<ExtractResult, ExtractionError> {
        let response: ExtractResponse = self.inner.clone().extract(request).await?.into_inner();

        if response.full_text.is_empty()
            && response.tables.is_empty()
            && response.equations.is_empty()
            && response.images.is_empty()
        {
            return Err(ExtractionError::InvalidResponse(
                "extraction returned no content (empty text, no tables, equations, or images)"
                    .to_string(),
            ));
        }

        let source_type = response
            .source_type
            .try_into()
            .unwrap_or(SourceType::Unknown);

        debug!(
            text_len = response.full_text.len(),
            tables = response.tables.len(),
            equations = response.equations.len(),
            images = response.images.len(),
            source_type = ?source_type,
            "extraction complete"
        );

        Ok(ExtractResult {
            full_text: response.full_text,
            tables: response.tables,
            equations: response.equations,
            images: response.images,
            metadata: response.metadata,
            source_type,
        })
    }

    async fn connect_unix(
        path: &Path,
        config: &ExtractionClientConfig,
    ) -> Result<Channel, tonic::transport::Error> {
        let path = path.to_path_buf();

        // tonic requires a URI even for UDS; the authority is ignored.
        let channel = Endpoint::from_static("http://[::]:50051")
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .connect_with_connector(service_fn(move |_: Uri| {
                let path = path.clone();
                async move {
                    let stream = tokio::net::UnixStream::connect(path).await?;
                    Ok::<_, std::io::Error>(TokioIo::new(stream))
                }
            }))
            .await?;

        Ok(channel)
    }

    async fn connect_tcp(
        addr: &str,
        config: &ExtractionClientConfig,
    ) -> Result<Channel, tonic::transport::Error> {
        let channel = Endpoint::from_shared(addr.to_string())?
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .connect()
            .await?;

        Ok(channel)
    }
}

/// Options for extraction requests.
#[derive(Debug, Clone, Default)]
pub struct ExtractOptions {
    /// Source type hint. `None` means auto-detect.
    pub source_type: Option<SourceType>,
    /// Whether to extract tables.
    pub extract_tables: bool,
    /// Whether to extract equations.
    pub extract_equations: bool,
    /// Whether to extract images/figures.
    pub extract_images: bool,
    /// Whether to use OCR for scanned content.
    pub use_ocr: bool,
}

impl ExtractOptions {
    /// Create options that extract all supported content types.
    ///
    /// OCR is left disabled because it is expensive and significantly
    /// increases extraction time.  Enable it explicitly when needed:
    /// `ExtractOptions { use_ocr: true, ..ExtractOptions::all() }`.
    pub fn all() -> Self {
        Self {
            source_type: None,
            extract_tables: true,
            extract_equations: true,
            extract_images: true,
            use_ocr: false,
        }
    }
}

/// Result of an extraction operation.
#[derive(Debug, Clone)]
pub struct ExtractResult {
    /// Full extracted text content.
    pub full_text: String,
    /// Extracted tables.
    pub tables: Vec<hades_proto::extraction::Table>,
    /// Extracted equations.
    pub equations: Vec<hades_proto::extraction::Equation>,
    /// Extracted image references.
    pub images: Vec<hades_proto::extraction::ImageRef>,
    /// Additional metadata from the extraction process.
    pub metadata: std::collections::HashMap<String, String>,
    /// Source type used for extraction.
    pub source_type: SourceType,
}

impl std::fmt::Debug for ExtractionClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtractionClient")
            .field("endpoint", &self.config.endpoint)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `?` at every gRPC call site depends on this conversion.
    ///
    /// It used to come from thiserror's `#[from]`. Boxing the variant to keep
    /// `clippy::result_large_err` quiet removed the derive, and the hand-written
    /// replacement was the only runtime-visible change in that commit and the
    /// one part nothing exercised. A refactor that drops the manual impl breaks
    /// every call site, and the suite would have stayed green.
    #[test]
    fn status_converts_and_keeps_its_message() {
        let err: ExtractionError = tonic::Status::not_found("no such checkpoint").into();
        assert!(matches!(err, ExtractionError::Status(_)));
        assert!(
            err.to_string().contains("no such checkpoint"),
            "boxing must not swallow the status message, got: {err}"
        );
    }

    /// The boxed status stays reachable as a source, which `#[from]` gave for
    /// free and `#[source]` has to be asked for.
    #[test]
    fn status_is_reachable_as_a_source() {
        let err: ExtractionError = tonic::Status::internal("backend fell over").into();
        let source = std::error::Error::source(&err).expect("status must remain the source");
        assert!(source.to_string().contains("backend fell over"));
    }

    #[test]
    fn parse_endpoint_accepts_the_four_spellings() {
        assert!(matches!(
            parse_endpoint("http://127.0.0.1:50052").unwrap(),
            ExtractionEndpoint::Tcp(ref a) if a == "http://127.0.0.1:50052"
        ));
        assert!(matches!(
            parse_endpoint("https://extractor.internal").unwrap(),
            ExtractionEndpoint::Tcp(_)
        ));
        assert!(matches!(
            parse_endpoint("unix:///run/hades/extractor.sock").unwrap(),
            ExtractionEndpoint::Unix(ref p) if p == Path::new("/run/hades/extractor.sock")
        ));
        assert!(matches!(
            parse_endpoint("/home/todd/.local/share/hades/run/extractor.sock").unwrap(),
            ExtractionEndpoint::Unix(ref p)
                if p == Path::new("/home/todd/.local/share/hades/run/extractor.sock")
        ));
    }

    #[test]
    fn parse_endpoint_rejects_a_relative_path_after_the_unix_scheme() {
        // `unix://` plus a relative path resolves from the working directory, so
        // the daemon and the CLI would reach different sockets from one value.
        let err = parse_endpoint("unix://run/extractor.sock").expect_err("must reject");
        assert!(err.to_string().contains("absolute"), "{err}");
    }

    #[test]
    fn parse_endpoint_rejects_a_relative_path_naming_the_value() {
        // Rejected rather than resolved against the cwd: a relative socket path
        // means something different from every directory the CLI is run in.
        let err = parse_endpoint("run/extractor.sock").expect_err("must reject");
        let message = err.to_string();
        assert!(message.contains("run/extractor.sock"), "{message}");
        assert!(message.contains(ENDPOINT_ENV), "{message}");
    }

    #[test]
    fn from_env_falls_back_to_the_compiled_in_socket() {
        // Env vars are process-global, so this test reads rather than writes:
        // asserting the fallback under a set variable would race any other test
        // that sets it. The unset case is the one worth pinning, because it is
        // what the system install depends on.
        if std::env::var_os(ENDPOINT_ENV).is_some() {
            return;
        }
        let config = ExtractionClientConfig::from_env().expect("unset env must succeed");
        assert!(matches!(
            config.endpoint,
            ExtractionEndpoint::Unix(ref p) if p == Path::new(DEFAULT_SOCKET)
        ));
    }
}
