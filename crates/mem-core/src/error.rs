use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid input: {0}")]
    Invalid(String),
    #[error("rejected: {0}")]
    Rejected(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("embedding error: {0}")]
    Embed(String),
    #[error("storage error: {0}")]
    Storage(String),
    #[error("index error: {0}")]
    Index(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("internal error: {0}")]
    Internal(String),
}

impl Error {
    /// Stable machine-readable code.
    pub fn code(&self) -> &'static str {
        match self {
            Error::Invalid(_) => "invalid",
            Error::Rejected(_) => "rejected",
            Error::NotFound(_) => "not_found",
            Error::Embed(_) => "embed",
            Error::Storage(_) => "storage",
            Error::Index(_) => "index",
            Error::Io(_) => "io",
            Error::Internal(_) => "internal",
        }
    }
}
