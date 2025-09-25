use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

use super::errors::OnnxClassifierError;

/// File-based artefact that must match a recorded checksum before loading.
#[derive(Debug, Clone)]
pub struct OnnxArtefact {
    /// Location of the artefact on disk.
    pub path: PathBuf,
    /// Expected SHA-256 checksum expressed as lowercase hexadecimal.
    pub sha256: String,
}

impl OnnxArtefact {
    /// Verifies the artefact checksum against the expected digest.
    ///
    /// # Errors
    ///
    /// Returns `ChecksumMismatch` when the computed digest does not match `sha256` and propagates I/O errors while reading the file.
    pub fn verify(&self) -> Result<(), OnnxClassifierError> {
        let actual = compute_sha256(&self.path)?;
        if actual == normalise_hex(&self.sha256) {
            Ok(())
        } else {
            Err(OnnxClassifierError::ChecksumMismatch {
                path: self.path.clone(),
                expected: normalise_hex(&self.sha256),
                actual,
            })
        }
    }
}

/// Computes the SHA-256 digest of the file at `path`.
///
/// # Errors
///
/// Returns I/O errors from opening or reading the file.
pub fn compute_sha256(path: &Path) -> Result<String, OnnxClassifierError> {
    let file = File::open(path).map_err(|source| OnnxClassifierError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8192];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|source| OnnxClassifierError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            break;
        }
        let chunk = buffer.get(..read).ok_or_else(|| OnnxClassifierError::Io {
            path: path.to_path_buf(),
            source: std::io::Error::other("read reported bytes beyond buffer length"),
        })?;
        hasher.update(chunk);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn normalise_hex(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::normalise_hex;

    #[test]
    fn normalise_hex_lowercases_and_trims() {
        assert_eq!(normalise_hex(" ABCDEF "), "abcdef");
    }
}
