use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

/// Parsed representation of a local or remote `.safetensors` checkpoint source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteSpec {
    /// Local filesystem path.
    LocalPath(PathBuf),
    /// `hf://<org>/<repo>[@<revision>]` - auto-discovers the `.safetensors` LoRA file via HF Hub API.
    HuggingFaceRepo {
        org: String,
        repo: String,
        revision: String,
    },
    /// `hf://<org>/<repo>[@<revision>]/<filename.safetensors>` - resolves directly on HF Hub.
    HuggingFaceFile {
        org: String,
        repo: String,
        revision: String,
        filename: String,
    },
    /// Direct `https://` or `http://` URL to a `.safetensors` file.
    DirectUrl { url: String },
}

/// Normalize HuggingFace web `/blob/` URLs into raw binary `/resolve/` URLs so copy-pasted
/// browser links download actual `.safetensors` weights rather than HTML pages.
pub fn normalize_direct_url(url: &str) -> String {
    let trimmed = url.trim();
    for prefix in [
        "https://huggingface.co/",
        "http://huggingface.co/",
        "https://hf.co/",
        "http://hf.co/",
    ] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            // Pattern: <org>/<repo>/blob/<rev>/<path...>
            let parts: Vec<&str> = rest.splitn(5, '/').collect();
            if parts.len() == 5 && parts[2] == "blob" && !parts[0].is_empty() && !parts[1].is_empty()
            {
                return format!(
                    "{}{}/{}/resolve/{}/{}",
                    prefix, parts[0], parts[1], parts[3], parts[4]
                );
            }
        }
    }
    trimmed.to_string()
}

fn validate_hf_segment(seg: &str, kind: &str) -> Result<(), String> {
    if seg.is_empty()
        || seg == "."
        || seg == ".."
        || seg.contains("..")
        || seg.contains('\0')
        || seg.contains('\r')
        || seg.contains('\n')
        || seg.chars().any(char::is_whitespace)
    {
        return Err(format!("Invalid HuggingFace {} segment: '{}'", kind, seg));
    }
    Ok(())
}

/// Parse a CLI checkpoint argument into a `RemoteSpec`.
///
/// Supported forms:
/// - `hf://<org>/<repo>` (or `hf://<org>/<repo>@<revision>`)
/// - `hf://<org>/<repo>/<filename.safetensors>` (including nested subpaths)
/// - `https://.../*.safetensors` (or `http://...`)
/// - Local filesystem paths
pub fn parse_source_spec(input: &str) -> Result<RemoteSpec, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("Checkpoint source path cannot be empty".to_string());
    }

    if let Some(rest) = trimmed.strip_prefix("hf://") {
        let clean = rest.trim_end_matches('/');
        let mut iter = clean.splitn(3, '/');
        let org = iter.next().unwrap_or("");
        let repo_and_rev = iter.next().unwrap_or("");
        if org.is_empty() || repo_and_rev.is_empty() {
            return Err(format!(
                "Invalid hf:// URI '{}': expected hf://<org>/<repo>[/<file.safetensors>]",
                trimmed
            ));
        }
        let (repo, revision) = match repo_and_rev.split_once('@') {
            Some((r, rev)) if !r.is_empty() && !rev.is_empty() => (r, rev),
            Some(_) => {
                return Err(format!(
                    "Invalid @revision in hf:// URI '{}': expected hf://<org>/<repo>@<revision>",
                    trimmed
                ))
            }
            None => (repo_and_rev, "main"),
        };

        validate_hf_segment(org, "org")?;
        validate_hf_segment(repo, "repo")?;
        validate_hf_segment(revision, "revision")?;

        if let Some(file_part) = iter.next() {
            let filename = file_part.trim_matches('/');
            if filename.is_empty() {
                return Ok(RemoteSpec::HuggingFaceRepo {
                    org: org.to_string(),
                    repo: repo.to_string(),
                    revision: revision.to_string(),
                });
            }
            if filename.split('/').any(|p| p.is_empty() || p == "." || p == "..")
                || filename.contains('\0')
                || filename.contains('\r')
                || filename.contains('\n')
            {
                return Err(format!(
                    "Invalid filename '{}' in hf:// URI '{}'",
                    filename, trimmed
                ));
            }
            return Ok(RemoteSpec::HuggingFaceFile {
                org: org.to_string(),
                repo: repo.to_string(),
                revision: revision.to_string(),
                filename: filename.to_string(),
            });
        }

        return Ok(RemoteSpec::HuggingFaceRepo {
            org: org.to_string(),
            repo: repo.to_string(),
            revision: revision.to_string(),
        });
    }

    if trimmed.starts_with("https://") || trimmed.starts_with("http://") {
        if trimmed.contains('\0') || trimmed.contains('\r') || trimmed.contains('\n') {
            return Err("URL contains invalid control characters".to_string());
        }
        return Ok(RemoteSpec::DirectUrl {
            url: normalize_direct_url(trimmed),
        });
    }

    Ok(RemoteSpec::LocalPath(PathBuf::from(input)))
}

/// Construct the HuggingFace Hub raw resolve URL for a file in `<org>/<repo>`.
pub fn hf_resolve_url(org: &str, repo: &str, revision: &str, filename: &str) -> String {
    format!(
        "https://huggingface.co/{}/{}/resolve/{}/{}",
        org, repo, revision, filename
    )
}

/// Construct the HuggingFace Hub model metadata API URL for `<org>/<repo>`.
pub fn hf_api_model_url(org: &str, repo: &str, revision: &str) -> String {
    if revision == "main" {
        format!("https://huggingface.co/api/models/{}/{}", org, repo)
    } else {
        format!(
            "https://huggingface.co/api/models/{}/{}/revision/{}",
            org, repo, revision
        )
    }
}

/// Return true if `url` targets `huggingface.co` or `hf.co` (where `HF_TOKEN` may be sent).
pub fn is_huggingface_url(url: &str) -> bool {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or("");
    let host = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    host == "huggingface.co"
        || host == "www.huggingface.co"
        || host == "hf.co"
        || host == "www.hf.co"
}

/// Resolve the cache root directory `${TMPDIR:-/tmp}/loraspectra-cache`.
pub fn resolve_cache_dir_from_tmpdir(tmpdir: Option<&str>) -> PathBuf {
    let base = tmpdir
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("/tmp");
    PathBuf::from(base).join("loraspectra-cache")
}

/// Return the active cache root directory from environment (`${TMPDIR:-/tmp}/loraspectra-cache`).
pub fn default_cache_dir() -> PathBuf {
    resolve_cache_dir_from_tmpdir(env::var("TMPDIR").ok().as_deref())
}

/// Read `HF_TOKEN` (falling back to `HUGGING_FACE_HUB_TOKEN`) if set and valid.
pub fn read_hf_token_from_env() -> Option<String> {
    env::var("HF_TOKEN")
        .ok()
        .or_else(|| env::var("HUGGING_FACE_HUB_TOKEN").ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && !s.contains('\r') && !s.contains('\n') && !s.contains('\0'))
}

/// Sanitize an identifier component so it is safe as a single filesystem path component.
pub fn sanitize_cache_component(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    while out.contains("..") {
        out = out.replace("..", "__");
    }
    let trimmed = out.trim_matches('.');
    if trimmed.is_empty() {
        "artifact".to_string()
    } else {
        trimmed.to_string()
    }
}

fn fnv1a_hex(input: &str) -> String {
    let mut hash: u64 = 14_695_981_039_346_656_037;
    for &b in input.as_bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    format!("{:016x}", hash)
}

/// Compute the deterministic local cache path under `cache_root` for a remote `.safetensors` resource.
pub fn cache_path_for_hf(
    cache_root: &Path,
    org: &str,
    repo: &str,
    revision: &str,
    filename: &str,
) -> PathBuf {
    let org_clean = sanitize_cache_component(org);
    let repo_clean = if revision == "main" {
        sanitize_cache_component(repo)
    } else {
        format!(
            "{}_rev_{}",
            sanitize_cache_component(repo),
            sanitize_cache_component(revision)
        )
    };
    let mut file_clean = sanitize_cache_component(&filename.replace('/', "__"));
    if !file_clean.ends_with(".safetensors") {
        file_clean.push_str(".safetensors");
    }
    cache_root.join(org_clean).join(repo_clean).join(file_clean)
}

/// Compute the deterministic local cache path under `cache_root` for a direct URL.
pub fn cache_path_for_url(cache_root: &Path, url: &str) -> PathBuf {
    let without_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);
    let no_query = without_scheme.split(['?', '#']).next().unwrap_or(without_scheme);
    let (host, path_part) = no_query.split_once('/').unwrap_or((no_query, "adapter.safetensors"));
    let host_clean = sanitize_cache_component(host);
    let leaf = path_part.rsplit('/').next().unwrap_or("adapter.safetensors");
    let mut leaf_clean = sanitize_cache_component(leaf);
    if !leaf_clean.ends_with(".safetensors") {
        leaf_clean.push_str(".safetensors");
    }
    let url_fingerprint = &fnv1a_hex(url)[..12];
    cache_root
        .join("url")
        .join(host_clean)
        .join(format!("{}_{}", url_fingerprint, leaf_clean))
}

/// Build arguments for `curl` to fetch a URL (either to stdout when `output_path` is `None`,
/// or directly to a file when `output_path` is `Some`).
///
/// Attaches `Authorization: Bearer <HF_TOKEN>` ONLY when `hf_token` is present AND `url` points
/// to a HuggingFace host (`huggingface.co` / `hf.co`).
pub fn build_curl_args(
    url: &str,
    output_path: Option<&Path>,
    hf_token: Option<&str>,
) -> Vec<String> {
    let mut args = vec!["-fsSL".to_string()];
    if let Some(token) = hf_token {
        if !token.is_empty() && is_huggingface_url(url) {
            args.push("-H".to_string());
            args.push(format!("Authorization: Bearer {}", token));
        }
    }
    if let Some(out) = output_path {
        args.push("-o".to_string());
        args.push(out.to_string_lossy().into_owned());
    }
    args.push("--".to_string());
    args.push(url.to_string());
    args
}

/// Extract all `"rfilename": "..."` entries from a HuggingFace `/api/models/<org>/<repo>` JSON payload
/// and select the best `.safetensors` LoRA file without external JSON dependencies.
pub fn select_safetensors_from_hf_api_json(json: &str) -> Option<String> {
    let rfilenames = extract_rfilenames_from_hf_json(json);
    let candidates: Vec<String> = rfilenames
        .into_iter()
        .filter(|name| {
            name.ends_with(".safetensors")
                && !name.starts_with('/')
                && !name.split('/').any(|seg| seg.is_empty() || seg == "." || seg == "..")
        })
        .collect();

    if candidates.is_empty() {
        return None;
    }

    // 1. Standard PEFT root filename: adapter_model.safetensors
    if let Some(exact) = candidates.iter().find(|c| c.as_str() == "adapter_model.safetensors") {
        return Some(exact.clone());
    }

    // 2. Nested PEFT filename: */adapter_model.safetensors
    if let Some(nested_peft) = candidates
        .iter()
        .find(|c| c.ends_with("/adapter_model.safetensors"))
    {
        return Some(nested_peft.clone());
    }

    let is_sharded_base = |name: &str| -> bool {
        let lower = name.to_ascii_lowercase();
        lower.contains("-of-0") || lower.contains("model.safetensors.index")
    };

    // 3. Root-level .safetensors file explicitly containing "lora" (Kohya / ComfyUI / FLUX convention)
    if let Some(root_lora) = candidates.iter().find(|c| {
        !c.contains('/') && c.to_ascii_lowercase().contains("lora") && !is_sharded_base(c)
    }) {
        return Some(root_lora.clone());
    }

    // 4. Root-level non-sharded .safetensors file (preferring non-"model.safetensors" if alternatives exist)
    if let Some(root_custom) = candidates.iter().find(|c| {
        !c.contains('/') && c.as_str() != "model.safetensors" && !is_sharded_base(c)
    }) {
        return Some(root_custom.clone());
    }
    if let Some(root_any) = candidates
        .iter()
        .find(|c| !c.contains('/') && !is_sharded_base(c))
    {
        return Some(root_any.clone());
    }

    // 5. Subpath .safetensors containing "lora"
    if let Some(sub_lora) = candidates
        .iter()
        .find(|c| c.to_ascii_lowercase().contains("lora") && !is_sharded_base(c))
    {
        return Some(sub_lora.clone());
    }

    // 6. First non-sharded candidate, or first candidate overall
    candidates
        .iter()
        .find(|c| !is_sharded_base(c))
        .or_else(|| candidates.first())
        .cloned()
}

fn extract_rfilenames_from_hf_json(json: &str) -> Vec<String> {
    let bytes = json.as_bytes();
    let needle = b"\"rfilename\"";
    let mut results = Vec::new();
    let mut i = 0usize;

    while i + needle.len() < bytes.len() {
        if &bytes[i..i + needle.len()] == needle {
            let mut cursor = i + needle.len();
            while cursor < bytes.len() && matches!(bytes[cursor], b' ' | b'\t' | b'\r' | b'\n') {
                cursor += 1;
            }
            if cursor < bytes.len() && bytes[cursor] == b':' {
                cursor += 1;
                while cursor < bytes.len() && matches!(bytes[cursor], b' ' | b'\t' | b'\r' | b'\n')
                {
                    cursor += 1;
                }
                if cursor < bytes.len() && bytes[cursor] == b'"' {
                    cursor += 1;
                    let mut val = String::new();
                    let mut valid = false;
                    while cursor < bytes.len() {
                        let b = bytes[cursor];
                        cursor += 1;
                        if b == b'"' {
                            valid = true;
                            break;
                        } else if b == b'\\' {
                            if cursor >= bytes.len() {
                                break;
                            }
                            let esc = bytes[cursor];
                            cursor += 1;
                            match esc {
                                b'"' => val.push('"'),
                                b'\\' => val.push('\\'),
                                b'/' => val.push('/'),
                                b'n' => val.push('\n'),
                                b'r' => val.push('\r'),
                                b't' => val.push('\t'),
                                other => val.push(other as char),
                            }
                        } else {
                            val.push(b as char);
                        }
                    }
                    if valid && !val.is_empty() {
                        results.push(val);
                    }
                    i = cursor;
                    continue;
                }
            }
        }
        i += 1;
    }
    results
}

fn is_valid_cached_safetensors(path: &Path) -> bool {
    match fs::metadata(path) {
        Ok(meta) => meta.is_file() && meta.len() >= 16,
        Err(_) => false,
    }
}

fn curl_fetch_string(url: &str, hf_token: Option<&str>) -> Result<String, String> {
    let args = build_curl_args(url, None, hf_token);
    let output = Command::new("curl")
        .args(&args)
        .output()
        .map_err(|e| format!("Failed to invoke system 'curl': {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "curl request failed for {} ({}): {}",
            url,
            output.status,
            stderr.trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|e| format!("Invalid UTF-8 response from {}: {}", url, e))
}

fn curl_download_to_cache(
    url: &str,
    dest_path: &Path,
    hf_token: Option<&str>,
) -> Result<(), String> {
    if let Some(parent) = dest_path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "Failed to create cache directory {}: {}",
                parent.display(),
                e
            )
        })?;
    }

    let tmp_path = dest_path.with_extension(format!("safetensors.part.{}", process::id()));
    let args = build_curl_args(url, Some(&tmp_path), hf_token);

    eprintln!("Downloading {} -> {} ...", url, dest_path.display());
    let output = Command::new("curl")
        .args(&args)
        .output()
        .map_err(|e| format!("Failed to invoke system 'curl': {}", e))?;

    if !output.status.success() {
        let _ = fs::remove_file(&tmp_path);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let auth_hint = if is_huggingface_url(url) && hf_token.is_none() {
            " (tip: set HF_TOKEN for gated/private HuggingFace repos)"
        } else {
            ""
        };
        return Err(format!(
            "Failed to download {}{}: {}",
            url,
            auth_hint,
            stderr.trim()
        ));
    }

    if !is_valid_cached_safetensors(&tmp_path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(format!(
            "Downloaded file from {} is empty or truncated (< 16 bytes)",
            url
        ));
    }

    fs::rename(&tmp_path, dest_path).map_err(|e| {
        let _ = fs::remove_file(&tmp_path);
        format!(
            "Failed to finalize cached file {}: {}",
            dest_path.display(),
            e
        )
    })
}

/// Resolve a local path, `hf://<org>/<repo>[/<file.safetensors>]`, or `https://...` URL
/// to a local `.safetensors` path on disk, caching remote downloads under
/// `${TMPDIR:-/tmp}/loraspectra-cache/`.
pub fn resolve_source_path(input: &str) -> Result<PathBuf, String> {
    resolve_source_path_with_cache(input, &default_cache_dir(), read_hf_token_from_env().as_deref())
}

/// Resolve a checkpoint source with an explicit `cache_root` and optional `hf_token`.
pub fn resolve_source_path_with_cache(
    input: &str,
    cache_root: &Path,
    hf_token: Option<&str>,
) -> Result<PathBuf, String> {
    let spec = parse_source_spec(input)?;
    match spec {
        RemoteSpec::LocalPath(p) => Ok(p),
        RemoteSpec::HuggingFaceFile {
            org,
            repo,
            revision,
            filename,
        } => {
            let cached = cache_path_for_hf(cache_root, &org, &repo, &revision, &filename);
            if is_valid_cached_safetensors(&cached) {
                return Ok(cached);
            }
            let url = hf_resolve_url(&org, &repo, &revision, &filename);
            curl_download_to_cache(&url, &cached, hf_token)?;
            Ok(cached)
        }
        RemoteSpec::HuggingFaceRepo {
            org,
            repo,
            revision,
        } => {
            // Fast path 1: check default PEFT adapter_model.safetensors in local cache
            let default_cached =
                cache_path_for_hf(cache_root, &org, &repo, &revision, "adapter_model.safetensors");
            if is_valid_cached_safetensors(&default_cached) {
                return Ok(default_cached);
            }

            // Fast path 2: check saved sibling resolution pointer for <org>/<repo>@<revision>
            let pointer_path = default_cached
                .parent()
                .unwrap_or(cache_root)
                .join(".resolved_target");
            if let Ok(saved_filename) = fs::read_to_string(&pointer_path) {
                let saved = saved_filename.trim();
                if !saved.is_empty() {
                    let cached = cache_path_for_hf(cache_root, &org, &repo, &revision, saved);
                    if is_valid_cached_safetensors(&cached) {
                        return Ok(cached);
                    }
                }
            }

            // Query HuggingFace Hub API siblings list to discover the LoRA .safetensors file;
            // fall back to adapter_model.safetensors if metadata query fails.
            let api_url = hf_api_model_url(&org, &repo, &revision);
            let filename = match curl_fetch_string(&api_url, hf_token) {
                Ok(json) => select_safetensors_from_hf_api_json(&json).ok_or_else(|| {
                    format!(
                        "No .safetensors files found in HuggingFace repository '{}/{}'",
                        org, repo
                    )
                })?,
                Err(_) => "adapter_model.safetensors".to_string(),
            };

            let cached = cache_path_for_hf(cache_root, &org, &repo, &revision, &filename);
            if is_valid_cached_safetensors(&cached) {
                let _ = fs::write(&pointer_path, &filename);
                return Ok(cached);
            }

            let resolve_url = hf_resolve_url(&org, &repo, &revision, &filename);
            curl_download_to_cache(&resolve_url, &cached, hf_token)?;
            let _ = fs::write(&pointer_path, &filename);
            Ok(cached)
        }
        RemoteSpec::DirectUrl { url } => {
            let cached = cache_path_for_url(cache_root, &url);
            if is_valid_cached_safetensors(&cached) {
                return Ok(cached);
            }
            curl_download_to_cache(&url, &cached, hf_token)?;
            Ok(cached)
        }
    }
}

/// Locate an optional companion `adapter_config.json` alongside a local or `hf://` `.safetensors`
/// checkpoint without failing if the checkpoint is a standalone Kohya/Diffusers archive.
pub fn resolve_companion_config_path(
    source_str: &str,
    resolved_safetensors: &Path,
) -> Option<PathBuf> {
    let spec = parse_source_spec(source_str).ok()?;
    match spec {
        RemoteSpec::LocalPath(_) => {
            let parent = resolved_safetensors.parent()?;
            // Avoid picking up stray files directly inside the shared system /tmp root unless
            // the user placed the checkpoint inside its own directory.
            if parent == env::temp_dir() {
                return None;
            }
            let candidate = parent.join("adapter_config.json");
            if candidate.is_file() {
                Some(candidate)
            } else {
                None
            }
        }
        RemoteSpec::HuggingFaceRepo {
            org,
            repo,
            revision,
        } => fetch_optional_hf_config(&org, &repo, &revision, "adapter_config.json", resolved_safetensors),
        RemoteSpec::HuggingFaceFile {
            org,
            repo,
            revision,
            filename,
        } => {
            let cfg_rel = match filename.rsplit_once('/') {
                Some((dir, _)) if !dir.is_empty() => format!("{}/adapter_config.json", dir),
                _ => "adapter_config.json".to_string(),
            };
            fetch_optional_hf_config(&org, &repo, &revision, &cfg_rel, resolved_safetensors)
        }
        RemoteSpec::DirectUrl { .. } => None,
    }
}

fn fetch_optional_hf_config(
    org: &str,
    repo: &str,
    revision: &str,
    cfg_rel: &str,
    resolved_safetensors: &Path,
) -> Option<PathBuf> {
    let parent = resolved_safetensors.parent()?;
    let dest = parent.join("adapter_config.json");
    if dest.is_file() {
        return Some(dest);
    }
    let neg_marker = parent.join(".no_adapter_config");
    if neg_marker.exists() {
        return None;
    }
    let url = hf_resolve_url(org, repo, revision, cfg_rel);
    let hf_token = read_hf_token_from_env();
    match curl_fetch_string(&url, hf_token.as_deref()) {
        Ok(body) if body.trim_start().starts_with('{') => {
            let _ = fs::write(&dest, body.as_bytes());
            if dest.is_file() {
                Some(dest)
            } else {
                None
            }
        }
        _ => {
            let _ = fs::write(&neg_marker, b"");
            None
        }
    }
}
