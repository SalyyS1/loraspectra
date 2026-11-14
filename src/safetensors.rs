use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

/// Supported floating-point storage formats in HuggingFace `.safetensors` files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
        }
    }

    pub fn byte_size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
        }
    }

    pub fn from_name(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_uppercase().as_str() {
            "F32" | "FP32" | "FLOAT32" => Ok(DType::F32),
            "F16" | "FP16" | "FLOAT16" => Ok(DType::F16),
            "BF16" | "BFLOAT16" => Ok(DType::BF16),
            other => Err(format!("Unsupported safetensors dtype: {}", other)),
        }
    }
}

/// A 2D matrix stored in row-major order in f64 precision for spectral computations,
/// retaining its original storage `DType`.
#[derive(Debug, Clone)]
pub struct Tensor2D {
    pub rows: usize,
    pub cols: usize,
    pub dtype: DType,
    pub data: Vec<f64>,
}

impl Tensor2D {
    pub fn zeros(rows: usize, cols: usize, dtype: DType) -> Self {
        Self {
            rows,
            cols,
            dtype,
            data: vec![0.0; rows * cols],
        }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    pub fn frobenius_norm_sq(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum()
    }
}

/// A matched LoRA adapter layer consisting of `A` (`r x d_in`) and `B` (`d_out x r`).
#[derive(Debug, Clone)]
pub struct LoraLayer {
    pub base_name: String,
    pub a_key: String,
    pub b_key: String,
    /// Matrix A of shape `(rank, d_in)`
    pub a: Tensor2D,
    /// Matrix B of shape `(d_out, rank)`
    pub b: Tensor2D,
}

impl LoraLayer {
    pub fn rank(&self) -> usize {
        self.a.rows
    }

    pub fn d_in(&self) -> usize {
        self.a.cols
    }

    pub fn d_out(&self) -> usize {
        self.b.rows
    }

    pub fn raw_byte_size(&self) -> usize {
        self.a.data.len() * self.a.dtype.byte_size() + self.b.data.len() * self.b.dtype.byte_size()
    }
}

/// Convert IEEE 754 binary16 (`u16`) to `f32`.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal f16
        let mut m = mant;
        let mut e = 127 - 14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03ff;
        return f32::from_bits(sign | (e << 23) | (m << 13));
    } else if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    let e = exp + (127 - 15);
    f32::from_bits(sign | (e << 23) | (mant << 13))
}

/// Convert `f32` to IEEE 754 binary16 (`u16`) with round-to-nearest-even.
#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        if mant != 0 {
            return sign | 0x7e00;
        }
        return sign | 0x7c00;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7c00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let m = (mant | 0x0080_0000) >> (1 - new_exp);
        let half_m = (m + 0x1000) >> 13;
        return sign | (half_m as u16);
    }

    let rounded = mant + 0x0fff + ((mant >> 13) & 1);
    let mut final_exp = new_exp as u16;
    let mut final_mant = (rounded >> 13) as u16;
    if final_mant >= 0x0400 {
        final_mant = 0;
        final_exp += 1;
        if final_exp >= 31 {
            return sign | 0x7c00;
        }
    }
    sign | (final_exp << 10) | (final_mant & 0x03ff)
}

/// Convert `bfloat16` (`u16`) to `f32`.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert `f32` to `bfloat16` (`u16`) with round-to-nearest-even.
#[inline]
pub fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    if (bits & 0x7f80_0000) == 0x7f80_0000 && (bits & 0x007f_ffff) != 0 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// Minimal header metadata for a single tensor inside `.safetensors`.
#[derive(Debug, Clone)]
struct RawTensorEntry {
    dtype: DType,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// Parse the JSON header of a `.safetensors` file without external crates.
fn parse_safetensors_header(json_str: &str) -> Result<BTreeMap<String, RawTensorEntry>, String> {
    let bytes = json_str.as_bytes();
    let mut i = 0;
    let skip_ws = |idx: &mut usize| {
        while *idx < bytes.len() && matches!(bytes[*idx], b' ' | b'\t' | b'\r' | b'\n') {
            *idx += 1;
        }
    };

    let parse_string = |idx: &mut usize| -> Result<String, String> {
        skip_ws(idx);
        if *idx >= bytes.len() || bytes[*idx] != b'"' {
            return Err(format!("Expected '\"' at byte {}", *idx));
        }
        *idx += 1;
        let mut out = String::new();
        while *idx < bytes.len() {
            let b = bytes[*idx];
            *idx += 1;
            if b == b'"' {
                return Ok(out);
            } else if b == b'\\' {
                if *idx >= bytes.len() {
                    return Err("Unexpected EOF in string escape".to_string());
                }
                let esc = bytes[*idx];
                *idx += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    other => out.push(other as char),
                }
            } else {
                out.push(b as char);
            }
        }
        Err("Unterminated string literal".to_string())
    };

    let parse_usize = |idx: &mut usize| -> Result<usize, String> {
        skip_ws(idx);
        let start = *idx;
        while *idx < bytes.len() && bytes[*idx].is_ascii_digit() {
            *idx += 1;
        }
        if start == *idx {
            return Err(format!("Expected integer at byte {}", start));
        }
        json_str[start..*idx]
            .parse::<usize>()
            .map_err(|e| e.to_string())
    };

    let skip_value = |idx: &mut usize| -> Result<(), String> {
        skip_ws(idx);
        let mut depth = 0usize;
        let mut in_str = false;
        while *idx < bytes.len() {
            let b = bytes[*idx];
            if in_str {
                if b == b'\\' {
                    *idx += 2;
                    continue;
                }
                if b == b'"' {
                    in_str = false;
                }
            } else {
                match b {
                    b'"' => in_str = true,
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        if depth == 0 {
                            break;
                        }
                        depth -= 1;
                        if depth == 0 {
                            *idx += 1;
                            break;
                        }
                    }
                    b',' if depth == 0 => break,
                    _ => {}
                }
            }
            *idx += 1;
        }
        Ok(())
    };

    skip_ws(&mut i);
    if i >= bytes.len() || bytes[i] != b'{' {
        return Err("Safetensors header must be a JSON object".to_string());
    }
    i += 1;

    let mut entries = BTreeMap::new();
    loop {
        skip_ws(&mut i);
        if i < bytes.len() && bytes[i] == b'}' {
            break;
        }
        let key = parse_string(&mut i)?;
        skip_ws(&mut i);
        if i >= bytes.len() || bytes[i] != b':' {
            return Err("Expected ':' after key".to_string());
        }
        i += 1;
        skip_ws(&mut i);

        if key == "__metadata__" {
            skip_value(&mut i)?;
        } else {
            if i >= bytes.len() || bytes[i] != b'{' {
                return Err(format!("Expected object for tensor {}", key));
            }
            i += 1;
            let mut dtype: Option<DType> = None;
            let mut shape: Vec<usize> = Vec::new();
            let mut offsets: (usize, usize) = (0, 0);

            loop {
                skip_ws(&mut i);
                if i < bytes.len() && bytes[i] == b'}' {
                    i += 1;
                    break;
                }
                let field = parse_string(&mut i)?;
                skip_ws(&mut i);
                if i >= bytes.len() || bytes[i] != b':' {
                    return Err("Expected ':' after tensor field".to_string());
                }
                i += 1;
                skip_ws(&mut i);

                match field.as_str() {
                    "dtype" => {
                        let dt_str = parse_string(&mut i)?;
                        dtype = Some(DType::from_name(&dt_str)?);
                    }
                    "shape" => {
                        if bytes[i] != b'[' {
                            return Err("Expected '[' for shape".to_string());
                        }
                        i += 1;
                        loop {
                            skip_ws(&mut i);
                            if bytes[i] == b']' {
                                i += 1;
                                break;
                            }
                            shape.push(parse_usize(&mut i)?);
                            skip_ws(&mut i);
                            if bytes[i] == b',' {
                                i += 1;
                            }
                        }
                    }
                    "data_offsets" => {
                        if bytes[i] != b'[' {
                            return Err("Expected '[' for data_offsets".to_string());
                        }
                        i += 1;
                        let s = parse_usize(&mut i)?;
                        skip_ws(&mut i);
                        if bytes[i] == b',' {
                            i += 1;
                        }
                        let e = parse_usize(&mut i)?;
                        skip_ws(&mut i);
                        if bytes[i] != b']' {
                            return Err("Expected ']' after data_offsets".to_string());
                        }
                        i += 1;
                        offsets = (s, e);
                    }
                    _ => {
                        skip_value(&mut i)?;
                    }
                }
                skip_ws(&mut i);
                if i < bytes.len() && bytes[i] == b',' {
                    i += 1;
                }
            }

            entries.insert(
                key,
                RawTensorEntry {
                    dtype: dtype.ok_or("Missing dtype in tensor entry")?,
                    shape,
                    start: offsets.0,
                    end: offsets.1,
                },
            );
        }

        skip_ws(&mut i);
        if i < bytes.len() && bytes[i] == b',' {
            i += 1;
        }
    }

    Ok(entries)
}

/// Maximum allowed `.safetensors` JSON header size (100 MB, matching the official `safetensors` spec).
pub const MAX_SAFETENSORS_HEADER_BYTES: usize = 100_000_000;

/// Read all 2D tensors (plus 1D/scalar LoRA `.alpha` and DoRA magnitude vectors) from a `.safetensors` file on disk.
pub fn read_safetensors(path: &Path) -> Result<BTreeMap<String, Tensor2D>, String> {
    let raw = fs::read(path).map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    if raw.len() < 8 {
        return Err("File too small to be a valid .safetensors archive".to_string());
    }
    let header_len = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    if header_len == 0
        || header_len > MAX_SAFETENSORS_HEADER_BYTES
        || 8usize
            .checked_add(header_len)
            .map_or(true, |end| end > raw.len())
    {
        return Err("Invalid safetensors header length".to_string());
    }
    let header_str = std::str::from_utf8(&raw[8..8 + header_len])
        .map_err(|e| format!("Invalid UTF-8 in safetensors header: {}", e))?;
    let entries = parse_safetensors_header(header_str)?;
    let body = &raw[8 + header_len..];

    let mut out = BTreeMap::new();
    for (name, entry) in entries {
        let (rows, cols) = if entry.shape.len() == 2 {
            (entry.shape[0], entry.shape[1])
        } else if (entry.shape.is_empty() || (entry.shape.len() == 1 && entry.shape[0] == 1))
            && name.ends_with(".alpha")
        {
            (1, 1)
        } else if entry.shape.len() == 1
            && (name.ends_with(".lora_magnitude_vector")
                || name.ends_with(".lora_magnitude_vector.weight")
                || name.ends_with(".lora_magnitude_vector.default.weight")
                || name.ends_with(".dora_scale")
                || name.ends_with(".magnitude"))
        {
            (entry.shape[0], 1)
        } else {
            continue;
        };
        let numel = rows * cols;
        let expected_bytes = numel * entry.dtype.byte_size();
        if entry.end < entry.start || entry.end > body.len() || (entry.end - entry.start) != expected_bytes {
            return Err(format!("Corrupt data offsets for tensor {}", name));
        }
        let slice = &body[entry.start..entry.end];
        let mut data = Vec::with_capacity(numel);
        match entry.dtype {
            DType::F32 => {
                for chunk in slice.chunks_exact(4) {
                    let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    data.push(v as f64);
                }
            }
            DType::F16 => {
                for chunk in slice.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    data.push(f16_to_f32(bits) as f64);
                }
            }
            DType::BF16 => {
                for chunk in slice.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    data.push(bf16_to_f32(bits) as f64);
                }
            }
        }
        out.insert(
            name,
            Tensor2D {
                rows,
                cols,
                dtype: entry.dtype,
                data,
            },
        );
    }
    Ok(out)
}

/// Write a map of 2D tensors to a spec-compliant `.safetensors` file (8-byte aligned header).
pub fn write_safetensors(path: &Path, tensors: &BTreeMap<String, Tensor2D>) -> io::Result<usize> {
    let mut header_parts = Vec::new();
    header_parts.push(
        "\"__metadata__\":{\"format\":\"pt\",\"generator\":\"loraspectra\"}".to_string(),
    );

    let mut current_offset = 0usize;
    for (name, tensor) in tensors {
        let byte_len = tensor.data.len() * tensor.dtype.byte_size();
        let start = current_offset;
        let end = start + byte_len;
        current_offset = end;
        if name.ends_with(".alpha") && tensor.rows == 1 && tensor.cols == 1 {
            header_parts.push(format!(
                "\"{}\":{{\"dtype\":\"{}\",\"shape\":[],\"data_offsets\":[{},{}]}}",
                name,
                tensor.dtype.as_str(),
                start,
                end
            ));
        } else if (name.ends_with(".lora_magnitude_vector")
            || name.ends_with(".lora_magnitude_vector.default.weight")
            || name.ends_with(".dora_scale"))
            && tensor.cols == 1
        {
            header_parts.push(format!(
                "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                name,
                tensor.dtype.as_str(),
                tensor.rows,
                start,
                end
            ));
        } else {
            header_parts.push(format!(
                "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{},{}],\"data_offsets\":[{},{}]}}",
                name,
                tensor.dtype.as_str(),
                tensor.rows,
                tensor.cols,
                start,
                end
            ));
        }
    }

    let mut header_json = format!("{{{}}}", header_parts.join(","));
    // Pad header with spaces to 8-byte alignment per safetensors specification
    let rem = header_json.len() % 8;
    if rem != 0 {
        for _ in 0..(8 - rem) {
            header_json.push(' ');
        }
    }

    let mut buf = Vec::with_capacity(8 + header_json.len() + current_offset);
    buf.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    buf.extend_from_slice(header_json.as_bytes());

    for tensor in tensors.values() {
        match tensor.dtype {
            DType::F32 => {
                for &v in &tensor.data {
                    buf.extend_from_slice(&(v as f32).to_le_bytes());
                }
            }
            DType::F16 => {
                for &v in &tensor.data {
                    buf.extend_from_slice(&f32_to_f16(v as f32).to_le_bytes());
                }
            }
            DType::BF16 => {
                for &v in &tensor.data {
                    buf.extend_from_slice(&f32_to_bf16(v as f32).to_le_bytes());
                }
            }
        }
    }

    let mut file = fs::File::create(path)?;
    file.write_all(&buf)?;
    Ok(buf.len())
}

/// Pair `.safetensors` keys into matched `(lora_A, lora_B)` layers across PEFT, Kohya/ComfyUI,
/// HuggingFace Diffusers, and OneTrainer naming conventions.
pub fn extract_lora_layers(tensors: &BTreeMap<String, Tensor2D>) -> Result<Vec<LoraLayer>, String> {
    let mut a_map: BTreeMap<String, (String, Tensor2D)> = BTreeMap::new();
    let mut b_map: BTreeMap<String, (String, Tensor2D)> = BTreeMap::new();

    for (key, tensor) in tensors {
        if let Some(raw_base) = key
            .strip_suffix(".lora_A.weight")
            .or_else(|| key.strip_suffix(".lora_A.default.weight"))
            .or_else(|| key.strip_suffix(".lora_embedding_A.weight"))
            .or_else(|| key.strip_suffix(".lora_embedding_A.default.weight"))
            .or_else(|| key.strip_suffix(".lora_embedding_A.default"))
            .or_else(|| key.strip_suffix(".lora_embedding_A"))
            .or_else(|| key.strip_suffix(".lora_down.weight"))
            .or_else(|| key.strip_suffix(".lora.down.weight"))
            .or_else(|| key.strip_suffix(".lora_linear_layer.down.weight"))
            .or_else(|| key.strip_suffix("_lora.down.weight"))
        {
            let base = raw_base
                .strip_suffix(".dora_layer")
                .or_else(|| raw_base.strip_suffix(".lora_layer"))
                .unwrap_or(raw_base);
            a_map.insert(base.to_string(), (key.clone(), tensor.clone()));
        } else if let Some(raw_base) = key
            .strip_suffix(".lora_B.weight")
            .or_else(|| key.strip_suffix(".lora_B.default.weight"))
            .or_else(|| key.strip_suffix(".lora_embedding_B.weight"))
            .or_else(|| key.strip_suffix(".lora_embedding_B.default.weight"))
            .or_else(|| key.strip_suffix(".lora_embedding_B.default"))
            .or_else(|| key.strip_suffix(".lora_embedding_B"))
            .or_else(|| key.strip_suffix(".lora_up.weight"))
            .or_else(|| key.strip_suffix(".lora.up.weight"))
            .or_else(|| key.strip_suffix(".lora_linear_layer.up.weight"))
            .or_else(|| key.strip_suffix("_lora.up.weight"))
        {
            let base = raw_base
                .strip_suffix(".dora_layer")
                .or_else(|| raw_base.strip_suffix(".lora_layer"))
                .unwrap_or(raw_base);
            b_map.insert(base.to_string(), (key.clone(), tensor.clone()));
        }
    }

    let mut layers = Vec::new();
    for (base_name, (a_key, a_tensor)) in a_map {
        if let Some((b_key, b_tensor)) = b_map.remove(&base_name) {
            if a_tensor.rows != b_tensor.cols {
                return Err(format!(
                    "Dimension mismatch in layer {}: A is ({}, {}), B is ({}, {})",
                    base_name, a_tensor.rows, a_tensor.cols, b_tensor.rows, b_tensor.cols
                ));
            }
            layers.push(LoraLayer {
                base_name,
                a_key,
                b_key,
                a: a_tensor,
                b: b_tensor,
            });
        }
    }
    Ok(layers)
}

/// Convert a slice of `LoraLayer` back into a `BTreeMap<String, Tensor2D>`.
pub fn layers_to_tensor_map(layers: &[LoraLayer]) -> BTreeMap<String, Tensor2D> {
    let mut map = BTreeMap::new();
    for layer in layers {
        map.insert(layer.a_key.clone(), layer.a.clone());
        map.insert(layer.b_key.clone(), layer.b.clone());
    }
    map
}

/// Cast the storage `DType` (`F32`, `F16`, or `BF16`) of all LoRA layers in place.
pub fn cast_layers_dtype(layers: &mut [LoraLayer], target_dtype: DType) {
    for layer in layers {
        layer.a.dtype = target_dtype;
        layer.b.dtype = target_dtype;
    }
}

/// Strip the standard HuggingFace PEFT `"base_model.model."` wrapper prefix so module keys match
/// `PreTrainedModel.named_modules()` entries inside `rank_pattern` and `alpha_pattern`.
pub fn strip_peft_base_prefix(name: &str) -> &str {
    name.strip_prefix("base_model.model.").unwrap_or(name)
}

/// Match a module `base_name` against a PEFT `rank_pattern` or `alpha_pattern` map following
/// `peft.utils.other.get_pattern_key` (`(.*\.)?({key})$` with longest-key precedence).
pub fn match_peft_pattern_value<T: Copy>(
    pattern_map: &BTreeMap<String, T>,
    base_name: &str,
) -> Option<T> {
    if let Some(&exact) = pattern_map.get(base_name) {
        return Some(exact);
    }
    let stripped = strip_peft_base_prefix(base_name);
    if let Some(&exact) = pattern_map.get(stripped) {
        return Some(exact);
    }
    let mut best: Option<(&str, T)> = None;
    for (pat, &val) in pattern_map {
        let pat_s = pat.as_str();
        let is_dot_suffix = stripped == pat_s
            || base_name == pat_s
            || (stripped.len() > pat_s.len()
                && stripped.ends_with(pat_s)
                && stripped.as_bytes()[stripped.len() - pat_s.len() - 1] == b'.')
            || (base_name.len() > pat_s.len()
                && base_name.ends_with(pat_s)
                && base_name.as_bytes()[base_name.len() - pat_s.len() - 1] == b'.');
        if is_dot_suffix {
            match best {
                Some((prev_pat, _)) if prev_pat.len() >= pat_s.len() => {}
                _ => best = Some((pat_s, val)),
            }
        }
    }
    best.map(|(_, v)| v)
}

/// Infer sorted unique leaf module names (e.g. `["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"]`)
/// from LoRA layers so `PeftModel.from_pretrained` injects adapters into all compressed target modules.
pub fn infer_peft_target_modules(layers: &[LoraLayer]) -> Vec<String> {
    let mut modules = std::collections::BTreeSet::new();
    for layer in layers {
        let stripped = strip_peft_base_prefix(&layer.base_name);
        let leaf = stripped.rsplit('.').next().unwrap_or(stripped);
        if !leaf.is_empty() {
            modules.insert(leaf.to_string());
        }
    }
    modules.into_iter().collect()
}

fn format_json_number(val: f64) -> String {
    let rounded = val.round();
    if (val - rounded).abs() < 1e-9 && rounded.abs() < (i64::MAX as f64) {
        format!("{}", rounded as i64)
    } else {
        let s = format!("{:.9}", val);
        let trimmed = s.trim_end_matches('0').trim_end_matches('.');
        if trimmed.is_empty() || trimmed == "-0" {
            "0".to_string()
        } else {
            trimmed.to_string()
        }
    }
}

fn escape_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Resolve pre-compression per-layer `(alpha, effective_scale, effective_rslora)` from explicit
/// `.alpha` tensors, CLI overrides, or a companion `PeftAdapterConfig`.
pub fn resolve_layer_scaling_params(
    layer: &LoraLayer,
    alphas: &BTreeMap<String, f64>,
    default_alpha_override: Option<f64>,
    peft_config: Option<&PeftAdapterConfig>,
    rslora_override: bool,
) -> (f64, f64, bool) {
    let orig_r = layer.rank().max(1) as f64;
    let use_rslora = rslora_override || peft_config.map(|c| c.use_rslora).unwrap_or(false);
    let denom = if use_rslora { orig_r.sqrt() } else { orig_r };

    let resolved_alpha = alphas
        .get(&layer.base_name)
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .or_else(|| default_alpha_override.filter(|v| v.is_finite() && *v > 0.0))
        .or_else(|| {
            peft_config.and_then(|cfg| {
                match_peft_pattern_value(&cfg.alpha_pattern, &layer.base_name)
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .or_else(|| cfg.lora_alpha.filter(|v| v.is_finite() && *v > 0.0))
            })
        })
        .unwrap_or(denom);

    let effective_scale = resolved_alpha / denom;
    (resolved_alpha, effective_scale, use_rslora)
}

/// Compute exact scale-preserving per-module `alpha_pattern` entries for compressed layers.
///
/// When `baked_scaling` is `false`, emits `alpha_l = s_l * k_l` (or `s_l * sqrt(k_l)` when
/// `out_use_rslora` is `true`) so HuggingFace PEFT's runtime multiplier `alpha_l / k_l`
/// (or `alpha_l / sqrt(k_l)`) matches the original layer's effective scale `s_l` to machine precision.
/// When `baked_scaling` is `true`, emits `alpha_l = k_l` (`out_use_rslora = false`, multiplier `1.0`).
pub fn compute_peft_alpha_pattern(
    layers: &[LoraLayer],
    pre_scales: &[f64],
    baked_scaling: bool,
    out_use_rslora: bool,
) -> BTreeMap<String, f64> {
    let mut alpha_pattern = BTreeMap::new();
    for (idx, layer) in layers.iter().enumerate() {
        let k = layer.rank().max(1) as f64;
        let clean = strip_peft_base_prefix(&layer.base_name).to_string();
        let alpha_val = if baked_scaling {
            k
        } else {
            let s = pre_scales.get(idx).copied().unwrap_or(1.0);
            if out_use_rslora {
                s * k.sqrt()
            } else {
                s * k
            }
        };
        alpha_pattern.insert(clean, alpha_val);
    }
    alpha_pattern
}

/// Build a complete HuggingFace PEFT `adapter_config.json` document preserving upstream metadata
/// (`base_model_name_or_path`, `task_type`, `target_modules`, `modules_to_save`, `use_dora`, etc.)
/// while emitting exact per-module `rank_pattern` and `alpha_pattern` maps.
pub fn build_peft_adapter_config_json(
    rank_pattern: &BTreeMap<String, usize>,
    alpha_pattern: &BTreeMap<String, f64>,
    default_rank: usize,
    out_use_rslora: bool,
    has_dora: bool,
    inferred_target_modules: &[String],
    source_config: Option<&PeftAdapterConfig>,
) -> String {
    let top_r = default_rank.max(1);
    let mut clean_ranks: BTreeMap<String, usize> = BTreeMap::new();
    for (name, &r) in rank_pattern {
        clean_ranks.insert(strip_peft_base_prefix(name).to_string(), r.max(1));
    }

    let mut clean_alphas: BTreeMap<String, f64> = BTreeMap::new();
    for (name, &a) in alpha_pattern {
        clean_alphas.insert(strip_peft_base_prefix(name).to_string(), a);
    }
    for (name, &r) in &clean_ranks {
        clean_alphas.entry(name.clone()).or_insert(r as f64);
    }

    // Representative top-level lora_alpha consistent with the first layer's post-compression scale
    let rep_scale = clean_ranks
        .iter()
        .next()
        .and_then(|(k, &r)| {
            clean_alphas.get(k).map(|&a| {
                let rf = r.max(1) as f64;
                if out_use_rslora {
                    a / rf.sqrt()
                } else {
                    a / rf
                }
            })
        })
        .unwrap_or(1.0);
    let top_alpha = if out_use_rslora {
        rep_scale * (top_r as f64).sqrt()
    } else {
        rep_scale * (top_r as f64)
    };

    let peft_type = source_config
        .and_then(|c| c.peft_type.as_deref())
        .unwrap_or("LORA");
    let inference_mode = source_config
        .and_then(|c| c.inference_mode)
        .unwrap_or(true);
    let bias = source_config
        .and_then(|c| c.bias.as_deref())
        .unwrap_or("none");
    let lora_dropout = source_config
        .and_then(|c| c.lora_dropout)
        .unwrap_or(0.0);
    let use_dora = has_dora || source_config.map(|c| c.use_dora).unwrap_or(false);

    let mut fields: Vec<String> = Vec::new();
    fields.push(format!("  \"peft_type\": \"{}\"", escape_json_str(peft_type)));
    if let Some(base_model) = source_config.and_then(|c| c.base_model_name_or_path.as_deref()) {
        fields.push(format!(
            "  \"base_model_name_or_path\": \"{}\"",
            escape_json_str(base_model)
        ));
    }
    if let Some(task_type) = source_config.and_then(|c| c.task_type.as_deref()) {
        fields.push(format!("  \"task_type\": \"{}\"", escape_json_str(task_type)));
    }
    fields.push(format!("  \"inference_mode\": {}", inference_mode));
    fields.push(format!("  \"r\": {}", top_r));
    fields.push(format!("  \"lora_alpha\": {}", format_json_number(top_alpha)));
    fields.push(format!(
        "  \"lora_dropout\": {}",
        format_json_number(lora_dropout)
    ));
    fields.push(format!("  \"bias\": \"{}\"", escape_json_str(bias)));
    fields.push(format!("  \"use_rslora\": {}", out_use_rslora));
    fields.push(format!("  \"use_dora\": {}", use_dora));
    if let Some(fan_in) = source_config.and_then(|c| c.fan_in_fan_out) {
        fields.push(format!("  \"fan_in_fan_out\": {}", fan_in));
    }

    if let Some(tm_str) = source_config.and_then(|c| c.target_modules_str.as_deref()) {
        fields.push(format!(
            "  \"target_modules\": \"{}\"",
            escape_json_str(tm_str)
        ));
    } else {
        let tm_list: &[String] = match source_config {
            Some(c) if !c.target_modules.is_empty() => &c.target_modules,
            _ => inferred_target_modules,
        };
        if !tm_list.is_empty() {
            let items: Vec<String> = tm_list
                .iter()
                .map(|m| format!("\"{}\"", escape_json_str(m)))
                .collect();
            fields.push(format!("  \"target_modules\": [{}]", items.join(", ")));
        }
    }

    if let Some(cfg) = source_config {
        if !cfg.modules_to_save.is_empty() {
            let items: Vec<String> = cfg
                .modules_to_save
                .iter()
                .map(|m| format!("\"{}\"", escape_json_str(m)))
                .collect();
            fields.push(format!("  \"modules_to_save\": [{}]", items.join(", ")));
        }
    }

    let rank_entries: Vec<String> = clean_ranks
        .iter()
        .map(|(k, &r)| format!("    \"{}\": {}", escape_json_str(k), r))
        .collect();
    fields.push(format!(
        "  \"rank_pattern\": {{\n{}\n  }}",
        rank_entries.join(",\n")
    ));

    let alpha_entries: Vec<String> = clean_alphas
        .iter()
        .map(|(k, &a)| format!("    \"{}\": {}", escape_json_str(k), format_json_number(a)))
        .collect();
    fields.push(format!(
        "  \"alpha_pattern\": {{\n{}\n  }}",
        alpha_entries.join(",\n")
    ));

    format!("{{\n{}\n}}\n", fields.join(",\n"))
}

/// Write a full HuggingFace PEFT `adapter_config.json` with exact `rank_pattern`, scale-preserving
/// `alpha_pattern`, `target_modules`, and preserved upstream metadata.
pub fn write_peft_adapter_config(
    path: &Path,
    rank_pattern: &BTreeMap<String, usize>,
    alpha_pattern: &BTreeMap<String, f64>,
    default_rank: usize,
    out_use_rslora: bool,
    has_dora: bool,
    inferred_target_modules: &[String],
    source_config: Option<&PeftAdapterConfig>,
) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let json = build_peft_adapter_config_json(
        rank_pattern,
        alpha_pattern,
        default_rank,
        out_use_rslora,
        has_dora,
        inferred_target_modules,
        source_config,
    );
    let mut file = fs::File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

/// Write a HuggingFace PEFT-compatible `adapter_config.json` containing `rank_pattern` and
/// `alpha_pattern` for dynamically compressed heterogeneous-rank LoRA adapters (fixes PEFT #3737).
pub fn write_peft_rank_pattern_config(
    path: &Path,
    rank_pattern: &BTreeMap<String, usize>,
    default_rank: usize,
) -> io::Result<()> {
    let mut alpha_pattern = BTreeMap::new();
    let mut leaf_set = std::collections::BTreeSet::new();
    for (name, &r) in rank_pattern {
        let clean = strip_peft_base_prefix(name);
        alpha_pattern.insert(clean.to_string(), r as f64);
        let leaf = clean.rsplit('.').next().unwrap_or(clean);
        if !leaf.is_empty() {
            leaf_set.insert(leaf.to_string());
        }
    }
    let inferred: Vec<String> = leaf_set.into_iter().collect();
    write_peft_adapter_config(
        path,
        rank_pattern,
        &alpha_pattern,
        default_rank,
        false,
        false,
        &inferred,
        None,
    )
}

/// Extract per-layer `.alpha` scalar values (`<base_name>.alpha`) from a `.safetensors` tensor map.
pub fn extract_lora_alphas(tensors: &BTreeMap<String, Tensor2D>) -> BTreeMap<String, f64> {
    let mut alphas = BTreeMap::new();
    for (key, tensor) in tensors {
        if let Some(base) = key.strip_suffix(".alpha") {
            if let Some(&val) = tensor.data.first() {
                alphas.insert(base.to_string(), val);
            }
        }
    }
    alphas
}

/// Fold per-layer LoRA scaling (`alpha / r` for standard LoRA, or `alpha / sqrt(r)` for rsLoRA)
/// directly into the low-rank factors `(B, A)` via `sqrt(scale)`, normalizing the effective
/// scale factor to `1.0` (`alpha = r`).
///
/// This solves heterogeneous `rank_pattern` and `rsLoRA` mis-scaling bugs in downstream inference
/// engines such as `vLLM` (#59799), `llama.cpp` (#31879), `Soup` (#2326), and `InvokeAI` (#10019).
pub fn bake_adapter_scaling(
    layers: &mut [LoraLayer],
    alphas: &BTreeMap<String, f64>,
    default_alpha: Option<f64>,
    rslora: bool,
) {
    for layer in layers {
        let r = layer.rank().max(1) as f64;
        let alpha = alphas
            .get(&layer.base_name)
            .copied()
            .or(default_alpha)
            .unwrap_or(r);
        let denom = if rslora { r.sqrt() } else { r };
        let scale = alpha / denom;
        if (scale - 1.0).abs() > 1e-12 {
            let mag_sqrt = scale.abs().sqrt();
            let sign = if scale < 0.0 { -1.0 } else { 1.0 };
            for v in &mut layer.b.data {
                *v *= sign * mag_sqrt;
            }
            for v in &mut layer.a.data {
                *v *= mag_sqrt;
            }
        }
    }
}

/// Extract per-layer DoRA magnitude vectors (`lora_magnitude_vector`, `dora_scale`, or `magnitude`) from a `.safetensors` map.
pub fn extract_dora_scales(
    tensors: &BTreeMap<String, Tensor2D>,
) -> BTreeMap<String, (String, Tensor2D)> {
    let mut scales = BTreeMap::new();
    for (key, tensor) in tensors {
        if let Some(raw_base) = key
            .strip_suffix(".lora_magnitude_vector.default.weight")
            .or_else(|| key.strip_suffix(".lora_magnitude_vector.weight"))
            .or_else(|| key.strip_suffix(".lora_magnitude_vector"))
            .or_else(|| key.strip_suffix(".dora_scale"))
            .or_else(|| key.strip_suffix(".magnitude"))
        {
            let base = raw_base
                .strip_suffix(".dora_layer")
                .or_else(|| raw_base.strip_suffix(".lora_layer"))
                .unwrap_or(raw_base);
            scales.insert(base.to_string(), (key.clone(), tensor.clone()));
        }
    }
    scales
}

/// Preserve DoRA magnitude vectors (`lora_magnitude_vector` / `dora_scale`) in an output tensor map,
/// optionally converting key suffixes and storage `DType`.
pub fn insert_dora_scales(
    map: &mut BTreeMap<String, Tensor2D>,
    dora_scales: &BTreeMap<String, (String, Tensor2D)>,
    target_format: Option<&str>,
    target_dtype: Option<DType>,
) {
    for (base_name, (orig_key, tensor)) in dora_scales {
        let mut t = tensor.clone();
        if let Some(dt) = target_dtype {
            t.dtype = dt;
        }
        let out_key = match target_format.map(|s| s.to_ascii_lowercase()) {
            Some(ref f) if f == "peft" => format!("{}.lora_magnitude_vector", base_name),
            Some(ref f) if f == "kohya" || f == "comfyui" || f == "diffusers" => {
                format!("{}.dora_scale", base_name)
            }
            _ => orig_key.clone(),
        };
        map.insert(out_key, t);
    }
}

/// Convert `a_key` and `b_key` naming conventions across HuggingFace PEFT, Kohya/ComfyUI, and Diffusers.
pub fn convert_layers_key_format(
    layers: &mut [LoraLayer],
    target_format: &str,
) -> Result<(), String> {
    let fmt = target_format.trim().to_ascii_lowercase();
    for layer in layers {
        match fmt.as_str() {
            "peft" => {
                layer.a_key = format!("{}.lora_A.weight", layer.base_name);
                layer.b_key = format!("{}.lora_B.weight", layer.base_name);
            }
            "kohya" | "comfyui" => {
                layer.a_key = format!("{}.lora_down.weight", layer.base_name);
                layer.b_key = format!("{}.lora_up.weight", layer.base_name);
            }
            "diffusers" => {
                layer.a_key = format!("{}.lora.down.weight", layer.base_name);
                layer.b_key = format!("{}.lora.up.weight", layer.base_name);
            }
            other => {
                return Err(format!(
                    "Unsupported --format '{}': expected peft, kohya, or diffusers",
                    other
                ));
            }
        }
    }
    Ok(())
}

/// Insert normalized `.alpha` scalar tensors (`alpha = rank`) into an output tensor map for
/// Kohya/ComfyUI/InvokeAI checkpoints that require explicit `.alpha` keys.
pub fn insert_unit_alpha_tensors(
    map: &mut BTreeMap<String, Tensor2D>,
    layers: &[LoraLayer],
) {
    for layer in layers {
        let alpha_key = format!("{}.alpha", layer.base_name);
        map.insert(
            alpha_key,
            Tensor2D {
                rows: 1,
                cols: 1,
                dtype: layer.a.dtype,
                data: vec![layer.rank() as f64],
            },
        );
    }
}

/// Summary of an unknown LoRA `.safetensors` checkpoint reverse-engineered from its tensor keys and shapes.
#[derive(Debug, Clone)]
pub struct AdapterArchitectureInfo {
    pub naming_format: &'static str,
    pub inferred_architecture: &'static str,
    pub layer_count: usize,
    pub min_rank: usize,
    pub max_rank: usize,
    pub is_heterogeneous_rank: bool,
    pub max_d_in: usize,
    pub max_d_out: usize,
    pub total_parameters: usize,
    pub total_bytes: usize,
    pub primary_dtype: DType,
    pub alpha_tensor_count: usize,
}

/// Reverse-engineer the naming format, base model family, rank distribution, and parameter budget
/// of a LoRA `.safetensors` file without requiring `adapter_config.json` (fixes rea #3665).
pub fn inspect_adapter_architecture(
    layers: &[LoraLayer],
    alphas: &BTreeMap<String, f64>,
) -> AdapterArchitectureInfo {
    let first_a = layers.first().map(|l| l.a_key.as_str()).unwrap_or("");
    let naming_format = if first_a.starts_with("lycoris_") {
        "SimpleTuner / LyCORIS"
    } else if first_a.ends_with(".lora_A.weight")
        || first_a.ends_with(".lora_A.default.weight")
        || first_a.contains(".lora_embedding_A")
    {
        "HuggingFace PEFT"
    } else if first_a.ends_with(".lora_down.weight") {
        "Kohya-ss / ComfyUI"
    } else if first_a.ends_with(".lora.down.weight")
        || first_a.ends_with(".lora_linear_layer.down.weight")
    {
        "HuggingFace Diffusers"
    } else if first_a.ends_with("_lora.down.weight") {
        "OneTrainer"
    } else {
        "Custom LoRA"
    };

    let mut min_rank = usize::MAX;
    let mut max_rank = 0usize;
    let mut max_d_in = 0usize;
    let mut max_d_out = 0usize;
    let mut total_parameters = 0usize;
    let mut total_bytes = 0usize;

    for l in layers {
        let r = l.rank();
        min_rank = min_rank.min(r);
        max_rank = max_rank.max(r);
        max_d_in = max_d_in.max(l.d_in());
        max_d_out = max_d_out.max(l.d_out());
        total_parameters += l.a.data.len() + l.b.data.len();
        total_bytes += l.raw_byte_size();
    }
    if layers.is_empty() {
        min_rank = 0;
    }

    let is_diffusion = layers.iter().any(|l| {
        l.base_name.contains("unet")
            || l.base_name.contains("double_blocks")
            || l.base_name.contains("single_blocks")
            || l.base_name.contains("down_blocks")
            || l.base_name.contains("transformer_blocks")
    });
    let is_minicpm_omni = layers.iter().any(|l| {
        l.base_name.contains("llm.model.")
            || l.base_name.contains("vpm.")
            || l.base_name.contains("apm.")
            || l.base_name.contains("resampler.")
    });
    let is_vlm = layers.iter().any(|l| {
        l.base_name.contains("language_model.")
            || l.base_name.contains("vision_tower.")
            || l.base_name.contains("multi_modal_projector.")
    });
    let max_dim = max_d_in.max(max_d_out);

    let inferred_architecture = if is_diffusion {
        if layers.iter().any(|l| {
            l.base_name.contains("double_stream_modulation")
                || l.base_name.contains("single_stream_modulation")
        }) {
            "Flux.2 DiT (Flow Transformer)"
        } else if layers
            .iter()
            .any(|l| l.base_name.contains("double_blocks") || l.base_name.contains("single_blocks"))
        {
            "Flux.1 DiT (12B Flow Transformer)"
        } else if max_dim >= 2048 {
            "Stable Diffusion XL (SDXL UNet / Text Encoder)"
        } else {
            "Stable Diffusion 1.5 / 2.x UNet"
        }
    } else if is_minicpm_omni {
        "MiniCPM-o / MiniCPM-V Multimodal LLM"
    } else if is_vlm {
        "Gemma-3 / Multimodal VLM"
    } else if max_dim >= 16384 {
        "Llama-3-70B / Qwen2.5-72B class LLM"
    } else if max_dim >= 13824 {
        "Llama-2-13B / Qwen2.5-14B class LLM"
    } else if max_dim >= 11008 || max_d_in == 4096 {
        "Llama-3-8B / Mistral-7B class LLM"
    } else if max_d_in == 3584 {
        "Qwen2.5-7B / Gemma-2-9B class LLM"
    } else if max_d_in >= 2048 {
        "Llama-3.2-1B-3B / Phi-3 / Qwen2.5-3B class LLM"
    } else {
        "Compact Transformer / Custom Model"
    };

    let primary_dtype = layers.first().map(|l| l.a.dtype).unwrap_or(DType::F16);

    AdapterArchitectureInfo {
        naming_format,
        inferred_architecture,
        layer_count: layers.len(),
        min_rank,
        max_rank,
        is_heterogeneous_rank: min_rank != max_rank,
        max_d_in,
        max_d_out,
        total_parameters,
        total_bytes,
        primary_dtype,
        alpha_tensor_count: alphas.len(),
    }
}

/// Parsed scaling and module metadata from a HuggingFace PEFT `adapter_config.json`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PeftAdapterConfig {
    pub r: Option<usize>,
    pub lora_alpha: Option<f64>,
    pub use_rslora: bool,
    pub use_dora: bool,
    pub rank_pattern: BTreeMap<String, usize>,
    pub alpha_pattern: BTreeMap<String, f64>,
    pub modules_to_save: Vec<String>,
    pub target_modules: Vec<String>,
    pub target_modules_str: Option<String>,
    pub base_model_name_or_path: Option<String>,
    pub task_type: Option<String>,
    pub peft_type: Option<String>,
    pub bias: Option<String>,
    pub inference_mode: Option<bool>,
    pub fan_in_fan_out: Option<bool>,
    pub lora_dropout: Option<f64>,
}

/// Parse a HuggingFace PEFT `adapter_config.json` document without external crates.
pub fn parse_peft_adapter_config(json_str: &str) -> Result<PeftAdapterConfig, String> {
    let bytes = json_str.as_bytes();
    let mut i = 0usize;

    let skip_ws = |idx: &mut usize| {
        while *idx < bytes.len() && matches!(bytes[*idx], b' ' | b'\t' | b'\r' | b'\n') {
            *idx += 1;
        }
    };

    let parse_str = |idx: &mut usize| -> Result<String, String> {
        skip_ws(idx);
        if *idx >= bytes.len() || bytes[*idx] != b'"' {
            return Err(format!("Expected '\"' at byte {}", *idx));
        }
        *idx += 1;
        let mut out = String::new();
        while *idx < bytes.len() {
            let b = bytes[*idx];
            *idx += 1;
            if b == b'"' {
                return Ok(out);
            } else if b == b'\\' {
                if *idx >= bytes.len() {
                    return Err("Unexpected EOF in JSON escape".to_string());
                }
                let esc = bytes[*idx];
                *idx += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    other => out.push(other as char),
                }
            } else {
                out.push(b as char);
            }
        }
        Err("Unterminated JSON string".to_string())
    };

    let parse_opt_str = |idx: &mut usize| -> Result<Option<String>, String> {
        skip_ws(idx);
        if *idx + 4 <= bytes.len() && &bytes[*idx..*idx + 4] == b"null" {
            *idx += 4;
            Ok(None)
        } else {
            Ok(Some(parse_str(idx)?))
        }
    };

    let parse_opt_f64 = |idx: &mut usize| -> Result<Option<f64>, String> {
        skip_ws(idx);
        if *idx + 4 <= bytes.len() && &bytes[*idx..*idx + 4] == b"null" {
            *idx += 4;
            return Ok(None);
        }
        let start = *idx;
        while *idx < bytes.len()
            && (bytes[*idx].is_ascii_digit()
                || matches!(bytes[*idx], b'-' | b'+' | b'.' | b'e' | b'E'))
        {
            *idx += 1;
        }
        if start == *idx {
            return Err(format!("Expected JSON number at byte {}", start));
        }
        let val = json_str[start..*idx]
            .parse::<f64>()
            .map_err(|e| format!("Invalid number '{}': {}", &json_str[start..*idx], e))?;
        Ok(Some(val))
    };

    let parse_opt_bool = |idx: &mut usize| -> Result<Option<bool>, String> {
        skip_ws(idx);
        if *idx + 4 <= bytes.len() && &bytes[*idx..*idx + 4] == b"true" {
            *idx += 4;
            Ok(Some(true))
        } else if *idx + 5 <= bytes.len() && &bytes[*idx..*idx + 5] == b"false" {
            *idx += 5;
            Ok(Some(false))
        } else if *idx + 4 <= bytes.len() && &bytes[*idx..*idx + 4] == b"null" {
            *idx += 4;
            Ok(None)
        } else {
            Err(format!("Expected boolean at byte {}", *idx))
        }
    };

    let skip_val = |idx: &mut usize| -> Result<(), String> {
        skip_ws(idx);
        let mut depth = 0usize;
        let mut in_str = false;
        while *idx < bytes.len() {
            let b = bytes[*idx];
            if in_str {
                if b == b'\\' {
                    *idx += 2;
                    continue;
                }
                if b == b'"' {
                    in_str = false;
                }
            } else {
                match b {
                    b'"' => in_str = true,
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        if depth == 0 {
                            break;
                        }
                        depth -= 1;
                        if depth == 0 {
                            *idx += 1;
                            break;
                        }
                    }
                    b',' if depth == 0 => break,
                    _ => {}
                }
            }
            *idx += 1;
        }
        Ok(())
    };

    skip_ws(&mut i);
    if i >= bytes.len() || bytes[i] != b'{' {
        return Err("adapter_config.json must be a JSON object".to_string());
    }
    i += 1;

    let mut cfg = PeftAdapterConfig::default();
    loop {
        skip_ws(&mut i);
        if i < bytes.len() && bytes[i] == b'}' {
            break;
        }
        if i >= bytes.len() {
            return Err("Unexpected EOF in adapter_config.json".to_string());
        }
        let key = parse_str(&mut i)?;
        skip_ws(&mut i);
        if i >= bytes.len() || bytes[i] != b':' {
            return Err("Expected ':' after key in adapter_config.json".to_string());
        }
        i += 1;
        skip_ws(&mut i);

        match key.as_str() {
            "r" => {
                if let Some(v) = parse_opt_f64(&mut i)? {
                    if v >= 1.0 {
                        cfg.r = Some(v.round() as usize);
                    }
                }
            }
            "lora_alpha" => {
                cfg.lora_alpha = parse_opt_f64(&mut i)?;
            }
            "lora_dropout" => {
                cfg.lora_dropout = parse_opt_f64(&mut i)?;
            }
            "use_rslora" => {
                cfg.use_rslora = parse_opt_bool(&mut i)?.unwrap_or(false);
            }
            "use_dora" => {
                cfg.use_dora = parse_opt_bool(&mut i)?.unwrap_or(false);
            }
            "inference_mode" => {
                cfg.inference_mode = parse_opt_bool(&mut i)?;
            }
            "fan_in_fan_out" => {
                cfg.fan_in_fan_out = parse_opt_bool(&mut i)?;
            }
            "base_model_name_or_path" => {
                cfg.base_model_name_or_path = parse_opt_str(&mut i)?;
            }
            "task_type" => {
                cfg.task_type = parse_opt_str(&mut i)?;
            }
            "peft_type" => {
                cfg.peft_type = parse_opt_str(&mut i)?;
            }
            "bias" => {
                cfg.bias = parse_opt_str(&mut i)?;
            }
            "target_modules" => {
                if i < bytes.len() && bytes[i] == b'[' {
                    i += 1;
                    loop {
                        skip_ws(&mut i);
                        if i < bytes.len() && bytes[i] == b']' {
                            i += 1;
                            break;
                        }
                        let m = parse_str(&mut i)?;
                        cfg.target_modules.push(m);
                        skip_ws(&mut i);
                        if i < bytes.len() && bytes[i] == b',' {
                            i += 1;
                        }
                    }
                } else if i < bytes.len() && bytes[i] == b'"' {
                    cfg.target_modules_str = Some(parse_str(&mut i)?);
                } else {
                    skip_val(&mut i)?;
                }
            }
            "rank_pattern" => {
                if i < bytes.len() && bytes[i] == b'{' {
                    i += 1;
                    loop {
                        skip_ws(&mut i);
                        if i < bytes.len() && bytes[i] == b'}' {
                            i += 1;
                            break;
                        }
                        let pat_key = parse_str(&mut i)?;
                        skip_ws(&mut i);
                        if i >= bytes.len() || bytes[i] != b':' {
                            return Err("Expected ':' in rank_pattern".to_string());
                        }
                        i += 1;
                        if let Some(v) = parse_opt_f64(&mut i)? {
                            if v >= 1.0 {
                                cfg.rank_pattern.insert(pat_key, v.round() as usize);
                            }
                        }
                        skip_ws(&mut i);
                        if i < bytes.len() && bytes[i] == b',' {
                            i += 1;
                        }
                    }
                } else {
                    skip_val(&mut i)?;
                }
            }
            "alpha_pattern" => {
                if i < bytes.len() && bytes[i] == b'{' {
                    i += 1;
                    loop {
                        skip_ws(&mut i);
                        if i < bytes.len() && bytes[i] == b'}' {
                            i += 1;
                            break;
                        }
                        let pat_key = parse_str(&mut i)?;
                        skip_ws(&mut i);
                        if i >= bytes.len() || bytes[i] != b':' {
                            return Err("Expected ':' in alpha_pattern".to_string());
                        }
                        i += 1;
                        if let Some(v) = parse_opt_f64(&mut i)? {
                            cfg.alpha_pattern.insert(pat_key, v);
                        }
                        skip_ws(&mut i);
                        if i < bytes.len() && bytes[i] == b',' {
                            i += 1;
                        }
                    }
                } else {
                    skip_val(&mut i)?;
                }
            }
            "modules_to_save" => {
                if i < bytes.len() && bytes[i] == b'[' {
                    i += 1;
                    loop {
                        skip_ws(&mut i);
                        if i < bytes.len() && bytes[i] == b']' {
                            i += 1;
                            break;
                        }
                        let m = parse_str(&mut i)?;
                        cfg.modules_to_save.push(m);
                        skip_ws(&mut i);
                        if i < bytes.len() && bytes[i] == b',' {
                            i += 1;
                        }
                    }
                } else {
                    skip_val(&mut i)?;
                }
            }
            _ => {
                skip_val(&mut i)?;
            }
        }
        skip_ws(&mut i);
        if i < bytes.len() && bytes[i] == b',' {
            i += 1;
        }
    }
    Ok(cfg)
}

/// Read and parse a HuggingFace PEFT `adapter_config.json` from disk.
pub fn read_peft_adapter_config(path: &Path) -> Result<PeftAdapterConfig, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    parse_peft_adapter_config(&content)
}
