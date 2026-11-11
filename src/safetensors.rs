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
        match s {
            "F32" => Ok(DType::F32),
            "F16" => Ok(DType::F16),
            "BF16" => Ok(DType::BF16),
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

/// Read all 2D tensors from a `.safetensors` file on disk.
pub fn read_safetensors(path: &Path) -> Result<BTreeMap<String, Tensor2D>, String> {
    let raw = fs::read(path).map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    if raw.len() < 8 {
        return Err("File too small to be a valid .safetensors archive".to_string());
    }
    let header_len = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    if 8 + header_len > raw.len() {
        return Err("Invalid safetensors header length".to_string());
    }
    let header_str = std::str::from_utf8(&raw[8..8 + header_len])
        .map_err(|e| format!("Invalid UTF-8 in safetensors header: {}", e))?;
    let entries = parse_safetensors_header(header_str)?;
    let body = &raw[8 + header_len..];

    let mut out = BTreeMap::new();
    for (name, entry) in entries {
        if entry.shape.len() != 2 {
            continue;
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
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
        if let Some(base) = key
            .strip_suffix(".lora_A.weight")
            .or_else(|| key.strip_suffix(".lora_A.default.weight"))
            .or_else(|| key.strip_suffix(".lora_down.weight"))
            .or_else(|| key.strip_suffix(".lora.down.weight"))
            .or_else(|| key.strip_suffix(".lora_linear_layer.down.weight"))
            .or_else(|| key.strip_suffix("_lora.down.weight"))
        {
            a_map.insert(base.to_string(), (key.clone(), tensor.clone()));
        } else if let Some(base) = key
            .strip_suffix(".lora_B.weight")
            .or_else(|| key.strip_suffix(".lora_B.default.weight"))
            .or_else(|| key.strip_suffix(".lora_up.weight"))
            .or_else(|| key.strip_suffix(".lora.up.weight"))
            .or_else(|| key.strip_suffix(".lora_linear_layer.up.weight"))
            .or_else(|| key.strip_suffix("_lora.up.weight"))
        {
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

/// Write a HuggingFace PEFT-compatible `adapter_config.json` containing `rank_pattern` and
/// `alpha_pattern` for dynamically compressed heterogeneous-rank LoRA adapters (fixes PEFT #3737).
pub fn write_peft_rank_pattern_config(
    path: &Path,
    rank_pattern: &BTreeMap<String, usize>,
    default_rank: usize,
) -> io::Result<()> {
    let mut entries = Vec::with_capacity(rank_pattern.len());
    for (name, &r) in rank_pattern {
        let clean_name = name
            .strip_prefix("base_model.model.")
            .unwrap_or(name.as_str());
        entries.push(format!("    \"{}\": {}", clean_name, r));
    }
    let pattern_body = entries.join(",\n");
    let json = format!(
        "{{\n  \"peft_type\": \"LORA\",\n  \"r\": {},\n  \"lora_alpha\": {},\n  \"rank_pattern\": {{\n{}\n  }},\n  \"alpha_pattern\": {{\n{}\n  }}\n}}\n",
        default_rank, default_rank, pattern_body, pattern_body
    );
    let mut file = fs::File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}
