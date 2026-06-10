use std::path::Path;

use crate::parser::{CodeUnit, UnitType};

/// Hard cap on embedding text length in characters. Code units longer than
/// this are truncated before tokenization — the tokenizer would truncate
/// anyway at the model's max sequence length, but doing it here avoids
/// tokenizing megabytes of code that would be thrown away.
const MAX_EMBEDDING_TEXT_CHARS: usize = 8 * 1024;
const TRUNCATION_MARKER: &str = "\n[...truncated...]\n";

/// Shorten a file path to keep only the filename and up to 3 parent folders.
/// This makes paths easier for language models to encode and process.
fn shorten_path(path: &Path) -> String {
    let components: Vec<_> = path.components().collect();
    let len = components.len();

    // Keep at most the last 4 components (3 folders + filename)
    let start = len.saturating_sub(4);
    let shortened: std::path::PathBuf = components[start..].iter().collect();

    shortened.display().to_string()
}

/// Normalize a path string for better embedding by separating words:
/// - Add spaces around path separators (/ and \)
/// - Replace underscores, hyphens, and dots with spaces
/// - Split CamelCase words (e.g., "MyClassName" -> "My Class Name")
/// - Remove extension from processed string (it's in the appended filename)
/// - Append the original filename at the end
fn normalize_path_for_embedding(path_str: &str) -> String {
    // Extract the original filename
    let original_filename = path_str.rsplit(['/', '\\']).next().unwrap_or(path_str);

    // Remove extension from path for processing
    let path_without_ext = if let Some(dot_pos) = path_str.rfind('.') {
        &path_str[..dot_pos]
    } else {
        path_str
    };

    let mut result = String::with_capacity(path_without_ext.len() * 2);
    let chars: Vec<char> = path_without_ext.chars().collect();

    for (i, &c) in chars.iter().enumerate() {
        match c {
            '/' | '\\' => {
                // Replace path separators with spaces
                if !result.ends_with(' ') && !result.is_empty() {
                    result.push(' ');
                }
            }
            '_' | '-' | '.' => {
                // Replace underscores, hyphens, and dots with spaces
                if !result.ends_with(' ') {
                    result.push(' ');
                }
            }
            c if c.is_uppercase() => {
                // For CamelCase: add space before uppercase if previous char was lowercase
                if i > 0 {
                    let prev = chars[i - 1];
                    if prev.is_lowercase() {
                        result.push(' ');
                    }
                }
                result.push(c);
            }
            _ => {
                result.push(c);
            }
        }
    }

    // Clean up any double spaces, trim, lowercase, and append original filename
    let normalized = result
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    format!("{} {}", normalized, original_filename)
}

fn count_chars(s: &str) -> usize {
    s.chars().count()
}

fn prefix_by_chars(s: &str, max_chars: usize) -> &str {
    if max_chars == 0 {
        return "";
    }

    match s.char_indices().nth(max_chars) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if count_chars(text) <= max_chars {
        return text.to_string();
    }

    let marker_chars = count_chars(TRUNCATION_MARKER);
    if max_chars <= marker_chars {
        return prefix_by_chars(TRUNCATION_MARKER, max_chars).to_string();
    }

    let kept = prefix_by_chars(text, max_chars - marker_chars).trim_end();
    format!("{kept}{TRUNCATION_MARKER}")
}

/// Build text representation combining all 5 analysis layers.
/// This rich text is what gets embedded by ColBERT for semantic search.
pub fn build_embedding_text(unit: &CodeUnit) -> String {
    // For RawCode and Constant units, return just the raw code content
    if unit.unit_type == UnitType::RawCode || unit.unit_type == UnitType::Constant {
        return truncate_text(&unit.code, MAX_EMBEDDING_TEXT_CHARS);
    }

    let mut parts = metadata_parts(unit);

    // === File Path (shortened for better LLM encoding) ===
    // Placed before Code intentionally: when the text is truncated at
    // MAX_EMBEDDING_TEXT_CHARS, the file path is preserved while only
    // the tail of the source code is lost.
    let file_part = format!(
        "File: {}",
        normalize_path_for_embedding(&shorten_path(&unit.file))
    );
    parts.push(file_part);

    // === Full Source Code ===
    if !unit.code.is_empty() {
        parts.push(format!("Code:\n{}", unit.code));
    }

    truncate_text(&parts.join("\n"), MAX_EMBEDDING_TEXT_CHARS)
}

/// Fingerprint of everything embedding-relevant about a unit, for incremental
/// re-embedding decisions: equal fingerprints mean re-encoding the unit would
/// produce an equivalent vector and identical content metadata.
///
/// Deliberate differences from the exact embedding input:
/// - The file path is excluded, so renames keep fingerprints valid and the
///   caller decides the rename policy.
/// - Nothing is truncated: two units whose texts differ only past the
///   embedding cap still get distinct fingerprints, so the stored metadata
///   (full code, FTS text) is refreshed even when the vector wouldn't change.
///
/// Line numbers are not part of the unit's content: a unit that merely shifts
/// keeps its fingerprint, and the caller refreshes its metadata in place.
pub fn unit_fingerprint(unit: &CodeUnit) -> u64 {
    use xxhash_rust::xxh3::xxh3_64;

    if unit.unit_type == UnitType::RawCode || unit.unit_type == UnitType::Constant {
        return xxh3_64(unit.code.as_bytes());
    }

    let mut parts = metadata_parts(unit);
    if !unit.code.is_empty() {
        parts.push(format!("Code:\n{}", unit.code));
    }
    xxh3_64(parts.join("\n").as_bytes())
}

/// The path-independent metadata lines of the embedding text (everything except
/// the `File:` line and the source code). Shared between [`build_embedding_text`]
/// and [`unit_fingerprint`] so the fingerprint tracks exactly what gets embedded.
fn metadata_parts(unit: &CodeUnit) -> Vec<String> {
    let mut parts = Vec::new();

    // === Layer 1: AST (Identity + Signature) ===
    let type_str = match unit.unit_type {
        UnitType::Function => "Function",
        UnitType::Method => "Method",
        UnitType::Class => "Class",
        UnitType::Constant => "Constant",
        UnitType::Document => "Document",
        UnitType::Section => "Section",
        UnitType::RawCode => "Code block",
    };
    parts.push(format!("{}: {}", type_str, unit.name));

    if !unit.signature.is_empty() {
        parts.push(format!("Signature: {}", unit.signature));
    }

    if let Some(parent) = &unit.extends {
        if !parent.is_empty() {
            parts.push(format!("Extends: {}", parent));
        }
    }

    if let Some(class_name) = &unit.parent_class {
        if !class_name.is_empty() {
            parts.push(format!("Class: {}", class_name));
        }
    }

    if let Some(doc) = &unit.docstring {
        if !doc.is_empty() {
            parts.push(format!("Description: {}", doc));
        }
    }

    if !unit.parameters.is_empty() {
        parts.push(format!("Parameters: {}", unit.parameters.join(", ")));
    }

    if let Some(ret) = &unit.return_type {
        if !ret.is_empty() {
            // For constants, show "Type:" instead of "Returns:"
            let label = if unit.unit_type == UnitType::Constant {
                "Type"
            } else {
                "Returns"
            };
            parts.push(format!("{}: {}", label, ret));
        }
    }

    // === Layer 2: Call Graph ===
    // Only outgoing calls: they derive from the unit's own body, so the text is a
    // pure function of file content. `called_by` is intentionally excluded — it is
    // computed per indexing batch (incremental runs only see changed files' units),
    // so the same unchanged function would embed differently depending on which
    // files happened to change alongside it. It remains in the metadata DB.
    if !unit.calls.is_empty() {
        parts.push(format!("Calls: {}", unit.calls.join(", ")));
    }

    // === Layer 4: Data Flow ===
    if !unit.variables.is_empty() {
        parts.push(format!("Variables: {}", unit.variables.join(", ")));
    }

    // === Layer 5: Dependencies ===
    if !unit.imports.is_empty() {
        parts.push(format!("Uses: {}", unit.imports.join(", ")));
    }

    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_path_separators() {
        assert_eq!(
            normalize_path_for_embedding("src/parser/mod.rs"),
            "src parser mod mod.rs"
        );
    }

    #[test]
    fn test_normalize_backslash_separators() {
        // Backslashes are replaced with spaces
        assert_eq!(
            normalize_path_for_embedding("src\\parser\\mod.rs"),
            "src parser mod mod.rs"
        );
    }

    #[test]
    fn test_normalize_underscores() {
        assert_eq!(
            normalize_path_for_embedding("my_file_name.py"),
            "my file name my_file_name.py"
        );
    }

    #[test]
    fn test_normalize_hyphens() {
        assert_eq!(
            normalize_path_for_embedding("my-file-name.py"),
            "my file name my-file-name.py"
        );
    }

    #[test]
    fn test_normalize_camel_case() {
        assert_eq!(
            normalize_path_for_embedding("MyClassName.ts"),
            "my class name MyClassName.ts"
        );
    }

    #[test]
    fn test_normalize_camel_case_lowercase_start() {
        assert_eq!(
            normalize_path_for_embedding("myClassName.ts"),
            "my class name myClassName.ts"
        );
    }

    #[test]
    fn test_normalize_combined() {
        assert_eq!(
            normalize_path_for_embedding("src/utils/HttpClientHelper.rs"),
            "src utils http client helper HttpClientHelper.rs"
        );
    }

    #[test]
    fn test_normalize_snake_case_path() {
        assert_eq!(
            normalize_path_for_embedding("src/my_module/file_utils.py"),
            "src my module file utils file_utils.py"
        );
    }

    #[test]
    fn test_normalize_mixed_separators() {
        assert_eq!(
            normalize_path_for_embedding("my_great-file.rs"),
            "my great file my_great-file.rs"
        );
    }

    #[test]
    fn test_normalize_empty_string() {
        assert_eq!(normalize_path_for_embedding(""), " ");
    }

    #[test]
    fn test_normalize_simple_filename() {
        assert_eq!(normalize_path_for_embedding("main.rs"), "main main.rs");
    }

    #[test]
    fn test_build_embedding_text_truncates_raw_code_early() {
        let mut unit = CodeUnit::new(
            "raw".to_string(),
            "src/huge.rs".into(),
            1,
            1,
            crate::parser::Language::Rust,
            UnitType::RawCode,
            None,
        );
        unit.code = "x".repeat(MAX_EMBEDDING_TEXT_CHARS + 100);

        let text = build_embedding_text(&unit);
        assert_eq!(count_chars(&text), MAX_EMBEDDING_TEXT_CHARS);
        assert!(text.contains("[...truncated...]"));
    }

    #[test]
    fn test_build_embedding_text_puts_file_before_code_and_truncates_tail() {
        let mut unit = CodeUnit::new(
            "huge_fn".to_string(),
            "src/some/deep/module/very_long_name.rs".into(),
            1,
            10,
            crate::parser::Language::Rust,
            UnitType::Function,
            None,
        );
        unit.signature = "fn huge_fn()".to_string();
        unit.code = "a".repeat(MAX_EMBEDDING_TEXT_CHARS + 500);

        let text = build_embedding_text(&unit);
        assert!(count_chars(&text) <= MAX_EMBEDDING_TEXT_CHARS);
        assert!(text.contains("[...truncated...]"));
        assert!(text.contains("File: "));
        assert!(text.contains("File: some deep module very long name very_long_name.rs"));
        let file_idx = text.find("File: ").unwrap();
        let code_idx = text.find("Code:\n").unwrap();
        assert!(file_idx < code_idx);
    }

    /// The embedding text must be a pure function of the unit's own file content.
    /// `called_by` is populated by build_call_graph over whatever batch of units
    /// happens to be indexed together, so baking it into the text made the same
    /// function embed differently across incremental runs.
    #[test]
    fn test_embedding_text_ignores_batch_local_called_by() {
        let mut unit = CodeUnit::new(
            "compute".to_string(),
            "src/math.rs".into(),
            1,
            5,
            crate::parser::Language::Rust,
            UnitType::Function,
            None,
        );
        unit.signature = "fn compute() -> i32".to_string();
        unit.code = "fn compute() -> i32 { helper() }".to_string();
        unit.calls = vec!["helper".to_string()];

        let without_callers = build_embedding_text(&unit);
        unit.called_by = vec!["caller_a".to_string(), "caller_b".to_string()];
        let with_callers = build_embedding_text(&unit);

        assert_eq!(without_callers, with_callers);
        assert!(!with_callers.contains("Called by"));
        assert!(with_callers.contains("Calls: helper"));
    }

    fn fingerprint_unit(file: &str, name: &str, code: &str) -> CodeUnit {
        let mut unit = CodeUnit::new(
            name.to_string(),
            file.into(),
            10,
            20,
            crate::parser::Language::Rust,
            UnitType::Function,
            None,
        );
        unit.signature = format!("fn {name}()");
        unit.code = code.to_string();
        unit
    }

    /// Fingerprints must survive everything that doesn't change the unit's own
    /// content: file renames (path excluded by design), line shifts, and
    /// batch-local called_by. They must change when the code changes.
    #[test]
    fn test_unit_fingerprint_stability() {
        let unit = fingerprint_unit("src/a.rs", "compute", "fn compute() { work() }");
        let base = unit_fingerprint(&unit);

        let mut renamed_file = unit.clone();
        renamed_file.file = "src/moved/elsewhere.rs".into();
        renamed_file.qualified_name = "src/moved/elsewhere.rs::compute".to_string();
        assert_eq!(
            base,
            unit_fingerprint(&renamed_file),
            "path must not matter"
        );

        let mut shifted = unit.clone();
        shifted.line = 100;
        shifted.end_line = 110;
        assert_eq!(
            base,
            unit_fingerprint(&shifted),
            "line numbers must not matter"
        );

        let mut with_callers = unit.clone();
        with_callers.called_by = vec!["main".to_string()];
        assert_eq!(
            base,
            unit_fingerprint(&with_callers),
            "called_by must not matter"
        );

        let mut edited = unit.clone();
        edited.code = "fn compute() { work(); more() }".to_string();
        assert_ne!(
            base,
            unit_fingerprint(&edited),
            "code changes must change it"
        );

        let mut redoc = unit.clone();
        redoc.docstring = Some("Computes things.".to_string());
        assert_ne!(
            base,
            unit_fingerprint(&redoc),
            "docstring changes must change it"
        );
    }

    /// Unlike the embedding text, the fingerprint must see past the truncation
    /// cap: a change in the tail of a huge unit doesn't move the vector, but the
    /// stored metadata (full code, FTS text) still needs refreshing.
    #[test]
    fn test_unit_fingerprint_sees_past_truncation_cap() {
        let long_code = "x".repeat(MAX_EMBEDDING_TEXT_CHARS + 100);
        let a = fingerprint_unit("src/a.rs", "huge", &long_code);
        let mut b = a.clone();
        b.code = format!("{}y", &long_code[..long_code.len() - 1]);

        assert_eq!(build_embedding_text(&a), build_embedding_text(&b));
        assert_ne!(unit_fingerprint(&a), unit_fingerprint(&b));
    }

    /// RawCode units embed as bare code (no name/signature lines), so their
    /// fingerprint is content-only: the same block keeps its fingerprint when a
    /// shift renames it (raw_code_<line>), letting the caller treat it as moved.
    #[test]
    fn test_unit_fingerprint_raw_code_is_content_only() {
        let mut a = fingerprint_unit("src/a.rs", "raw_code_5", "let x = 1;");
        a.unit_type = UnitType::RawCode;
        let mut b = fingerprint_unit("src/a.rs", "raw_code_42", "let x = 1;");
        b.unit_type = UnitType::RawCode;
        b.line = 42;
        b.end_line = 42;

        assert_eq!(unit_fingerprint(&a), unit_fingerprint(&b));
    }

    #[test]
    fn test_normalize_consecutive_separators() {
        // Multiple underscores/hyphens should collapse to single space
        assert_eq!(
            normalize_path_for_embedding("my__file--name.rs"),
            "my file name my__file--name.rs"
        );
    }

    #[test]
    fn test_normalize_no_extension() {
        assert_eq!(
            normalize_path_for_embedding("src/Makefile"),
            "src makefile Makefile"
        );
    }
}
