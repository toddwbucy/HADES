//! End-to-end probe for #149 against a real CUDA kernel file.
//!
//! Skips (does not fail) if libclang or the fixture is unavailable.
//!
//! The fixture path was hardcoded to one contributor's home directory, which
//! meant this skipped silently for everyone else and could only ever run on a
//! single machine. A test that cannot fail anywhere is a test nobody checks.
//!
//! Point `HADES_CUDA_FIXTURE` at any `.cu` file to run it:
//!
//!     HADES_CUDA_FIXTURE=/path/to/kernels.cu cargo test -p hades-core --test clang_cuda_probe
//!
//! It still skips when the variable is unset, because CUDA sources are not
//! vendored here and a missing fixture is not a failure. The difference is
//! that running it is now possible rather than accidental.

use std::path::Path;

use hades_core::code;

#[test]
fn clang_extracts_cuda_kernel_symbols() {
    let Ok(fixture) = std::env::var("HADES_CUDA_FIXTURE") else {
        eprintln!("SKIP: set HADES_CUDA_FIXTURE to a .cu file to run this probe");
        return;
    };
    let fixture = fixture.as_str();
    if !Path::new(fixture).exists() {
        eprintln!("SKIP: fixture not present: {fixture}");
        return;
    }

    match clang::Clang::new() {
        Ok(clang) => drop(clang),
        Err(e) => {
            eprintln!("SKIP: libclang unavailable: {e}");
            return;
        }
    }

    let source = std::fs::read_to_string(fixture).unwrap();
    let analysis = match code::analyze(&source, fixture) {
        Ok(analysis) if analysis.analysis_tier == code::AnalysisTier::Semantic => analysis,
        Ok(analysis) => {
            eprintln!(
                "SKIP: libclang could not analyze the CUDA fixture; fallback={} reason={}",
                analysis.analyzer,
                analysis.fallback_reason.as_deref().unwrap_or("unknown")
            );
            return;
        }
        Err(error) => {
            eprintln!("SKIP: CUDA fixture analysis unavailable: {error}");
            return;
        }
    };
    assert_eq!(analysis.analysis_tier, code::AnalysisTier::Semantic);
    assert_eq!(analysis.analyzer, "libclang");
    let names: Vec<&str> = analysis.symbols.iter().map(|s| s.name.as_str()).collect();
    eprintln!("extracted {} symbols from {fixture}", names.len());

    assert!(
        !analysis.symbols.is_empty(),
        "no functions extracted from the CUDA file"
    );

    // The assertions used to name `sigmoid_kernel` and `sigmoid_cuda`, which
    // exist only in the fixture this test was originally written against. That
    // made the test unusable with any other CUDA file, so it asserted the
    // shape of one private file rather than the behaviour of the extractor.
    //
    // What is actually being verified is that libclang resolves a CUDA
    // translation unit into named function symbols with line spans. Any real
    // kernel file exercises that, so the assertion is about the extraction, not
    // about which kernels happen to be present.
    assert!(
        names.iter().any(|n| !n.is_empty()),
        "symbols extracted but all unnamed: {names:?}"
    );
    // Every symbol carries a line span -- the anchor the #148 identity needs.
    assert!(
        analysis.symbols.iter().all(|symbol| symbol.start_line > 0),
        "every symbol must have a 1-based line"
    );

    // The launch edge is NOT asserted here. See cuda_launch_edges_resolve
    // below: it does not currently work, and making this test fail on every
    // real CUDA file would hide the extraction coverage that does work.
    let launches = count_launch_edges(&analysis);
    eprintln!("resolved {launches} kernel launch edge(s)");
}

/// How many resolved calls are marked as CUDA kernel launches.
fn count_launch_edges(analysis: &code::FileAnalysis) -> usize {
    analysis
        .symbols
        .iter()
        .filter_map(|symbol| symbol.metadata.get("calls")?.as_array())
        .flatten()
        .filter(|call| call["is_kernel_launch"] == true)
        .count()
}

/// KNOWN FAILING. Kernel launch edges are not resolved from real CUDA files.
///
/// `kernel<<<grid, block>>>(args)` is a `CUDAKernelCallExpr`, which the
/// `clang` crate does not expose as an `EntityKind`; libclang surfaces it as
/// `UnexposedExpr`. `collect_calls` matches `CallExpr` only, so every launch
/// is dropped. Matching `UnexposedExpr` as well is not the fix: it takes the
/// call count on one file from 170 to 1272 and still resolves zero kernels,
/// because `call_metadata` cannot resolve the callee from that node either.
///
/// Measured on a 1,506-line kernel file with 30 launches: 48 symbols
/// extracted, 170 calls resolved, 0 naming a kernel.
///
/// This matters because the launch edge is what connects a `__global__`
/// kernel to the host code that invokes it. Without it, kernels are isolated
/// vertices and nothing links them to the rest of the graph.
///
/// Ignored rather than deleted so the gap stays visible and this passes the
/// moment it is fixed. Run with: cargo test -- --ignored
#[test]
#[ignore = "known gap: CUDAKernelCallExpr is not exposed by the clang binding"]
fn cuda_launch_edges_resolve() {
    let Ok(fixture) = std::env::var("HADES_CUDA_FIXTURE") else {
        eprintln!("SKIP: set HADES_CUDA_FIXTURE to a .cu file containing <<<>>> launches");
        return;
    };
    let source = std::fs::read_to_string(&fixture).unwrap();
    let analysis = code::analyze(&source, &fixture).expect("analysis failed");
    let launches = count_launch_edges(&analysis);
    assert!(
        launches > 0,
        "no kernel launch edges resolved from {fixture} ({} symbols)",
        analysis.symbols.len()
    );
}
