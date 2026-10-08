//! Guards the shared shader boundary used by the WGPU, Metal, and DirectX renderers.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("gpui crate must live at <workspace>/crates/gpui")
        .to_path_buf()
}

fn collect_renderer_shader_sources(root: &Path, sources: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(root)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", root.display()));

    for entry in entries {
        let entry = entry.unwrap_or_else(|error| {
            panic!(
                "failed to inspect an entry under {}: {error}",
                root.display()
            )
        });
        let path = entry.path();
        let file_type = entry
            .file_type()
            .unwrap_or_else(|error| panic!("failed to inspect {}: {error}", path.display()));

        if file_type.is_dir() {
            collect_renderer_shader_sources(&path, sources);
        } else if matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("wgsl" | "metal" | "hlsl")
        ) {
            sources.push(path);
        }
    }
}

fn relative_paths(root: &Path, paths: impl IntoIterator<Item = PathBuf>) -> Vec<String> {
    let mut relative: Vec<_> = paths
        .into_iter()
        .map(|path| {
            path.strip_prefix(root)
                .unwrap_or(&path)
                .display()
                .to_string()
        })
        .collect();
    relative.sort();
    relative
}

#[test]
fn renderer_backends_share_rust_authored_shaders() {
    let root = workspace_root();

    let native_renderer_modules = [
        "crates/gpui_apple/src/metal_renderer.rs",
        "crates/gpui_windows/src/directx_renderer.rs",
    ];
    for relative in native_renderer_modules {
        let path = root.join(relative);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        assert!(
            source.contains("NATIVE_SHADERS"),
            "{} must consume generated shared shader artifacts",
            path.display()
        );
    }

    let macos_root = root.join("crates/gpui_macos/src");
    assert!(
        !macos_root.join("metal_renderer.rs").exists(),
        "the extracted Apple renderer must not retain a macOS compatibility module"
    );
    let macos_lib = fs::read_to_string(macos_root.join("gpui_macos.rs"))
        .expect("failed to read the macOS crate root");
    assert!(
        !macos_lib.contains("pub mod metal_renderer"),
        "the macOS crate must not re-export the extracted renderer"
    );

    let mut platform_shader_sources = Vec::new();
    for relative in [
        "crates/gpui_apple/src",
        "crates/gpui_macos/src",
        "crates/gpui_windows/src",
    ] {
        let path = root.join(relative);
        if path.exists() {
            collect_renderer_shader_sources(&path, &mut platform_shader_sources);
        }
    }
    assert!(
        platform_shader_sources.is_empty(),
        "platform-local renderer shader source bypasses shared generation: {}",
        relative_paths(&root, platform_shader_sources).join(", ")
    );

    let rust_shader_module = root.join("crates/gpui_render/src/shaders/mod.rs");
    assert!(
        rust_shader_module.is_file(),
        "renderers must keep their typed Rust shader module in gpui_render"
    );

    let mut standalone_shader_sources = Vec::new();
    for relative in ["crates/gpui_render/src", "crates/gpui_wgpu/src"] {
        collect_renderer_shader_sources(&root.join(relative), &mut standalone_shader_sources);
    }
    assert!(
        standalone_shader_sources.is_empty(),
        "standalone shader sources bypass typed generation: {}",
        relative_paths(&root, standalone_shader_sources).join(", ")
    );
}

#[test]
fn directx_registers_and_instance_views_follow_generated_shader_contracts() {
    let path = workspace_root().join("crates/gpui_windows/src/directx_renderer.rs");
    let source = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));

    for binding in [
        "shader_interface::DATA_BUFFER_BINDING",
        "shader_interface::PRIMARY_TEXTURE_BINDING",
        "shader_interface::PRIMARY_SAMPLER_BINDING",
        "shader_interface::SURFACE_SAMPLER_BINDING",
    ] {
        assert!(
            source.contains(binding),
            "{} must derive native register slots from {binding}",
            path.display()
        );
    }
    assert!(
        source.contains("self.pipelines.surfaces.params_buffer"),
        "{} must bind the surface pipeline's generated uniform buffer",
        path.display()
    );
    assert!(
        source.contains("PSSetSamplers(SURFACE_SAMPLER_REGISTER"),
        "{} must bind the surface sampler at its generated slot",
        path.display()
    );
    assert!(
        !source.contains("create_buffer_view_range"),
        "{} must reuse each pipeline's whole-buffer SRV and select batches with first-instance",
        path.display()
    );
}

#[test]
fn native_renderer_fallbacks_and_intermediates_are_lazy() {
    let root = workspace_root();
    let wgpu_context_path = root.join("crates/gpui_wgpu/src/wgpu_context.rs");
    let wgpu_context = fs::read_to_string(&wgpu_context_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", wgpu_context_path.display()));
    assert!(
        wgpu_context.contains("enum NativeBackend"),
        "{} must represent one native backend as a closed type",
        wgpu_context_path.display()
    );
    assert!(
        wgpu_context.contains("impl From<NativeBackend> for wgpu::Backends")
            && wgpu_context.contains("backends: self.into()")
            && wgpu_context.contains("try_in_preference_order"),
        "{} must initialize one backend per fallback attempt",
        wgpu_context_path.display()
    );
    assert!(
        !wgpu_context.contains("Backends::VULKAN | wgpu::Backends::GL"),
        "{} must not eagerly construct the GL fallback beside Vulkan",
        wgpu_context_path.display()
    );

    let directx_path = root.join("crates/gpui_windows/src/directx_renderer.rs");
    let directx = fs::read_to_string(&directx_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", directx_path.display()));
    assert!(
        directx.contains("blur: Option<BlurResources>"),
        "{} must make large filter targets optional",
        directx_path.display()
    );
    assert!(
        directx.contains("path: Option<PathResources>"),
        "{} must make large path targets optional",
        directx_path.display()
    );
    assert!(
        directx.contains("ensure_blur_resources(device, requirements.isolated_target_count)"),
        "{} must allocate filter targets from the typed scene requirements",
        directx_path.display()
    );
    assert!(
        directx.contains("ShaderModule::Blur.bytecode()")
            && directx.contains("create_vertex_shader(device, bytecode.vertex)"),
        "{} must create DirectX shaders from generated bytecode",
        directx_path.display()
    );
    assert!(
        directx.contains("surface_views: FxHashMap<usize, CachedSurfaceView>"),
        "{} must reuse capture texture views while their surfaces remain active",
        directx_path.display()
    );
}

#[test]
fn windows_default_renderer_has_no_wgpu_path() {
    let root = workspace_root();
    // Inspect Cargo's feature definitions rather than rejecting the optional
    // dependency's name. This also works when the audit runs on macOS or Linux.
    let output = Command::new(env!("CARGO"))
        .current_dir(&root)
        .args([
            "metadata",
            "--format-version=1",
            "--no-deps",
            "--offline",
            "--locked",
        ])
        .output()
        .expect("failed to inspect workspace manifests");
    assert!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("invalid Cargo metadata");
    let packages = metadata["packages"].as_array().expect("workspace packages");
    let package = |name: &str| {
        packages
            .iter()
            .find(|package| package["name"] == name)
            .unwrap_or_else(|| panic!("missing workspace package {name}"))
    };
    let windows = package("gpui_ce_windows");
    let platform = package("gpui_ce_platform");
    let dependencies = windows["dependencies"]
        .as_array()
        .expect("Windows dependencies");
    assert!(
        !dependencies
            .iter()
            .any(|dependency| dependency["name"] == "wgpu"),
        "Windows must use the shared optional WGPU renderer"
    );
    let wgpu = dependencies
        .iter()
        .find(|dependency| dependency["rename"] == "gpui_wgpu")
        .expect("optional Windows WGPU dependency");
    assert_eq!(wgpu["name"], "gpui_ce_wgpu");
    assert_eq!(wgpu["optional"], true, "WGPU must remain optional");
    assert_eq!(wgpu["target"], "cfg(target_os = \"windows\")");
    assert_eq!(
        windows["features"]["default"],
        serde_json::json!(["gpui/default"])
    );
    assert_eq!(platform["features"]["default"], serde_json::json!([]));
    assert_eq!(
        windows["features"]["windows-wgpu"],
        serde_json::json!(["dep:gpui_wgpu"])
    );
    assert_eq!(
        platform["features"]["windows-wgpu"],
        serde_json::json!(["gpui_windows/windows-wgpu"])
    );

    let renderer_path = root.join("crates/gpui_windows/src/renderer.rs");
    let renderer = fs::read_to_string(&renderer_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", renderer_path.display()));
    assert!(
        renderer.contains("Dx11(DirectXRenderer)"),
        "{} must retain the native DirectX renderer",
        renderer_path.display()
    );
    assert!(
        renderer
            .lines()
            .zip(renderer.lines().skip(1))
            .any(|(guard, variant)| {
                guard.trim() == "#[cfg(feature = \"windows-wgpu\")]"
                    && variant.trim().starts_with("Dx12(")
            }),
        "{} must gate the DX12 renderer behind windows-wgpu",
        renderer_path.display()
    );
}
