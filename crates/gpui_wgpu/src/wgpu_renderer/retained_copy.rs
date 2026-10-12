//! Conservative full-image presentation copy of the retained frame.

pub(super) fn format_supported(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
    )
}

pub(super) fn alpha_supported(alpha: wgpu::CompositeAlphaMode) -> bool {
    matches!(
        alpha,
        wgpu::CompositeAlphaMode::Opaque | wgpu::CompositeAlphaMode::PreMultiplied
    )
}

#[derive(Clone, Copy)]
struct CopyDescriptor {
    size: wgpu::Extent3d,
    format: wgpu::TextureFormat,
    dimension: wgpu::TextureDimension,
    samples: u32,
    mips: u32,
    usage: wgpu::TextureUsages,
}

impl From<&wgpu::Texture> for CopyDescriptor {
    fn from(texture: &wgpu::Texture) -> Self {
        Self {
            size: texture.size(),
            format: texture.format(),
            dimension: texture.dimension(),
            samples: texture.sample_count(),
            mips: texture.mip_level_count(),
            usage: texture.usage(),
        }
    }
}

fn descriptor_matches(
    descriptor: CopyDescriptor,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    usage: wgpu::TextureUsages,
) -> bool {
    format_supported(format)
        && width > 0
        && height > 0
        && descriptor.format == format
        && descriptor.dimension == wgpu::TextureDimension::D2
        && descriptor.size
            == (wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            })
        && descriptor.samples == 1
        && descriptor.mips == 1
        && descriptor.usage.contains(usage)
}

pub(super) fn compatible(
    source: &wgpu::Texture,
    destination: &wgpu::Texture,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    alpha: wgpu::CompositeAlphaMode,
) -> bool {
    source != destination
        && alpha_supported(alpha)
        && descriptor_matches(
            source.into(),
            format,
            width,
            height,
            wgpu::TextureUsages::COPY_SRC,
        )
        && descriptor_matches(
            destination.into(),
            format,
            width,
            height,
            wgpu::TextureUsages::COPY_DST,
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presentation_copy_rejects_conversion_multisampling_layers_and_missing_usage() {
        let valid = CopyDescriptor {
            size: wgpu::Extent3d {
                width: 201,
                height: 103,
                depth_or_array_layers: 1,
            },
            format: wgpu::TextureFormat::Bgra8Unorm,
            dimension: wgpu::TextureDimension::D2,
            samples: 1,
            mips: 1,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        };
        let accepts = |descriptor| {
            descriptor_matches(
                descriptor,
                valid.format,
                201,
                103,
                wgpu::TextureUsages::COPY_DST,
            )
        };
        assert!(accepts(valid));
        for invalid in [
            CopyDescriptor {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                ..valid
            },
            CopyDescriptor {
                format: wgpu::TextureFormat::Rgba8Unorm,
                ..valid
            },
            CopyDescriptor {
                samples: 4,
                ..valid
            },
            CopyDescriptor { mips: 2, ..valid },
            CopyDescriptor {
                size: wgpu::Extent3d {
                    depth_or_array_layers: 2,
                    ..valid.size
                },
                ..valid
            },
            CopyDescriptor {
                size: wgpu::Extent3d {
                    width: 200,
                    ..valid.size
                },
                ..valid
            },
            CopyDescriptor {
                dimension: wgpu::TextureDimension::D3,
                ..valid
            },
            CopyDescriptor {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                ..valid
            },
        ] {
            assert!(!accepts(invalid));
        }
        assert!(!format_supported(wgpu::TextureFormat::Rgba16Float));
        assert!(!format_supported(wgpu::TextureFormat::Rgba8UnormSrgb));
        assert!(alpha_supported(wgpu::CompositeAlphaMode::PreMultiplied));
        assert!(!alpha_supported(wgpu::CompositeAlphaMode::PostMultiplied));
        assert!(!alpha_supported(wgpu::CompositeAlphaMode::Inherit));
    }
}
