
pub const vector_data: &[u8] = include_bytes!("vector_data");

pub struct VectorField {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl VectorField {
    pub fn from_bytes(
        device: &wgpu::Device, queue: &wgpu::Queue, 
        bytes: &[u8], edge_dimensions: (u32, u32, u32)
    ) -> Self {
        //RGBA32f data
        let mut data = vec![0f32; (edge_dimensions.0 * edge_dimensions.1 * edge_dimensions.2 * 4) as usize];

        //copy data from raw floats, adding a 4th 'alpha' channel padding.
        for i in 0..(edge_dimensions.0 * edge_dimensions.1 * edge_dimensions.2) {
            let raw_index: usize = (i * 3 * 4) as usize; //(i * 3): 3 float packs, * 4: 4 bytes per float
            let data_index: usize = (i * 4) as usize; //(i * 4): RGBA = 4 float packs
            
            let floats = (
                bytemuck::try_pod_read_unaligned::<f32>(&bytes[raw_index..(raw_index + 4)]).expect(&format!("couldn't cast slice to float! interpreting vector {}", i)),
                bytemuck::try_pod_read_unaligned::<f32>(&bytes[(raw_index + 4)..(raw_index + 8)]).expect(&format!("couldn't cast slice to float! interpreting vector {}", i)),
                bytemuck::try_pod_read_unaligned::<f32>(&bytes[(raw_index + 8)..(raw_index + 12)]).expect(&format!("couldn't cast slice to float! interpreting vector {}", i)),
            );

            data[data_index + 0] = floats.0;
            data[data_index + 1] = floats.1;
            data[data_index + 2] = floats.2;
        }

        let texture_size = wgpu::Extent3d {
            width: edge_dimensions.0,
            height: edge_dimensions.1,
            depth_or_array_layers: edge_dimensions.2,
        };

        let texture = device.create_texture(
            &wgpu::TextureDescriptor {
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                label: Some("vector field"),
                view_formats: &[],
            }
        );
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect:wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(edge_dimensions.0 * 4 * 4),
                rows_per_image: Some(edge_dimensions.1),
            },
            texture_size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self { texture, view, sampler }
    }
}