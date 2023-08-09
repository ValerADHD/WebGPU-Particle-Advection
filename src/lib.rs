#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;

use cgmath::prelude::*;

mod texture;
mod data;

use winit::{
    event::*,
    window::{Window, WindowBuilder},
    event_loop::{ControlFlow, EventLoop},
};

use std::time::Instant;

use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex { 
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { //position
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute { //tex_coords
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ]
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex { position: [0.5, 0.5, 0.0], tex_coords: [1.0, 0.0], }, // C
    Vertex { position: [-0.5, 0.5, 0.0], tex_coords: [0.0, 0.0], }, // A
    Vertex { position: [0.5, -0.5, 0.0], tex_coords: [1.0, 1.0], }, // C
    Vertex { position: [-0.5, -0.5, 0.0], tex_coords: [0.0, 1.0], }, // B
];

struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,

    orbit_radius: f32,
    orbit_angles_deg: cgmath::Vector2<f32>,
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }

    fn orbit_target(&mut self, target: cgmath::Point3<f32>, view_angles_deg: cgmath::Vector2<f32>) {
        self.target = target;
        self.orbit_angles_deg = view_angles_deg;

        self.eye = {
            let rotation_mat = cgmath::Matrix3::from_angle_y(cgmath::Deg(view_angles_deg.y)) * cgmath::Matrix3::from_angle_x(cgmath::Deg(view_angles_deg.x));

            let eye_offset = rotation_mat * (cgmath::Vector3::unit_z() * self.orbit_radius);

            target + eye_offset
        };
    }
}

struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    scroll_delta: f32,
}

impl CameraController {
    fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            scroll_delta: 0.0,
        }
    }

    fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input: KeyboardInput {
                    state,
                    virtual_keycode: Some(keycode),
                    ..
                },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::W | VirtualKeyCode::Up => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::A | VirtualKeyCode::Left => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::D | VirtualKeyCode::Right => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,

        }
    }

    fn update_camera(&self, camera: &mut Camera) {
        let horizontal_net_input = if self.is_left_pressed { 1 } else { 0 } + if self.is_right_pressed { -1 } else { 0 };
        let vertical_net_input = if self.is_forward_pressed { 1 } else { 0 } + if self.is_backward_pressed { -1 } else { 0 };

        camera.orbit_target(
            camera.target, 
            camera.orbit_angles_deg + cgmath::Vector2::new(vertical_net_input as f32 * self.speed, horizontal_net_input as f32 * self.speed)
        );
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

const NUM_INSTANCES_PER_ROW: u64 = 1000;

struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation)).into()
        }
    }
}

impl InstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance, //per instance attribute, instead of per-vertex
            attributes: &[ //mat4 = 4 vec4s, so we set 4 attributes
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ]
        }
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,

    diffuse_bind_group: wgpu::BindGroup,
    diffuse_texture: texture::Texture,

    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_uniform_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,

    compute_bind_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    
    vector_field_bind_group: wgpu::BindGroup,
    vector_field_texture: data::VectorField,

    clear_color: wgpu::Color,
}

const FORCE_HIGH_PERFORMANCE: bool = true;

impl State {
    async fn new(window: Window) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: if FORCE_HIGH_PERFORMANCE { wgpu::PowerPreference::HighPerformance } else { wgpu::PowerPreference::default() },
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                label: None,
            },
            None,
        ).await.expect("Unable to create device!");

        let surface_caps = surface.get_capabilities(&adapter);

        //sRGB surface
        let surface_format = surface_caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let diffuse_bytes = include_bytes!("happy-tree.png");
        
        let diffuse_texture = texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy-tree.png").expect("Unable to create texture from image!");

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry { //texture
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry { //sampler
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    }
                ],
                label: Some("texture_bind_group_layout"),
            });
        let texture_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                    },
                ],
                label: Some("diffuse_bind_group"),
            }
        );
        
        let mut camera = Camera {
            eye: (0.0, 0.0, 1.0).into(),
            target: (0.5, 0.5, 0.5).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,

            orbit_radius: 2.0,
            orbit_angles_deg: (0.0, 0.0).into(),
        };

        let camera_controller = CameraController::new(10.0);

        camera.orbit_target(camera.target, (-45.0, 45.0).into());

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_uniform_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("camera_bind_group_layout"),
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("camera_bind_group"),
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let render_pipeline_layout = 
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[]
            });
        
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    Vertex::desc(), InstanceRaw::desc(),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );
        let num_vertices: u32 = VERTICES.len() as u32;
        
        let mut instances: Vec<Instance> = Vec::new();

        for x in 0..NUM_INSTANCES_PER_ROW {
            for z in 0..NUM_INSTANCES_PER_ROW {
                let position = cgmath::Vector3::new((x as f32) / NUM_INSTANCES_PER_ROW as f32, 0.5, (z as f32) / NUM_INSTANCES_PER_ROW as f32);

                let rotation = cgmath::Quaternion::from_angle_x(cgmath::Deg(0.0));

                instances.push(Instance { position, rotation });
            }
        }

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
        });

        let vector_field_texture = data::VectorField::from_bytes(&device, &queue, data::vector_data, (50, 50, 50));

        let vector_field_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry { //texture
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D3,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry { //sampler
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    }
                ],
                label: Some("vector_field_bind_group_layout"),
            });
        let vector_field_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &vector_field_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&vector_field_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&vector_field_texture.sampler),
                    },
                ],
                label: Some("vector_field_bind_group"),
            }
        );

        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ]
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: instance_buffer.as_entire_binding(),
                }
            ]
        });

        let compute_shader = device.create_shader_module(wgpu::include_wgsl!("compute.wgsl"));

        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute pipeline layout"),
            bind_group_layouts: &[&compute_bind_group_layout, &vector_field_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        });

        Self {
            size,
            surface,
            device,
            queue,
            config,
            window,
            render_pipeline,
            vertex_buffer,
            num_vertices,

            diffuse_bind_group: texture_bind_group,
            diffuse_texture,

            camera,
            camera_controller,
            camera_uniform,
            camera_uniform_buffer,
            camera_bind_group,

            instances,
            instance_buffer,

            compute_bind_group,
            compute_pipeline,

            vector_field_bind_group,
            vector_field_texture,

            clear_color: wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0 },
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera_controller.process_events(event)
    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(&self.camera_uniform_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Compute Command Encoder") });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Compute Pass") });
        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
        compute_pass.set_bind_group(1, &self.vector_field_bind_group, &[]);
        compute_pass.dispatch_workgroups(((NUM_INSTANCES_PER_ROW * NUM_INSTANCES_PER_ROW) as f32 / 64f32).ceil() as u32, 1, 1);
        
        drop(compute_pass);
        
        self.queue.submit([encoder.finish()]);
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.clear_color),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
        render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..)); //instance positions
        render_pass.draw(0..self.num_vertices, 0..self.instances.len() as u32);

        //we no longer need the render pass object, we drop it because it has a 
        //mutable reference to our command encoder.
        drop(render_pass);

        self.queue.submit([encoder.finish()]);
        output.present();

        Ok(())
    }
}

#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch="wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger!");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    use winit::dpi::PhysicalSize;
    window.set_inner_size(PhysicalSize::new(1280, 1280));

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document()) //get document
            .and_then(|doc| {
                let dest = doc.get_element_by_id("wasm-example")?; //get parent element in DOM
                dest.append_child(&window.canvas()).ok()?; //attach to parent element
                Some(())
            })
            .expect("Couldn't append canvas to document body");
    }

    let mut state = State::new(window).await;

    let mut frame_start: Instant;
    cfg_if::cfg_if! {
        if #[cfg(target_arch="wasm32")] {} else {
            frame_start = Instant::now();
        }
    }
    

    event_loop.run(move |event, _, control_flow| {
        match event {
            // event is for correct window
            Event::WindowEvent { window_id, ref event } if window_id == state.window.id() =>
            if !state.input(event) { // let state handle inputs first
                match event {
                    // match events for our window only
                    WindowEvent::CloseRequested | 
                    WindowEvent::KeyboardInput {
                        input: KeyboardInput {
                            state: ElementState::Pressed, virtual_keycode: Some(VirtualKeyCode::Escape), .. 
                        },
                        ..
                    } => { // either close requested or escape key pressed
                        *control_flow = ControlFlow::Exit
                    }
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        //state.clear_color = wgpu::Color { r: position.x / 400.0, g: position.y / 400.0, b: 0.3, a: 1.0 };
                    }
                    _ => {},
                }
            },
            Event::RedrawRequested(window_id) if window_id == state.window.id() => {
                state.update();
                match state.render() {
                    Ok(_) => {},
                    // if surface is lost, call resize to force reconfigure
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    Err(e) => eprintln!("{:?}", e),
                };
                cfg_if::cfg_if! {
                    if #[cfg(target_arch="wasm32")] {} else {
                        let frame_time_millis: f64 = { 
                            let elapsed = frame_start.elapsed().as_micros();
                            (elapsed as f64) / 1000f64
                        };
                        let fps = 1000f64 / frame_time_millis;
        
                        state.window().set_title(&format!("frame time: {} ms | {} FPS", frame_time_millis, fps));
                        frame_start = Instant::now();
                    }
                }
                
            },
            Event::MainEventsCleared => {
                // manually request a redraw after each loop of handling events.
                state.window().request_redraw();
            }
            _ => {},
        }
    }
    );
}