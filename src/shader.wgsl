// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
};

const particle_radius = 0.01;

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    var base = camera.view_proj * model_matrix * vec4<f32>(0.0, 0.0, 0.0, 1.0);

    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = base + vec4<f32>(model.position * particle_radius * 1.5, 0.0);

    return out;
}

//Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_3d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.tex_coords * 2.0 - vec2f(1.0);
    let dist = length(uv);
    let strength = 1.0 - (dist - 1.0) / dpdx(uv).x;
    if strength < 0.9 { discard; }
    return vec4f(strength, strength, strength, 1.0);
}