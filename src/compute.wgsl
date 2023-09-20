@group(0)
@binding(0)
var<storage, read_write> instance_matrices: array<mat4x4<f32>>;

@group(1) @binding(0)
var vector_field_texture: texture_3d<f32>;
@group(1) @binding(1)
var vector_field_sampler: sampler;

fn trilinear_sample(position: vec3f, texture: texture_3d<f32>) -> vec4<f32> {
    let p000: vec3<i32> = vec3<i32>(floor(position));
    let s000 = textureLoad(texture, p000, 0);
    let s001 = textureLoad(texture, p000 + vec3<i32>(0, 0, 1), 0);
    let s010 = textureLoad(texture, p000 + vec3<i32>(0, 1, 0), 0);
    let s011 = textureLoad(texture, p000 + vec3<i32>(0, 1, 1), 0);
    let s100 = textureLoad(texture, p000 + vec3<i32>(1, 0, 0), 0);
    let s101 = textureLoad(texture, p000 + vec3<i32>(1, 0, 1), 0);
    let s110 = textureLoad(texture, p000 + vec3<i32>(1, 1, 0), 0);
    let s111 = textureLoad(texture, p000 + vec3<i32>(1, 1, 1), 0);
    let t = position - vec3<f32>(p000);
    let x00 = mix(s000, s100, t.x);
    let x01 = mix(s001, s101, t.x);
    let x10 = mix(s010, s110, t.x);
    let x11 = mix(s011, s111, t.x);
    let y0 = mix(x00, x10, t.y);
    let y1 = mix(x01, x11, t.y);
    let z = mix(y0, y1, t.z);
    return z;
}

fn rk4_eval(pos: vec4f, vector_field: texture_3d<f32>, step_size: f32) -> vec4f {
    let k1 = trilinear_sample(pos.xyz, vector_field);
    let k2 = trilinear_sample((pos + k1 * step_size * 0.5).xyz, vector_field);
    let k3 = trilinear_sample((pos + k2 * step_size * 0.5).xyz, vector_field);
    let k4 = trilinear_sample((pos + k3 * step_size).xyz, vector_field);
    let step = step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
    return step;
}

fn create_basis(origin: vec4f, forward: vec3f, up: vec3f) -> mat4x4<f32> {
    let right = cross(forward, up);
    let up_basis = cross(right, forward);
    return mat4x4<f32>(vec4f(right, 0.0), vec4f(up_basis, 0.0), vec4f(forward, 0.0), origin);
}

const NUM_STEPS = 100;
const STEP_SIZE = 0.0005;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var pos = instance_matrices[global_id.x][3];
    var step: vec4f;
    for(var i: i32 = 0; i < NUM_STEPS; i++) {
        step = rk4_eval(pos * 50.0, vector_field_texture, STEP_SIZE);
        pos += step;
    }
    let basis = create_basis(pos, normalize(step).xyz, vec3f(0.0, 1.0, 0.0));
    instance_matrices[global_id.x] = basis;
}