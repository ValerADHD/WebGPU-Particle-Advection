# WebGPU-Particle-Advection
test implementation of particle advection via WebGPU compute shaders

## Building

### Native
To build for the native version, make sure you have [cargo](https://www.rust-lang.org/tools/install) installed and simply run `cargo run` in the base project folder.

### Web
This project uses [wasm-pack](https://github.com/rustwasm/wasm-pack) to automate the wasm-bindgen workflow. Once installed, it allows you to run 

`wasm-pack build --target web`

to generate the WASM binary required to run the project.
Building to properly utilize WebGPU on the web requires additional flags. To ensure that the project uses WebGPU, the environment variable RUSTFLAGS must contain

`--cfg=web_sys_unstable_apis`

before building the project. 
`wasm-build.bat` will set this for you and build the project automatically. For this project specifically, once you have used wasm-pack to build your WASM binary you can host a server from the included `index.html` to view the output.
