fn main() {
    println!("cargo:rerun-if-changed=schema/zero.capnp");
    
    capnpc::CompilerCommand::new()
        .src_prefix("schema")
        .file("schema/zero.capnp")
        .run()
        .expect("compiling schema");
}
