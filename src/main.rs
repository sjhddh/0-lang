//! ZeroLang CLI - The first language for machines, by machines

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;
use clap::{Parser, Subcommand, ValueEnum};

use zerolang::zero_capnp::{graph, node, proof, Operation};
use zerolang::{compute_hash, RuntimeGraph, Tensor, VM};

/// ZeroLang CLI - The first language for machines, by machines
#[derive(Parser)]
#[command(name = "zero")]
#[command(about = "Agent-to-Agent programming language toolchain", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a Zero graph file (.0)
    Generate {
        /// Output file path
        output: PathBuf,
        /// Type of graph to generate
        #[arg(short, long, default_value = "hello")]
        graph_type: GraphType,
    },
    /// Execute a Zero graph file using the 0-VM
    Execute {
        /// Input file path
        input: PathBuf,
    },
    /// Inspect a Zero graph file (human-readable debug view)
    Inspect {
        /// Input file path
        input: PathBuf,
    },
}

#[derive(Clone, ValueEnum)]
enum GraphType {
    /// The Genesis Block - Hello World embedding
    Hello,
    /// Simple math: 1.0 + 2.0 = 3.0
    Math,
}

// ============================================================================
// GENERATORS
// ============================================================================

/// The "Hello" concept encoded as a 768-dimensional embedding vector.
fn hello_embedding() -> Vec<f32> {
    let mut embedding = vec![0.0f32; 768];
    
    // Encode "HELLO" as ASCII values normalized to [0, 1]
    embedding[0] = 72.0 / 255.0;  // H
    embedding[1] = 69.0 / 255.0;  // E
    embedding[2] = 76.0 / 255.0;  // L
    embedding[3] = 76.0 / 255.0;  // L
    embedding[4] = 79.0 / 255.0;  // O
    
    // Fill remaining dimensions with a recognizable pattern
    for i in 5..768 {
        embedding[i] = ((i as f32) * 0.1).sin() * 0.5 + 0.5;
    }
    
    embedding
}

/// Generate the "Hello World" graph - the Genesis Block of ZeroLang
fn generate_hello_world(output: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let mut message = Builder::new_default();
    
    {
        let mut graph_builder = message.init_root::<graph::Builder>();
        graph_builder.set_version(0);
        
        let embedding = hello_embedding();
        let embedding_bytes: Vec<u8> = embedding.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let node_hash = compute_hash(&embedding_bytes);
        
        let mut nodes = graph_builder.reborrow().init_nodes(1);
        
        {
            let mut node_builder = nodes.reborrow().get(0);
            {
                let mut id_builder = node_builder.reborrow().init_id();
                id_builder.set_hash(&node_hash);
            }
            {
                let mut tensor_builder = node_builder.init_constant();
                let mut shape = tensor_builder.reborrow().init_shape(1);
                shape.set(0, 768);
                let mut data = tensor_builder.reborrow().init_data(768);
                for (i, &val) in embedding.iter().enumerate() {
                    data.set(i as u32, val);
                }
                tensor_builder.set_confidence(1.0);
            }
        }
        
        {
            let mut entry = graph_builder.reborrow().init_entry_point();
            entry.set_hash(&node_hash);
        }
        {
            let mut outputs = graph_builder.reborrow().init_outputs(1);
            outputs.reborrow().get(0).set_hash(&node_hash);
        }
        {
            let mut proofs = graph_builder.reborrow().init_proofs(1);
            let proof_builder = proofs.reborrow().get(0);
            let mut halting = proof_builder.init_halting();
            halting.set_max_steps(1);
            halting.set_fuel_budget(1);
        }
        {
            let mut metadata = graph_builder.reborrow().get_metadata();
            metadata.set_created_by(b"zerolang-genesis");
            metadata.set_created_at(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs());
            metadata.set_description(b"The Genesis Block - Hello World in ZeroLang");
        }
    }
    
    let file = File::create(output)?;
    let mut writer = BufWriter::new(file);
    serialize::write_message(&mut writer, &message)?;
    
    println!("Generated: {}", output.display());
    println!("  Type: Genesis Block (Hello World)");
    println!("  Nodes: 1 (Constant)");
    println!("  Output: Tensor<768> with confidence 1.0");
    
    Ok(())
}

/// Generate a simple math graph: 1.0 + 2.0 = 3.0
fn generate_simple_math(output: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let mut message = Builder::new_default();
    
    {
        let mut graph_builder = message.init_root::<graph::Builder>();
        graph_builder.set_version(0);
        
        // Create tensors for hashing
        let tensor_a = Tensor::scalar(1.0, 1.0);
        let tensor_b = Tensor::scalar(2.0, 1.0);
        
        let hash_a = compute_hash(&tensor_a.to_bytes());
        let hash_b = compute_hash(&tensor_b.to_bytes());
        
        // Hash for the operation node (hash of operation + inputs)
        let mut op_content = vec![0u8]; // Operation::Add = 0
        op_content.extend(&hash_a);
        op_content.extend(&hash_b);
        let hash_result = compute_hash(&op_content);
        
        // Create 3 nodes: A=1.0, B=2.0, Result=A+B
        let mut nodes = graph_builder.reborrow().init_nodes(3);
        
        // Node A: Constant 1.0
        {
            let mut node_builder = nodes.reborrow().get(0);
            node_builder.reborrow().init_id().set_hash(&hash_a);
            let mut tensor_builder = node_builder.init_constant();
            tensor_builder.reborrow().init_shape(1).set(0, 1);
            tensor_builder.reborrow().init_data(1).set(0, 1.0);
            tensor_builder.set_confidence(1.0);
        }
        
        // Node B: Constant 2.0
        {
            let mut node_builder = nodes.reborrow().get(1);
            node_builder.reborrow().init_id().set_hash(&hash_b);
            let mut tensor_builder = node_builder.init_constant();
            tensor_builder.reborrow().init_shape(1).set(0, 1);
            tensor_builder.reborrow().init_data(1).set(0, 2.0);
            tensor_builder.set_confidence(1.0);
        }
        
        // Node Result: A + B
        {
            let mut node_builder = nodes.reborrow().get(2);
            node_builder.reborrow().init_id().set_hash(&hash_result);
            let mut op_builder = node_builder.init_operation();
            op_builder.set_op(Operation::Add);
            let mut inputs = op_builder.reborrow().init_inputs(2);
            inputs.reborrow().get(0).set_hash(&hash_a);
            inputs.reborrow().get(1).set_hash(&hash_b);
        }
        
        // Entry point is first constant
        graph_builder.reborrow().init_entry_point().set_hash(&hash_a);
        
        // Output is the result node
        {
            let mut outputs = graph_builder.reborrow().init_outputs(1);
            outputs.reborrow().get(0).set_hash(&hash_result);
        }
        
        // Halting proof
        {
            let mut proofs = graph_builder.reborrow().init_proofs(1);
            let proof_builder = proofs.reborrow().get(0);
            let mut halting = proof_builder.init_halting();
            halting.set_max_steps(3);
            halting.set_fuel_budget(1);
        }
        
        // Metadata
        {
            let mut metadata = graph_builder.reborrow().get_metadata();
            metadata.set_created_by(b"zerolang-math");
            metadata.set_created_at(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs());
            metadata.set_description(b"Simple Math: 1.0 + 2.0 = 3.0");
        }
    }
    
    let file = File::create(output)?;
    let mut writer = BufWriter::new(file);
    serialize::write_message(&mut writer, &message)?;
    
    println!("Generated: {}", output.display());
    println!("  Type: Simple Math (1.0 + 2.0)");
    println!("  Nodes: 3 (2 Constants + 1 Add Operation)");
    println!("  Expected Output: 3.0");
    
    Ok(())
}

// ============================================================================
// EXECUTOR
// ============================================================================

/// Execute a Zero graph file using the 0-VM
fn execute_graph(input: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading: {}", input.display());
    
    // Load the graph
    let graph = RuntimeGraph::load_from_file(input)?;
    
    println!("  Protocol Version: {}", graph.version);
    println!("  Nodes: {}", graph.node_count());
    
    // Create VM and execute
    let mut vm = VM::new();
    let outputs = vm.execute(&graph)?;
    
    println!("\n  Execution Statistics:");
    println!("    Operations: {}", vm.ops_executed());
    println!("    Remaining Fuel: {}", vm.remaining_fuel());
    
    println!("\n  Outputs ({}):", outputs.len());
    for (i, tensor) in outputs.iter().enumerate() {
        println!("    [{}] Shape: {:?}", i, tensor.shape);
        println!("        Confidence: {:.2}%", tensor.confidence * 100.0);
        
        if tensor.is_scalar() {
            println!("        Value: {}", tensor.as_scalar());
        } else if tensor.numel() <= 10 {
            println!("        Data: {:?}", tensor.data);
        } else {
            // For large tensors, show preview
            let preview: Vec<f32> = tensor.data.iter().take(5).copied().collect();
            println!("        Data: {:?}... ({} elements)", preview, tensor.numel());
            
            // Try to decode "HELLO" if it looks like an embedding
            if tensor.shape == vec![768] && tensor.data.len() >= 5 {
                let decoded: String = (0..5)
                    .map(|i| (tensor.data[i] * 255.0) as u8 as char)
                    .collect();
                println!("        Decoded: \"{}\"", decoded);
            }
        }
    }
    
    println!("\nExecution complete.");
    Ok(())
}

// ============================================================================
// INSPECTOR
// ============================================================================

/// Inspect a Zero graph file (detailed debug view)
fn inspect_graph(input: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(input)?;
    let reader = BufReader::new(file);
    let message_reader = serialize::read_message(reader, ReaderOptions::new())?;
    let graph_reader = message_reader.get_root::<graph::Reader>()?;
    
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    ZERO GRAPH INSPECTOR                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("File: {}", input.display());
    println!("Protocol Version: {}", graph_reader.get_version());
    println!();
    
    // Metadata
    let metadata = graph_reader.get_metadata();
    println!("┌─ METADATA ─────────────────────────────────────────────────────");
    if let Ok(created_by) = metadata.get_created_by() {
        println!("│ Created By: {}", String::from_utf8_lossy(created_by));
    }
    println!("│ Created At: {}", metadata.get_created_at());
    if let Ok(desc) = metadata.get_description() {
        println!("│ Description: {}", String::from_utf8_lossy(desc));
    }
    println!("└────────────────────────────────────────────────────────────────");
    println!();
    
    // Nodes
    let nodes = graph_reader.get_nodes()?;
    println!("┌─ NODES ({}) ──────────────────────────────────────────────────", nodes.len());
    for (i, node) in nodes.iter().enumerate() {
        let node_id = node.get_id()?;
        let hash = hex::encode(node_id.get_hash()?);
        
        print!("│ [{}] ", i);
        print!("Hash: {}... ", &hash[..16]);
        
        match node.which()? {
            node::Constant(tensor_reader) => {
                let tensor = tensor_reader?;
                let shape: Vec<u32> = tensor.get_shape()?.iter().collect();
                let data = tensor.get_data()?;
                if shape == vec![1] && data.len() == 1 {
                    println!("CONSTANT Scalar({}) conf={:.2}", data.get(0), tensor.get_confidence());
                } else {
                    println!("CONSTANT Tensor<{:?}> conf={:.2}", shape, tensor.get_confidence());
                }
            }
            node::Operation(op) => {
                println!("OPERATION {:?} inputs={}", op.get_op()?, op.get_inputs()?.len());
            }
            node::External(ext) => {
                println!("EXTERNAL uri={:?}", ext.get_uri()?);
            }
            node::Branch(br) => {
                println!("BRANCH threshold={:.2}", br.get_threshold());
            }
        }
    }
    println!("└────────────────────────────────────────────────────────────────");
    println!();
    
    // Proofs
    let proofs = graph_reader.get_proofs()?;
    println!("┌─ PROOFS ({}) ──────────────────────────────────────────────────", proofs.len());
    for (i, proof) in proofs.iter().enumerate() {
        print!("│ [{}] ", i);
        match proof.which()? {
            proof::Halting(h) => {
                println!("HALTING max_steps={} fuel={}", h.get_max_steps(), h.get_fuel_budget());
            }
            proof::ShapeValid(_) => println!("SHAPE_VALID"),
            proof::Signature(_) => println!("SIGNATURE"),
            proof::None(()) => println!("NONE (unsafe)"),
        }
    }
    println!("└────────────────────────────────────────────────────────────────");
    println!();
    
    // Entry point
    let entry = graph_reader.get_entry_point()?;
    println!("Entry Point: {}...", &hex::encode(entry.get_hash()?)[..16]);
    
    // Outputs
    let outputs = graph_reader.get_outputs()?;
    println!("Outputs: {}", outputs.len());
    
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    
    Ok(())
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    match &cli.command {
        Commands::Generate { output, graph_type } => {
            match graph_type {
                GraphType::Hello => generate_hello_world(output)?,
                GraphType::Math => generate_simple_math(output)?,
            }
        }
        Commands::Execute { input } => execute_graph(input)?,
        Commands::Inspect { input } => inspect_graph(input)?,
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hello_embedding_is_deterministic() {
        let e1 = hello_embedding();
        let e2 = hello_embedding();
        assert_eq!(e1, e2);
    }
    
    #[test]
    fn test_hello_embedding_decodes_to_hello() {
        let embedding = hello_embedding();
        let decoded: String = (0..5)
            .map(|i| (embedding[i] * 255.0) as u8 as char)
            .collect();
        assert_eq!(decoded, "HELLO");
    }
}
