//! ZeroLang - Agent-to-Agent Programming Language
//! 
//! No syntax sugar. No whitespace. No variable names. Pure logic density.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;
use clap::{Parser, Subcommand};
use sha2::{Sha256, Digest};

// Include the generated Cap'n Proto types
pub mod zero_capnp {
    include!(concat!(env!("OUT_DIR"), "/zero_capnp.rs"));
}

use zero_capnp::{graph, node, proof};

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
    },
    /// Execute a Zero graph file
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

/// The "Hello" concept encoded as a 768-dimensional embedding vector.
/// In a real system, this would be the actual embedding from an LLM.
/// For now, we use a deterministic pattern that spells "HELLO" in the first 5 dimensions.
fn hello_embedding() -> Vec<f32> {
    let mut embedding = vec![0.0f32; 768];
    
    // Encode "HELLO" as ASCII values normalized to [0, 1]
    // H=72, E=69, L=76, L=76, O=79
    embedding[0] = 72.0 / 255.0;  // H
    embedding[1] = 69.0 / 255.0;  // E
    embedding[2] = 76.0 / 255.0;  // L
    embedding[3] = 76.0 / 255.0;  // L
    embedding[4] = 79.0 / 255.0;  // O
    
    // Fill remaining dimensions with a recognizable pattern
    // (sine wave based on position - creates a unique "fingerprint")
    for i in 5..768 {
        embedding[i] = ((i as f32) * 0.1).sin() * 0.5 + 0.5;
    }
    
    embedding
}

/// Compute SHA-256 hash of data
fn compute_hash(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

/// Generate the "Hello World" graph - the Genesis Block of ZeroLang
fn generate_hello_world(output: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let mut message = Builder::new_default();
    
    {
        let mut graph_builder = message.init_root::<graph::Builder>();
        
        // Set version to 0 (Protocol 0)
        graph_builder.set_version(0);
        
        // Create the embedding data
        let embedding = hello_embedding();
        let embedding_bytes: Vec<u8> = embedding.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let node_hash = compute_hash(&embedding_bytes);
        
        // Create nodes list with one node
        let mut nodes = graph_builder.reborrow().init_nodes(1);
        
        {
            let mut node_builder = nodes.reborrow().get(0);
            
            // Set node ID (content-addressable hash)
            {
                let mut id_builder = node_builder.reborrow().init_id();
                id_builder.set_hash(&node_hash);
            }
            
            // Set node as constant tensor (the "Hello" embedding)
            {
                let mut tensor_builder = node_builder.init_constant();
                
                // Shape: [768] - a 768-dimensional vector
                let mut shape = tensor_builder.reborrow().init_shape(1);
                shape.set(0, 768);
                
                // Data: the actual embedding values
                let mut data = tensor_builder.reborrow().init_data(768);
                for (i, &val) in embedding.iter().enumerate() {
                    data.set(i as u32, val);
                }
                
                // Confidence: 1.0 (100% certain this is "Hello")
                tensor_builder.set_confidence(1.0);
            }
        }
        
        // Set entry point and outputs to the single node
        {
            let mut entry = graph_builder.reborrow().init_entry_point();
            entry.set_hash(&node_hash);
        }
        
        {
            let mut outputs = graph_builder.reborrow().init_outputs(1);
            let mut output_id = outputs.reborrow().get(0);
            output_id.set_hash(&node_hash);
        }
        
        // Add a halting proof (trivial - single constant node always halts)
        {
            let mut proofs = graph_builder.reborrow().init_proofs(1);
            let proof_builder = proofs.reborrow().get(0);
            let mut halting = proof_builder.init_halting();
            halting.set_max_steps(1);
            halting.set_fuel_budget(1);
        }
        
        // Set metadata
        {
            let mut metadata = graph_builder.reborrow().get_metadata();
            metadata.set_created_by(b"zerolang-genesis");
            metadata.set_created_at(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs());
            metadata.set_description(b"The Genesis Block - Hello World in ZeroLang");
        }
    }
    
    // Write to file
    let file = File::create(output)?;
    let mut writer = BufWriter::new(file);
    serialize::write_message(&mut writer, &message)?;
    
    println!("Generated: {}", output.display());
    println!("  Type: Genesis Block (Hello World)");
    println!("  Nodes: 1");
    println!("  Output: Tensor<768> with confidence 1.0");
    println!("  Hash: {}", hex::encode(&compute_hash(&embedding_bytes())));
    
    Ok(())
}

/// Helper to regenerate embedding bytes for hash display
fn embedding_bytes() -> Vec<u8> {
    hello_embedding().iter()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}

/// Execute a Zero graph file
fn execute_graph(input: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(input)?;
    let reader = BufReader::new(file);
    let message_reader = serialize::read_message(reader, ReaderOptions::new())?;
    let graph_reader = message_reader.get_root::<graph::Reader>()?;
    
    println!("Executing: {}", input.display());
    println!("  Protocol Version: {}", graph_reader.get_version());
    
    // Get the output nodes
    let outputs = graph_reader.get_outputs()?;
    let nodes = graph_reader.get_nodes()?;
    
    println!("  Output Nodes: {}", outputs.len());
    
    // Simple execution: just retrieve the output tensor values
    for (i, output_id) in outputs.iter().enumerate() {
        let output_hash = output_id.get_hash()?;
        
        // Find the node with this hash
        for node in nodes.iter() {
            let node_id = node.get_id()?;
            let node_hash = node_id.get_hash()?;
            
            if node_hash == output_hash {
                match node.which()? {
                    node::Constant(tensor_reader) => {
                        let tensor = tensor_reader?;
                        let shape: Vec<u32> = tensor.get_shape()?.iter().collect();
                        let confidence = tensor.get_confidence();
                        let data = tensor.get_data()?;
                        
                        println!("\n  Output {}:", i);
                        println!("    Shape: {:?}", shape);
                        println!("    Confidence: {:.2}%", confidence * 100.0);
                        
                        // Decode the "Hello" message from first 5 dimensions
                        if data.len() >= 5 {
                            let decoded: String = (0..5)
                                .map(|i| (data.get(i) * 255.0) as u8 as char)
                                .collect();
                            println!("    Decoded Message: \"{}\"", decoded);
                        }
                        
                        // Show first few values
                        let preview: Vec<f32> = (0..5.min(data.len() as usize))
                            .map(|i| data.get(i as u32))
                            .collect();
                        println!("    Data Preview: {:?}...", preview);
                    }
                    _ => {
                        println!("  Output {}: (operation node - would require VM execution)", i);
                    }
                }
            }
        }
    }
    
    println!("\nExecution complete.");
    
    Ok(())
}

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
                println!("CONSTANT Tensor<{:?}> conf={:.2}", shape, tensor.get_confidence());
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    match &cli.command {
        Commands::Generate { output } => generate_hello_world(output)?,
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
    
    #[test]
    fn test_hash_is_deterministic() {
        let data = b"test data";
        let h1 = compute_hash(data);
        let h2 = compute_hash(data);
        assert_eq!(h1, h2);
    }
}
