//! VM - The ZeroLang Virtual Machine
//!
//! Executes Zero graphs by topologically sorting nodes and
//! evaluating operations on tensors.

use std::collections::HashMap;

use crate::graph::{GraphError, NodeHash, Op, RuntimeGraph, RuntimeNode};
use crate::Tensor;

/// The ZeroLang Virtual Machine
pub struct VM {
    /// Memory: stores computed tensor values by node hash
    memory: HashMap<NodeHash, Tensor>,
    /// Execution fuel (max operations before halt)
    fuel: u64,
    /// Operations executed
    ops_executed: u64,
}

impl VM {
    /// Create a new VM with default fuel
    pub fn new() -> Self {
        Self {
            memory: HashMap::new(),
            fuel: 1_000_000, // Default: 1 million operations
            ops_executed: 0,
        }
    }

    /// Create a VM with specified fuel budget
    pub fn with_fuel(fuel: u64) -> Self {
        Self {
            memory: HashMap::new(),
            fuel,
            ops_executed: 0,
        }
    }

    /// Execute a graph and return the output tensors
    pub fn execute(&mut self, graph: &RuntimeGraph) -> Result<Vec<Tensor>, VMError> {
        // Clear memory from previous executions
        self.memory.clear();
        self.ops_executed = 0;

        // Get topological order
        let order = graph
            .topological_sort()
            .map_err(|e| VMError::GraphError(e))?;

        // Execute nodes in order
        for hash in order {
            self.execute_node(&hash, graph)?;
        }

        // Collect outputs
        let mut outputs = Vec::new();
        for output_hash in &graph.outputs {
            let tensor = self
                .memory
                .get(output_hash)
                .ok_or_else(|| VMError::NodeNotComputed(hex::encode(output_hash)))?
                .clone();
            outputs.push(tensor);
        }

        Ok(outputs)
    }

    /// Execute a single node
    fn execute_node(&mut self, hash: &NodeHash, graph: &RuntimeGraph) -> Result<(), VMError> {
        // Check fuel
        if self.ops_executed >= self.fuel {
            return Err(VMError::OutOfFuel);
        }

        let node = graph
            .nodes
            .get(hash)
            .ok_or_else(|| VMError::NodeNotFound(hex::encode(hash)))?;

        let result = match node {
            RuntimeNode::Constant(tensor) => {
                // Constants just get stored directly
                tensor.clone()
            }
            RuntimeNode::Operation { op, inputs } => {
                self.ops_executed += 1;
                self.execute_operation(*op, inputs)?
            }
            RuntimeNode::Branch {
                condition,
                threshold,
                true_branch,
                false_branch,
            } => {
                self.ops_executed += 1;
                self.execute_branch(condition, *threshold, true_branch, false_branch)?
            }
        };

        self.memory.insert(hash.clone(), result);
        Ok(())
    }

    /// Execute an operation on input tensors
    fn execute_operation(&self, op: Op, inputs: &[NodeHash]) -> Result<Tensor, VMError> {
        // Fetch input tensors
        let input_tensors: Result<Vec<&Tensor>, VMError> = inputs
            .iter()
            .map(|h| {
                self.memory
                    .get(h)
                    .ok_or_else(|| VMError::NodeNotComputed(hex::encode(h)))
            })
            .collect();
        let input_tensors = input_tensors?;

        match op {
            // Binary operations
            Op::Add => {
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                input_tensors[0]
                    .checked_add(input_tensors[1])
                    .map_err(|e| VMError::TensorError(e.to_string()))
            }
            Op::Sub => {
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                input_tensors[0]
                    .checked_sub(input_tensors[1])
                    .map_err(|e| VMError::TensorError(e.to_string()))
            }
            Op::Mul => {
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                input_tensors[0]
                    .checked_mul(input_tensors[1])
                    .map_err(|e| VMError::TensorError(e.to_string()))
            }
            Op::Div => {
                if input_tensors.len() != 2 {
                    return Err(VMError::WrongInputCount {
                        expected: 2,
                        got: input_tensors.len(),
                    });
                }
                input_tensors[0]
                    .checked_div(input_tensors[1])
                    .map_err(|e| VMError::TensorError(e.to_string()))
            }

            // Unary operations
            Op::Relu => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                Ok(input_tensors[0].relu())
            }
            Op::Sigmoid => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                Ok(input_tensors[0].sigmoid())
            }
            Op::Tanh => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                Ok(input_tensors[0].tanh())
            }

            // Reductions
            Op::Sum => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                Ok(input_tensors[0].sum())
            }
            Op::Mean => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                Ok(input_tensors[0].mean())
            }

            // Special
            Op::Identity => {
                if input_tensors.len() != 1 {
                    return Err(VMError::WrongInputCount {
                        expected: 1,
                        got: input_tensors.len(),
                    });
                }
                Ok(input_tensors[0].clone())
            }
        }
    }

    /// Execute a branch node
    fn execute_branch(
        &self,
        condition: &NodeHash,
        threshold: f32,
        true_branch: &NodeHash,
        false_branch: &NodeHash,
    ) -> Result<Tensor, VMError> {
        // Get condition tensor (must be scalar)
        let cond_tensor = self
            .memory
            .get(condition)
            .ok_or_else(|| VMError::NodeNotComputed(hex::encode(condition)))?;

        if !cond_tensor.is_scalar() {
            return Err(VMError::BranchConditionNotScalar);
        }

        let cond_value = cond_tensor.as_scalar();

        // Choose branch based on threshold
        let branch_hash = if cond_value >= threshold {
            true_branch
        } else {
            false_branch
        };

        // Return the tensor from the chosen branch
        self.memory
            .get(branch_hash)
            .ok_or_else(|| VMError::NodeNotComputed(hex::encode(branch_hash)))
            .map(|t| t.clone())
    }

    /// Get the number of operations executed
    pub fn ops_executed(&self) -> u64 {
        self.ops_executed
    }

    /// Get remaining fuel
    pub fn remaining_fuel(&self) -> u64 {
        self.fuel.saturating_sub(self.ops_executed)
    }
}

impl Default for VM {
    fn default() -> Self {
        Self::new()
    }
}

/// VM execution errors
#[derive(Debug)]
pub enum VMError {
    GraphError(GraphError),
    NodeNotFound(String),
    NodeNotComputed(String),
    OutOfFuel,
    WrongInputCount { expected: usize, got: usize },
    TensorError(String),
    BranchConditionNotScalar,
}

impl std::fmt::Display for VMError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VMError::GraphError(e) => write!(f, "Graph error: {}", e),
            VMError::NodeNotFound(h) => write!(f, "Node not found: {}", h),
            VMError::NodeNotComputed(h) => write!(f, "Node not computed: {}", h),
            VMError::OutOfFuel => write!(f, "Out of fuel (exceeded max operations)"),
            VMError::WrongInputCount { expected, got } => {
                write!(f, "Wrong input count: expected {}, got {}", expected, got)
            }
            VMError::TensorError(e) => write!(f, "Tensor error: {}", e),
            VMError::BranchConditionNotScalar => write!(f, "Branch condition must be a scalar"),
        }
    }
}

impl std::error::Error for VMError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::RuntimeNode;

    #[test]
    fn test_vm_constant_execution() {
        let mut graph = RuntimeGraph {
            nodes: HashMap::new(),
            entry_point: vec![1, 2, 3],
            outputs: vec![vec![1, 2, 3]],
            version: 0,
        };

        graph.nodes.insert(
            vec![1, 2, 3],
            RuntimeNode::Constant(Tensor::scalar(42.0, 1.0)),
        );

        let mut vm = VM::new();
        let outputs = vm.execute(&graph).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_scalar(), 42.0);
    }

    #[test]
    fn test_vm_addition() {
        let mut graph = RuntimeGraph {
            nodes: HashMap::new(),
            entry_point: vec![1],
            outputs: vec![vec![3]],
            version: 0,
        };

        // Node 1: Constant 10.0
        graph
            .nodes
            .insert(vec![1], RuntimeNode::Constant(Tensor::scalar(10.0, 1.0)));

        // Node 2: Constant 20.0
        graph
            .nodes
            .insert(vec![2], RuntimeNode::Constant(Tensor::scalar(20.0, 0.9)));

        // Node 3: Add Node1 + Node2
        graph.nodes.insert(
            vec![3],
            RuntimeNode::Operation {
                op: Op::Add,
                inputs: vec![vec![1], vec![2]],
            },
        );

        let mut vm = VM::new();
        let outputs = vm.execute(&graph).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_scalar(), 30.0);
        assert_eq!(outputs[0].confidence, 0.9); // min(1.0, 0.9)
    }
}
