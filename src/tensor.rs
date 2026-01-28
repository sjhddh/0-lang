//! Tensor - The fundamental data type in ZeroLang
//!
//! Replaces int, float, bool, string with probabilistic vectors.
//! Every value carries a confidence score.

use std::ops::{Add, Sub, Mul, Div};

/// The fundamental data type in ZeroLang.
/// 
/// In traditional languages, you have `int`, `float`, `bool`, `string`.
/// In ZeroLang, everything is a Tensor with a confidence score.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Dimensions of the tensor, e.g., [768] for embedding, [1] for scalar
    pub shape: Vec<u32>,
    /// Flattened tensor data
    pub data: Vec<f32>,
    /// Meta-confidence in this tensor's validity [0.0, 1.0]
    /// This is the "Schrödinger" probability - how certain are we about this value?
    pub confidence: f32,
}

impl Tensor {
    /// Create a new tensor with given shape, data, and confidence
    pub fn new(shape: Vec<u32>, data: Vec<f32>, confidence: f32) -> Self {
        Self { shape, data, confidence }
    }

    /// Create a scalar tensor (shape [1]) with a single value
    pub fn scalar(value: f32, confidence: f32) -> Self {
        Self {
            shape: vec![1],
            data: vec![value],
            confidence,
        }
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<u32>, confidence: f32) -> Self {
        let size: usize = shape.iter().map(|&d| d as usize).product();
        Self {
            shape,
            data: vec![0.0; size],
            confidence,
        }
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: Vec<u32>, confidence: f32) -> Self {
        let size: usize = shape.iter().map(|&d| d as usize).product();
        Self {
            shape,
            data: vec![1.0; size],
            confidence,
        }
    }

    /// Create a tensor from a vector (1D)
    pub fn from_vec(data: Vec<f32>, confidence: f32) -> Self {
        let len = data.len() as u32;
        Self {
            shape: vec![len],
            data,
            confidence,
        }
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// Check if this is a scalar (shape [1])
    pub fn is_scalar(&self) -> bool {
        self.shape == vec![1]
    }

    /// Get scalar value (panics if not a scalar)
    pub fn as_scalar(&self) -> f32 {
        assert!(self.is_scalar(), "Tensor is not a scalar");
        self.data[0]
    }

    /// Compute confidence propagation for binary operations
    /// Using min() to be conservative - result is only as confident as the least confident input
    fn propagate_confidence(a: f32, b: f32) -> f32 {
        a.min(b)
    }

    /// Element-wise operation helper
    fn elementwise_op<F>(&self, other: &Tensor, op: F) -> Result<Tensor, TensorError>
    where
        F: Fn(f32, f32) -> f32,
    {
        // Check shape compatibility
        if self.shape != other.shape {
            // Allow broadcasting for scalars
            if self.is_scalar() {
                let scalar = self.data[0];
                let data: Vec<f32> = other.data.iter().map(|&x| op(scalar, x)).collect();
                return Ok(Tensor {
                    shape: other.shape.clone(),
                    data,
                    confidence: Self::propagate_confidence(self.confidence, other.confidence),
                });
            } else if other.is_scalar() {
                let scalar = other.data[0];
                let data: Vec<f32> = self.data.iter().map(|&x| op(x, scalar)).collect();
                return Ok(Tensor {
                    shape: self.shape.clone(),
                    data,
                    confidence: Self::propagate_confidence(self.confidence, other.confidence),
                });
            }
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }

        let data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| op(a, b))
            .collect();

        Ok(Tensor {
            shape: self.shape.clone(),
            data,
            confidence: Self::propagate_confidence(self.confidence, other.confidence),
        })
    }

    /// Checked addition
    pub fn checked_add(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.elementwise_op(other, |a, b| a + b)
    }

    /// Checked subtraction
    pub fn checked_sub(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.elementwise_op(other, |a, b| a - b)
    }

    /// Checked multiplication
    pub fn checked_mul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.elementwise_op(other, |a, b| a * b)
    }

    /// Checked division
    pub fn checked_div(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.elementwise_op(other, |a, b| if b != 0.0 { a / b } else { f32::NAN })
    }

    /// Sum all elements, returning a scalar
    pub fn sum(&self) -> Tensor {
        let total: f32 = self.data.iter().sum();
        Tensor::scalar(total, self.confidence)
    }

    /// Mean of all elements, returning a scalar
    pub fn mean(&self) -> Tensor {
        let total: f32 = self.data.iter().sum();
        let count = self.data.len() as f32;
        Tensor::scalar(total / count, self.confidence)
    }

    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.max(0.0)).collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            confidence: self.confidence,
        }
    }

    /// Sigmoid activation: 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            confidence: self.confidence,
        }
    }

    /// Tanh activation
    pub fn tanh(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.tanh()).collect();
        Tensor {
            shape: self.shape.clone(),
            data,
            confidence: self.confidence,
        }
    }

    /// Serialize tensor data to bytes (for hashing)
    pub fn to_bytes(&self) -> Vec<u8> {
        self.data.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect()
    }
}

/// Implement Add trait for convenient syntax: tensor_a + tensor_b
impl Add for Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Tensor {
        Tensor::checked_add(&self, &other).expect("Shape mismatch in Add")
    }
}

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        Tensor::checked_add(self, other).expect("Shape mismatch in Add")
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Tensor {
        Tensor::checked_sub(&self, &other).expect("Shape mismatch in Sub")
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Tensor {
        Tensor::checked_mul(&self, &other).expect("Shape mismatch in Mul")
    }
}

impl Div for Tensor {
    type Output = Tensor;

    fn div(self, other: Tensor) -> Tensor {
        Tensor::checked_div(&self, &other).expect("Shape mismatch in Div")
    }
}

/// Tensor operation errors
#[derive(Debug, Clone)]
pub enum TensorError {
    ShapeMismatch { expected: Vec<u32>, got: Vec<u32> },
    InvalidShape { reason: String },
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            TensorError::InvalidShape { reason } => {
                write!(f, "Invalid shape: {}", reason)
            }
        }
    }
}

impl std::error::Error for TensorError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_creation() {
        let t = Tensor::scalar(3.14, 1.0);
        assert_eq!(t.shape, vec![1]);
        assert_eq!(t.data, vec![3.14]);
        assert!(t.is_scalar());
    }

    #[test]
    fn test_addition() {
        let a = Tensor::scalar(1.0, 1.0);
        let b = Tensor::scalar(2.0, 0.9);
        let c = a + b;
        assert_eq!(c.as_scalar(), 3.0);
        assert_eq!(c.confidence, 0.9); // min(1.0, 0.9)
    }

    #[test]
    fn test_confidence_propagation() {
        let a = Tensor::scalar(5.0, 0.8);
        let b = Tensor::scalar(3.0, 0.6);
        let c = a + b;
        assert_eq!(c.confidence, 0.6); // min(0.8, 0.6)
    }

    #[test]
    fn test_vector_addition() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], 1.0);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], 1.0);
        let c = a + b;
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_scalar_broadcast() {
        let scalar = Tensor::scalar(2.0, 1.0);
        let vec = Tensor::from_vec(vec![1.0, 2.0, 3.0], 1.0);
        let result = scalar.checked_mul(&vec).unwrap();
        assert_eq!(result.data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_relu() {
        let t = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], 1.0);
        let r = t.relu();
        assert_eq!(r.data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sum() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 0.95);
        let s = t.sum();
        assert_eq!(s.as_scalar(), 10.0);
        assert_eq!(s.confidence, 0.95);
    }
}
