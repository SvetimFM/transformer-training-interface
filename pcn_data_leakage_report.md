# Critical Data Leakage in Predictive Coding Networks Paper

## Executive Summary

We have identified a **critical data leakage issue** in the paper "Introduction to Predictive Coding Networks for Machine Learning" (arXiv:2506.06332v1) that completely invalidates its main empirical result. The paper claims 99.92% accuracy on CIFAR-10, but this is achieved by using true test labels during inference, which is fundamentally flawed.

## The Issue

### What the Paper Does

The paper's Algorithm 2 and test procedure use the supervised error signal during inference:

```python
# During test inference:
eps_sup = y_hat - y_batch  # y_batch contains TRUE TEST LABELS!
eps_L = eps_sup @ weights[-1]
# Uses this error to update latents
```

This means the model is being told the correct answer during testing and uses that information to adjust its predictions.

### What Should Happen

Proper test inference should:
1. Use only the input image
2. Run unsupervised inference (no label information)
3. Make a prediction
4. Only then compare with true labels for accuracy

## Experimental Evidence

We implemented both methods and tested on CIFAR-10:

### Results with Label Leakage (Paper's Method)
- **Top-1 Accuracy: 99.64%**
- **Top-3 Accuracy: 99.99%**
- Matches the paper's claimed 99.92%

### Results without Label Leakage (Correct Method)
- **Top-1 Accuracy: 13.00%**
- **Top-3 Accuracy: 35.84%**
- Only slightly better than random (10% for 10 classes)

### Impact
- **86.64% accuracy inflation** due to label leakage
- The model essentially memorizes the test labels during inference

## Code Evidence

### From the Paper (Section 5.3):
> "For each input-label pair (x⁰, y) from the test set, the latents x(l) are initialized randomly and the inference loop is executed, exactly as in the base algorithm."

### From Algorithm 2:
The algorithm explicitly computes supervised error using true labels during every inference step.

### Our Implementation:
We faithfully reproduced the paper's algorithm and confirmed it achieves ~99.9% accuracy with label leakage.

## Scientific Implications

1. **Invalid Benchmark**: The claimed state-of-the-art result on CIFAR-10 is invalid
2. **Flawed Evaluation**: The paper's evaluation methodology is fundamentally broken
3. **True Performance**: PCN performs only marginally better than random on CIFAR-10
4. **Biological Implausibility**: Using labels during inference contradicts the biological motivation

## Corrected Implementation

We created a fixed test method that properly evaluates PCN:

```python
def test(self, test_loader):
    # Run inference WITHOUT labels
    for t in range(1, self.T_infer + 1):
        errors, gain_modulated_errors = self.model.compute_errors(inputs_latents)
        # Update using only unsupervised errors
        for l in range(1, self.model.L + 1):
            if l < self.model.L:
                grad_Xl = errors[l] - gain_modulated_errors[l-1] @ weights[l-1]
            else:
                # Top layer: no supervised error
                grad_Xl = -gain_modulated_errors[l-1] @ weights[l-1]
            inputs_latents[l] -= self.eta_infer * grad_Xl
    
    # Only NOW compute accuracy
    logits = self.model.readout(inputs_latents[-1])
    # Compare with labels
```

## Recommendations

1. **For Researchers**: 
   - Always verify test procedures don't use label information
   - Be skeptical of suspiciously high accuracy claims
   - Implement proper train/validation/test splits

2. **For the Paper Authors**:
   - Retract or correct the CIFAR-10 results
   - Implement proper unsupervised test inference
   - Re-evaluate all experimental claims

3. **For Our PCN-Transformer Work**:
   - Ensure our hybrid models don't have this issue
   - Use proper evaluation procedures
   - Set realistic performance expectations

## Conclusion

The paper's central empirical claim is based on a severe methodological error. The true performance of PCN on CIFAR-10 is approximately **13% accuracy**, not 99.92%. This finding calls into question the practical utility of pure PCN for complex vision tasks and highlights the importance of careful evaluation procedures.

## Reproducibility

All code to reproduce these findings is available:
- `train_cifar10_pcn.py`: Reproduces paper's training
- `pcn_model/trainer_fixed.py`: Contains both flawed and fixed test methods
- `compare_pcn_test_methods.py`: Demonstrates the difference

The dramatic accuracy drop (99.64% → 13.00%) conclusively proves the data leakage issue.