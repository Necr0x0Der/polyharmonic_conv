# Convolutional Polyharmonic Cascade

## References and goal

This project is a convolutional extension of the "Polyharmonic Cascade" cycle by Yu. N. Bakhvalov:

1. Bakhvalov, Y. N. (2025). *Solving a Machine Learning Regression Problem Based on the Theory of Random Functions.* arXiv:2512.12731.  
   https://arxiv.org/abs/2512.12731

2. Bakhvalov, Y. N. (2025). *Polyharmonic Spline Packages: Composition, Efficient Procedures for Computation and Differentiation.* arXiv:2512.16718.  
   https://arxiv.org/abs/2512.16718

3. Bakhvalov, Y. N. (2025). *Polyharmonic Cascade.* arXiv:2512.17671.  
   https://arxiv.org/abs/2512.17671

4. Bakhvalov, Y. N. (2025). *Initialization of a Polyharmonic Cascade, Launch and Testing.* arXiv:2512.19524.  
   https://arxiv.org/abs/2512.19524

The goal of this document is to extend the ideas from these papers to a **convolutional setting**: turn the polyharmonic package into a local, translation-equivariant convolutional layer, build a small MNIST classifier from it, and test a linear-solver optimizer for the shared parameters.

This README is self-contained: it defines the kernel, constellation, package, convolutional layer, classifier, training procedures, results, and limitations. The references are given for context and attribution.

## Background: what the original papers propose

The original papers treat regression as inference over a random function. Under assumptions of translation, rotation, and scale invariance plus Gaussianity, the natural covariance is the polyharmonic kernel

\[
k(\tau)=\|\tau\|^2(\ln\|\tau\|-b)+c,
\]

which is a generalized polyharmonic / thin-plate spline. The regression solution is

\[
f(x)=\sum_i \lambda_i k(x_i-x),
\qquad
\lambda=(K+\sigma^2 I)^{-1}Y.
\]

This is essentially Gaussian Process Regression / Kernel Ridge Regression with a kernel derived from symmetry assumptions rather than chosen empirically.

A "package" is a family of such functions defined on a common set of key points, called a constellation. If the constellation is \(C\), then a package can compute many functions at once using a shared inverse matrix

\[
U=(K(C,C)+\sigma^2 I)^{-1}.
\]

A cascade stacks packages into a deep architecture. Training in the original papers is not SGD over the coefficients; instead one solves linear systems for the function values at the constellation points.

This repository keeps that idea but makes the package **local and convolutional**.

- a package is applied to image patches;
- the constellation is defined in patch space;
- the trainable values `Y` are shared across all spatial positions;
- this makes the layer translation-equivariant;
- the optimizer solves linear systems for changes in `Y` instead of applying gradient descent directly.

## Files

- `polyharmonic_conv.py` — `PolyConvLayer`, `PolyCascadeLayer`, `PolyConvClassifier`.
- `train_mnist_polyconv.py` — MNIST training with Adam (baseline).
- `generalized_solver.py` — MNIST training with a generalized Gauss-Newton-like linear solver.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes local data and Python caches.
- `README.md` — this file.

## Convolutional polyharmonic layer

For a patch \(x_s \in \mathbb{R}^{p^2 C_{\text{in}}}\), the layer computes

\[
y_s = K(x_s, C)\,U\,Y,
\]

where:

- \(C\) is a fixed constellation in patch space;
- \(K(x_s,C)\) is the polyharmonic kernel matrix;
- \(U=(K(C,C)+\sigma^2 I)^{-1}\);
- \(Y\) is a shared matrix of function values at the constellation points.

`Y` is the same for every spatial position, so the layer is translation-equivariant: when the image shifts, the patches shift, but `C` and `Y` do not.

The output is reshaped back into a feature map:

\[
[B, C_{\text{out}}, H_{\text{out}}, W_{\text{out}}].
\]

This is not a standard linear convolution. The kernel depends on the patch content, so the layer is nonlinear in the input and linear in `Y`. It is better described as a **convolutional RBF / polyharmonic layer** with weight sharing.

Implementation note: because the constellation is the hyperoctahedron plus its center, the squared distances to its points can be computed from \(\|x\|^2\) and the individual coordinates. The code uses this structure instead of a full patch–constellation `cdist`. It also computes \(UY\) once and then \(K(UY)\), which is \(O(NkC_{\text{out}})\) rather than the much larger \(O(Nk^2)\).

## Classifier

The default `PolyConvClassifier` uses **four** convolutional polyharmonic layers:

- layer 1: 1 → 16 channels, 3×3 patches, stride 1, same padding;
- layer 2: 16 → 24 channels, 3×3 patches, stride 1, same padding;
- layer 3: 24 → 32 channels, 3×3 patches, stride 1, same padding;
- layer 4: 32 → 32 channels, 3×3 patches, stride 1, same padding;
- global average pooling;
- two fully connected polyharmonic layers: 32 → 32 → 10.

With \(p=3\), stride \(1\), and `same` padding, the spatial feature-map size stays exactly \(28\times28\) in every convolutional layer. Thus an arbitrary number of convolutional layers can be added without changing the spatial size.

Total trainable parameters: about **42,290** for this default configuration.

The model uses `tanh` after each layer only as a range-normalization / stability device. It is not part of the original theoretical derivation.

## Configuration and CLI

Both training scripts accept architecture and training parameters.

### Common architecture parameters

| Parameter | Meaning | Default |
|---|---|---|
| `--layers` | number of convolutional polyharmonic layers | `4` |
| `--channels` | comma-separated channel widths, e.g. `16,24,32,32` | generated from `--layers` |
| `--patch-size` | patch / kernel size \(p\) | `3` |
| `--stride` | default patch stride \(s\) | `1` |
| `--strides` | optional comma-separated per-layer strides, e.g. `2,2,1` | uses `--stride` for all |
| `--padding` | `0` for valid, or `same` for \(p/2\) padding | `same` |
| `--head-dims` | comma-separated fully connected polyharmonic output dims | `32,10` |
| `--batch-size` | batch size | `128` |
| `--seed` | random seed | `0` |

If `--channels` is supplied, its length defines the number of convolutional layers. If `--layers` is also supplied, it must match `len(--channels)`. If `--channels` is omitted, the script uses `default_channels(--layers)`.

### Adam training parameters

| Parameter | Meaning | Default |
|---|---|---|
| `--epochs` | number of epochs | `8` |
| `--lr` | Adam learning rate | `1e-3` |
| `--device` | `cpu` or `cuda` | `cuda` |

### Generalized-solver parameters

| Parameter | Meaning | Default |
|---|---|---|
| `--epochs` | number of epochs | `8` |
| `--alpha` | regularization in the linear solve | `1e-2` |
| `--step` | step multiplier for \(\Delta Y\) | `1.0` |
| `--device` | `cpu` or `cuda` | `cuda` |

## Padding and layer shapes

The convolutional layer receives

\[
[B, C_{\text{in}}, H, W].
\]

For patch size \(p\), stride \(s\), and padding \(d\), the output spatial size is

\[
H_{\text{out}}
=
\left\lfloor \frac{H + 2d - p}{s} \right\rfloor + 1,
\qquad
W_{\text{out}}
=
\left\lfloor \frac{W + 2d - p}{s} \right\rfloor + 1.
\]

With `--padding 0` this is a valid convolution: the spatial size shrinks.

With `--padding same`, the code uses

\[
d = \left\lfloor \frac{p}{2} \right\rfloor.
\]

For an odd patch size \(p\) and stride \(s=1\), this gives

\[
H+2d-p+1 = H+\left(2\left\lfloor p/2\right\rfloor-p\right)+1 = H,
\]

so the spatial size is exactly preserved. For \(p=3\), \(s=1\), \(H=28\):

- valid padding: \(H_{\text{out}}=26\);
- same padding: \(d=1\), \(H_{\text{out}}=28\).

For even \(p\), symmetric `same` padding grows the output by one pixel at stride 1; the exact preservation statement is for odd patch sizes. The default \(p=3\) is odd, so the default architecture keeps the feature map at \(28\times28\) through any number of convolutional layers.

If `--stride` is greater than 1, or `--strides` is supplied, the spatial size generally shrinks and is no longer guaranteed to be constant.

Each subsequent convolutional layer has:

- input channels \(C_{\text{in}}\) equal to the previous layer's output channels;
- patch vector dimension \(n=p^2 C_{\text{in}}\);
- constellation size \(k=2n+1\);
- output channels \(C_{\text{out}}\);
- output shape \([B, C_{\text{out}}, H_{\text{out}}, W_{\text{out}}]\).

After the last convolutional layer, global average pooling produces a vector \([B, C_{\text{last}}]\). The fully connected polyharmonic head layers then map

\[
C_{\text{last}} \to \text{head dims}.
\]

So the "kernel size" is supported as `--patch-size`. A larger patch means a larger receptive field, but also a larger patch vector \(p^2 C_{\text{in}}\), a larger constellation \(2p^2 C_{\text{in}}+1\), and more computation.

## Training modes

### 1. Adam baseline

`train_mnist_polyconv.py` trains the whole model with standard backpropagation and Adam.

Measured examples on MNIST:

| Configuration | Epochs | Test accuracy |
|---|---|---|
| default 4 layers, stride 1, same padding | 1 | about **31.7%** (early, not converged) |
| default 4 layers, stride 1, same padding | longer runs | doesn't converge |
| 3 layers, stride 2, valid padding | 8 | about **96.5%** |
| 3 layers, `--strides 2,2,1`, valid padding | 8 | about **96.9%** |
| 3 layers, `--strides 2,2,1`, valid padding, lr \(2\cdot10^{-3}\) | 15 | about **97.8%** |

This is a proof-of-concept, not SOTA. A normal CNN on MNIST is around 99%, and the original fully connected cascade in the paper reports about 98.3%.

### 2. Generalized linear-solver optimizer

`generalized_solver.py` implements a **Gauss-Newton-like linear-solver update** for shared `Y`.

For each layer \(l\), let

\[
A_l = H_l Y_l,
\]

where \(H_l\) is the linear map from `Y` to the layer's pre-activation. Given the gradient of the loss with respect to \(A_l\),

\[
g_l = \frac{\partial L}{\partial A_l},
\]

the update for each output channel \(t\) is obtained by solving

\[
\left(H_l^T H_l + \alpha I\right)\Delta Y_l[:,t]
=
-H_l^T g_l[:,t].
\]

Then

\[
Y_l \leftarrow Y_l + \text{step}\cdot \Delta Y_l.
\]

For convolutional layers, \(H_l\) is assembled from all patch positions. For fully connected layers, it is assembled from all batch examples. The update is a linear solve, not SGD/Adam.

### What the generalized solver actually does, step by step

1. Run a forward pass through all polyharmonic layers and store each layer's pre-activation
   \[
   A_l = H_l Y_l.
   \]
2. Compute the loss and, using PyTorch autograd, obtain
   \[
   g_l = \frac{\partial L}{\partial A_l}.
   \]
   This is the only place where gradients are used.
3. For every layer \(l\), assemble \(H_l\):
   - for a convolutional layer, one row per patch position;
   - for a fully connected layer, one row per batch example.
4. For each output channel \(t\), solve
   \[
   \left(H_l^T H_l + \alpha I\right)\Delta Y_l[:,t]
   =
   -H_l^T g_l[:,t].
   \]
5. Update
   \[
   Y_l \leftarrow Y_l + \text{step}\cdot \Delta Y_l.
   \]

So the optimizer does **not** call `loss.backward()` followed by `optimizer.step()`. It computes gradients for the linearization, then solves linear systems for the parameter changes.

### What is generalized and what is not

The generalized part is:

- `Y` is shared across all spatial positions in a convolutional layer;
- the linear solve is performed per layer and per output channel;
- the same idea of "optimize function values at constellation points" is kept.

The part that is **not** the exact paper algorithm is:

- the paper uses a synchronized Lagrange-multiplier system with a vector \(B\);
- here the update is a per-layer Gauss-Newton-like linear solve;
- there is no exact reproduction of the paper's
  \[
  \Omega_\tau=(H_\tau H_\tau^T)\circ(G_\tau G_\tau^T)
  \]
  and
  \[
  \Delta Y_\tau=H_\tau^T(G_\tau\circ(B\mathbf 1)).
  \]
  These formulas would need to be adapted to shared convolutional `Y` and to the multi-output setting.

On MNIST, for the earlier 3-layer stride-2 valid-padding configuration, 8 epochs with \(\alpha=10^{-2}\) and step \(=1.0\):

- final test accuracy: about **97.6%**.

The solver is more sensitive than Adam and is slower for the large 4-layer stride-1 constellation sizes. For the default architecture it should be treated as an experimental optimizer that needs tuning of `alpha` and `step`; it is not a proof of the exact solver from the paper.

### Reproducing the 97% results

The 97%+ numbers were obtained **before** the default was changed to 4 layers / stride 1 / `same` padding. They belong to the older 3-layer stride-2 configuration.

Generalized solver, 8 epochs:

```bash
python generalized_solver.py \
  --layers 3 \
  --channels 16,24,32 \
  --patch-size 3 \
  --stride 2 \
  --padding 0 \
  --head-dims 32,10 \
  --epochs 8 \
  --alpha 1e-2 \
  --step 1.0
```

This configuration reached about **97.6%** test accuracy.

UPDATE. The following configuration

```bash
python generalized_solver.py \
  --layers 4 \
  --channels 12,16,24,32 \
  --patch-size 3 \
  --stride 2 \
  --padding 1 \
  --head-dims 32,10 \
  --epochs 10 \
  --alpha 1e-2 \
  --step 1.0
```

reaches **98.65%** test accuracy.

With Adam, 15 epochs:

```bash
python train_mnist_polyconv.py \
  --layers 3 \
  --channels 16,24,32 \
  --patch-size 3 \
  --strides 2,2,1 \
  --padding 0 \
  --head-dims 32,10 \
  --epochs 15 \
  --lr 2e-3
```

This configuration reached about **97.8%** test accuracy.

The current default 4-layer stride-1 `same`-padding model is a different and harder optimization problem. It keeps \(28\times28\) feature maps and uses much larger constellations (\(k=577\) in the last layer instead of \(k=433\)). Overlapping stride-1 patches also make the rows of \(H\) strongly correlated, so \(H^T H\) is worse conditioned and the linear-solver update can overshoot. If the loss grows, lower `--lr` for Adam or lower `--step` / increase `--alpha` for the generalized solver. Adding gradient clipping is also a reasonable next step. Generally, with stride=1 the method tends to diverge.

## Important limitations

1. **The solver is a generalization, not the exact paper algorithm.**  
   The paper's solver uses Lagrange multipliers and synchronized updates across a cascade. Here, the solver performs a per-layer Gauss-Newton-like linear solve with per-output-channel updates. It preserves the "linear solve instead of SGD" idea, but it is an approximation.

2. **It still uses autograd gradients.**  
   The optimizer does not use SGD, but it does compute derivatives with PyTorch autograd. It is not a fully gradient-free method.

3. **No convergence guarantees.**  
   The linearized update can be unstable for large steps or small \(\alpha\). It works in this MNIST prototype but needs tuning.

4. **`tanh` is an extra stabilization layer.**  
   The original cascade is globally smooth without activations. Here `tanh` keeps intermediate values in a bounded range so the fixed constellation remains usable.

5. **The layer is not a standard convolution.**  
   It is a local polyharmonic kernel expansion with shared `Y`. It is translation-equivariant, but it is not a conventional learned conv filter.

6. **MNIST only.**  
   This is a small demonstrator. It does not establish scalability to natural images or large architectures.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run from the repository root.

Default Adam baseline:

```bash
python train_mnist_polyconv.py
```

Default generalized-solver run:

```bash
python generalized_solver.py
```

To use CUDA:

```bash
python train_mnist_polyconv.py --device cuda
python generalized_solver.py --device cuda
```

Examples with explicit parameters:

```bash
# Default architecture, written out explicitly:
# 4 convolutional layers, 3x3 patches, stride 1, same padding,
# spatial size stays 28x28.
python train_mnist_polyconv.py \
  --layers 4 \
  --channels 16,24,32,32 \
  --patch-size 3 \
  --stride 1 \
  --padding same \
  --head-dims 32,10 \
  --epochs 8 \
  --lr 1e-3

# Same default architecture with the generalized linear solver:
python generalized_solver.py \
  --layers 4 \
  --channels 16,24,32,32 \
  --patch-size 3 \
  --stride 1 \
  --padding same \
  --head-dims 32,10 \
  --epochs 4 \
  --alpha 1e-2 \
  --step 1.0

# Arbitrary number of layers with constant spatial size:
python train_mnist_polyconv.py \
  --layers 5 \
  --channels 16,24,32,32,32 \
  --patch-size 3 \
  --stride 1 \
  --padding same \
  --head-dims 32,10 \
  --epochs 8

# Comparison: shrinking spatial size with stride 2 and valid padding.
python train_mnist_polyconv.py \
  --layers 3 \
  --channels 16,24,32 \
  --patch-size 3 \
  --stride 2 \
  --padding 0 \
  --head-dims 32,10 \
  --epochs 8
```

The scripts download MNIST into a relative `./data` directory.

The default 4-layer stride-1 model is computationally heavier than the stride-2 comparison.

## Possible next steps

- Tune patch size, stride, number of channels, `alpha`, and `step`.
- Compare with a standard CNN of similar parameter count.
- Implement a closer version of the paper's synchronized Lagrange-multiplier solver for shared `Y`.
- Try 5×5 patches or multi-scale constellations.
- Test on CIFAR-10 or a small image dataset.
- Use the layer as a decoder/predictor in an autoencoder or JEPA-like setup.

## Summary

The prototype shows that a polyharmonic package can be turned into a **translation-equivariant convolutional layer** by sharing `Y` across spatial positions. The default classifier uses four such layers with stride 1 and `same` padding, so an arbitrary number of convolutional layers can be added without changing the \(28\times28\) feature-map size. A deep MNIST classifier built from such layers can be trained with both Adam and a generalized linear-solver optimizer. The default 4-layer stride-1 configuration is computationally heavier and converges more slowly than the stride-2 comparisons; its long-run accuracy is not reported here. The main open question is whether the exact non-SGD solver from the original papers can be extended cleanly to convolutional weight sharing and larger architectures.
