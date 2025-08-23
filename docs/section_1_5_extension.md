# Section 1.5: Extension to the Stochastic Setting

## Breakdown of PGD in the Stochastic Setting

Perturbed Gradient Descent (PGD) relies on two deterministic checks to decide when a saddle has been reached and whether a perturbation successfully escaped it:

1. **Gradient Norm Condition**: `\|\nabla f(x_t)\| \leq g_\text{thres}`.
2. **Function Decrease Condition**: `f(x_t) - f(\tilde{x}_t^\text{noise}) > -f_\text{thres}`.

When gradients are estimated from mini-batches these tests become unreliable.

### Gradient Norm Condition

Mini-batch gradients `\nabla f_{B_t}(x_t)` satisfy `\mathbb{E}[\nabla f_{B_t}(x_t)] = \nabla f(x_t)` but exhibit high variance. A single draw may produce a small norm even when `\|\nabla f(x_t)\|` is large, falsely triggering a perturbation. Conversely, a large accidental norm may mask a true saddle. Formally, for a batch of size `b`

\[
\mathbb{E}[\|\nabla f_{B_t}(x_t)\|^2] = \|\nabla f(x_t)\|^2 + \frac{\sigma^2(x_t)}{b},
\]

where `\sigma^2(x_t)` is the variance of individual sample gradients. The `\sigma^2/b` term can dominate near a saddle, making `\|\nabla f_{B_t}(x_t)\|` a poor proxy for the true norm.

### Function Decrease Condition

The PGD check `f(x_t) - f(\tilde{x}_t^\text{noise}) > -f_\text{thres}` relies on monotonic decrease of the objective under gradient descent. SGD lacks this property: with probability proportional to gradient variance, a step may *increase* `f`, so the observed decrease (or lack thereof) after perturbation no longer correlates with proximity to a saddle. This renders the function-value test ineffective for deciding whether perturbation succeeded.

## Stochastic Perturbed Gradient Descent (S-PGD)

To stabilise perturbation decisions we modify PGD as follows:

1. **Exponential Moving Average (EMA) of Gradient Norms**: Maintain `v_t = (1-\beta) v_{t-1} + \beta \|g_t\|^2` with `g_t` the stochastic gradient. Trigger perturbation only when `\sqrt{v_t} \le g_\text{thres}`. The EMA reduces variance by averaging over recent batches.
2. **Variance-Reduced Gradient Check**: When the EMA is small, compute a large batch (or SVRG-style) gradient `\bar{g}_t`. If `\|\bar{g}_t\| \le g_\text{thres}` the algorithm accepts that it is near a saddle and perturbs.
3. **Post-Perturbation Escape Test**: After perturbation, monitor the EMA for `m` steps. If `\sqrt{v_t}` rises above `c_\text{esc} g_\text{thres}` (with `c_\text{esc} > 1`), the algorithm concludes it left the saddle region; otherwise a new perturbation is attempted.

### Pseudocode

```text
Input: step size \eta, thresholds g_thres, c_esc, perturb radius r, EMA parameter \beta, full-batch period m
Initialize x_0, v_0 = 0
for t = 0,1,2,... do
    sample mini-batch B_t
    g_t = stochastic_gradient(x_t, B_t)
    v_t = (1-\beta) v_{t-1} + \beta \|g_t\|^2
    if \sqrt{v_t} \le g_thres then
        \bar{g}_t = large_batch_gradient(x_t)
        if \|\bar{g}_t\| \le g_thres then
            x_t = x_t + UniformBall(r)
            for i = 1,...,m do
                sample mini-batch B_{t+i}
                g_{t+i} = stochastic_gradient(x_{t+i}, B_{t+i})
                v_{t+i} = (1-\beta) v_{t+i-1} + \beta \|g_{t+i}\|^2
                x_{t+i+1} = x_{t+i} - \eta g_{t+i}
            if \sqrt{v_{t+m}} < c_esc g_thres then
                repeat perturbation
    x_{t+1} = x_t - \eta g_t
```

## Justification of Design Choices

* **EMA of Gradient Norms** mitigates variance: `\mathrm{Var}[\sqrt{v_t}] \le \beta/(2-\beta) \cdot \sigma^2/b`, ensuring perturbation decisions reflect the true gradient norm with high probability.
* **Variance-Reduced Check** avoids false perturbations by verifying small gradients using `\bar{g}_t` whose variance is `\sigma^2/B` for large batch size `B`, sharply concentrating around `\nabla f(x_t)`.
* **Escape Test via EMA Increase** replaces unreliable function values. In regions of negative curvature the gradient norm grows after leaving a saddle, so monitoring `v_t` detects successful escape without assuming monotonic `f`.

## Proof Sketch for Convergence

1. **EMA Concentration**: Using Hoeffding or Bernstein inequalities, show that `\sqrt{v_t}` deviates from `\|\nabla f(x_t)\|` by at most `O(\sqrt{\sigma^2 \log(1/\delta)/(b\beta)})` with probability `1-\delta`.
2. **Accurate Saddle Detection**: Combine the above with the variance-reduced check to guarantee that perturbations are invoked only when `\|\nabla f(x_t)\| \le 2 g_\text{thres}`.
3. **Escape from Strict Saddles**: Adapt the deterministic PGD analysis. Conditioned on triggering a perturbation at a strict saddle, show that within `m` steps the EMA increases past `c_\text{esc} g_\text{thres}` with probability at least `1-\epsilon`. This uses the same random perturbation arguments as PGD, plus concentration of the EMA.
4. **Descent Outside Saddle Regions**: Standard SGD descent lemmas ensure expected function decrease whenever `\sqrt{v_t} > g_\text{thres}`.
5. **Overall Convergence**: Combining the above, S-PGD performs a finite number of successful perturbations before reaching a point where `\|\nabla f(x)\| \le g_\text{thres}` and the Hessian eigenvalues are `\ge -\sqrt{\rho \epsilon}`, i.e., a second-order stationary point.

This sketch leverages concentration inequalities and variance-reduction to control stochastic noise, allowing the deterministic PGD guarantees to extend to mini-batch gradients.
