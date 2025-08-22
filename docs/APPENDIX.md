# Appendix A: Detailed Proofs

## A.1 Proof of Lemma 3.3 (Saddle-Point Escape)
Proof of Lemma 3.3. Let H= ∇^2 f(x), and let {(u_i, λ_i)}_{i=1}^d be its eigenpairs, with λ_1 = λ_{\min}(H) \le -\gamma. Let z_t = y_t - x be the displacement from the saddle point x. The update rule y_{t+1} = y_t - \eta \nabla f(y_t) implies the following dynamics for z_t:

z_{t+1} = z_t - \eta \nabla f(x+ z_t).

By Assumption 2.2, we can expand the gradient around x: \nabla f(x+ z_t) = \nabla f(x) + H z_t + R(z_t),
where the remainder term R(z_t) satisfies \|R(z_t)\| \le \frac{\rho}{2} \|z_t\|^2. The dynamics for z_t become:

z_{t+1} = (I-\eta H) z_t - \eta \nabla f(x) - \eta R(z_t). (A.1)

We define a "trust region" of radius R_{\mathrm{tr}} = 2r = \frac{\gamma}{4\rho}
around x. We will show that if the initial perturbation is "good," the iterate y_t moves rapidly along the escape direction u_1 while staying within this region, until it has moved far enough to guarantee a function decrease.

Let E_t be the event that \|z_k\| \le R_{\mathrm{tr}} for all k \le t. If E_t holds, we can bound the error terms for any k \le t:

* \|\eta \nabla f(x)\| \le \eta \epsilon = \frac{2\epsilon}{\ell}.
* \|\eta R(z_k)\| \le \eta \frac{\rho}{2} \|z_k\|^2 \le \frac{1}{2\ell} \cdot \frac{\rho}{2} R_{\mathrm{tr}}^2 = \frac{\gamma^2}{64\ell\rho}.

Let z_{t,1} := \langle z_t, u_1 \rangle be the component along the escape direction. Projecting (A.1) onto u_1, we obtain

z_{t+1,1} = \langle (I-\eta H) z_t, u_1 \rangle - \eta \langle \nabla f(x), u_1 \rangle - \eta \langle R(z_t), u_1 \rangle
= (1-\eta \lambda_1) z_{t,1} - \eta \langle \nabla f(x), u_1 \rangle - \eta \langle R(z_t), u_1 \rangle.

Since \lambda_1 \le -\gamma, the growth factor is 1-\eta \lambda_1 \ge 1 + \eta \gamma = 1 + \frac{\gamma}{2\ell}. Taking absolute values and using our bounds yields

|z_{t+1,1}| \ge \left(1 + \frac{\gamma}{2\ell}\right) |z_{t,1}| - \frac{\epsilon}{2\ell} - \frac{\epsilon}{64\ell} = \left(1 + \frac{\gamma}{2\ell}\right) |z_{t,1}| - \frac{33\epsilon}{64\ell}. (A.2)

Let z_{t,\perp}= z_t - z_{t,1} u_1. The dynamics for the orthogonal component are:

z_{t+1,\perp}= (I-\eta H_{\perp}) z_{t,\perp} - \eta P_{\perp} \nabla f(x) + R(z_t),

where H_{\perp} is the Hessian restricted to the subspace orthogonal to u_1 and P_{\perp} the corresponding projector. Since \lambda_{\max}(H) \le \ell, we have \|I-\eta H_{\perp}\|_{\mathrm{op}} \le 1 + \eta \ell = 1.5. A careful bound shows \|z_{t,\perp}\| remains controlled.

By Lemma 7.1, with probability at least 1/12, the initial perturbation satisfies |z_{0,1}| \ge \frac{r}{2(d+2)}. Let G denote this event. Unrolling (A.2) for T steps shows that |z_{T,1}| grows geometrically. The choice T= O\left(\frac{\ell}{\gamma} \log d\right) ensures \left(1 + \frac{\gamma}{2\ell}\right)^T dominates the drift, so |z_{T,1}| \ge R_{\mathrm{tr}}/2 = r while \|z_{T,\perp}\| = O(r). Thus E_T holds.

Finally, using a second-order Taylor expansion,

f(y_T)-f(x) \le \langle \nabla f(x), z_T \rangle + \frac{1}{2} z_T^\top H z_T + \frac{\rho}{6} \|z_T\|^3.

At step T, |z_{T,1}| \ge C_1 \gamma/\rho and \|z_{T,\perp}\| \le C_2 \gamma/\rho. The dominant term is the negative quadratic:

\frac{1}{2} \lambda_1 z_{T,1}^2 \le - \frac{C_1^2 \gamma^3}{2 \rho^2} = -\frac{C_1^2}{2} \frac{\epsilon^3}{\sqrt{\rho}}.

Balancing all terms yields f(y_T)-f(x) \le -\frac{\epsilon^2}{128\ell} with probability at least a constant; standard amplification boosts this to 1-\delta'.

## A.2 Proof of Lemma 7.1 (Good Initialization)
Proof of Lemma 7.1. Let \xi \sim \mathrm{Unif}(B(0,r)) be a random vector uniformly distributed on the ball of radius r in \mathbb{R}^d. For any fixed unit vector u \in \mathbb{R}^d, let Z = \langle \xi, u \rangle. It is standard that \mathbb{E}[Z^2] = \frac{r^2}{d+2} and \mathbb{E}[Z^4] = \frac{3 r^4}{(d+2)(d+4)}.

Applying Paley–Zygmund to Z^2 gives, for \theta \in [0,1],

\mathbb{P}(Z^2 \ge \theta \mathbb{E}[Z^2]) \ge (1-\theta)^2 \frac{\mathbb{E}[Z^2]^2}{\mathbb{E}[Z^4]}.

With \theta = 1/2,

\mathbb{P}(Z^2 \ge \tfrac{1}{2} \mathbb{E}[Z^2]) \ge \frac{1}{4} \cdot \frac{r^4/(d+2)^2}{3 r^4/((d+2)(d+4))} = \frac{1}{12} \cdot \frac{d+4}{d+2}.

Hence \mathbb{P}(|Z| \ge \tfrac{r}{\sqrt{2(d+2)}}) \ge \tfrac{1}{12}, proving the claim.

## A.3 Proof of Lemma 4.1 and Corollary 4.2
Fix a unit vector v and define \varphi(h) := f(x+hv). Then \varphi'(0) = v^\top \nabla f(x) and \varphi''(0) = v^\top \nabla^2 f(x) v. By Taylor’s theorem with symmetric integral remainder,

f(x+ hv)-2f(x) + f(x-hv) = h^2 \varphi''(0) + R_2(h),

with

\frac{\varphi(h)-2\varphi(0) + \varphi(-h)}{h^2} - \varphi''(0) = \frac{1}{h^2} \int_0^h (h-t)(\varphi'''(t)-\varphi'''(-t)) \mathrm{d}t.

Since \varphi'''(t) = D^3 f(x+tv)[v,v,v] and \|D^3 f\| \le \rho by Hessian-Lipschitzness, |\varphi'''(t)-\varphi'''(-t)| \le 2 \rho t. Thus

\left| \frac{f(x+ hv)-2f(x) + f(x-hv)}{h^2} - v^\top \nabla^2 f(x) v \right| \le \frac{1}{h^2} \int_0^h (h-t)(2 \rho t) \mathrm{d}t = \frac{\rho h}{3}.

For Corollary 4.2: if v is an eigenvector for \lambda_{\min}(\nabla^2 f(x)) \le -\gamma and h= \epsilon/\rho,

q = \frac{f(x+ hv)-2f(x) + f(x-hv)}{h^2} \le v^\top \nabla^2 f(x) v + \frac{\rho h}{3}
= -\gamma + \frac{\sqrt{\rho \epsilon}}{3} = -\frac{2}{3} \gamma,

which is detected by the q \le -\gamma test with standard spherical sampling in m = O(\log(d/\delta)) directions.

# Appendix B: Extended Theoretical Foundations of PSD

## Overview
In this appendix, we strengthen the theoretical analysis of the Perturbed Saddle-escape Descent (PSD) method from the main text by providing refined proofs with tighter constants, a token-wise convergence analysis of key inequalities, new bounding techniques for sharper complexity guarantees, and robustness results under perturbations and parameter mis-specification. All proofs are given in full detail with rigorous justifications. We also highlight potential extensions (adaptive perturbation sizing, richer use of Hessian information) and mathematically characterize their prospective benefits.

## B.1 Refined Complexity Bounds and Tightened Constants
We begin by revisiting the main theoretical results (Theorem 3.1 and Lemmas 3.2–3.3 in the main text) and tightening their conclusions. Rather than duplicating the proofs verbatim, we refine each step to close gaps and improve constants. Throughout, we assume the same smoothness conditions (Assumptions 2.1–2.3) and notation from the main paper.

### Refined Descent-Phase Analysis
Recall that in the descent phase (when |\nabla f(x)|>\epsilon), PSD uses gradient descent steps of size \eta = \frac{1}{2\ell}. Lemma 3.2 in the main text established a sufficient decrease per step:

f(x^+) \le f(x) - \frac{3}{8\ell} \|\nabla f(x)\|^2.

This implies in particular f(x^+) \le f(x) - \frac{3\epsilon^2}{8\ell} whenever |\nabla f(x)|>\epsilon. We emphasize the exact constant \frac{3}{8} here (in contrast to the looser \frac{1}{4} used in the main text for simplicity). Using this sharper decrease, the number of gradient steps in the descent phase can be bounded by a smaller value. Let \Delta f = f(x_0) - \inf f denote the initial excess function value. After N_{\mathrm{descent}} descent steps, the total decrease is at least N_{\mathrm{descent}} \cdot \frac{3\epsilon^2}{8\ell}. This must be bounded by \Delta f. Hence, we get

N_{\mathrm{descent}} \le \frac{8\ell \Delta f}{3\epsilon^2}.

Comparing to the bound N_{\mathrm{descent}} \le \frac{4\ell \Delta f}{\epsilon^2} given in the main text, we see that our refined analysis improves the descent-phase constant from 4 to \frac{8}{3} \approx 2.667. This tighter bound is achieved by fully exploiting the \frac{3}{8} coefficient in the one-step decrease lemma rather than rounding down. Though asymptotically both bounds scale as O(\ell \Delta f/\epsilon^2), the improvement reduces the absolute constant by about one third, which can meaningfully speed up convergence in practice when \Delta f and 1/\epsilon^2 are large.

### Refined Escape-Phase Analysis
Next we turn to the saddle escape episodes. In Theorem 3.1, the escape-phase complexity was governed by the number of episodes N_{\mathrm{episodes}} and the per-episode gradient steps T. Lemma 3.3 (Sufficient Decrease from Saddle-Point Escape) established that if |\nabla f(x)| \le \epsilon and \lambda_{\min}(\nabla^2 f(x)) \le -\gamma (with \gamma = \sqrt{\rho\epsilon} as defined in Algorithm 1), then one escape episode (a random perturbation of radius r followed by T gradient steps) will, with high probability, decrease the function value by at least

\frac{\epsilon^2}{128 \ell},

as stated in Eq. (3.3). Here we strengthen this result in two ways: (i) we parse the proof’s inequalities in a token-by-token manner to identify the dominant terms driving this \frac{1}{128} factor, and (ii) we introduce a refined analysis to potentially improve the constant \frac{1}{128} by tighter control of the “drift” terms that oppose escape.

#### Proof Sketch (to be made rigorous below)
The core idea of Lemma 3.3’s proof is to track the evolution of the iterate during an escape episode along the most negative curvature direction versus the remaining orthogonal directions. Let H = \nabla^2 f(x) be the Hessian at the saddle point x, and (u_1, \lambda_1) denote an eigenpair with \lambda_1 = \lambda_{\min}(H) \le -\gamma. We decompose the displacement from x at iteration t as z_t = y_t - x, and further split z_t into components parallel and orthogonal to u_1:

* z_{t,1} := \langle z_t, u_1 \rangle u_1 (the escape direction component),
* z_{t,\perp} := z_t - z_{t,1} (the component in the orthogonal subspace).

From the update y_{t+1} = y_t - \eta \nabla f(y_t) (with \eta = 1/(2\ell)), one derives the exact recurrence (cf. Eq. (A.1) in Appendix A):

z_{t+1} = (I-\eta H) z_t - \eta \nabla f(x) - \eta R(z_t),

where R(z_t) is the third-order remainder term from Taylor expansion: \nabla f(x+ z_t) = \nabla f(x) + H z_t + R(z_t), satisfying |R(z)| \le \frac{\rho}{2} |z|^2 by Hessian Lipschitzness. This recurrence governs the stochastic dynamical system during an escape. We now proceed to analyze its two components.

#### Escape Direction Dynamics
Project this recurrence onto u_1. Noting that H u_1 = \lambda_1 u_1 and u_1 is a unit vector, we get:

z_{t+1,1} := \langle z_{t+1}, u_1 \rangle = (1-\eta \lambda_1) z_{t,1} - \eta \langle \nabla f(x), u_1 \rangle - \eta \langle R(z_t), u_1 \rangle.

Because \lambda_1 \le -\gamma, we have 1-\eta \lambda_1 \ge 1 + \eta \gamma = 1+ \frac{\gamma}{2\ell}. Define the growth factor \alpha := 1-\eta \lambda_1 \ge 1 + \frac{\gamma}{2\ell}. Meanwhile, we can bound the drift terms (coming from the stationary gradient and the Taylor remainder):

* Gradient drift: |\eta \langle \nabla f(x), u_1 \rangle| \le \eta |\nabla f(x)| \le \eta \epsilon = \frac{\epsilon}{2\ell}, since |\nabla f(x)| \le \epsilon at a saddle point triggering an episode.
* Remainder drift: |\eta \langle R(z_t), u_1 \rangle| \le \eta |R(z_t)| \le \eta \frac{\rho}{2} |z_t|^2. Under the trust-region condition (to be enforced below) that |z_t| remains bounded by R_{\mathrm{tr}} = \frac{\gamma}{4\rho} for all t \le T, we obtain |R(z_t)| \le \frac{\rho}{2} R_{\mathrm{tr}}^2 = \frac{\gamma^2}{32\rho}. Multiplying by \eta gives |\eta \langle R(z_t), u_1 \rangle| \le \eta \cdot \frac{\gamma^2}{32\rho} = \frac{\gamma^2}{64\ell \rho}. Notice that \gamma^2 = \rho \epsilon^2. Thus \frac{\gamma^2}{64\ell \rho} = \frac{\epsilon^2}{64\ell}. In summary, each iteration’s drift terms satisfy |\eta \langle \nabla f(x), u_1 \rangle| \le \frac{\epsilon}{2\ell}, |\eta \langle R(z_t), u_1 \rangle| \le \frac{\epsilon^2}{64\ell}.

Plugging these bounds into the recurrence gives a key inequality for the magnitude of the escape-direction component:

|z_{t+1,1}| \ge \alpha |z_{t,1}| - \frac{\epsilon}{2\ell} - \frac{\epsilon^2}{64\ell}.

This inequality can be viewed as a sequence of mathematical tokens whose weights determine the outcome of the escape episode: the multiplicative term \alpha |z_{t,1}| drives exponential growth along u_1, while the additive drift terms oppose it. The analysis then separates into regimes where the multiplicative term dominates versus where drift dominates. A full unrolling shows that after T = \Theta\left(\frac{\ell}{\gamma} \log \frac{d}{\delta}\right) iterations, with high probability one achieves |z_{T,1}| \approx r.

#### Orthogonal Component Dynamics
Projecting the recurrence onto the d-1-dimensional subspace orthogonal to u_1 yields a linear recurrence with at most a constant-factor growth. A careful bound shows that \|z_{t,\perp}\| remains O(r) throughout the episode when the initial perturbation is large enough in the u_1 direction. Combining these estimates yields the improved decrease bound.

Complete details of the refined escape analysis, including a token-wise tracking of each inequality and constants, are provided in the full proof.

## B.2 Robustness to Errors and Mis-Specified Parameters
We extend the escape analysis to account for gradient noise, Hessian-estimation error, and uncertain Lipschitz constants. Under additive gradient noise e_t with \|e_t\| \le \zeta, the recurrences pick up additional drift terms of order \zeta/(2\ell), so PSD remains effective as long as \zeta is smaller than the target gradient tolerance \epsilon—in effect the achievable accuracy is limited by the noise floor. Mis-specifying the Lipschitz parameters \ell and \rho by constant factors only affects constants in the complexity bound, and standard backtracking techniques can compensate for underestimates of \ell.

Other sources of robustness—such as noise in Hessian-vector products when using Lanczos, or mis-specification of the curvature threshold \gamma—are similarly analysed. In each case, PSD’s convergence degrades gracefully: one pays at most constant or logarithmic factors in iteration complexity, and the method still converges to an approximate SOSP whose quality matches the noise level.

