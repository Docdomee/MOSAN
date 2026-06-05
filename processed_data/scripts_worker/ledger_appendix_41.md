# Ledger Appendix — Cycle 41

## Strategic Lesson

**RESEARCH CHRONICLE**

**Current State**
The current peak performance is **HO=0.8467** (CV=0.8152), achieved using the `attention_d_cwt_v_high_cap_fixed.py` architecture with `3D_VIDEO` representation. 

**Dominant Hyperparameters**
*   **Batch Size:** Small (8) is required to prevent OOM.
*   **Capacity:** Leaner configurations (e.g., filter count=32) are stable; high-capacity settings (filters=128, layers=5) caused severe performance degradation (HO=0.5890).
*   **Optimization:** Optuna sweeps are effective for peak discovery but may mask underlying instability.

**Confirmed Dead-Ends**
*   **High-Capacity Scaling:** Simply increasing layers and filters does not improve HO and currently degrades results.

**Open Hypotheses**
*   Whether the peak HO is a result of genuine architectural superiority or a lucky seed in a high-variance environment.
*   The optimal balance between capacity and stability to bridge the CV/HO gap without triggering crashes.

**CRITICAL WARNING (From Critic)**
**Numerical Instability:** The model exhibits catastrophic variance (HO swings from 0.84 to 0.33). Treating Optuna peaks as "success" is cherry-picking. The system is hyper-sensitive or numerically unstable; current "successes" may be illusory. Prioritize stability and variance reduction over chasing isolated HO peaks.

## Critique

**Critique:** *   **False Correlation:** Attributing performance to "lower filter counts" is premature. You failed at high capacity, but didn't prove lean configs are the *driver* of the 0.8467 peak—only that they avoided OOM.
*   **Instability Blind Spot:** You are ignoring a catastrophic variance pattern. An HO swing from 0.84 to 0.33 indicates the model is numerically unstable or hyper-sensitive; treating this as a "success" via Optuna is cherry-picking.
*   **Overfitting Denial:** A narrow CV/HO gap is irrelevant if the variance is this high. You are masking systemic instability as "overfitting risk."
