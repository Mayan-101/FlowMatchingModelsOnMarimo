# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anywidget",
#     "equinox==0.13.8",
#     "jax[cuda]==0.10.2",
#     "marimo>=0.23.9",
#     "matplotlib==3.11.0",
#     "numpy>=2.4.6",
#     "optax==0.2.8",
#     "plotly==6.8.0",
#     "torchvision",
#     "traitlets==5.15.1",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="full",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <style>
    input[type="range"] {
        accent-color: #2563eb !important;
    }
    </style>

    # Back to Basics: Let Denoising Generative Models Denoise

    **Paper:** [Tianhong Li, Kaiming He (2026)](https://www.alphaxiv.org/abs/2511.13720) &nbsp;·&nbsp;
    **Notebook by:** Mayan S Hiremath

    Weren't we denoising before? But... what does one really mean by
    "denoising"? This notebook explores and tries to add intuition to
    the core ideas of the paper — a big question in denoising
    diffusion models about *what the network should actually predict?*  —
    along with the architectural decisions that make **Just Image
    Transformers (JiT)** special.
    ---

    **Note*:** For the best experience ensure the notebook is connected to GPU instance.
    """)
    return


@app.cell(hide_code=True)
def _():
    import json
    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import io
    import base64

    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax
    import marimo as mo
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import traitlets
    import anywidget
    from torchvision import datasets
    import jax.tree_util as jtu

    # Print baseline verification
    print("JAX Version:", jax.__version__)
    print("Available Devices:", jax.devices())          
    print("Default Backend:", jax.default_backend())
    return (
        anywidget,
        base64,
        datasets,
        eqx,
        io,
        jax,
        jnp,
        json,
        jtu,
        mo,
        np,
        optax,
        plt,
        traitlets,
    )


@app.cell(hide_code=True)
def _(datasets, np):
    _mnist_dataset = datasets.MNIST(root='./data', train=True, download=True)
    MNIST_X_RAW = (_mnist_dataset.data.numpy().astype(np.float32) / 255.0 - 0.5) / 0.5
    MNIST_Y_RAW = _mnist_dataset.targets.numpy().astype(np.int32)
    return MNIST_X_RAW, MNIST_Y_RAW


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A Refresher to Diffusion

    Denoising Diffusion models work by deconstructing data — adding noise to it
    via a **Forward Process** — and then learning to reconstruct the
    data or undoing the noising (denoising) via a **Reverse Process** during training.
    During inference, the model generates new data sample by denoising a point from pure noise. This is called **Sampling **.

    For this notebook, we use a flow-based approach.

    ### The Forward Process

    The forward process is done by an interpolation between the data
    distribution and the noise distribution:

    $$x_t = t\,x + (1-t)\,\epsilon$$

    Where:

    * $x \sim P_{\text{data}}$
    * $\epsilon \sim \mathcal{N}(0, I)$
    * $x_t \sim P_t$

    Here's what the forward noising process looks like — watch how the
    data distribution gets progressively destroyed as $t$ moves from
    data to noise:
    """)
    return


@app.cell(hide_code=True)
def _(jnp, np):
    # Initialize the spatial target clusters matrix
    _N_BOUNDARY = 4000
    _rng = np.random.RandomState(42)
    _half = _N_BOUNDARY // 2
    _b1 = _rng.randn(_half, 2) * 0.45 + np.array([5.0, 2.3])
    _b2 = _rng.randn(_half, 2) * 0.45 + np.array([5.0, -2.3])

    TOY1_X_TARGET = np.vstack([_b1, _b2]).astype(np.float32)
    TOY1_X_TARGET_jnp = jnp.array(TOY1_X_TARGET, dtype=jnp.bfloat16)

    _rng_eps = np.random.RandomState(7)
    TOY1_ALL_EPS = _rng_eps.randn(_N_BOUNDARY, 2).astype(np.float32)
    return TOY1_ALL_EPS, TOY1_X_TARGET, TOY1_X_TARGET_jnp


@app.cell(hide_code=True)
def _(TOY1_ALL_EPS, TOY1_X_TARGET, anywidget, json, traitlets):
    class Toy1ForwardNoisingWidget(anywidget.AnyWidget):
        _esm = """
        export default {
            render({ model, el }) {
                let x0 = [];
                let eps = [];
                let state = { playing: false, interval_id: null };

                el.innerHTML = `
                <div style="display: flex; flex-direction: column; align-items: center; gap: 16px; font-family: system-ui, sans-serif; background: #f8fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; max-width: 460px; margin: 0 auto; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);">
                    <div style="text-align: center; width: 100%;">
                        <div style="font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 2px;">Forward Process</div>
                        <div style="font-size: 11px; color: #64748b; margin-bottom: 12px;">Visualizing the forward noising process from data (t=1.0) to noise (t=0.0)</div>
                        <canvas id="scatterCanvas" width="400" height="340" style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; display: block; margin: 0 auto;"></canvas>
                    </div>

                    <div style="display: flex; align-items: center; gap: 12px; width: 100%; background: #f1f5f9; padding: 8px 16px; border-radius: 8px; box-sizing: border-box;">
                        <button id="playBtn" style="background: #f1f5f9; border: 1px solid #2563eb; border-radius: 6px; padding: 6px 14px; font-size: 13px; font-weight: 500; color: #2563eb; cursor: pointer; display: flex; align-items: center; gap: 4px; min-width: 70px; justify-content: center;">▶ play</button>
                        <input type="range" id="timeSlider" min="0" max="50" step="1" value="0" style="flex-grow: 1; accent-color: #2563eb; cursor: pointer; margin: 0;">
                        <span id="timeLabel" style="font-family: monospace; font-size: 13px; color: #475569; font-weight: 600; min-width: 65px; text-align: right;">t = 1.00</span>
                    </div>
                </div>
                `;

                let ctx_s = el.querySelector("#scatterCanvas").getContext("2d");
                let slider = el.querySelector("#timeSlider");
                let btn = el.querySelector("#playBtn");
                let lbl = el.querySelector("#timeLabel");

                function to_canvas(x, y, w=400) {
                    let cx = ((x - (-4.0)) / 12.0) * w;
                    let cy = 340 - (((y - (-5.0)) / 10.0) * 340);
                    return [cx, cy];
                }

                function draw_frame(step_idx) {
                    let t = 1.0 - step_idx * 0.02;
                    ctx_s.clearRect(0, 0, 400, 340);
                    lbl.textContent = `t = ${t.toFixed(2)}`;
                    if (!x0 || x0.length === 0) return;

                    ctx_s.fillStyle = "rgba(137, 99, 220, 0.35)";
                    for (let i = 0; i < x0.length; i++) {
                        let z_x = t * x0[i][0] + (1.0 - t) * eps[i][0];
                        let z_y = t * x0[i][1] + (1.0 - t) * eps[i][1];
                        let [cx, cy] = to_canvas(z_x, z_y, 400);
                        ctx_s.beginPath();
                        ctx_s.arc(cx, cy, 2, 0, 2 * Math.PI);
                        ctx_s.fill();
                    }
                }

                function tick() {
                    let step = parseInt(slider.value) + 1;
                    if (step > 50) step = 0;
                    slider.value = String(step);
                    draw_frame(step);
                }

                btn.addEventListener("click", () => {
                    if (state.playing) {
                        clearInterval(state.interval_id);
                        state.playing = false;
                        btn.textContent = "▶ play";
                    } else {
                        state.interval_id = setInterval(tick , 80);
                        state.playing = true;
                        btn.textContent = "⏸ pause";
                    }
                });

                slider.addEventListener("input", () => {
                    if (state.playing) {
                        clearInterval(state.interval_id);
                        state.playing = false;
                        btn.textContent = "▶ play";
                    }
                    draw_frame(parseInt(slider.value));
                });

                function update() {
                    let config = JSON.parse(model.get("config_json"));
                    if (config && config.X_TARGET) {
                        x0 = config.X_TARGET;
                        eps = config.ALL_EPS;
                        draw_frame(parseInt(slider.value));
                    }
                }

                model.on("change:config_json", update);
                update();
            }
        }
        """
        config_json = traitlets.Unicode("{}").tag(sync=True)

    toy1_fwd_widget = (Toy1ForwardNoisingWidget())
    toy1_fwd_widget.config_json = json.dumps({
        "X_TARGET": TOY1_X_TARGET[::3].tolist(),
        "ALL_EPS": TOY1_ALL_EPS[::3].tolist()
    })
    return (toy1_fwd_widget,)


@app.cell(hide_code=True)
def _(toy1_fwd_widget):
    toy1_fwd_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The Velocity Field & Loss

    Flow-based methods define a velocity field:

    $$\frac{d}{dt}(x_t) = x - \epsilon = v$$

    and minimize a loss function:

    $$L = \mathbb{E}_{t, x, \epsilon} \| v_\theta(x_t, t) - v \|_2^2$$

    Looking at this loss function objective, the model can learn in
    three different ways. It can learn the velocity field itself — we
    call it **$v$-pred**. It can learn the clean data directly, then use
    that to compute velocity — we call it **$x$-pred**. Or it can learn
    the noise and subtract it — we call it **$\epsilon$-pred**.

    | | Network predicts | Velocity Reparameterized | Loss |
    |---|---|---|---|
    | **$v$-pred** | $v_\theta := \text{net}_\theta(x_t, t)$ | $v_\theta$ | $\lVert v_\theta - (x-\epsilon) \rVert_2^2$ |
    | **$x$-pred** | $x_\theta := \text{net}_\theta(x_t, t)$ | $v_\theta = \dfrac{x_\theta - x_t}{1-t}$ | $\lVert v_\theta - (x-\epsilon) \rVert_2^2$ |
    | **$\epsilon$-pred** | $\epsilon_\theta := \text{net}_\theta(x_t, t)$ | $v_\theta = \dfrac{x_t - \epsilon_\theta}{t}$ | $\lVert v_\theta - (x-\epsilon) \rVert_2^2$ |
    """)
    return


@app.cell(hide_code=True)
def _(MNIST_X_RAW, MNIST_Y_RAW, np):
    # Grab one real MNIST "7" and derive the handful of views the diagram
    # below needs: two nearby points in t along the same forward trajectory
    # (for v-pred), the noisy input alone (for x-pred), and the clean image
    # plus the raw noise (for eps-pred).
    _digit_idx = int(np.where(MNIST_Y_RAW == 7)[0][0])
    _clean = np.clip((MNIST_X_RAW[_digit_idx] * 0.5) + 0.5, 0.0, 1.0)

    _rng = np.random.RandomState(3)
    _eps = np.clip(_rng.normal(0.5, 0.28, size=_clean.shape), 0.0, 1.0)

    _t = 0.55
    _dt = 0.18
    _x_t = np.clip(_t * _clean + (1 - _t) * _eps, 0.0, 1.0)
    _x_t_dt = np.clip((_t + _dt) * _clean + (1 - _t - _dt) * _eps, 0.0, 1.0)

    NATURE_CLEAN = _clean.tolist()
    NATURE_NOISE = _eps.tolist()
    NATURE_X_T = _x_t.tolist()
    NATURE_X_T_DT = _x_t_dt.tolist()
    return NATURE_CLEAN, NATURE_NOISE, NATURE_X_T, NATURE_X_T_DT


@app.cell(hide_code=True)
def _(
    NATURE_CLEAN,
    NATURE_NOISE,
    NATURE_X_T,
    NATURE_X_T_DT,
    anywidget,
    json,
    traitlets,
):
    class NaturePredictionDiagramWidget(anywidget.AnyWidget):
        _esm = """
        export default {
            render({ model, el }) {
                el.innerHTML = `
                <div style="display: flex; flex-direction: column; gap: 22px; font-family: system-ui, sans-serif; background: #f8fafc; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; max-width: 760px; margin: 0 auto; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);">
                    <div style="font-size: 15px; font-weight: 700; color: #0f172a;">Three ways to denoise the same image</div>
                    <div id="rowV" class="predRow"></div>
                    <div id="rowX" class="predRow"></div>
                    <div id="rowE" class="predRow"></div>
                </div>
                `;

                function makeRow(container, label, formula) {
                    let row = document.createElement("div");
                    row.style.cssText = "display:flex; flex-direction:column; gap:8px; border-top:1px solid #e2e8f0; padding-top:16px;";
                    let top = document.createElement("div");
                    top.style.cssText = "display:flex; align-items:center; gap:20px; flex-wrap:wrap;";
                    top.innerHTML = `<span style="font-size:14px; font-weight:700; color:#1e293b; min-width:78px;">${label}</span>`;
                    row.appendChild(top);
                    let formulaDiv = document.createElement("div");
                    formulaDiv.style.cssText = "font-size:12px; color:#64748b; font-family: ui-monospace, monospace; padding-left: 98px;";
                    formulaDiv.innerHTML = formula;
                    row.appendChild(formulaDiv);
                    container.replaceChildren(row);
                    return top;
                }

                function drawDigit(canvas, pixels) {
                    let ctx = canvas.getContext("2d");
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    if (!pixels) return;
                    let n = pixels.length, cell = canvas.width / n;
                    for (let r = 0; r < n; r++) {
                        for (let c = 0; c < n; c++) {
                            let v = Math.floor(Math.max(0, Math.min(1, pixels[r][c])) * 255);
                            ctx.fillStyle = `rgb(${v},${v},${v})`;
                            ctx.fillRect(c * cell, r * cell, cell + 0.5, cell + 0.5);
                        }
                    }
                }

                function digitBox(caption) {
                    let wrap = document.createElement("div");
                    wrap.style.cssText = "display:flex; flex-direction:column; align-items:center; gap:4px;";
                    let canvas = document.createElement("canvas");
                    canvas.width = 64; canvas.height = 64;
                    canvas.style.cssText = "background:#000; border-radius:4px; border:1px solid #cbd5e1;";
                    let cap = document.createElement("div");
                    cap.style.cssText = "font-size:11px; color:#64748b; font-family: ui-monospace, monospace;";
                    cap.innerHTML = caption;
                    wrap.appendChild(canvas);
                    wrap.appendChild(cap);
                    return { wrap, canvas };
                }

                function opSpan(text) {
                    let s = document.createElement("span");
                    s.style.cssText = "font-size:18px; color:#94a3b8; font-weight:600;";
                    s.innerHTML = text;
                    return s;
                }

                function update() {
                    let cfg = JSON.parse(model.get("digits_json"));
                    if (!cfg || !cfg.clean) return;

                    // ── row 1: v-pred — xt -> x_(t+dt) - xt ──
                    let topV = makeRow(el.querySelector("#rowV"), "v-pred", "x<sub>t</sub> &rarr; x<sub>t+&Delta;t</sub> &minus; x<sub>t</sub>");
                    let b1 = digitBox("x<sub>t</sub>");
                    let b2 = digitBox("x<sub>t+&Delta;t</sub>");
                    let b3 = digitBox("x<sub>t</sub>");
                    topV.appendChild(b1.wrap);
                    topV.appendChild(opSpan("&rarr;"));
                    topV.appendChild(b2.wrap);
                    topV.appendChild(opSpan("&minus;"));
                    topV.appendChild(b3.wrap);
                    drawDigit(b1.canvas, cfg.x_t);
                    drawDigit(b2.canvas, cfg.x_t_dt);
                    drawDigit(b3.canvas, cfg.x_t);

                    // ── row 2: x-pred — xt -> x0 ──
                    let topX = makeRow(el.querySelector("#rowX"), "x-pred", "x<sub>t</sub> &rarr; x<sub>0</sub>");
                    let c1 = digitBox("x<sub>t</sub>");
                    let c2 = digitBox("x<sub>0</sub>");
                    topX.appendChild(c1.wrap);
                    topX.appendChild(opSpan("&rarr;"));
                    topX.appendChild(c2.wrap);
                    drawDigit(c1.canvas, cfg.x_t);
                    drawDigit(c2.canvas, cfg.clean);

                    // ── row 3: eps-pred — xt -> eps ──
                    let topE = makeRow(el.querySelector("#rowE"), "ε-pred", "x<sub>t</sub> &rarr; &epsilon;");
                    let d1 = digitBox("x<sub>t</sub>");
                    let d2 = digitBox("&epsilon;");
                    topE.appendChild(d1.wrap);
                    topE.appendChild(opSpan("&rarr;"));
                    topE.appendChild(d2.wrap);
                    drawDigit(d1.canvas, cfg.x_t);
                    drawDigit(d2.canvas, cfg.noise);
                }

                model.on("change:digits_json", update);
                update();
            }
        }
        """
        digits_json = traitlets.Unicode("{}").tag(sync=True)

    nature_prediction_widget = NaturePredictionDiagramWidget()
    nature_prediction_widget.digits_json = json.dumps({
        "clean": NATURE_CLEAN,
        "noise": NATURE_NOISE,
        "x_t": NATURE_X_T,
        "x_t_dt": NATURE_X_T_DT,
    })
    return (nature_prediction_widget,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How do these "Denoise"?

    Each of the three
    objectives above is really a different, concrete operation:

    * **$v$-pred**: Denoises by steering the samples towards a lower noise direction (i.e. towards clean data):
    * **$x$-pred**: Denoises the samples but predicts clean in one-shot:
    * **$\epsilon$-pred**: Denoises by subtracting the noise directly from the sample:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Note*:** The x<sub>t+$\Delta$t</sub> is just for illustrative purposes only to indicate the moves to less noise
    """)
    return


@app.cell(hide_code=True)
def _(nature_prediction_widget):
    nature_prediction_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lets see how these loss objectives perform

    To see this we can perform a toy experiment, visualizing the velocity field each of the objectives produce
    """)
    return


@app.cell(hide_code=True)
def toy1_controls(mo):
    toy1_form = (
        mo.md(
            """
            ### Toy 1 Configuration controls
            {n_epochs}

            {hidden_dim}

            {n_layers}

            {batch_size}

            {lr}
            """
        )
        .batch(
            n_epochs=mo.ui.slider(10, 500, step=10, value=150, label="Epochs", show_value=True),
            hidden_dim=mo.ui.slider(64, 512, step=64, value=256, label="Hidden Dimension", show_value=True),
            n_layers=mo.ui.slider(2, 10, step=1, value=5, label="Hidden Layers", show_value=True),
            batch_size=mo.ui.slider(64, 1024, step=64, value=256, label="Batch Size", show_value=True),
            lr=mo.ui.dropdown(options=["0.0001", "0.0005", "0.001", "0.005"], value="0.0005", label="Learning Rate"),
        )
        .form(submit_button_label="Run Experiment 1")
    )

    mo.vstack([
        mo.md("## Train $v$-pred, $x$-pred & $\\epsilon$-pred side by side"),
        toy1_form
    ])
    return (toy1_form,)


@app.cell(hide_code=True)
def toy1_config_setup(jax, jnp, toy1_form):
    _val = toy1_form.value or {}
    toy1_config = {
        "key": jax.random.PRNGKey(0),
        "n_boundary": 4000,
        "n_t_steps": 51,
        "hidden_dim": int(_val.get("hidden_dim", 256)),
        "n_layers": int(_val.get("n_layers", 5)),
        "batch_size": int(_val.get("batch_size", 256)),
        "n_epochs": int(_val.get("n_epochs", 150)),
        "lr": float(_val.get("lr", 5e-4)),
        "n_reverse": 120,
        "modes": ["x_pred", "eps_pred", "v_pred"],
        "t_vals": jnp.linspace(0.0, 1.0, 51)
    }
    return (toy1_config,)


@app.cell(hide_code=True)
def _(eqx, jax, jnp, toy1_config):
    class Toy1FlowMLP(eqx.Module):
        layers: list

        def __init__(self, *, key):
            _dims = [3] + [toy1_config["hidden_dim"]] * (toy1_config["n_layers"] - 1) + [2]
            _keys = jax.random.split(key, len(_dims) - 1)
            self.layers = [
                eqx.nn.Linear(_dims[i], _dims[i+1], key=_keys[i])
                for i in range(len(_dims) - 1)
            ]

        def __call__(self, x, t):
            _h = jnp.concatenate([x, jnp.atleast_1d(t)])
            for _layer in self.layers[:-1]:
                _h = jax.nn.relu(_layer(_h))
            return self.layers[-1](_h)

    return (Toy1FlowMLP,)


@app.cell(hide_code=True)
def _(
    TOY1_X_TARGET_jnp,
    Toy1FlowMLP,
    eqx,
    jax,
    jnp,
    jtu,
    mo,
    optax,
    toy1_config,
    toy1_form,
):
    if toy1_form.value is not None:
        _opt = optax.adamw(toy1_config["lr"])

        def _v_from_pred(pred, z, t, mode_idx):
            return jax.lax.switch(
                mode_idx,
                [
                    lambda: (pred - z) / jnp.clip(1.0 - t, 0.05),
                    lambda: (z - pred) / jnp.clip(t, 0.05),
                    lambda: pred,
                ]
            )

        def _loss_fn(arrays, static, x_batch, t_batch, key, mode_idx):
            _model = eqx.combine(arrays, static)
            _eps = jax.random.normal(key, x_batch.shape)
            _t_col = t_batch[:, None]
            _z = _t_col * x_batch + (1.0 - _t_col) * _eps
            _v_true = x_batch - _eps
            _out = jax.vmap(_model)(_z, t_batch)
            _v_pred = jax.vmap(lambda p, zi, ti: _v_from_pred(p, zi, ti, mode_idx))(_out, _z, t_batch)
            return jnp.mean((_v_pred - _v_true) ** 2)

        _init_keys = jax.random.split(toy1_config["key"], 3)
        _models = [Toy1FlowMLP(key=_k) for _k in _init_keys]

        _arrays_list = []
        for _m in _models:
            _a, _static = eqx.partition(_m, eqx.is_inexact_array)
            _arrays_list.append(_a)

        _stacked_arrays = jtu.tree_map(lambda *args: jnp.stack(args), *_arrays_list)
        _stacked_opt = jtu.tree_map(lambda *args: jnp.stack(args), *[_opt.init(_a) for _a in _arrays_list])

        _mode_idxs = jnp.array([0, 1, 2])
        _step_keys = jax.random.split(toy1_config["key"], 3)
        _N = TOY1_X_TARGET_jnp.shape[0]
        _steps_per_epoch = _N // toy1_config["batch_size"]

        @jax.jit
        def _step(arrays, opt_state, x_batch, t_batch, step_keys):
            def single_mode_step(arrays_m, opt_state_m, key_m, mode_idx_m):
                loss, grads = jax.value_and_grad(_loss_fn)(arrays_m, _static, x_batch, t_batch, key_m, mode_idx_m)
                updates, new_opt = _opt.update(grads, opt_state_m, params=arrays_m)
                return eqx.apply_updates(arrays_m, updates), new_opt, loss

            _next_keys = jax.vmap(lambda k: jax.random.split(k)[0])(step_keys)
            _proc_keys = jax.vmap(lambda k: jax.random.split(k)[1])(step_keys)
            new_arrays, new_opt_state, losses = jax.vmap(single_mode_step)(arrays, opt_state, _proc_keys, _mode_idxs)
            return new_arrays, new_opt_state, losses, _next_keys

        _key = toy1_config["key"]
        for _ep in mo.status.progress_bar(
            range(toy1_config["n_epochs"]),
            title="Toy Experiment 1 Training",
            subtitle="This may take about 15-20s (default)",
            show_eta=True,
            show_rate=True,
        ):
            for _ in range(_steps_per_epoch):
                _key, _idx_k, _t_k = jax.random.split(_key, 3)
                _idx = jax.random.randint(_idx_k, (toy1_config["batch_size"],), 0, _N)
                _stacked_arrays, _stacked_opt, _lv, _step_keys = _step(
                    _stacked_arrays, _stacked_opt, TOY1_X_TARGET_jnp[_idx], jax.random.uniform(_t_k, (toy1_config["batch_size"],)), _step_keys
                )

        TOY1_TRAINED_MODELS = {}
        for _i, _mode in enumerate(toy1_config["modes"]):
            _a_m = jtu.tree_map(lambda x: x[_i], _stacked_arrays)
            TOY1_TRAINED_MODELS[_mode] = eqx.combine(_a_m, _static)

        # Free compiled training artifacts from device cache
        jax.clear_caches()
    else:
        TOY1_TRAINED_MODELS = None
    return (TOY1_TRAINED_MODELS,)


@app.cell(hide_code=True)
def _(TOY1_TRAINED_MODELS, jax, jnp, np, toy1_config):
    if TOY1_TRAINED_MODELS is not None:
        _key_pts = jax.random.split(toy1_config["key"])[1]
        _z0 = np.array(jax.random.normal(_key_pts, (toy1_config["n_reverse"], 2)))

        TOY1_REV_TRAJS = {}
        _t_vals_np = np.array(toy1_config["t_vals"])

        for _mode in toy1_config["modes"]:
            _model = TOY1_TRAINED_MODELS[_mode]
            _traj = [np.array(_z0)]
            _z_curr = jnp.array(_z0)

            for _si in range(len(_t_vals_np) - 1):
                _t_cur = float(_t_vals_np[_si])
                _pred = np.array(jax.vmap(_model, in_axes=(0, None))(_z_curr, _t_cur))
                _z_np = np.array(_z_curr)

                if _mode == "x_pred":
                    _vp = (_pred - _z_np) / max(1.0 - _t_cur, 0.01)
                elif _mode == "eps_pred":
                    _vp = (_z_np - _pred) / max(_t_cur, 0.01)
                else:
                    _vp = _pred

                _dt = _t_vals_np[_si + 1] - _t_vals_np[_si]
                _z_curr = _z_curr + _dt * jnp.array(_vp)
                _traj.append(np.array(_z_curr))

            TOY1_REV_TRAJS[_mode] = np.stack(_traj, axis=1).tolist()

        _gx = np.linspace(-4.0, 8.0, 25)
        _gy = np.linspace(-5.0, 5.0, 25)
        _GX, _GY = np.meshgrid(_gx, _gy)
        _pts_mesh = np.stack([_GX.ravel(), _GY.ravel()], axis=1).astype(np.float32)
        _pts_jnp = jnp.array(_pts_mesh)

        TOY1_VELOCITY_FIELDS = {}
        for _mode in toy1_config["modes"]:
            _model = TOY1_TRAINED_MODELS[_mode]
            _mode_grids = []
            for _t in _t_vals_np:
                _out = np.array(jax.vmap(_model, in_axes=(0, None))(_pts_jnp, _t))
                if _mode == "x_pred":
                    _v = (_out - _pts_mesh) / max(1.0 - float(_t), 0.01)
                elif _mode == "eps_pred":
                    _v = (_pts_mesh - _out) / max(float(_t), 0.01)
                else:
                    _v = _out
                _mode_grids.append(_v.tolist())
            TOY1_VELOCITY_FIELDS[_mode] = _mode_grids

        # Free inference caches after reverse flow computation
        jax.clear_caches()
    else:
        TOY1_REV_TRAJS = {}
        TOY1_VELOCITY_FIELDS = {}
    return TOY1_REV_TRAJS, TOY1_VELOCITY_FIELDS


@app.cell(hide_code=True)
def _(
    TOY1_REV_TRAJS,
    TOY1_VELOCITY_FIELDS,
    TOY1_X_TARGET,
    anywidget,
    json,
    toy1_config,
    traitlets,
):
    class Toy1ReverseFlowWidget(anywidget.AnyWidget):
        _esm = """
        export default {
            render({ model, el }) {
                let trajs_all = {}, vfields_all = {}, modes = [], x_target = [];
                let active_mode = "v_pred";
                let state = { playing: false, interval_id: null };

                el.innerHTML = `
                <div style="display: flex; flex-direction: column; align-items: center; gap: 16px; font-family: system-ui, sans-serif; background: #f8fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; max-width: 950px; margin: 0 auto;">
                    <div style="display: flex; gap: 16px; align-items: center; width: 100%; justify-content: space-between; border-bottom: 1px solid #e2e8f0; padding-bottom: 12px;">
                        <div style="font-size: 15px; font-weight: 700; color: #1e293b;">Reverse Flow Velocity Field & Distribution Map</div>
                        <div id="modeContainer" style="display: flex; gap: 6px;"></div>
                    </div>

                    <div style="display: flex; gap: 24px; justify-content: center; width: 100%; flex-wrap: wrap;">
                        <div style="text-align: center;">
                            <div style="font-size: 13px; font-weight: 600; color: #475569; margin-bottom: 6px;">Velocity Field Map</div>
                            <canvas id="vectorCanvas" width="450" height="340" style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;"></canvas>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 13px; font-weight: 600; color: #475569; margin-bottom: 6px;">Density Profile Slice</div>
                            <canvas id="sliceCanvas" width="400" height="340" style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;"></canvas>
                        </div>
                    </div>

                    <div style="display: flex; align-items: center; gap: 12px; width: 100%; max-width: 600px; background: #f1f5f9; padding: 8px 16px; border-radius: 8px;">
                        <button id="revPlayBtn" style="background: #f1f5f9; border: 1px solid #2563eb; border-radius: 6px; padding: 6px 14px; font-size: 13px; font-weight: 500; color: #2563eb; cursor: pointer; display: flex; align-items: center; gap: 4px; min-width: 70px; justify-content: center;">▶ play</button>
                        <input type="range" id="revSlider" min="0" max="50" step="1" value="0" style="flex-grow: 1; accent-color: #2563eb; cursor: pointer;">
                        <span id="revLabel" style="font-family: monospace; font-size: 13px; color: #475569; font-weight: 600; min-width: 65px; text-align: right;">t = 0.00</span>
                    </div>
                </div>
                `;

                let mode_box = el.querySelector("#modeContainer");
                let ctx_v = el.querySelector("#vectorCanvas").getContext("2d");
                let ctx_s = el.querySelector("#sliceCanvas").getContext("2d");
                let slider = el.querySelector("#revSlider");
                let p_btn = el.querySelector("#revPlayBtn");
                let lbl = el.querySelector("#revLabel");

                function linspace(start, end, steps) {
                    let arr = [];
                    let step = (end - start) / (steps - 1);
                    for (let i = 0; i < steps; i++) arr.push(start + step * i);
                    return arr;
                }

                function to_canvas(x, y, w=450) {
                    let cx = ((x - (-4.0)) / 12.0) * w;
                    let cy = 340 - (((y - (-5.0)) / 10.0) * 340);
                    return [cx, cy];
                }

                function draw_frame(step_idx) {
                    let m = active_mode;
                    let t_val = step_idx * 0.02;
                    lbl.textContent = `t = ${t_val.toFixed(2)}`;

                    ctx_v.clearRect(0, 0, 450, 340);
                    ctx_s.clearRect(0, 0, 400, 340);

                    if (!trajs_all || !trajs_all[m]) return;

                    // --- SHADOW PLOT BACKGROUND PARTICLES ---
                    if (x_target && x_target.length > 0) {
                        // Changed to a lighter purple than the active particles
                        ctx_v.fillStyle = "rgba(137, 99, 220, 0.15)"; 
                        for (let i = 0; i < x_target.length; i++) {
                            let [scx, scy] = to_canvas(x_target[i][0], x_target[i][1], 450);
                            if (0 <= scx && scx <= 450 && 0 <= scy && scy <= 340) {
                                ctx_v.beginPath();
                                ctx_v.arc(scx, scy, 2, 0, 2 * Math.PI);
                                ctx_v.fill();
                            }
                        }
                    }

                    let vf = vfields_all[m][step_idx];
                    let gx_c = linspace(-4.0, 8.0, 25);
                    let gy_c = linspace(-5.0, 5.0, 25);

                    let max_mag = 16.0;

                    let idx = 0;
                    for (let g_y of gy_c) {
                        for (let g_x of gx_c) {
                            let vx = vf[idx][0], vy = vf[idx][1];
                            let v_mag = Math.sqrt(vx**2 + vy**2);
                            let [cx, cy] = to_canvas(g_x, g_y, 450);

                            let ratio = Math.min(1.0, Math.max(0.0, v_mag / max_mag));

                            let r_col = Math.floor(189 + ratio * (0 - 189));
                            let g_col = Math.floor(230 + ratio * (54 - 230));
                            let b_col = Math.floor(255 + ratio * (179 - 255));

                            let color_str = `rgba(${r_col}, ${g_col}, ${b_col}, 0.85)`;
                            ctx_v.strokeStyle = color_str;
                            ctx_v.fillStyle = color_str;
                            ctx_v.lineWidth = 1.35;

                            let mag_norm = v_mag + 1e-6;
                            let [ex, ey] = to_canvas(g_x + (vx/mag_norm)*0.38, g_y + (vy/mag_norm)*0.38, 450);

                            ctx_v.beginPath();
                            ctx_v.moveTo(cx, cy);
                            ctx_v.lineTo(ex, ey);
                            ctx_v.stroke();

                            let angle = Math.atan2(ey - cy, ex - cx);
                            let headlen = 4.5;
                            ctx_v.beginPath();
                            ctx_v.moveTo(ex, ey);
                            ctx_v.lineTo(ex - headlen * Math.cos(angle - Math.PI / 6), ey - headlen * Math.sin(angle - Math.PI / 6));
                            ctx_v.lineTo(ex - headlen * Math.cos(angle + Math.PI / 6), ey - headlen * Math.sin(angle + Math.PI / 6));
                            ctx_v.fill();
                            idx++;
                        }
                    }

                    ctx_v.fillStyle = "#ffffff";
                    ctx_v.fillRect(314, 0, 110, 45);
                    ctx_v.strokeStyle = "#cbd5e1";
                    ctx_v.strokeRect(314, 0, 110, 45);

                    let let_grad = ctx_v.createLinearGradient(320, 0, 420, 0);
                    let_grad.addColorStop(0, "rgba(189, 230, 255, 1.0)");
                    let_grad.addColorStop(1, "rgba(0, 54, 179, 100)");
                    ctx_v.fillStyle = let_grad;
                    ctx_v.fillRect(320, 20, 100, 8);

                    ctx_v.fillStyle = "#475569";
                    ctx_v.font = "bold 9px system-ui";
                    ctx_v.fillText(" Magnitude Scale", 316, 15);
                    ctx_v.font = "9px monospace";
                    ctx_v.fillText("0.0", 320, 42);
                    ctx_v.fillText("16.0", 402, 42);

                    let trajs = trajs_all[m];
                    let trail_len = 4;
                    for (let p = 0; p < trajs.length; p++) {
                        let trail_start = Math.max(0, step_idx - trail_len);
                        if (step_idx > trail_start) {
                            ctx_v.beginPath();
                            ctx_v.lineWidth = 1.0;
                            for (let ti = trail_start; ti <= step_idx; ti++) {
                                let [tcx, tcy] = to_canvas(trajs[p][ti][0], trajs[p][ti][1], 450);
                                if (ti === trail_start) ctx_v.moveTo(tcx, tcy);
                                else ctx_v.lineTo(tcx, tcy);
                            }
                            ctx_v.strokeStyle = "rgba(137, 99, 220, 0.6)";
                            ctx_v.stroke();
                        }
                    }

                    ctx_v.fillStyle = "#8963DC";
                    for (let p = 0; p < trajs.length; p++) {
                        let [cx, cy] = to_canvas(trajs[p][step_idx][0], trajs[p][step_idx][1], 450);
                        ctx_v.beginPath();
                        ctx_v.arc(cx, cy, 2.5, 0, 2 * Math.PI);
                        ctx_v.fill();
                    }

                    let eval_grid = linspace(-5.0, 5.0, 100);
                    let density = new Array(100).fill(0);
                    let bw = 0.40;
                    for (let p = 0; p < trajs.length; p++) {
                        let y = trajs[p][step_idx][1];
                        for (let i = 0; i < eval_grid.length; i++) {
                            let diff = (eval_grid[i] - y) / bw;
                            density[i] += Math.exp(-0.5 * diff * diff) / (bw * Math.sqrt(2 * Math.PI));
                        }
                    }
                    for (let i = 0; i < density.length; i++) density[i] /= trajs.length;

                    // --- DISTRIBUTION PLOT (DENSITY PROFILE SLICE) ---
                    ctx_s.strokeStyle = "#8963DC";              // Solid Purple line mapping particle color
                    ctx_s.fillStyle = "rgba(137, 99, 220, 0.20)"; // Transparent Purple matching curve fill
                    ctx_s.lineWidth = 2;
                    ctx_s.beginPath();
                    for (let i = 0; i < eval_grid.length; i++) {
                        let cx = 2 + (density[i] / 0.85) * 320;
                        let [, cy] = to_canvas(0.0, eval_grid[i], 400);
                        if (i === 0) ctx_s.moveTo(2, cy);
                        else ctx_s.lineTo(cx, cy);
                    }
                    let [, last_cy] = to_canvas(0.0, eval_grid[eval_grid.length - 1], 400);
                    ctx_s.lineTo(2, last_cy);
                    ctx_s.closePath();
                    ctx_s.fill();
                    ctx_s.stroke();

                    ctx_s.strokeStyle = "#cbd5e1";
                    ctx_s.lineWidth = 1;
                    ctx_s.beginPath();
                    ctx_s.moveTo(0.1, 0);
                    ctx_s.lineTo(0.1, 340);
                    ctx_s.stroke();
                }

                function tick() {
                    let step = parseInt(slider.value) + 1;
                    if (step > 50) step = 0;
                    slider.value = String(step);
                    draw_frame(step);
                }

                p_btn.addEventListener("click", () => {
                    if (state.playing) {
                        clearInterval(state.interval_id);
                        state.playing = false;
                        p_btn.textContent = "▶ play";
                    } else {
                        state.interval_id = setInterval(tick , 80);
                        state.playing = true;
                        p_btn.textContent = "⏸ pause";
                    }
                });

                slider.addEventListener("input", () => {
                    draw_frame(parseInt(slider.value));
                });

                function init_modes() {
                    mode_box.innerHTML = "";
                    modes.forEach(m => {
                        let btn_m = document.createElement("button");
                        let displayName = m;
                        if (m === "x_pred") displayName = "x-pred";
                        else if (m === "v_pred") displayName = "v-pred";
                        else if (m === "eps_pred") displayName = "ε-pred";
                        btn_m.textContent = displayName;
                        btn_m.style.cssText = "padding: 4px 10px; font-size: 11px; font-weight:600; border-radius:4px; border:1px solid #cbd5e1; cursor:pointer; background:" + (m === active_mode ? "#2563eb; color:white; border-color:#2563eb;" : "#fff; color:#475569;");
                        btn_m.addEventListener("click", (e) => {
                            active_mode = m;
                            mode_box.querySelectorAll("button").forEach(b => {
                                b.style.background = "#fff";
                                b.style.color = "#475569";
                                b.style.borderColor = "#cbd5e1";
                            });
                            e.target.style.background = "#2563eb";
                            e.target.style.borderColor = "#2563eb";
                            e.target.style.color = "white";
                            draw_frame(parseInt(slider.value));
                        });
                        mode_box.appendChild(btn_m);
                    });
                }

                function update() {
                    let config = JSON.parse(model.get("config_json"));
                    if (config && config.trajs) {
                        trajs_all = config.trajs;
                        vfields_all = config.vfields;
                        modes = config.modes;
                        x_target = config.x_target || [];
                        init_modes();
                        draw_frame(parseInt(slider.value));
                    }
                }

                model.on("change:config_json", update);
                update();
            }
        }
        """
        config_json = traitlets.Unicode("{}").tag(sync=True)

    toy1_rev_widget = Toy1ReverseFlowWidget()
    toy1_rev_widget.config_json = json.dumps({
        "trajs": TOY1_REV_TRAJS,
        "vfields": TOY1_VELOCITY_FIELDS,
        "modes": toy1_config["modes"],
        "x_target": TOY1_X_TARGET[::3].tolist()
    })
    return (toy1_rev_widget,)


@app.cell(hide_code=True)
def _(TOY1_TRAINED_MODELS, mo, toy1_rev_widget):
    if TOY1_TRAINED_MODELS is not None:
        _o = mo.vstack([
            toy1_rev_widget,
        ])
    else: 
        _o = None
    _o
    return


@app.cell(hide_code=True)
def _(TOY1_VELOCITY_FIELDS, TOY1_X_TARGET, anywidget, json, traitlets):
    class Toy1VectorTracerWidget(anywidget.AnyWidget):
        _esm = """
        export default {
            render({ model, el }) {
                let vfields_all = {}, x_target = [];
                let active_mode = "v_pred";
                let rx_start = -2.0 + Math.random() * 4.0;
                let ry_start = -2.0 + Math.random() * 4.0;
                let current_probe = [rx_start, ry_start];

                 el.innerHTML = `
                <div style="display: flex; flex-direction: column; align-items: center; gap: 12px; font-family: system-ui, sans-serif; background: #f8fafc; padding: 18px; border-radius: 12px; border: 1px solid #e2e8f0; max-width: 520px; margin: 0 auto; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);">
                    <div style="text-align: center; width: 100%;">
                        <div style="font-size: 15px; font-weight: 700; color: #0f172a;">Interactive Trajectory Tracer</div>
                        <div style="font-size: 12px; color: #64748b; margin-top: 3px; max-width: 420px; margin-left: auto; margin-right: auto; margin-bottom: 8px;">Click anywhere in the field to trace how a point flows toward the data distribution</div>
                        <div id="tracerModeContainer" style="display: flex; gap: 6px; justify-content: center; margin-bottom: 4px;"></div>
                    </div>

                    <div style="position: relative; width: 460px; height: 350px;">
                        <canvas id="probeCanvas" width="460" height="350" style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; cursor: crosshair;"></canvas>
                    </div>

                    <div id="readoutPanel" style="font-family: monospace; font-size: 12px; color: #2563eb; font-weight: 600; background: #eff6ff; padding: 8px 16px; border-radius: 6px; border: 1px solid #dbeafe; width: 100%; text-align: center; box-sizing: border-box;">
                        Trace Seed Source Coordinate : (${rx_start.toFixed(2)}, ${ry_start.toFixed(2)})
                    </div>
                </div>
                `;

                let canvas = el.querySelector("#probeCanvas");
                let ctx = canvas.getContext("2d");
                let readout = el.querySelector("#readoutPanel");
                let mode_container = el.querySelector("#tracerModeContainer");

                function to_canvas(x, y) {
                    let cx = ((x - (-4.0)) / 12.0) * 460;
                    let cy = 350 - (((y - (-5.0)) / 10.0) * 350);
                    return [cx, cy];
                }

                function to_math(cx, cy) {
                    let mx = -4.0 + (cx / 460) * 12.0;
                    let my = -5.0 + ((350 - cy) / 350) * 10.0;
                    return [mx, my];
                }

                function draw_base_workspace() {
                    ctx.clearRect(0, 0, 460, 350);
                    let m = active_mode;

                    if (!vfields_all || !vfields_all[m]) return;

                    // --- PURPLE TARGET POINT CLOUD ---
                    if (x_target && x_target.length > 0) {
                        ctx.fillStyle = "rgba(137, 99, 220, 0.15)";
                        for (let i = 0; i < x_target.length; i++) {
                            let [scx, scy] = to_canvas(x_target[i][0], x_target[i][1]);
                            if (0 <= scx && scx <= 460 && 0 <= scy && scy <= 350) {
                                ctx.beginPath();
                                ctx.arc(scx, scy, 2.0, 0, 2 * Math.PI);
                                ctx.fill();
                            }
                        }
                    }

                    // Compute path step sequences
                    let traj_pts = [current_probe];
                    let curr = [current_probe[0], current_probe[1]];
                    let vf = vfields_all[m];

                    // Grid dimensions in vfield
                    let steps_x = 25, steps_y = 25;
                    let min_x = -4.0, max_x = 8.0;
                    let min_y = -5.0, max_y = 5.0;

                    let dt = 0.02; 
                    for (let step = 0; step < 50; step++) {
                        let tx = curr[0], ty = curr[1];
                        if (tx < min_x || tx > max_x || ty < min_y || ty > max_y) break;

                        // Get the velocity field slice for the current time step
                        let vf_step = vf[step];
                        if (!vf_step) break;

                        // bilinear interpolation
                        let px = ((tx - min_x) / (max_x - min_x)) * (steps_x - 1);
                        let py = ((ty - min_y) / (max_y - min_y)) * (steps_y - 1);

                        let ix = Math.floor(px), iy = Math.floor(py);
                        let fx = px - ix, fy = py - iy;

                        ix = Math.max(0, Math.min(steps_x - 2, ix));
                        iy = Math.max(0, Math.min(steps_y - 2, iy));

                        // lookup indices
                        let idx00 = iy * steps_x + ix;
                        let idx10 = iy * steps_x + (ix + 1);
                        let idx01 = (iy + 1) * steps_x + ix;
                        let idx11 = (iy + 1) * steps_x + (ix + 1);

                        let v00 = vf_step[idx00], v10 = vf_step[idx10], v01 = vf_step[idx01], v11 = vf_step[idx11];
                        if (!v00 || !v10 || !v01 || !v11) break;

                        let vx = (1-fx)*(1-fy)*v00[0] + fx*(1-fy)*v10[0] + (1-fx)*fy*v01[0] + fx*fy*v11[0];
                        let vy = (1-fx)*(1-fy)*v00[1] + fx*(1-fy)*v10[1] + (1-fx)*fy*v01[1] + fx*fy*v11[1];

                        curr[0] += vx * dt;
                        curr[1] += vy * dt;
                        traj_pts.push([curr[0], curr[1]]);
                    }

                    // Render path line
                    ctx.beginPath();
                    ctx.lineWidth = 2.0;
                    ctx.strokeStyle = "#2563eb";
                    for (let i = 0; i < traj_pts.length; i++) {
                        let [tcx, tcy] = to_canvas(traj_pts[i][0], traj_pts[i][1]);
                        if (i === 0) ctx.moveTo(tcx, tcy);
                        else ctx.lineTo(tcx, tcy);
                    }
                    ctx.stroke();

                    // Render start point (probe)
                    let [pcx, pcy] = to_canvas(current_probe[0], current_probe[1]);
                    ctx.fillStyle = "#ef4444";
                    ctx.beginPath();
                    ctx.arc(pcx, pcy, 5.0, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.strokeStyle = "#ffffff";
                    ctx.lineWidth = 1.5;
                    ctx.stroke();

                    // --- BLACK TARGET LABEL ---
                    if (traj_pts.length > 0) {
                        let x_pred_last = traj_pts[traj_pts.length - 1];
                        let [s_xp, s_yp] = to_canvas(x_pred_last[0], x_pred_last[1]);
                        ctx.fillStyle = "#000000"; 
                        ctx.font = "bold 11px system-ui";
                        ctx.fillText("target", s_xp + 8, s_yp + 4);
                    }
                }

                canvas.addEventListener("click", (e) => {
                    let rect = canvas.getBoundingClientRect();
                    let click_cx = e.clientX - rect.left;
                    let click_cy = e.clientY - rect.top;
                    let [mx, my] = to_math(click_cx, click_cy);

                    current_probe = [mx, my];
                    readout.textContent = `Trace Seed Source Coordinate : (${mx.toFixed(2)}, ${my.toFixed(2)})`;
                    draw_base_workspace();
                });

                function init_modes() {
                    mode_container.innerHTML = "";
                    ["x_pred", "eps_pred", "v_pred"].forEach(m => {
                        let btn_m = document.createElement("button");
                        let displayName = m;
                        if (m === "x_pred") displayName = "x-pred";
                        else if (m === "v_pred") displayName = "v-pred";
                        else if (m === "eps_pred") displayName = "ε-pred";
                        btn_m.textContent = displayName;
                        btn_m.style.cssText = "padding: 4px 10px; font-size: 11px; font-weight:600; border-radius:4px; border:1px solid #cbd5e1; cursor:pointer; background:" + (m === active_mode ? "#2563eb; color:white; border-color:#2563eb;" : "#fff; color:#475569;");
                        btn_m.addEventListener("click", (e) => {
                            active_mode = m;
                            mode_container.querySelectorAll("button").forEach(b => {
                                b.style.background = "#fff";
                                b.style.color = "#475569";
                                b.style.borderColor = "#cbd5e1";
                            });
                            e.target.style.background = "#2563eb";
                            e.target.style.color = "white";
                            e.target.style.borderColor = "#2563eb";
                            draw_base_workspace();
                        });
                        mode_container.appendChild(btn_m);
                    });
                }

                function update() {
                    let config = JSON.parse(model.get("config_json"));
                    if (config && config.vfields) {
                        vfields_all = config.vfields;
                        x_target = config.X_TARGET || [];
                        let rx = -2.0 + Math.random() * 4.0;
                        let ry = -2.0 + Math.random() * 4.0;
                        current_probe = [rx, ry];
                        readout.textContent = `Trace Seed Source Coordinate: (${rx.toFixed(2)}, ${ry.toFixed(2)})`;
                        init_modes();
                        draw_base_workspace();
                    }
                }

                model.on("change:config_json", update);
                update();
            }
        }
        """
        config_json = traitlets.Unicode("{}").tag(sync=True)

    toy1_tracer_widget = Toy1VectorTracerWidget()
    toy1_tracer_widget.config_json = json.dumps({
        "vfields": TOY1_VELOCITY_FIELDS,
        "X_TARGET": TOY1_X_TARGET[::4].tolist()
    })
    return (toy1_tracer_widget,)


@app.cell(hide_code=True)
def _(TOY1_TRAINED_MODELS, toy1_tracer_widget):
    if TOY1_TRAINED_MODELS is not None:
        _o = toy1_tracer_widget
    else:
        _o = None
    _o
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        **Key Insight:** We were able to find that denoising can be done in multiple ways.
        """
    ).callout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which Prediction Objective is Best?

    Often, we deal with data in high dimensions — so which of these
    objectives actually leads to the best results?

    To understand this, we first need to look at how data behaves in high dimensions. Real-world, "clean" data typically possesses inherent structure—deterministic features such as the central hole in a handwritten "0" or the rigid, linear stroke of a "1." In mathematical terms, these features can be modeled as invariant quantities or functional constraints within a higher-dimensional space. This gives rise to the **Manifold Hypothesis**: the idea that high-dimensional data actually concentrates near a much lower-dimensional, embedded subspace called a **manifold**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What does a manifold actually look like?

    A manifold is just a surface that locally looks flat (Euclidean),
    even if it's curved or twisted globally. Below are three simple
    examples — a sphere, a torus, and a plane
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, plt):
    # Resolution for surfaces
    res = 60

    # Sphere (x² + y² + z² = 1)
    u = np.linspace(0, np.pi, res)
    v = np.linspace(0, 2 * np.pi, res)
    U, V = np.meshgrid(u, v)
    sphere_x = np.sin(U) * np.cos(V)
    sphere_y = np.sin(U) * np.sin(V)
    sphere_z = np.cos(U)

    # Torus ((√(x²+y²) − R)² + z² = r²)
    R, r = 1.0, 0.35
    u2 = np.linspace(0, 2 * np.pi, res)
    v2 = np.linspace(0, 2 * np.pi, res)
    U2, V2 = np.meshgrid(u2, v2)
    torus_x = (R + r * np.cos(V2)) * np.cos(U2)
    torus_y = (R + r * np.cos(V2)) * np.sin(U2)
    torus_z = r * np.sin(V2)

    # Plane (z = 0)
    p = np.linspace(-1.5, 1.5, 10)
    plane_x, plane_y = np.meshgrid(p, p)
    plane_z = np.zeros_like(plane_x)

    fig = plt.figure(figsize=(10, 3.5))

    # Sphere
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(sphere_x, sphere_y, sphere_z, cmap='Blues', edgecolor='none', rstride=1, cstride=1, antialiased=True)
    ax1.set_title("Sphere", fontsize=11, fontweight='bold', color='#0f172a', pad=0)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_zlim(-1.5, 1.5)
    ax1.set_box_aspect((1, 1, 1))
    ax1.set_axis_off()
    ax1.view_init(elev=20, azim=45)

    # Torus
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(torus_x, torus_y, torus_z, cmap='Purples', edgecolor='none', rstride=1, cstride=1, antialiased=True)
    ax2.set_title("Torus", fontsize=11, fontweight='bold', color='#0f172a', pad=0)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_zlim(-1.5, 1.5)
    ax2.set_box_aspect((1, 1, 1))
    ax2.set_axis_off()
    ax2.view_init(elev=35, azim=45)

    # Plane
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(plane_x, plane_y, plane_z, cmap='Greens_r', edgecolor='none', rstride=1, cstride=1, antialiased=True)
    ax3.set_title("Plane", fontsize=11, fontweight='bold', color='#0f172a', pad=0)
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_zlim(-1.5, 1.5)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_axis_off()
    ax3.view_init(elev=20, azim=45)

    mo.center(fig)
    return


@app.cell(hide_code=True)
def _(
    MNIST_X_RAW,
    MNIST_Y_RAW,
    anywidget,
    base64,
    io,
    json,
    np,
    plt,
    traitlets,
):
    # Isolate 5 completely distinct topological variants of the digit 7
    _sevens = np.where(MNIST_Y_RAW == 7)[0]
    _img_A = MNIST_X_RAW[_sevens[0]]         # Base "7" (Loop Start & Loop Closure point)
    _img_LeftArc = MNIST_X_RAW[_sevens[4]]   # Left arc unique data variant
    _img_B = MNIST_X_RAW[_sevens[8]]         # Bottom loop vertex extreme variant
    _img_RightArc = MNIST_X_RAW[_sevens[12]] # Right arc unique data variant
    _img_C = MNIST_X_RAW[_sevens[16]]        # "Different different 7" (Stem terminal node)

    _digit_frames = []
    _path_x, _path_y, _path_z = [], [], []

    # Manifold surface floats uniformly at Z = 1 relative to the ground plane axis
    def get_z(x, y):
        return 1.0 + 0.18 * np.sin(x) * np.cos(y) - 0.04 * x

    # ── GENERATE A GEOMETRICALLY CORRECT UPRIGHT "6" TRAJECTORY (45 TOTAL FRAMES) ──

    # Phase 1: Clockwise Circular Loop base of the "6" (Frames 0 to 24)
    cx_c, cy_c = 0.2, 0.0
    r_c = 0.45
    for i in range(25):
        phi = (i / 24.0) * 2.0 * np.pi
        theta = np.pi - phi  
        px = cx_c + r_c * np.cos(theta)
        py = cy_c + r_c * np.sin(theta)
        pz = get_z(px, py)

        if i <= 6:  
            alpha = i / 6.0
            img = (1.0 - alpha) * _img_A + alpha * _img_LeftArc
        elif i <= 12:  
            alpha = (i - 6) / 6.0
            img = (1.0 - alpha) * _img_LeftArc + alpha * _img_B
        elif i <= 18:  
            alpha = (i - 12) / 6.0
            img = (1.0 - alpha) * _img_B + alpha * _img_RightArc
        else:  
            alpha = (i - 18) / 6.0
            img = (1.0 - alpha) * _img_RightArc + alpha * _img_A

        _digit_frames.append(np.clip((img + 1.0) / 2.0, 0.0, 1.0).tolist())
        _path_x.append(float(px))
        _path_y.append(float(py))
        _path_z.append(float(pz))

    # Phase 2: Mathematically Perfect Linear Tangent Stem of the "6" (Frames 25 to 34)
    start_x, start_y = -0.25, 0.0
    end_x, end_y = -0.25, 0.8
    for i in range(1, 11):
        alpha = i / 10.0
        px = (1.0 - alpha) * start_x + alpha * end_x
        py = (1.0 - alpha) * start_y + alpha * end_y
        pz = get_z(px, py)

        img = (1.0 - alpha) * _img_A + alpha * _img_C

        _digit_frames.append(np.clip((img + 1.0) / 2.0, 0.0, 1.0).tolist())
        _path_x.append(float(px))
        _path_y.append(float(py))
        _path_z.append(float(pz))

    # Phase 3: Vertical Extrapolation Lift-Off (Frames 35 to 44)
    base_z = get_z(end_x, end_y)
    for i in range(1, 11):
        alpha = i / 10.0
        px = end_x
        py = end_y
        pz = base_z + alpha * 1.0

        noise = np.random.RandomState(42 + i).randn(*_img_C.shape) * (alpha * 0.42)
        img = _img_C + noise

        _digit_frames.append(np.clip((img + 1.0) / 2.0, 0.0, 1.0).tolist())
        _path_x.append(float(px))
        _path_y.append(float(py))
        _path_z.append(float(pz))

    # Build 3D Manifold Wireframe Grid to match axis boundary limits
    _X_m = np.linspace(-1.5, 1.5, 15)
    _Y_m = np.linspace(-1.5, 1.5, 15)
    _X_grid, _Y_grid = np.meshgrid(_X_m, _Y_m)
    _Z_grid = get_z(_X_grid, _Y_grid)

    # ── Precompute Matplotlib 3D Image Frames for high fidelity rendering ──
    _plot_frames = []
    for _s in range(45):
        _fig = plt.figure(figsize=(4.5, 3.5), dpi=95)
        _ax = _fig.add_subplot(111, projection='3d')

        # Draw wireframe manifold sheet
        _ax.plot_surface(_X_grid, _Y_grid, _Z_grid, color='#4f46e5', alpha=0.08, edgecolor='none')
        _ax.plot(_X_grid[0,:], _Y_grid[0,:], _Z_grid[0,:], 'k--', alpha=0.15)
        _ax.plot(_X_grid[-1,:], _Y_grid[-1,:], _Z_grid[-1,:], 'k--', alpha=0.15)
        _ax.plot(_X_grid[:,0], _Y_grid[:,0], _Z_grid[:,0], 'k--', alpha=0.15)
        _ax.plot(_X_grid[:,-1], _Y_grid[:,-1], _Z_grid[:,-1], 'k--', alpha=0.15)

        # Draw trajectory paths
        _ax.plot(_path_x[:_s+1], _path_y[:_s+1], _path_z[:_s+1], color='#1e1b4b', linewidth=2.2)

        # Color state logic for node tracker dot elements
        _dot_color = "#dc2626" if _s > 34 else ("#1e40af" if _s == 0 or _s == 24 else "#059669")
        _ax.scatter([_path_x[_s]], [_path_y[_s]], [_path_z[_s]], color=_dot_color, s=50, zorder=12, edgecolor='#ffffff', linewidth=1)

        # Scale limits configuration
        _ax.set_xlim(-1.5, 1.5)
        _ax.set_ylim(-1.5, 1.5)
        _ax.set_zlim(0.5, 2.5)

        # Transparent Pane style configuration alignment
        _ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        _ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        _ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Synchronize Camera Angle Matrix exactly to user specification reference
        _ax.view_init(elev=25, azim=-60)
        _fig.tight_layout()

        _buf = io.BytesIO()
        _fig.savefig(_buf, format='png', bbox_inches='tight', transparent=True)
        plt.close(_fig)
        _buf.seek(0)
        _plot_frames.append("data:image/png;base64," + base64.b64encode(_buf.read()).decode('utf-8'))

    class ManifoldHypothesisWidget(anywidget.AnyWidget):
        _esm = """
        export default {
            render({ model, el }) {
                let plots = [], digits = [];
                let state = { playing: false, interval_id: null };

                el.innerHTML = `
                <div style="display: flex; flex-direction: column; align-items: center; gap: 16px; font-family: system-ui, sans-serif; background: #f8fafc; padding: 15px; border-radius: 12px; border: 1px solid #e2e8f0; max-width: 740px; margin: 20px auto; box-shadow: 0 4px 10px rgb(0 0 0 / 0.05);">
                    <div style="text-align: center; width: 100%;">
                        <div style="font-size: 15px; font-weight: 700; color: #0f172a; margin-bottom: 2px;">Representation of Data Manifold as a 2D surface in 3D space</div>
                        <div id="statusLabel" style="font-size: 12px; font-weight: 600; margin-bottom: 8px; min-height: 18px;">Initializing view fields...</div>
                    </div>

                    <div style="display: flex; gap: 20px; align-items: center; justify-content: center; width: 100%; flex-wrap: wrap;">
                        <div style="background: #f8fafc; border-radius: 8px; padding: 4px; border: 1px solid #e2e8f0;">
                            <img id="manifoldImg" style="width: 380px; height: auto; display: block;" />
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 11px; font-weight: 700; color: #64748b; margin-bottom: 6px;">Latent Representation</div>
                            <canvas id="manifoldDigitCanvas" width="168" height="168" style="background: #000; border-radius: 8px; border: 1px solid #cbd5e1;"></canvas>
                        </div>
                    </div>

                    <div style="display: flex; align-items: center; gap: 12px; width: 100%; background: #f1f5f9; padding: 8px 16px; border-radius: 8px; box-sizing: border-box;">
                        <button id="manPlayBtn" style="background: #f1f5f9; border: 1px solid #2563eb; border-radius: 6px; padding: 6px 14px; font-size: 13px; font-weight: 500; color: #2563eb; cursor: pointer; min-width: 70px;">▶ play</button>
                        <input type="range" id="manSlider" min="0" max="44" step="1" value="0" style="flex-grow: 1; accent-color: #2563eb; cursor: pointer; margin: 0;">
                        <span id="manLabel" style="font-family: monospace; font-size: 13px; color: #475569; font-weight: 600; min-width: 85px; text-align: right;">Step 0</span>
                </div>
                </div>
                `;

                let img_el = el.querySelector("#manifoldImg");
                let canvasDigit = el.querySelector("#manifoldDigitCanvas");
                let ctxDigit = canvasDigit.getContext("2d");
                let slider = el.querySelector("#manSlider");
                let btn = el.querySelector("#manPlayBtn");
                let lbl = el.querySelector("#manLabel");
                let status_lbl = el.querySelector("#statusLabel");

                function draw_frame(idx) {
                    if (!plots || plots.length === 0) return;
                    lbl.textContent = ``;
                    img_el.src = plots[idx];

                    if (idx === 0) {
                        status_lbl.textContent = "On Manifold";
                        status_lbl.style.color = "#1e40af";
                    } else if (idx < 12) {
                        status_lbl.textContent = "On Manifold";
                        status_lbl.style.color = "#059669";
                    } else if (idx === 12) {
                        status_lbl.textContent = "On Manifold";
                        status_lbl.style.color = "#b45309";
                    } else if (idx < 24) {
                        status_lbl.textContent = "On Manifold";
                        status_lbl.style.color = "#0891b2";
                    } else if (idx === 24) {
                        status_lbl.textContent = "On Manifold";
                        status_lbl.style.color = "#1e40af";
                    } else if (idx <= 34) {
                        status_lbl.textContent = "On Manifold";
                        status_lbl.style.color = "#4f46e5";
                    } else {
                        status_lbl.textContent = "Off Manifold ";
                        status_lbl.style.color = "#dc2626";
                    }

                    ctxDigit.clearRect(0, 0, 168, 168);
                    for (let r = 0; r < 28; r++) {
                        for (let c = 0; c < 28; c++) {
                            let v = Math.floor(digits[idx][r][c] * 255);
                            ctxDigit.fillStyle = `rgb(${v},${v},${v})`;
                            ctxDigit.fillRect(c * 6, r * 6, 6, 6);
                        }
                    }
                }

                function tick() {
                    let step = parseInt(slider.value) + 1;
                    if (step > 44) step = 0;
                    slider.value = String(step);
                    draw_frame(step);
                }

                btn.addEventListener("click", () => {
                    if (state.playing) {
                        clearInterval(state.interval_id);
                        state.playing = false;
                        btn.textContent = "▶ play";
                    } else {
                        state.interval_id = setInterval(tick, 240);
                        state.playing = true;
                        btn.textContent = "⏸ pause";
                    }
                });

                slider.addEventListener("input", () => {
                    if (state.playing) {
                        clearInterval(state.interval_id);
                        state.playing = false;
                        btn.textContent = "▶ play";
                    }
                    draw_frame(parseInt(slider.value));
                });

                function update() {
                    let config = JSON.parse(model.get("config_json"));
                    if (config && config.plots) {
                        plots = config.plots;
                        digits = config.digits;
                        draw_frame(parseInt(slider.value));
                    }
                }

                model.on("change:config_json", update);
                update();
            }
        }
        """
        config_json = traitlets.Unicode("{}").tag(sync=True)
        current_idx = traitlets.Int(0).tag(sync=True)

    manifold_widget = ManifoldHypothesisWidget()
    manifold_widget.config_json = json.dumps({
        "plots": _plot_frames,
        "digits": _digit_frames
    })
    return (manifold_widget,)


@app.cell(hide_code=True)
def _(manifold_widget):
    manifold_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's run a toy experiment, looking into how different denoising
    techniques scale with high dimensional data with reference to the
    manifold hypothesis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Setup: Embedding Low-Dimensional Data in High Dimensions

    To test the manifold hypothesis experimentally, we embed $d=2$
    dimensional data into progressively higher ambient dimensions $D$.
    The following is the setup:

    1. Generate structured 2D data (a spiral) — this is our "clean data manifold"
    2. Construct a random orthogonal projection matrix $P \in \mathbb{R}^{D \times d}$ satisfying $I = P^T P$
    3. Embed the data via $\hat{x} = Px$, lifting it into $\mathbb{R}^D$
    4. Train all three prediction objectives ($x$-pred, $v$-pred, $\epsilon$-pred) at each dimension $D$

    As $D$ grows far beyond $d=2$, we can directly observe which
    objective handles the increasing gap between data dimensionality
    and ambient dimensionality.
    """)
    return


@app.cell(hide_code=True)
def toy2_controls(mo):
    toy2_form = (
        mo.md(
            """
            ### Toy 2 Configuration controls
            {n_epochs}

            {hidden_dim}

            {n_layers}

            {batch_size}

            {lr}
            """
        )
        .batch(
            n_epochs=mo.ui.slider(100, 500, step=10, value=300, label="Epochs", show_value=True),
            hidden_dim=mo.ui.slider(64, 512, step=64, value=256, label="Hidden Dimension", show_value=True),
            n_layers=mo.ui.slider(2, 10, step=1, value=5, label="Hidden Layers", show_value=True),
            batch_size=mo.ui.slider(64, 1024, step=64, value=512, label="Batch Size", show_value=True),
            lr=mo.ui.dropdown(options=["0.0001", "0.0005", "0.001", "0.005"], value="0.001", label="Learning Rate"),
        )
        .form(submit_button_label="Run Experiment 2")
    )

    mo.vstack([
        mo.md("## Toy 2: $x$-pred vs. $v$-pred vs. $\\epsilon$-pred as ambient dimension D grows"),
        toy2_form
    ])
    return (toy2_form,)


@app.cell(hide_code=True)
def toy2_config_setup(jax, toy2_form):
    _val = toy2_form.value or {}
    toy2_config = {
        "d_values": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        "modes": ['x_pred', 'eps_pred', 'v_pred'],
        "d_latent": 2,
        "hidden_dim": int(_val.get("hidden_dim", 256)),
        "n_layers": int(_val.get("n_layers", 5)),
        "n_samples": 20000,
        "batch_size": int(_val.get("batch_size", 512)),
        "n_epochs": int(_val.get("n_epochs", 300)),
        "lr": float(_val.get("lr", 1e-3)),
        "based_key": jax.random.PRNGKey(3407)
    }
    return (toy2_config,)


@app.cell(hide_code=True)
def _(jnp, np, toy2_config):
    def sample_latent_data(n):
        _rng = np.random.RandomState(42)
        _t = 2 * np.pi * (2 + 1.5 * _rng.rand(n)) 
        _x = _t * np.cos(_t) + 0.125 * _rng.randn(n)
        _z = _t * np.sin(_t) + 0.125 * _rng.randn(n)
        _X = np.stack([_x, _z], axis=1).astype(np.float32)
        _X = 2.0 * (_X - _X.min(axis=0)) / (_X.max(axis=0) - _X.min(axis=0)) - 1.0
        return jnp.array(_X)

    def create_projection(d, D_val, seed=42):
        _rng = np.random.RandomState(seed)
        _P = _rng.randn(D_val, d).astype(np.float32)
        _Q, _ = np.linalg.qr(_P, mode='reduced')
        return jnp.array(_Q)

    TOY2_X_LATENT = sample_latent_data(toy2_config["n_samples"])
    TOY2_PROJECTIONS = {D_val: create_projection(toy2_config["d_latent"], D_val) for D_val in toy2_config["d_values"]}
    return TOY2_PROJECTIONS, TOY2_X_LATENT


@app.cell(hide_code=True)
def _(TOY2_X_LATENT, mo, np):
    import plotly.express as plox

    _data = np.array(TOY2_X_LATENT)

    # Create an interactive scatter plot using Plotly Express
    _fig = plox.scatter(
        x=_data[:, 0],
        y=_data[:, 1],
        template="plotly_white",
        labels={"x": "x", "y": "y"}
    )
    _fig.update_traces(
        marker=dict(
            size=4,
            opacity=0.7,
            color="#2563eb"
        )
    )
    _fig.update_yaxes(scaleanchor="x", scaleratio=1)
    _fig.update_layout(
        width=400,
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    _html = mo.Html(
        f"""
        <div style="
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            font-family: system-ui, -apple-system, sans-serif; 
            background: #f8fafc; 
            padding: 24px; 
            border-radius: 12px; 
            border: 1px solid #e2e8f0; 
            max-width: 520px; 
            margin: 0 auto; 
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        ">
            <div style="text-align: center; width: 100%; margin-bottom: 16px;">
                <div style="font-size: 15px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
                    Our Target Latent Distribution (d = 2)
                </div>
            </div>

            <div style="background: #ffffff; padding: 4px; border-radius: 8px; border: 1px solid #e2e8f0; display: inline-block;">
                {mo.as_html(_fig)}
            </div>
        </div>
        """
    )
    _html
    return


@app.cell(hide_code=True)
def _(eqx, jax, jnp):
    class Toy2MLP(eqx.Module):
        layers: list
        dropout_p: float

        def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout_p=0.1, *, key):
            _keys = jax.random.split(key, n_layers)
            _layers = [eqx.nn.Linear(input_dim, hidden_dim, key=_keys[0])]
            for i in range(n_layers - 2):
                _layers.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=_keys[i + 1]))
            _layers.append(eqx.nn.Linear(hidden_dim, output_dim, key=_keys[-1]))
            self.layers = _layers
            self.dropout_p = dropout_p

        def __call__(self, z, t, key):
            _t_arr = jnp.atleast_1d(t)
            _h = jnp.concatenate([z, _t_arr], axis=-1)
            for _layer in self.layers[:-1]:
                _h = jax.nn.relu(_layer(_h))
                key, _subkey = jax.random.split(key)
                _mask = jax.random.bernoulli(_subkey, 1.0 - self.dropout_p, _h.shape)
                _h = _h * _mask / (1.0 - self.dropout_p)
            return self.layers[-1](_h)

        def infer(self, z, t):
            _t_arr = jnp.atleast_1d(t)
            _h = jnp.concatenate([z, _t_arr], axis=-1)
            for _layer in self.layers[:-1]:
                _h = jax.nn.relu(_layer(_h))
            return self.layers[-1](_h)

    return (Toy2MLP,)


@app.cell(hide_code=True)
def _(
    TOY2_PROJECTIONS,
    TOY2_X_LATENT,
    Toy2MLP,
    eqx,
    jax,
    jnp,
    mo,
    np,
    optax,
    toy2_config,
    toy2_form,
):
    max_d = max(toy2_config["d_values"])
    optimizer = optax.adam(learning_rate=toy2_config["lr"], b1=0.9, b2=0.999)

    def single_model_train_step(arrays, static, opt_state, x_batch, t_batch, active_dim, mode_idx, key):
        def loss_fn(arrs):
            model = eqx.combine(arrs, static)
            noise_key, dropout_key = jax.random.split(key)
            dim_mask = jnp.arange(max_d) < active_dim
            eps = jax.random.normal(noise_key, x_batch.shape) * dim_mask
            t_exp = t_batch[:, None]
            z = t_exp * x_batch + (1.0 - t_exp) * eps
            v_true = x_batch - eps

            batch_keys = jax.vmap(lambda i: jax.random.fold_in(dropout_key, i))(jnp.arange(x_batch.shape[0]))
            out = jax.vmap(model)(z, t_batch, batch_keys) * dim_mask

            v0 = (out - z) / jnp.clip(1.0 - t_exp, 0.1)
            v1 = (z - out) / jnp.clip(t_exp, 0.1)
            v2 = out

            v_pred = jax.lax.switch(mode_idx, [lambda _: v0, lambda _: v1, lambda _: v2], None) * dim_mask
            se = jnp.sum((v_pred - v_true)**2, axis=-1)
            return jnp.mean(se / active_dim)

        loss, grads = jax.value_and_grad(loss_fn)(arrays)
        updates, new_opt_state = optimizer.update(grads, opt_state, params=arrays)
        new_arrays = eqx.apply_updates(arrays, updates)
        return new_arrays, new_opt_state, loss

    @eqx.filter_jit
    def train_epoch_sweep(mega_arrays, static, mega_opt_states, mega_x_obs, mega_active_dims, mega_mode_indices, carry_key, steps_per_epoch):
        def scan_body(carry, _step_idx):
            arrs, os, loop_key = carry
            loop_key, idx_key, t_key, step_key = jax.random.split(loop_key, 4)
            indices = jax.random.randint(idx_key, (toy2_config["batch_size"],), 0, toy2_config["n_samples"])
            t_batch = jax.random.uniform(t_key, (toy2_config["batch_size"],))
            step_keys = jax.random.split(step_key, mega_active_dims.shape[0])

            def mapped_update(a, o, x_data, act_dim, m_idx, skey):
                return single_model_train_step(a, static, o, x_data[indices], t_batch, act_dim, m_idx, skey)

            new_arrs, new_os, losses = jax.vmap(mapped_update)(
                arrs, os, mega_x_obs, mega_active_dims, mega_mode_indices, step_keys
            )
            return (new_arrs, new_os, loop_key), losses

        final_carry, losses_history = jax.lax.scan(scan_body, (mega_arrays, mega_opt_states, carry_key), jnp.arange(steps_per_epoch))
        return final_carry[0], final_carry[1], final_carry[2], losses_history

    @eqx.filter_jit
    def generate_mega(mega_arrays, static, mega_active_dims, mega_mode_indices, key):
        def generate_single(arrs, active_dim, mode_idx, skey):
            model = eqx.combine(arrs, static)
            dim_mask = jnp.arange(max_d) < active_dim
            z0 = jax.random.normal(skey, (toy2_config["n_samples"], max_d)) * dim_mask
            t_vals = jnp.linspace(0.0, 1.0, 50)

            def velocity_fn(z, t):
                out = jax.vmap(lambda zi, ti: model.infer(zi, ti))(z, jnp.full((toy2_config["n_samples"],), t))
                out = out * dim_mask
                v0 = (out - z) / jnp.clip(1.0 - t, 0.1)
                v1 = (z - out) / jnp.clip(t, 0.1)
                v2 = out
                return jax.lax.switch(mode_idx, [lambda _: v0, lambda _: v1, lambda _: v2], None) * dim_mask

            def ode_step(i, z):
                t_val = t_vals[i]
                t_next = t_vals[i + 1]
                h = t_next - t_val
                k1 = velocity_fn(z, t_val)
                k2 = velocity_fn(z + 0.5 * h * k1, t_val + 0.5 * h)
                k3 = velocity_fn(z + 0.5 * h * k2, t_val + 0.5 * h)
                k4 = velocity_fn(z + h * k3, t_next)
                return z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            x_gen = jax.lax.fori_loop(0, t_vals.shape[0] - 1, ode_step, z0)
            return x_gen

        mega_keys = jax.random.split(key, mega_active_dims.shape[0])
        return jax.vmap(generate_single)(mega_arrays, mega_active_dims, mega_mode_indices, mega_keys)

    if toy2_form.value is not None:
        # ── Execute Kernel Sweeps ──
        ref_model = Toy2MLP(input_dim=max_d + 1, hidden_dim=toy2_config["hidden_dim"], n_layers=toy2_config["n_layers"], output_dim=max_d, key=jax.random.PRNGKey(0))
        _, STATIC_GRAPH = eqx.partition(ref_model, eqx.is_inexact_array)

        mode_int_map = {'x_pred': 0, 'eps_pred': 1, 'v_pred': 2}
        mega_x_obs, mega_active_dims, mega_mode_indices, CONFIGS = [], [], [], []

        for D_val in toy2_config["d_values"]:
            x_obs = jnp.dot(TOY2_X_LATENT, TOY2_PROJECTIONS[D_val].T)
            padded_x = jnp.pad(x_obs, ((0, 0), (0, max_d - D_val)), mode='constant')
            for mode in toy2_config["modes"]:
                CONFIGS.append((mode, D_val))
                mega_x_obs.append(padded_x)
                mega_active_dims.append(D_val)
                mega_mode_indices.append(mode_int_map[mode])

        MEGA_X_OBS = jnp.stack(mega_x_obs)
        MEGA_ACTIVE_DIMS = jnp.array(mega_active_dims)
        MEGA_MODE_INDICES = jnp.array(mega_mode_indices, dtype=jnp.int32)

        init_key, train_key, gen_key = jax.random.split(toy2_config["based_key"], 3)
        init_keys = jax.random.split(init_key, len(CONFIGS))

        def init_fn(k):
            m = Toy2MLP(input_dim=max_d + 1, hidden_dim=toy2_config["hidden_dim"], n_layers=toy2_config["n_layers"], output_dim=max_d, key=k)
            a, _ = eqx.partition(m, eqx.is_inexact_array)
            return a, optimizer.init(a)

        MEGA_ARRAYS, MEGA_OPT_STATES = jax.vmap(init_fn)(init_keys)

        steps_per_epoch = toy2_config["n_samples"] // toy2_config["batch_size"]
        current_carry_key = train_key
        current_arrays = MEGA_ARRAYS
        current_opt_states = MEGA_OPT_STATES
        all_losses_list = []

        for ep in mo.status.progress_bar(
            range(toy2_config["n_epochs"]),
            title="Toy 2 Experiment training",
            show_eta=True,
            show_rate=True,
        ):
            current_arrays, current_opt_states, current_carry_key, epoch_losses = train_epoch_sweep(
                current_arrays, STATIC_GRAPH, current_opt_states, MEGA_X_OBS, MEGA_ACTIVE_DIMS, MEGA_MODE_INDICES, current_carry_key, steps_per_epoch
            )
            all_losses_list.append(epoch_losses)

        FINAL_ARRAYS = current_arrays
        ALL_LOSSES = jnp.concatenate(all_losses_list, axis=0)
        X_GEN_MEGA = generate_mega(FINAL_ARRAYS, STATIC_GRAPH, MEGA_ACTIVE_DIMS, MEGA_MODE_INDICES, gen_key)

        # ── Packing downsampled profiles for JSON UI transmission ──
        TOY2_PACKED_SAMPLES = {}
        TOY2_PACKED_LOSSES = {}
        all_losses_np = np.array(ALL_LOSSES).T

        for _i, (mode, D_val) in enumerate(CONFIGS):
            epoch_losses = all_losses_np[_i].reshape(toy2_config["n_epochs"], steps_per_epoch).mean(axis=1)
            TOY2_PACKED_LOSSES[f"{mode}_{D_val}"] = epoch_losses.tolist()

            x_gen_padded = X_GEN_MEGA[_i]
            x_gen_actual = x_gen_padded[:, :D_val]
            x_gen_2d = np.array(jnp.dot(x_gen_actual, TOY2_PROJECTIONS[D_val]))
            TOY2_PACKED_SAMPLES[f"{mode}_{D_val}"] = x_gen_2d[::16].tolist()

            TOY2_PACKED_LATENT = np.array(TOY2_X_LATENT)[::16].tolist()

            TOY2_TRAINED_MODELS = FINAL_ARRAYS

            jax.clear_caches()
    else:
        TOY2_PACKED_LATENT = []
        TOY2_PACKED_LOSSES = {}
        TOY2_PACKED_SAMPLES = {}
        TOY2_TRAINED_MODELS = None
    return (
        TOY2_PACKED_LATENT,
        TOY2_PACKED_LOSSES,
        TOY2_PACKED_SAMPLES,
        TOY2_TRAINED_MODELS,
    )


@app.cell(hide_code=True)
def _(
    TOY2_PACKED_LATENT,
    TOY2_PACKED_SAMPLES,
    anywidget,
    json,
    toy2_config,
    traitlets,
):
    class Toy2DimensionSweepWidget(anywidget.AnyWidget):
        _esm = """
        export default {
            render({ model, el }) {
                let d_vals = [], samples_all = {}, true_data = [];

                el.innerHTML = `
                <div style="display: flex; flex-direction: column; align-items: center; gap: 20px; font-family: system-ui, sans-serif; background: #f8fafc; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; max-width: 1000px; margin: 0 auto; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);">
                    <div style="width: 100%; border-bottom: 1px solid #e2e8f0; padding-bottom: 12px; text-align: center;">
                        <div style="font-size: 16px; font-weight: 700; color: #0f172a;">x-pred vs. v-pred vs. ε-pred across ambient dimension D</div>
                        <div style="font-size: 12px; color: #64748b; margin-top: 4px;">Compare structural convergence of prediction targets across dimension (D)</div>
                    </div>

                    <div style="display: flex; align-items: center; gap: 14px; width: 100%; max-width: 550px; background: #f1f5f9; padding: 10px 16px; border-radius: 8px; box-sizing: border-box; justify-content: center;">
                        <span style="font-size: 13px; font-weight: 700; color: #475569; min-width: 110px;">Ambient dimension D:</span>
                        <input type="range" id="widgetDimSlider" min="0" max="10" step="1" value="0" style="flex-grow: 1; accent-color: #2563eb; cursor: pointer; margin: 0;">
                        <span id="dimLabel" style="font-family: monospace; font-size: 14px; color: #2563eb; font-weight: 700; min-width: 60px; text-align: right;">D = 2</span>
                    </div>

                    <div style="display: flex; flex-direction: row; gap: 12px; width: 100%; justify-content: center; overflow-x: auto; padding-bottom: 6px;">
                        <div class="canvas-card" style="text-align:center; background:#f8fafc; padding:8px; border-radius:8px; border:1px solid #e2e8f0;">
                            <div style="font-size:11px; font-weight:700; color:#475569; margin-bottom:4px;">True Latent Target</div>
                            <canvas id="canvas_true" width="200" height="200" style="background:#ffffff; border-radius:4px; border:1px solid #cbd5e1;"></canvas>
                        </div>
                        <div class="canvas-card" style="text-align:center; background:#f8fafc; padding:8px; border-radius:8px; border:1px solid #e2e8f0;">
                            <div style="font-size:11px; font-weight:700; color:#475569; margin-bottom:4px;">x-pred</div>
                            <canvas id="canvas_x" width="200" height="200" style="background:#ffffff; border-radius:4px; border:1px solid #cbd5e1;"></canvas>
                        </div>
                        <div class="canvas-card" style="text-align:center; background:#f8fafc; padding:8px; border-radius:8px; border:1px solid #e2e8f0;">
                            <div style="font-size:11px; font-weight:700; color:#475569; margin-bottom:4px;">ε-pred</div>
                            <canvas id="canvas_eps" width="200" height="200" style="background:#ffffff; border-radius:4px; border:1px solid #cbd5e1;"></canvas>
                        </div>
                        <div class="canvas-card" style="text-align:center; background:#f8fafc; padding:8px; border-radius:8px; border:1px solid #e2e8f0;">
                            <div style="font-size:11px; font-weight:700; color:#475569; margin-bottom:4px;">v-pred</div>
                            <canvas id="canvas_v" width="200" height="200" style="background:#ffffff; border-radius:4px; border:1px solid #cbd5e1;"></canvas>
                        </div>
                    </div>
                </div>
                `;

                let slider = el.querySelector("#widgetDimSlider");
                let d_lbl = el.querySelector("#dimLabel");

                let ctx_true = el.querySelector("#canvas_true").getContext("2d");
                let ctx_x = el.querySelector("#canvas_x").getContext("2d");
                let ctx_eps = el.querySelector("#canvas_eps").getContext("2d");
                let ctx_v = el.querySelector("#canvas_v").getContext("2d");

                function map_scat(x, y) {
                    let cx = 10 + ((x - (-2.0)) / 4.0) * 200;
                    let cy = 10 + (200 - ((y - (-2.0)) / 4.0) * 200);
                    return [cx, cy];
                }

                function draw_scatter(ctx, pts, color) {
                    ctx.clearRect(0, 0, 200, 200);
                    ctx.fillStyle = color;
                    if (!pts) return;
                    for (let i = 0; i < pts.length; i++) {
                        let [cx, cy] = map_scat(pts[i][0], pts[i][1]);
                        if (cx < 0 || cx > 200 || cy < 0 || cy > 200) continue;
                        ctx.beginPath();
                        ctx.arc(cx, cy, 1.3, 0, 2 * Math.PI);
                        ctx.fill();
                    }
                }

                function update_dashboard(d_idx) {
                    if (d_vals.length === 0) return;
                    let D_val = d_vals[d_idx];
                    d_lbl.textContent = `D = ${D_val}`;

                    draw_scatter(ctx_true, true_data, "rgba(139, 92, 246, 0.45)");
                    draw_scatter(ctx_x, samples_all[`x_pred_${D_val}`], "rgba(37, 99, 235, 0.45)");
                    draw_scatter(ctx_eps, samples_all[`eps_pred_${D_val}`], "rgba(249, 115, 22, 0.45)");
                    draw_scatter(ctx_v, samples_all[`v_pred_${D_val}`], "rgba(34, 197, 94, 0.45)");
                }

                slider.addEventListener("input", (e) => {
                    let idx = parseInt(e.target.value);
                    model.set("current_idx", idx);
                    model.save_changes();
                    update_dashboard(idx);
                });

                function update() {
                    let cfg = JSON.parse(model.get("config_json"));
                    if (cfg && cfg.d_values) {
                        d_vals = cfg.d_values;
                        samples_all = cfg.samples;
                        true_data = cfg.true_data;
                        let c_idx = model.get("current_idx");
                        slider.value = String(c_idx);
                        update_dashboard(c_idx);
                    }
                }

                model.on("change:config_json", update);
                update();
            }
        }
        """
        config_json = traitlets.Unicode("{}").tag(sync=True)
        current_idx = traitlets.Int(0).tag(sync=True)

    toy2_sweep_widget = Toy2DimensionSweepWidget()
    toy2_sweep_widget.config_json = json.dumps({
        "d_values": toy2_config["d_values"],
        "samples": TOY2_PACKED_SAMPLES,
        "true_data": TOY2_PACKED_LATENT
    })
    return (toy2_sweep_widget,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Below you can see how different ambient dimension converge to the data manifold
    """)
    return


@app.cell(hide_code=True)
def _(TOY2_TRAINED_MODELS, toy2_sweep_widget):
    if TOY2_TRAINED_MODELS is not None:
        _o = toy2_sweep_widget
    else: 
        _o = None
    _o
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Here we can see how training stability varies with the increase in ambient dimension
    """)
    return


@app.cell(hide_code=True)
def _(TOY2_PACKED_LOSSES, TOY2_TRAINED_MODELS, mo, plt, toy2_config):
    if TOY2_TRAINED_MODELS is not None:
        _d_values = toy2_config["d_values"]
        _fig3, _ax3 = plt.subplots(figsize=(8, 5))

        _mode_colors = {
            'x_pred':   '#2563eb',
            'v_pred':   '#22c55e',
            'eps_pred': '#f97316',
        }
        _mode_markers = {
            'x_pred':   'o',
            'v_pred':   '^',
            'eps_pred': 's',
        }

        for _mode in ["x_pred", "v_pred"]:
            _final_losses = []
            for _d_val in _d_values:
                _curve = TOY2_PACKED_LOSSES[f"{_mode}_{_d_val}"]
                _final_losses.append(_curve[-1])

            _ax3.plot(
                _d_values,
                _final_losses,
                marker=_mode_markers[_mode],
                color=_mode_colors[_mode],
                linewidth=1.8,
                label=f"{_mode}"
            )

        _ax3.set_title("Final Loss vs Ambient Dimension (D)", fontsize=12, pad=10)
        _ax3.set_xlabel("Dimension D (Log Scale)")
        _ax3.set_ylabel("Final Loss")

        # Use log scale for clear visualization across 2 to 2048
        _ax3.set_xscale("log")
        _ax3.set_xticks(_d_values)
        _ax3.set_xticklabels([str(_d) for _d in _d_values], rotation=45)

        _ax3.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        _ax3.legend(fontsize=10, loc="best")

        _fig3.tight_layout()
        _fig3_centered = mo.center(_fig3)
    else:
        _fig3_centered = None
    _fig3_centered
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analysis via the Manifold Hypothesis

    It can be observed that the **$x$-pred** loss remains consistent in
    learning the data manifold at higher ambient dimensions, while
    **v-pred** dominates where $d \approx D$, whereas
    **$\epsilon$-pred** is inferior to both.

    As noise $\epsilon \sim \mathcal{N}(0, I)$ spans over nearly all of
    the ambient space $\mathbb{R}^D$, while structured clean data does
    not:

    | Prediction Objective | Target | Dimensionality | Manifold Status |
    | --- | --- | --- | --- |
    | **$x$-pred** (Data prediction) | $x$ | $d$-dimension (Lower dimension) | **On-manifold** |
    | **$v$-pred** (Velocity prediction) | $x - \epsilon$ | $D$-dimension (Ambient space) | **Off-manifold** |
    | **$\epsilon$-pred** (Noise prediction) | $\epsilon$ | $D$-dimension (Ambient space) | **Off-manifold** |

    *(where $d \ll D$)*

    Because $v$-pred and $\epsilon$-pred are forced to learn noise
    targets that span the entire ambient space, they struggle to
    navigate high-dimensional spaces cleanly. Meanwhile, $x$-pred
    directly focuses on learning the much lower-dimensional "clean"
    data manifold.

    Here's that same "on-manifold vs. off-manifold" argument made
    concrete, geometrically — clean data sitting **on** the manifold
    surface, with its noisy corruptions floating **off** of it:
    """)
    return


@app.cell(hide_code=True)
def _(MNIST_X_RAW, MNIST_Y_RAW, np):
    # Grab one real MNIST "3" and derive a clean / noisy pair purely for
    # illustrating the input -> x-pred / v-pred relationship on the manifold.
    _digit_idx = int(np.where(MNIST_Y_RAW == 3)[0][0])
    _clean_01 = np.clip((MNIST_X_RAW[_digit_idx] * 0.5) + 0.5, 0.0, 1.0)

    _rng = np.random.RandomState(7)
    _input_01 = np.clip(_clean_01 + _rng.normal(0, 0.18, size=_clean_01.shape), 0.0, 1.0)
    _noisy_01 = np.clip(_rng.normal(0, 1.0, size=_clean_01.shape) * 0.5 + _clean_01 * 0.5, 0.0, 1.0)

    CONCEPT_DIGIT_INPUT = _input_01.tolist()
    CONCEPT_DIGIT_X = _clean_01.tolist()
    CONCEPT_DIGIT_V = _noisy_01.tolist()
    return CONCEPT_DIGIT_INPUT, CONCEPT_DIGIT_V, CONCEPT_DIGIT_X


@app.cell(hide_code=True)
def _(
    CONCEPT_DIGIT_INPUT,
    CONCEPT_DIGIT_V,
    CONCEPT_DIGIT_X,
    anywidget,
    json,
    traitlets,
):
    class ManifoldConceptWidget(anywidget.AnyWidget):
        _esm = """
        export default {
            render({ model, el }) {
                el.innerHTML = `
                <div style="position: relative; width: 100%; max-width: 720px; margin: 0 auto; font-family: system-ui, sans-serif; background: #f8fafc; border-radius: 12px; border: 1px solid #cbd5e1; overflow: hidden;">
                    <canvas id="conceptCanvas" width="700" height="460" style="display:block; width:100%; height:auto;"></canvas>
                </div>
                `;

                let canvas = el.querySelector("#conceptCanvas");
                let ctx = canvas.getContext("2d");
                let W = canvas.width, H = canvas.height;

                let digits = { input: null, x: null, v: null };

                function update() {
                    let cfg = JSON.parse(model.get("digits_json"));
                    digits.input = cfg.input;
                    digits.x = cfg.x;
                    digits.v = cfg.v;
                    render();
                }

                // ── fixed camera (no drag controls, no hover reveal) ─
                const rotY = -0.55, rotX = 0.0;

                // ── 3D projection helpers ───────────────────────────
                let cx = W / 2, cy = H / 2 + 40, baseScale = 46, dist = 9;

                function project(x, y, z) {
                    let cosY = Math.cos(rotY), sinY = Math.sin(rotY);
                    let x1 = x * cosY - z * sinY;
                    let z1 = x * sinY + z * cosY;
                    let cosX = Math.cos(rotX), sinX = Math.sin(rotX);
                    let y1 = y * cosX - z1 * sinX;
                    let z2 = y * sinX + z1 * cosX;
                    let k = dist / (dist + z2);
                    return { x: cx + x1 * baseScale * k, y: cy - y1 * baseScale * k, scale: k, depth: z2 };
                }

                // ── manifold mesh (wavy grid, like a draped sheet) ──
                const GRID_N = 16;
                const GRID_EXTENT_X = 4.4;
                const GRID_EXTENT_Y = 2.2;
                function manifoldHeight(gx, gy) {
                    return 0.55 * Math.sin((gx + 2.0) * 0.55) * Math.cos(gy * 0.7) - 0.15 * gx;
                }

                // Mesh rendering colors updated to be visible on light grey
                function drawManifold() {
                    let quads = [];
                    for (let i = 0; i < GRID_N; i++) {
                        for (let j = 0; j < GRID_N; j++) {
                            let x0 = -GRID_EXTENT_X + (2 * GRID_EXTENT_X * i) / GRID_N;
                            let x1 = -GRID_EXTENT_X + (2 * GRID_EXTENT_X * (i + 1)) / GRID_N;
                            let y0 = -GRID_EXTENT_Y + (2 * GRID_EXTENT_Y * j) / GRID_N;
                            let y1 = -GRID_EXTENT_Y + (2 * GRID_EXTENT_Y * (j + 1)) / GRID_N;

                            let p00 = project(x0, manifoldHeight(x0, y0) - 1.6, y0);
                            let p10 = project(x1, manifoldHeight(x1, y0) - 1.6, y0);
                            let p11 = project(x1, manifoldHeight(x1, y1) - 1.6, y1);
                            let p01 = project(x0, manifoldHeight(x0, y1) - 1.6, y1);

                            let avgDepth = (p00.depth + p10.depth + p11.depth + p01.depth) / 4;
                            quads.push({ pts: [p00, p10, p11, p01], depth: avgDepth });
                        }
                    }
                    quads.sort((a, b) => b.depth - a.depth);

                    for (let q of quads) {
                        ctx.beginPath();
                        ctx.moveTo(q.pts[0].x, q.pts[0].y);
                        for (let k = 1; k < 4; k++) ctx.lineTo(q.pts[k].x, q.pts[k].y);
                        ctx.closePath();
                        ctx.fillStyle = "rgba(100, 116, 139, 0.08)";
                        ctx.fill();
                        ctx.strokeStyle = "rgba(148, 163, 184, 0.45)";
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    }
                }

                const DATA_GX = 0.3, DATA_GY = -0.3;
                const DATA_X = DATA_GX;
                const DATA_Z = DATA_GY;
                const DATA_Y = manifoldHeight(DATA_GX, DATA_GY) - 1.6;

                const CLOUD_CENTER_X = DATA_X + 1.5;
                const CLOUD_CENTER_Y = DATA_Y + 1.7;
                const CLOUD_CENTER_Z = DATA_Z + 0.7;

                function seededRandom(seed) {
                    let s = seed;
                    return function () {
                        s = (s * 1664525 + 1013904223) % 4294967296;
                        return s / 4294967296;
                    };
                }
                let _rand = seededRandom(42);
                function _gaussian() {
                    let u1 = _rand(), u2 = _rand();
                    return Math.sqrt(-2 * Math.log(u1 + 1e-9)) * Math.cos(2 * Math.PI * u2);
                }

                const CLOUD_RADIUS = 0.85;
                const CLOUD_N = 26;
                const V_CLOUD_IDX = 0;
                const CLOUD = [];
                for (let i = 0; i < CLOUD_N; i++) {
                    if (i === V_CLOUD_IDX) {
                        CLOUD.push({ x: CLOUD_CENTER_X, y: CLOUD_CENTER_Y, z: CLOUD_CENTER_Z });
                        continue;
                    }
                    let dx = _gaussian() * CLOUD_RADIUS;
                    let dy = _gaussian() * CLOUD_RADIUS;
                    let dz = _gaussian() * CLOUD_RADIUS;
                    CLOUD.push({ x: CLOUD_CENTER_X + dx, y: CLOUD_CENTER_Y + dy, z: CLOUD_CENTER_Z + dz });
                }

                function drawDigit(pixels, screenX, screenY, size) {
                    if (!pixels) return;
                    let n = pixels.length;
                    let cell = size / n;
                    let startX = screenX - size / 2, startY = screenY - size / 2;
                    ctx.save();
                    ctx.shadowColor = "rgba(0,0,0,0.3)";
                    ctx.shadowBlur = 6;
                    ctx.fillStyle = "#000";
                    ctx.fillRect(startX - 2, startY - 2, size + 4, size + 4);
                    ctx.restore();
                    for (let r = 0; r < n; r++) {
                        for (let c = 0; c < n; c++) {
                            let v = Math.floor(pixels[r][c] * 255);
                            ctx.fillStyle = `rgb(${v},${v},${v})`;
                            ctx.fillRect(startX + c * cell, startY + r * cell, cell + 0.6, cell + 0.6);
                        }
                    }
                    ctx.strokeStyle = "rgba(255,255,255,0.25)";
                    ctx.lineWidth = 1;
                    ctx.strokeRect(startX, startY, size, size);
                }

                function curvedArrow(p0, p1, bend, color) {
                    let mx = (p0.x + p1.x) / 2, my = (p0.y + p1.y) / 2;
                    let dx = p1.x - p0.x, dy = p1.y - p0.y;
                    let len = Math.sqrt(dx * dx + dy * dy) || 1;
                    let nx = -dy / len, ny = dx / len;
                    let ctrlX = mx + nx * bend, ctrlY = my + ny * bend;

                    ctx.beginPath();
                    ctx.moveTo(p0.x, p0.y);
                    ctx.quadraticCurveTo(ctrlX, ctrlY, p1.x, p1.y);
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 1.6;
                    ctx.stroke();

                    // arrowhead
                    let t = 0.94;
                    let ax = (1 - t) * (1 - t) * p0.x + 2 * (1 - t) * t * ctrlX + t * t * p1.x;
                    let ay = (1 - t) * (1 - t) * p0.y + 2 * (1 - t) * t * ctrlY + t * t * p1.y;
                    let ang = Math.atan2(p1.y - ay, p1.x - ax);
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p1.x - 8 * Math.cos(ang - 0.4), p1.y - 8 * Math.sin(ang - 0.4));
                    ctx.lineTo(p1.x - 8 * Math.cos(ang + 0.4), p1.y - 8 * Math.sin(ang + 0.4));
                    ctx.closePath();
                    ctx.fillStyle = color;
                    ctx.fill();

                    return { x: ctrlX, y: ctrlY };
                }

                function labelPill(x, y, text) {
                    ctx.font = "bold 11px system-ui, sans-serif";
                    let padX = 8, padY = 5;
                    let w = ctx.measureText(text).width + padX * 2;
                    let h = 20;
                    ctx.fillStyle = "#f1f5f9";
                    ctx.strokeStyle = "#2563eb";
                    ctx.lineWidth = 1.2;
                    roundRect(x - w / 2, y - h / 2, w, h, 6);
                    ctx.fill();
                    ctx.stroke();
                    ctx.fillStyle = "#2563eb";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText(text, x, y + 0.5);
                }

                function roundRect(x, y, w, h, r) {
                    ctx.beginPath();
                    ctx.moveTo(x + r, y);
                    ctx.arcTo(x + w, y, x + w, y + h, r);
                    ctx.arcTo(x + w, y + h, x, y + h, r);
                    ctx.arcTo(x, y + h, x, y, r);
                    ctx.arcTo(x, y, x + w, y, r);
                    ctx.closePath();
                }

                function render() {
                    ctx.clearRect(0, 0, W, H);

                    drawManifold();

                    // "image manifold" label, always shown
                    ctx.save();
                    ctx.font = "13px system-ui, sans-serif";
                    ctx.fillStyle = "#475569";
                    ctx.textAlign = "center";
                    let lp = project(0, -2.7, -1.0);
                    ctx.fillText("image manifold", lp.x, lp.y);
                    ctx.restore();

                    // node positions in 3D
                    let pInput = project(-2.6, 2.0, -1.6);

                    let pX = project(DATA_GX, manifoldHeight(DATA_GX, DATA_GY) - 1.6, DATA_GY);

                    for (let i = 0; i < CLOUD.length; i++) {
                        if (i === V_CLOUD_IDX) continue;
                        let c = CLOUD[i];
                        let pc = project(c.x, c.y, c.z);
                        ctx.beginPath();
                        ctx.arc(pc.x, pc.y, 2.2 * pc.scale, 0, Math.PI * 2);
                        ctx.fillStyle = "rgba(71, 85, 105, 0.5)";
                        ctx.fill();
                    }

                    let vMember = CLOUD[V_CLOUD_IDX];
                    let pV = project(vMember.x, vMember.y, vMember.z);
                    let pEps = { x: (pV.x + pX.x) / 2, y: (pV.y + pX.y) / 2 - 10 };

                    let cV = curvedArrow(pInput, pV, -60, "#64748b");
                    let cX = curvedArrow(pInput, pX, 45, "#64748b");
                    labelPill(cV.x, cV.y - 6, "v-pred");
                    labelPill(cX.x, cX.y + 6, "x-pred");

                    ctx.save();
                    ctx.setLineDash([4, 4]);
                    ctx.strokeStyle = "rgba(71, 85, 105, 0.5)";
                    ctx.lineWidth = 1.4;
                    ctx.beginPath();
                    ctx.moveTo(pV.x, pV.y);
                    ctx.lineTo(pX.x, pX.y);
                    ctx.stroke();
                    ctx.restore();
                    ctx.fillStyle = "#2563eb";
                    ctx.font = "bold 15px system-ui, sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText("ε", pEps.x, pEps.y);

                    for (let p of [pInput, pV, pX]) {
                        ctx.beginPath();
                        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
                        ctx.fillStyle = "#2563eb";
                        ctx.fill();
                    }

                    drawDigit(digits.input, pInput.x, pInput.y - 50, 66 * pInput.scale);
                    drawDigit(digits.v, pV.x + 8, pV.y - 58, 66 * pV.scale);
                    drawDigit(digits.x, pX.x, pX.y + 66, 66 * pX.scale);

                    ctx.fillStyle = "#475569";
                    ctx.font = "13px system-ui, sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText("input", pInput.x, pInput.y - 100);
                }

                model.on("change:digits_json", update);
                update();
            }
        }
        """
        digits_json = traitlets.Unicode("{}").tag(sync=True)

    manifold_concept_widget = ManifoldConceptWidget()
    manifold_concept_widget.digits_json = json.dumps({
        "input": CONCEPT_DIGIT_V,
        "x": CONCEPT_DIGIT_X,
        "v": CONCEPT_DIGIT_INPUT,
    })
    return (manifold_concept_widget,)


@app.cell(hide_code=True)
def _(manifold_concept_widget, mo):
    mo.vstack(
        [mo.md(
        """### The diagram below is indicative that the distrbution learnt by model via $v$-pred is **Off-manifold** """

    )   , 
        manifold_concept_widget  ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summary of Performance

    | Objective | High Ambient Dimension ($d \ll D$) | Low Ambient Dimension ($d \approx D$) |
    | --- | --- | --- |
    | **$x$-pred** | ✓ | ✗ |
    | **$v$-pred** | ✗ | ✓ |
    | **$\epsilon$-pred** | ✗ | ✗ |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        **Key Insight:** We were able to deduce that **$x$-pred** loss is scalable
        """
    ).callout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Application to Image Generation

    Let's apply these concepts to image generation, taking MNIST as a
    simple illustrative example. For image generation we work in a
    high-dimensional **pixel space**, meaning we should primarily use
    the **$x$-pred loss**. It's worth noting, though, that depending on
    the configuration — if the patch dimension is changed such that
    the ambient dimension $D$ becomes low — then **$v$-pred** yields
    better results. That's exactly the crossover we just watched in
    Toy 2.

    ## Architectural Selection

    With that in mind, which architecture should we use? CNNs? UNet?
    Or Transformers?

    To decide, we need to understand the nature of pixel-space
    prediction ($x$-prediction). Following the Manifold Hypothesis, we
    don't get any structural prior evidence about specific symmetries
    or configurations on the clean data manifold. This means we should
    choose an architecture with the **least structural assumptions**.

    * **CNNs and UNets** come with heavily integrated domain-specific
      inductive biases, such as local connectivity and translation
      equivariance.
    * **Transformers** serve as an optimal architecture here — they
      preserve global context purely via attention, offer massive
      scalability, and let us borrow extensive optimization techniques
      developed in NLP (e.g., `qk-norm`, `RoPE`, `RMSNorm`, and
      attention residuals).

    Since, there is a use of transformers and pixel space (clean data) prediction , hence the name **Just Image Transformers**
    Let's train it on MNIST
    """)
    return


@app.cell(hide_code=True)
def _(MNIST_X_RAW, MNIST_Y_RAW, anywidget, json, mnist_config, np, traitlets):
    # Isolate a single clear sample (digit 7) to use as the fixed reference profile
    _sevens = np.where(MNIST_Y_RAW == 7)[0]
    _raw_img = MNIST_X_RAW[_sevens[0]]  # Shape: (28, 28)

    # Manually apply the 2-pixel uniform zero padding to match the preprocessing module logic
    _padded_img = np.pad(_raw_img, ((2, 2), (2, 2)), mode='constant', constant_values=-1.0)

    # Normalize input range from [-1, 1] into [0, 1] for precise HTML5 Canvas rendering
    _raw_norm = np.clip((_raw_img + 1.0) / 2.0, 0.0, 1.0)
    _padded_norm = np.clip((_padded_img + 1.0) / 2.0, 0.0, 1.0)

    _sample_data = {
        "raw": _raw_norm.tolist(),
        "padded": _padded_norm.tolist()
    }

    class TokenizationPipelineWidget(anywidget.AnyWidget):
        _esm = """
        export default {
            render({ model, el }) {
                let sample = null;
                let active_P = 2;

                el.innerHTML = `
                <div style="display: flex; flex-direction: column; align-items: center; gap: 20px; font-family: system-ui, sans-serif; background: #f8fafc; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; max-width: 900px; margin: 20px auto; box-shadow: 0 4px 10px rgb(0 0 0 / 0.05);">
                    <div style="text-align: center; width: 100%; border-bottom: 1px solid #e2e8f0; padding-bottom: 16px;">
                        <div style="font-size: 15px; font-weight: 700; color: #0f172a;">Preprocessing & Tokenization </div>
                    </div>

                    <div style="display: flex; align-items: center; gap: 16px; width: 100%; background: #f8fafc; padding: 12px 16px; border-radius: 8px; box-sizing: border-box; justify-content: center;">
                        <span style="font-size: 13px; font-weight: 600; color: #475569;">Select Token Patch Size:</span>
                        <div style="display: flex; gap: 8px;">
                            <button id="btnP1" style="padding: 6px 14px; font-size: 12px; font-weight: 600; border-radius: 6px; border: 1px solid #cbd5e1; background: #ffffff; color: #475569; cursor: pointer; transition: all 0.2s;">1 × 1</button>
                            <button id="btnP2" style="padding: 6px 14px; font-size: 12px; font-weight: 600; border-radius: 6px; border: 1px solid #3b82f6; background: #3b82f6; color: #ffffff; cursor: pointer; transition: all 0.2s;">2 × 2</button>
                            <button id="btnP4" style="padding: 6px 14px; font-size: 12px; font-weight: 600; border-radius: 6px; border: 1px solid #cbd5e1; background: #ffffff; color: #475569; cursor: pointer; transition: all 0.2s;">4 × 4</button>
                            <button id="btnP8" style="padding: 6px 14px; font-size: 12px; font-weight: 600; border-radius: 6px; border: 1px solid #cbd5e1; background: #ffffff; color: #475569; cursor: pointer; transition: all 0.2s;">8 × 8</button>
                        </div>
                    </div>

                    <div style="display: flex; gap: 24px; justify-content: center; width: 100%; flex-wrap: wrap; align-items: flex-start;">
                        <div style="text-align: center;">
                            <div style="font-size: 11px; font-weight: 700; color: #475569; margin-bottom: 8px; height: 32px; display: flex; flex-direction: column; justify-content: center;">1. Raw MNIST Input<br><span style="font-weight:400; color:#64748b;">(28 × 28 Core)</span></div>
                            <div style="background: #f1f5f9; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0; height: 272px; width: 200px; display: flex; align-items: center; justify-content: center; box-sizing: border-box;">
                                <canvas id="canvasRaw" width="168" height="168" style="background: #000; border-radius: 4px; display: block;"></canvas>
                            </div>
                        </div>

                        <div style="text-align: center;">
                            <div style="font-size: 11px; font-weight: 700; color: #475569; margin-bottom: 8px; height: 32px; display: flex; flex-direction: column; justify-content: center;">2. Border Padding<br><span style="font-weight:400; color:#64748b;">(32 × 32 Pixel Space)</span></div>
                            <div style="background: #f1f5f9; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0; height: 272px; width: 224px; display: flex; align-items: center; justify-content: center; box-sizing: border-box;">
                                <canvas id="canvasPadded" width="192" height="192" style="background: #000; border-radius: 4px; display: block;"></canvas>
                            </div>
                        </div>

                        <div style="text-align: center;">
                            <div style="font-size: 11px; font-weight: 700; color: #475569; margin-bottom: 8px; height: 32px; display: flex; flex-direction: column; justify-content: center;">3. Continuous Token Patches<br><span style="font-weight:400; color:#64748b;">(Isolated Tokens)</span></div>
                            <div style="background: #f1f5f9; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0; height: 272px; width: 272px; display: flex; align-items: center; justify-content: center; box-sizing: border-box;">
                                <canvas id="canvasPatches" width="240" height="240" style="background: transparent; display: block;"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                `;

                const btn1 = el.querySelector("#btnP1");
                const btn2 = el.querySelector("#btnP2");
                const btn4 = el.querySelector("#btnP4");
                const btn8 = el.querySelector("#btnP8");

                const c_raw = el.querySelector("#canvasRaw");
                const c_pad = el.querySelector("#canvasPadded");
                const c_pat = el.querySelector("#canvasPatches");

                const ctx_raw = c_raw.getContext("2d");
                const ctx_pad = c_pad.getContext("2d");
                const ctx_pat = c_pat.getContext("2d");

                function draw_pipeline_frame() {
                    if (!sample) return;
                    const P = active_P;
                    const raw_mat = sample.raw;
                    const pad_mat = sample.padded;

                    // 1. Render Raw 28x28 Core Space
                    ctx_raw.clearRect(0, 0, 168, 168);
                    for (let r = 0; r < 28; r++) {
                        for (let c = 0; c < 28; c++) {
                            let v = Math.floor(raw_mat[r][c] * 255);
                            ctx_raw.fillStyle = `rgb(${v},${v},${v})`;
                            ctx_raw.fillRect(c * 6, r * 6, 6, 6);
                        }
                    }

                    // 2. Render Padded 32x32 Boundary Grid
                    ctx_pad.clearRect(0, 0, 192, 192);
                    for (let r = 0; r < 32; r++) {
                        for (let c = 0; c < 32; c++) {
                            let v = Math.floor(pad_mat[r][c] * 255);
                            ctx_pad.fillStyle = `rgb(${v},${v},${v})`;
                            ctx_pad.fillRect(c * 6, r * 6, 6, 6);
                        }
                    }

                    ctx_pad.strokeStyle = "#3b82f6";
                    ctx_pad.lineWidth = 2;
                    ctx_pad.strokeRect(0, 0, 192, 192);

                    ctx_pad.strokeStyle = "#ef4444";
                    ctx_pad.lineWidth = 1.5;
                    ctx_pad.setLineDash([4, 4]);
                    ctx_pad.strokeRect(12, 12, 168, 168);
                    ctx_pad.setLineDash([]);

                    ctx_pad.fillStyle = "rgba(239, 68, 68, 0.2)";
                    ctx_pad.fillRect(0, 0, 192, 12);    
                    ctx_pad.fillRect(0, 180, 192, 12);  
                    ctx_pad.fillRect(0, 12, 12, 168);   
                    ctx_pad.fillRect(180, 12, 12, 168); 

                    // 3. Dynamic Patches drawn over transparent background
                    ctx_pat.clearRect(0, 0, 240, 240);

                    const num_patches = 32 / P;
                    const pixel_size = 5;
                    let gap = 1;
                    if (P === 1) gap = 1;
                    else if (P === 2) gap = 3;
                    else if (P === 4) gap = 7;
                    else if (P === 8) gap = 15;

                    const patch_width = P * pixel_size;
                    const total_dim = num_patches * patch_width + (num_patches - 1) * gap;
                    const offset = Math.floor((240 - total_dim) / 2);

                    for (let patch_r = 0; patch_r < num_patches; patch_r++) {
                        for (let patch_c = 0; patch_c < num_patches; patch_c++) {
                            const start_x = offset + patch_c * (patch_width + gap);
                            const start_y = offset + patch_r * (patch_width + gap);

                            ctx_pat.fillStyle = "#000000";
                            ctx_pat.fillRect(start_x, start_y, patch_width, patch_width);

                            for (let pr = 0; pr < P; pr++) {
                                for (let pc = 0; pc < P; pc++) {
                                    const r = patch_r * P + pr;
                                    const c = patch_c * P + pc;
                                    const v = Math.floor(pad_mat[r][c] * 255);

                                    ctx_pat.fillStyle = `rgb(${v},${v},${v})`;
                                    ctx_pat.fillRect(start_x + pc * pixel_size, start_y + pr * pixel_size, pixel_size, pixel_size);
                                }
                            }

                            ctx_pat.strokeStyle = "#f1f5f9";
                            ctx_pat.lineWidth = 1.0;
                            ctx_pat.strokeRect(start_x, start_y, patch_width, patch_width);
                        }
                    }
                }

                function update_active_button(target_val, target_btn) {
                    active_P = target_val;
                    for (let b of [btn1, btn2, btn4, btn8]) {
                        b.style.background = "#ffffff";
                        b.style.color = "#475569";
                        b.style.borderColor = "#cbd5e1";
                    }
                    target_btn.style.background = "#3b82f6";
                    target_btn.style.color = "#ffffff";
                    target_btn.style.borderColor = "#f1f5f9";
                    draw_pipeline_frame();
                }

                btn1.addEventListener("click", () => update_active_button(1, btn1));
                btn2.addEventListener("click", () => update_active_button(2, btn2));
                btn4.addEventListener("click", () => update_active_button(4, btn4));
                btn8.addEventListener("click", () => update_active_button(8, btn8));

                function update() {
                    const config = JSON.parse(model.get("config_json"));
                    if (config && config.sample) {
                        sample = config.sample;
                        active_P = config.patch_size || 2;

                        // Update active button state initially
                        const active_btn = active_P === 1 ? btn1 : (active_P === 2 ? btn2 : (active_P === 4 ? btn4 : btn8));
                        update_active_button(active_P, active_btn);
                    }
                }

                model.on("change:config_json", update);
                update();
            }
        }
        """
        config_json = traitlets.Unicode("{}").tag(sync=True)

    tokenization_pipeline_widget = TokenizationPipelineWidget()
    tokenization_pipeline_widget.config_json = json.dumps({
        "sample": _sample_data,
        "patch_size": mnist_config["patch_size"]
    })
    return (tokenization_pipeline_widget,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Given below is the tokenization strategy we will be using:
    The beauty of JiT is that, it's a simple, large-patch Transformers on pixels which uses no tokenizer, no pre-training, and no extra loss.
    """)
    return


@app.cell(hide_code=True)
def _(tokenization_pipeline_widget):
    tokenization_pipeline_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Note*:**  Padding was added to make the images 32x32 , as computers love powers of 2 and it gave more patch size "freedom"
    """)
    return


@app.cell(hide_code=True)
def _(MNIST_X_RAW, MNIST_Y_RAW, anywidget, json, np, traitlets):
    _sixs = np.where(MNIST_Y_RAW == 6)[0]
    _raw = MNIST_X_RAW[_sixs[0]]
    _padded = np.pad(_raw, ((2, 2), (2, 2)), mode='constant', constant_values=-1.0)
    _clean_norm = np.clip((_padded + 1.0) / 2.0, 0.0, 1.0)

    # Standard static noise
    _rng = np.random.RandomState(42)
    _noise = np.clip(_rng.normal(0.5, 0.28, size=_clean_norm.shape), 0.0, 1.0)

    _arch_data = {
        "clean": _clean_norm.tolist(),
        "noise": _noise.tolist()
    }
    class JiTModelArchitectureWidget(anywidget.AnyWidget):
        _esm = """
        export default {
            render({ model, el }) {
                let sample = null;

                el.innerHTML = `
                <div style="display: flex; flex-direction: column; align-items: center; gap: 12px; font-family: system-ui, sans-serif; background: #f8fafc; padding: 18px; border-radius: 12px; border: 1px solid #e2e8f0; max-width: 900px; margin: 15px auto; color: #0f172a; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05); box-sizing: border-box;">
                    <div style="text-align: center; width: 100%; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 6px;">
                        <div style="font-size: 13px; font-weight: 700; color: #2563eb; letter-spacing: 0.5px;">JiT-Tiny Architecture</div>
                    </div>

                    <div style="display: flex; flex-direction: row; gap: 16px; width: 100%; justify-content: center; align-items: center; box-sizing: border-box;">
                        <!-- Left: Input -->
                        <div style="display: flex; flex-direction: column; align-items: center; gap: 8px; background: #f8fafc; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0; width: 130px; box-sizing: border-box;">
                            <span style="font-size: 10px; font-weight: 700; color: #475569; text-align: center;">1. Input Image (32×32)</span>
                            <canvas id="flowCanvasInput" width="96" height="96" style="background:#000; border-radius:4px; display:block; border: 1px solid #cbd5e1;"></canvas>
                            <span style="font-size: 9px; font-weight: 700; color: #64748b; margin-top: 2px;">Input Patches (4×4)</span>
                            <div id="inputPatchesContainer" style="display: flex; gap: 2px; justify-content: center; width: 110px; height: 12px; overflow: hidden; flex-wrap: wrap;"></div>
                        </div>

                        <!-- Connector 1 -->
                        <svg width="24" height="24" style="overflow: visible;">
                            <defs>
                                <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                                    <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#64748b" />
                                </marker>
                                <marker id="arrow-blue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                                    <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2563eb" />
                                </marker>
                            </defs>
                            <line x1="0" y1="12" x2="24" y2="12" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)" />
                        </svg>

                        <!-- Middle: SVG Diagram -->
                        <div style="background: #f8fafc; padding: 8px; border-radius: 8px; border: 1px solid #e2e8f0; display: flex; align-items: center; justify-content: center; box-sizing: border-box; height: 300px;">
                            <svg id="architectureSvg" width="340" height="280" style="background: transparent; overflow: visible;"></svg>
                        </div>

                        <!-- Connector 2 -->
                        <svg width="24" height="24" style="overflow: visible;">
                            <line x1="0" y1="12" x2="24" y2="12" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow-blue)" />
                        </svg>

                        <!-- Right: Output -->
                        <div style="display: flex; flex-direction: column; align-items: center; gap: 8px; background: #f8fafc; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0; width: 130px; box-sizing: border-box;">
                            <span style="font-size: 10px; font-weight: 700; color: #475569; text-align: center;">4. Output Image</span>
                            <canvas id="flowCanvasOutput" width="96" height="96" style="background:#000; border-radius:4px; display:block; border: 1px solid #cbd5e1;"></canvas>
                            <span style="font-size: 9px; font-weight: 700; color: #64748b; margin-top: 2px;">Predicted Patches</span>
                            <div id="outputPatchesContainer" style="display: flex; gap: 2px; justify-content: center; width: 110px; height: 12px; overflow: hidden; flex-wrap: wrap;"></div>
                        </div>
                    </div>
                </div>
                `;

                const c_in = el.querySelector("#flowCanvasInput");
                const c_out = el.querySelector("#flowCanvasOutput");
                const ctx_in = c_in.getContext("2d");
                const ctx_out = c_out.getContext("2d");

                const p_in_container = el.querySelector("#inputPatchesContainer");
                const p_out_container = el.querySelector("#outputPatchesContainer");

                // Initialize 16 tiny patch canvases (representing a sample of patches)
                const p_in_canvases = [];
                const p_out_canvases = [];
                for (let i = 0; i < 16; i++) {
                    let canv = document.createElement("canvas");
                    canv.width = 4;
                    canv.height = 4;
                    canv.style.background = "#000";
                    canv.style.borderRadius = "1px";
                    canv.style.border = "1px solid #cbd5e1";
                    canv.style.width = "10px";
                    canv.style.height = "10px";
                    p_in_container.appendChild(canv);
                    p_in_canvases.push(canv);

                    let canv_out = document.createElement("canvas");
                    canv_out.width = 4;
                    canv_out.height = 4;
                    canv_out.style.background = "#000";
                    canv_out.style.borderRadius = "1px";
                    canv_out.style.border = "1px solid #cbd5e1";
                    canv_out.style.width = "10px";
                    canv_out.style.height = "10px";
                    p_out_container.appendChild(canv_out);
                    p_out_canvases.push(canv_out);
                }

                // Render Model Block Diagram using SVG
                const svg = el.querySelector("#architectureSvg");
                svg.innerHTML = `
                <defs>
                    <marker id="arrow-svg" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#94a3b8" />
                    </marker>
                </defs>

                <!-- Input connection path (shortened to y2=12 to pull back) -->
                <line x1="170" y1="2" x2="170" y2="12" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow-svg)" />

                <!-- 1. Embedding Layer -->
                <rect x="100" y="15" width="140" height="20" rx="4" fill="#f1f5f9" stroke="#8963DC" stroke-width="1.5" />
                <text x="170" y="27" font-size="9" font-weight="700" fill="#8963DC" text-anchor="middle">Embedding</text>

                <!-- Connection to loop block (shortened to y2=43 to pull back) -->
                <line x1="170" y1="35" x2="170" y2="43" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow-svg)" />

                <!-- Transformer Block Outline (Dotted Loop) -->
                <rect x="50" y="46" width="240" height="178" rx="8" fill="none" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="4,4" />
                <text x="284" y="218" font-size="8.5" font-weight="700" fill="#94a3b8" text-anchor="end">× N</text>

                <!-- Internal Backbone Flow Line -->
                <line x1="170" y1="46" x2="170" y2="224" stroke="rgba(148,163,184,0.2)" stroke-width="1.5" />

                <!-- A. Normalization Block 1 -->
                <rect x="100" y="56" width="140" height="18" rx="3" fill="#f1f5f9" stroke="#475569" stroke-width="1.2" />
                <text x="170" y="68" font-size="8.5" font-weight="600" fill="#475569" text-anchor="middle">Normalize</text>

                <!-- B. MHA Block -->
                <rect x="100" y="86" width="140" height="22" rx="3" fill="#f1f5f9" stroke="#895CE6" stroke-width="1.5" />
                <text x="170" y="100" font-size="9" font-weight="700" fill="#895CE6" text-anchor="middle">Attention</text>

                <!-- RoPE Side Input Box -->
                <rect x="252" y="86" width="30" height="18" rx="3" fill="#f1f5f9" stroke="#FCB21E" stroke-width="1.2" />
                <text x="267" y="98" font-size="8.5" font-weight="700" fill="#FCB21E" text-anchor="middle">RoPE</text>
                <path d="M 252 95 L 243 95" fill="none" stroke="#FCB21E" stroke-width="1.2" marker-end="url(#arrow-svg)" />

                <!-- Residual Addition Circle 1 -->
                <circle cx="170" cy="122" r="6" fill="#f1f5f9" stroke="#475569" stroke-width="1" />
                <text x="170" y="125" font-size="9" font-weight="bold" fill="#475569" text-anchor="middle">+</text>

                <!-- Residual Left Loop 1 (shortened to x=162 to pull back) -->
                <path d="M 170 51 L 82 51 L 82 122 L 162 122" fill="none" stroke="#94a3b8" stroke-width="1" marker-end="url(#arrow-svg)" />

                <!-- C. Normalization Block 2 -->
                <rect x="100" y="136" width="140" height="18" rx="3" fill="#f1f5f9" stroke="#475569" stroke-width="1.2" />
                <text x="170" y="148" font-size="8.5" font-weight="600" fill="#475569" text-anchor="middle">Normalize</text>

                <!-- D. FFN  -->
                <rect x="100" y="166" width="140" height="22" rx="3" fill="#f1f5f9" stroke="#000000" stroke-width="1.5" />
                <text x="170" y="180" font-size="9" font-weight="700" fill="#000000" text-anchor="middle">FFN</text>

                <!-- Residual Addition Circle 2 -->
                <circle cx="170" cy="202" r="6" fill="#f1f5f9" stroke="#475569" stroke-width="1" />
                <text x="170" y="205" font-size="9" font-weight="bold" fill="#475569" text-anchor="middle">+</text>

                <!-- Residual Left Loop 2 (shortened to x=162 to pull back) -->
                <path d="M 170 131 L 82 131 L 82 202 L 162 202" fill="none" stroke="#94a3b8" stroke-width="1" marker-end="url(#arrow-svg)" />

                <!-- Connection out of Block (shortened to y2=233 to pull back) -->
                <line x1="170" y1="224" x2="170" y2="233" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow-svg)" />

                <!-- 3. Final Norm -->
                <rect x="100" y="236" width="140" height="18" rx="3" fill="#f1f5f9" stroke="#475569" stroke-width="1.2" />
                <text x="170" y="248" font-size="8.5" font-weight="600" fill="#475569" text-anchor="middle">Normalize</text>

                <!-- Connection to Linear  -->
                <line x1="170" y1="254" x2="170" y2="261" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arrow-svg)" />

                <!-- 4. Linear Predict Output Layer -->
                <rect x="100" y="264" width="140" height="20" rx="4" fill="#f1f5f9" stroke="#130CED" stroke-width="1.5" />
                <text x="170" y="276" font-size="9" font-weight="700" fill="#130CED" text-anchor="middle">Linear</text>
                `;

                // Draw static digits
                function drawDigit(ctx, pixels) {
                    ctx.clearRect(0, 0, 96, 96);
                    if (!pixels) return;

                    // 1. Draw raw pixels (pixel size is 96/32 = 3px)
                    for (let r = 0; r < 32; r++) {
                        for (let c = 0; c < 32; c++) {
                            let v = Math.floor(pixels[r][c] * 255);
                            ctx.fillStyle = `rgb(${v},${v},${v})`;
                            ctx.fillRect(c * 3, r * 3, 3, 3);
                        }
                    }

                    // 2. Draw 4x4 Grid overlay (separated in 8 blocks of 12px each)
                    ctx.strokeStyle = "rgba(59, 130, 246, 0.22)";
                    ctx.lineWidth = 0.8;
                    for (let grid = 0; grid <= 8; grid++) {
                        let coord = grid * 12; 
                        ctx.beginPath();
                        ctx.moveTo(coord, 0);
                        ctx.lineTo(coord, 96);
                        ctx.stroke();

                        ctx.beginPath();
                        ctx.moveTo(0, coord);
                        ctx.lineTo(96, coord);
                        ctx.stroke();
                    }
                }

                function drawPatch(canvas, r_start, c_start, matrix) {
                    let ctx = canvas.getContext("2d");
                    ctx.clearRect(0, 0, 4, 4);
                    for (let r = 0; r < 4; r++) {
                        for (let c = 0; c < 4; c++) {
                            let val = Math.floor(matrix[r_start + r][c_start + c] * 255);
                            ctx.fillStyle = `rgb(${val},${val},${val})`;
                            ctx.fillRect(c, r, 1, 1);
                        }
                    }
                }

                function update() {
                    let config = JSON.parse(model.get("config_json"));
                    if (config && config.sample) {
                        sample = config.sample;
                        let clean = sample.clean;
                        let noise = sample.noise;

                        // Create input noisy image at a fixed noise level t=0.5
                        let x_t = [];
                        for (let r = 0; r < 32; r++) {
                            let row = [];
                            for (let c = 0; c < 32; c++) {
                                let val = 0.5 * clean[r][c] + 0.5 * noise[r][c];
                                row.push(Math.max(0.0, Math.min(1.0, val)));
                            }
                            x_t.push(row);
                        }

                        drawDigit(ctx_in, x_t);
                        drawDigit(ctx_out, clean); // Show clean image on the output

                        // Update tiny patch canvases with first 16 patches
                        for (let i = 0; i < 16; i++) {
                            let pr = Math.floor(i / 4) * 4;
                            let pc = (i % 4) * 4;
                            drawPatch(p_in_canvases[i], pr + 8, pc + 8, x_t);
                            drawPatch(p_out_canvases[i], pr + 8, pc + 8, clean);
                        }
                    }
                }

                model.on("change:config_json", update);
                update();
            }
        }
        """
        config_json = traitlets.Unicode("{}").tag(sync=True)

    jit_model_architecture_widget = JiTModelArchitectureWidget()
    jit_model_architecture_widget.config_json = json.dumps({
        "sample": _arch_data
    })
    return (jit_model_architecture_widget,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Given below is the architecture we will be using for training. Let's call it **Jit-Tiny**
    """)
    return


@app.cell(hide_code=True)
def _(jit_model_architecture_widget):
    jit_model_architecture_widget
    return


@app.cell(hide_code=True)
def mnist_controls(mo):
    mnist_form = (
        mo.md(
            """
            ### MNIST JiT-Tiny Attention Controls
            {num_epochs}

            {batch_size}

            {learning_rate}

            {emb_dim}

            {n_layers}

            {n_heads}

            {patch_size}

            {bottleneck_dim}

            {sampling_steps}
            """
        )
        .batch(
            num_epochs=mo.ui.slider(5, 50, step=5, value=15, label="Epochs", show_value=True),
            batch_size=mo.ui.slider(64, 1024, step=64, value=512, label="Batch Size", show_value=True),
            learning_rate=mo.ui.dropdown(options=["0.0001", "0.0002", "0.0005", "0.001"], value="0.0002", label="Learning Rate"),
            emb_dim=mo.ui.dropdown(options=["64", "128", "192", "256"], value="192", label="d_model (Embedding Size)"),
            n_layers=mo.ui.slider(2, 12, step=1, value=6, label="Attention Layers", show_value=True),
            n_heads=mo.ui.slider(2, 12, step=2, value=6, label="Attention Heads", show_value=True),
            patch_size=mo.ui.dropdown(options=["2", "4", "8"], value="4", label="Patch Size"),
            bottleneck_dim=mo.ui.slider(16, 128, step=16, value=32, label="Bottleneck Dimension", show_value=True),
            sampling_steps=mo.ui.slider(10, 100, step=10, value=50, label="Sampling Steps", show_value=True),
        )
        .form(submit_button_label="Run MNIST Training Sequence")
    )

    mo.vstack([
        mo.md("## Train JiT-Tiny on MNIST"),
        mnist_form
    ])
    return (mnist_form,)


@app.cell(hide_code=True)
def mnist_config_setup(mnist_form):
    _val = mnist_form.value or {}
    mnist_config = {
        "in_channels": 1,
        "image_size": 32,
        "patch_size": int(_val.get("patch_size", 2)),
        "emb_dim": int(_val.get("emb_dim", 192)),
        "n_layers": int(_val.get("n_layers", 6)),
        "n_heads": int(_val.get("n_heads", 6)),
        "bottleneck_dim": int(_val.get("bottleneck_dim", 16)),
        "class_tokens": 4,
        "dropout_rate": 0.1,
        "batch_size": int(_val.get("batch_size", 512)),
        "learning_rate": float(_val.get("learning_rate", 2e-4)),
        "num_epochs": int(_val.get("num_epochs", 15)),
        "warmup_epochs": 5,
        "sampling_steps": int(_val.get("sampling_steps", 50)),
    }
    return (mnist_config,)


@app.cell(hide_code=True)
def _(PCA_DATA, anywidget, json, traitlets):
    class PcaWidget(anywidget.AnyWidget):
        _esm = """
        export default {
            render({ model, el }) {
                let data = {};
                let state = { playing: false, interval_id: null };
                const W = 460, H = 350;

                el.innerHTML = `
                <div id="container" style="display:flex; flex-direction:column; align-items:center; gap:16px;
                            font-family:system-ui,sans-serif; background:#f8fafc; padding:24px;
                            border-radius:12px; border:1px solid #e2e8f0; max-width:520px;
                            margin:0 auto; box-shadow:0 4px 6px -1px rgb(0 0 0/.05);">
                    <div style="text-align:center; width:100%;">
                        <div id="title" style="font-size:15px; font-weight:700; color:#0f172a; margin-bottom:2px;"></div>
                        <div id="subtitle" style="font-size:11px; color:#64748b; margin-bottom:12px;"></div>
                        <canvas id="pcaCanvas" width="${W}" height="${H}"
                                style="border-radius:8px; display:block; margin:0 auto;">
                        </canvas>
                    </div>
                    <div style="display:flex; align-items:center; gap:12px; width:100%;
                                background:#f1f5f9; padding:8px 16px; border-radius:8px;
                                box-sizing:border-box;">
                        <button id="playBtn"
                                style="background:#f1f5f9; border:1px solid #2563eb;
                                       border-radius:6px; padding:6px 14px; font-size:13px;
                                       font-weight:500; color:#2563eb; cursor:pointer;
                                       display:flex; align-items:center; gap:4px;
                                       min-width:70px; justify-content:center;">▶ play</button>
                        <input type="range" id="slider" min="0" max="50" step="1" value="0"
                               style="flex-grow:1; cursor:pointer; margin:0; accent-color:#2563eb;">
                        <span id="label"
                              style="font-family:monospace; font-size:13px; color:#475569;
                                     font-weight:600; min-width:65px; text-align:right;">
                            t = 0.00
                        </span>
                    </div>
                    <div id="legend" style="display:flex; gap:20px; font-size:12px;"></div>
                </div>
                `;

                const titleEl = el.querySelector("#title");
                const subtitleEl = el.querySelector("#subtitle");
                const canvas = el.querySelector("#pcaCanvas");
                const ctx = canvas.getContext("2d");
                const slider = el.querySelector("#slider");
                const btn = el.querySelector("#playBtn");
                const lbl = el.querySelector("#label");
                const legendEl = el.querySelector("#legend");

                let xMin, xMax, yMin, yMax;

                function computeExtent() {
                    let allX = [], allY = [];
                    if (data.mode === "forward") {
                        for (let i = 0; i < data.proj.length; i++) {
                            allX.push(data.proj[i][0], data.noise[i][0]);
                            allY.push(data.proj[i][1], data.noise[i][1]);
                        }
                    } else {
                        for (let i = 0; i < data.gt_proj.length; i++) {
                            allX.push(data.gt_proj[i][0]);
                            allY.push(data.gt_proj[i][1]);
                        }
                        for (let p = 0; p < data.traj.length; p++) {
                            let last = data.traj[p].length - 1;
                            allX.push(data.traj[p][0][0], data.traj[p][last][0]);
                            allY.push(data.traj[p][0][1], data.traj[p][last][1]);
                        }
                    }
                    let mnX = Math.min(...allX), mxX = Math.max(...allX);
                    let mnY = Math.min(...allY), mxY = Math.max(...allY);
                    let padX = (mxX - mnX) * 0.12;
                    let padY = (mxY - mnY) * 0.12;
                    xMin = mnX - padX;  xMax = mxX + padX;
                    yMin = mnY - padY;  yMax = mxY + padY;
                }

                function toCanvas(x, y) {
                    let cx = ((x - xMin) / (xMax - xMin)) * W;
                    let cy = H - ((y - yMin) / (yMax - yMin)) * H;
                    return [cx, cy];
                }

                function draw(step) {
                    let maxSteps = parseInt(slider.max);
                    let t_val = step / maxSteps;

                    let display_t = data.mode === "forward" ? (1.0 - step * 0.02) : t_val;
                    lbl.textContent = `t = ${display_t.toFixed(2)}`;
                    ctx.clearRect(0, 0, W, H);

                    ctx.strokeStyle = data.theme === "dark" ? "rgba(148,163,184,0.12)" : "rgba(148,163,184,0.18)";
                    ctx.lineWidth = 0.5;
                    let [ox, oy] = toCanvas(0, 0);
                    ctx.beginPath(); ctx.moveTo(ox, 0); ctx.lineTo(ox, H); ctx.stroke();
                    ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(W, oy); ctx.stroke();

                    if (data.mode === "forward") {
                        if (!data.proj || data.proj.length === 0) return;
                        for (let i = 0; i < data.proj.length; i++) {
                            let px = display_t * data.proj[i][0] + (1 - display_t) * data.noise[i][0];
                            let py = display_t * data.proj[i][1] + (1 - display_t) * data.noise[i][1];
                            let [cx, cy] = toCanvas(px, py);
                            if (cx < -5 || cx > W + 5 || cy < -5 || cy > H + 5) continue;

                            let is6 = data.labels[i] === 6;
                            let glow = 0.15 + 0.35 * display_t;
                            ctx.fillStyle = is6
                                ? `rgba(37,99,235,${glow})`
                                : `rgba(139,92,246,${glow})`;
                            ctx.beginPath();
                            ctx.arc(cx, cy, 2.2, 0, 2 * Math.PI);
                            ctx.fill();
                        }
                    } else {
                        if (!data.traj || data.traj.length === 0) return;

                        for (let i = 0; i < data.gt_proj.length; i++) {
                            let [cx, cy] = toCanvas(data.gt_proj[i][0], data.gt_proj[i][1]);
                            ctx.fillStyle = "rgba(148,163,184,0.18)";
                            ctx.beginPath();
                            ctx.arc(cx, cy, 2.0, 0, 2 * Math.PI);
                            ctx.fill();
                        }

                        let trail_len = 4;
                        let trail_start = Math.max(0, step - trail_len);
                        if (step > trail_start) {
                            ctx.lineWidth = 0.8;
                            for (let p = 0; p < data.traj.length; p++) {
                                let is6 = data.labels_rev[p] === 6;
                                ctx.strokeStyle = is6
                                    ? "rgba(37,99,235,0.25)"
                                    : "rgba(139,92,246,0.25)";
                                ctx.beginPath();
                                for (let ti = trail_start; ti <= step; ti++) {
                                    let [cx, cy] = toCanvas(data.traj[p][ti][0], data.traj[p][ti][1]);
                                    if (ti === trail_start) ctx.moveTo(cx, cy);
                                    else ctx.lineTo(cx, cy);
                                }
                                ctx.stroke();
                            }
                        }

                        let glow = 0.4 + 0.5 * t_val;
                        for (let p = 0; p < data.traj.length; p++) {
                            let [cx, cy] = toCanvas(data.traj[p][step][0], data.traj[p][step][1]);
                            if (cx < -5 || cx > W + 5 || cy < -5 || cy > H + 5) continue;
                            let is6 = data.labels_rev[p] === 6;
                            ctx.fillStyle = is6
                                ? `rgba(37,99,235,${glow})`
                                : `rgba(139,92,246,${glow})`;
                            ctx.beginPath();
                            ctx.arc(cx, cy, 2.5, 0, 2 * Math.PI);
                            ctx.fill();
                        }
                    }
                }

                function tick() {
                    let maxSteps = parseInt(slider.max);
                    let s = parseInt(slider.value) + 1;
                    if (s > maxSteps) s = 0;
                    slider.value = String(s);
                    draw(s);
                }

                btn.addEventListener("click", () => {
                    if (state.playing) {
                        clearInterval(state.interval_id);
                        state.playing = false;
                        btn.textContent = "▶ play";
                    } else {
                        state.interval_id = setInterval(tick, data.mode === "forward" ? 80 : 120);
                        state.playing = true;
                        btn.textContent = "⏸ pause";
                    }
                });

                slider.addEventListener("input", () => {
                    if (state.playing) {
                        clearInterval(state.interval_id);
                        state.playing = false;
                        btn.textContent = "▶ play";
                    }
                    draw(parseInt(slider.value));
                });

                function update() {
                    let cfg = JSON.parse(model.get("config_json"));
                    if (cfg && cfg.mode) {
                        data = cfg;
                        titleEl.textContent = data.title;
                        subtitleEl.textContent = data.subtitle;
                        slider.max = String(data.mode === "forward" ? 50 : data.n_steps - 1);
                        slider.style.accentColor = data.accent_color;

                        if (data.theme === "dark") {
                            canvas.style.background = "#0f172a";
                            canvas.style.border = "1px solid #334155";
                            legendEl.style.color = "#94a3b8";
                        } else {
                            canvas.style.background = "#f8fafc";
                            canvas.style.border = "1px solid #e2e8f0";
                            legendEl.style.color = "#64748b";
                        }

                        legendEl.innerHTML = "";
                        data.legend.forEach(item => {
                            let span = document.createElement("span");
                            span.innerHTML = `<span style="color:${item.color};">●</span> ${item.label}`;
                            legendEl.appendChild(span);
                        });

                        computeExtent();
                        draw(parseInt(slider.value));
                    }
                }

                model.on("change:config_json", update);
                update();
            }
        }
        """
        config_json = traitlets.Unicode("{}").tag(sync=True)

    pca_fwd_widget = PcaWidget()
    pca_fwd_widget.config_json = json.dumps({
        "mode": "forward",
        "title": "Forward Process",
        "subtitle": "Visualizing the forward noising process via PCA of Digits 6 & 7 ",
        "theme": "light",
        "accent_color": "#2563eb",
        "legend": [
            {"color": "#2563eb", "label": "digit 6"},
            {"color": "#8b5cf6", "label": "digit 7"}
        ],
        "proj": PCA_DATA["proj"],
        "labels": PCA_DATA["labels"],
        "noise": PCA_DATA["noise"]
    })
    return PcaWidget, pca_fwd_widget


@app.cell(hide_code=True)
def _(MNIST_X_RAW, MNIST_Y_RAW, np):
    # ── PCA on MNIST digits ──────────────────────────────────────
    _mask_6 = MNIST_Y_RAW == 6
    _mask_7 = MNIST_Y_RAW == 7
    _mask = _mask_6 | _mask_7

    _X = MNIST_X_RAW[_mask]                          # (N, 28, 28)
    _labels = MNIST_Y_RAW[_mask]                      # 6 or 7

    _X_flat = _X.reshape(len(_X), -1).astype(np.float64)   # (N, 784)

    # Center and compute top-2 PCA directions
    PCA_MEAN = _X_flat.mean(axis=0)
    _X_c = _X_flat - PCA_MEAN
    _cov = np.cov(_X_c, rowvar=False)
    _eigvals, _eigvecs = np.linalg.eigh(_cov)
    PCA_TOP2 = _eigvecs[:, -2:][:, ::-1].astype(np.float32)                 # (784, 2)

    _proj = (_X_c @ PCA_TOP2).astype(np.float32)          # (N, 2)

    # Subsample for performance (keep ~800 per digit)
    _rng = np.random.RandomState(99)
    _idx = _rng.choice(len(_proj), size=min(1600, len(_proj)), replace=False)
    _proj_sub = _proj[_idx]
    _labels_sub = _labels[_idx]

    # Matched Gaussian noise (same shape, unit normal)
    _noise = _rng.randn(len(_proj_sub), 2).astype(np.float32)

    PCA_DATA = {
        "proj": _proj_sub.tolist(),
        "labels": _labels_sub.tolist(),
        "noise": _noise.tolist(),
    }
    return PCA_DATA, PCA_MEAN, PCA_TOP2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Let's visualize the forward process as in the first experiment
    """)
    return


@app.cell(hide_code=True)
def _(pca_fwd_widget):
    pca_fwd_widget
    return


@app.cell(hide_code=True)
def _(eqx, jax, jnp):
    @eqx.filter_jit
    def get_timestep_embedding(t: jax.Array, freq_bands: jax.Array) -> jax.Array:
        _t = jnp.asarray(t, dtype=jnp.float32).reshape(())
        _args = _t * freq_bands
        return jnp.concatenate([jnp.sin(_args), jnp.cos(_args)])

    def precompute_frequencies(dim, max_pos, theta=10000.0):
        _inv_freq = 1.0 / theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32)[:dim // 2] / dim)
        _t = jnp.arange(0, max_pos, dtype=jnp.float32)
        _freqs = jnp.outer(_t, _inv_freq)
        return (jnp.cos(_freqs), jnp.sin(_freqs))

    def calculate_rope(x, cos_freq, sin_freq):
        _sin = jax.lax.expand_dims(sin_freq, (1,))
        _cos = jax.lax.expand_dims(cos_freq, (1,))
        _x1 = x[..., 0::2]
        _x2 = x[..., 1::2]
        _pos_embed = jnp.stack([_x1 * _cos - _x2 * _sin, _x1 * _sin + _x2 * _cos], axis=-1)
        _pos_embed = jax.lax.collapse(_pos_embed, -2)
        return _pos_embed.astype(x.dtype)

    class RMSNorm(eqx.Module):
        weight: jax.Array
        eps: float

        def __init__(self, dim: int, eps: float=1e-06):
            self.weight = jnp.ones((dim,))
            self.eps = eps

        def __call__(self, x: jax.Array) -> jax.Array:
            _rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
            return x / _rms * self.weight

    class MnistFeedForward(eqx.Module):
        mlp_gate: eqx.nn.Linear
        mlp_value: eqx.nn.Linear
        output_linear: eqx.nn.Linear
        dropout: eqx.nn.Dropout

        def __init__(self, emb_dim: int, dropout_rate: float, key):
            _k1, _k2, _k3 = jax.random.split(key, 3)
            _hidden_dim = int(4 * emb_dim * 2 / 3)
            self.mlp_gate = eqx.nn.Linear(emb_dim, _hidden_dim, use_bias=False, key=_k1)
            self.mlp_value = eqx.nn.Linear(emb_dim, _hidden_dim, use_bias=False, key=_k2)
            self.output_linear = eqx.nn.Linear(_hidden_dim, emb_dim, key=_k3)
            self.dropout = eqx.nn.Dropout(dropout_rate)

        def __call__(self, x: jax.Array, key=None) -> jax.Array:
            _g = self.mlp_gate(x)
            _v = self.mlp_value(x)
            _x = jax.nn.silu(_g) * _v
            _x = self.output_linear(_x)
            return self.dropout(_x, key=key, inference=key is None)

    class MnistAttnBlock(eqx.Module):
        mha: eqx.nn.MultiheadAttention
        ffn: MnistFeedForward
        ln1: RMSNorm
        ln2: RMSNorm
        q_norm: RMSNorm
        k_norm: RMSNorm
        adaLN_proj: eqx.nn.Linear
        dropout: eqx.nn.Dropout
        n_heads: int
        head_dim: int

        def __init__(self, n_heads: int, emb_dim: int, head_dim: int, cond_dim: int, dropout_rate: float, key):
            _keys = jax.random.split(key, 3)
            self.mha = eqx.nn.MultiheadAttention(num_heads=n_heads, query_size=emb_dim, use_query_bias=True, use_key_bias=True, use_value_bias=True, use_output_bias=True, dropout_p=dropout_rate, key=_keys[0])
            self.ffn = MnistFeedForward(emb_dim, dropout_rate, key=_keys[1])
            self.ln1 = RMSNorm(emb_dim)
            self.ln2 = RMSNorm(emb_dim)
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

            _adaLN_linear = eqx.nn.Linear(cond_dim, 6 * emb_dim, use_bias=True, key=_keys[2])
            _adaLN_linear = eqx.tree_at(lambda l: l.weight, _adaLN_linear, jnp.zeros_like(_adaLN_linear.weight))
            _adaLN_linear = eqx.tree_at(lambda l: l.bias, _adaLN_linear, jnp.zeros_like(_adaLN_linear.bias))
            self.adaLN_proj = _adaLN_linear
            self.dropout = eqx.nn.Dropout(dropout_rate)
            self.n_heads = n_heads
            self.head_dim = head_dim

        def __call__(self, x: jax.Array, cos_freq: jax.Array, sin_freq: jax.Array, cond_emb: jax.Array, key=None):
            _T, _ = x.shape
            _k1, _k2, _k3 = (None, None, None)
            if key is not None:
                _k1, _k2, _k3 = jax.random.split(key, 3)
            _modulation = self.adaLN_proj(cond_emb)
            _shift_1, _scale_1, _scale_attn, _shift_2, _scale_2, _scale_mlp = jnp.split(_modulation, 6, axis=-1)

            _norm_x = jax.vmap(self.ln1)(x)
            _norm_x = _norm_x * (1.0 + _scale_1[None, :]) + _shift_1[None, :]

            def process_heads(q, k, v):
                q = jax.vmap(jax.vmap(self.q_norm))(q)
                k = jax.vmap(jax.vmap(self.k_norm))(k)
                q = calculate_rope(q, cos_freq, sin_freq)
                k = calculate_rope(k, cos_freq, sin_freq)
                return (q, k, v)

            _attn_out = self.mha(query=_norm_x, key_=_norm_x, value=_norm_x, key=_k1, inference=key is None, process_heads=process_heads)
            _attn_out = self.dropout(_attn_out, key=_k2, inference=key is None)
            x = x + _attn_out * _scale_attn[None, :]

            _norm_x2 = jax.vmap(self.ln2)(x)
            _norm_x2 = _norm_x2 * (1.0 + _scale_2[None, :]) + _shift_2[None, :]
            if key is not None:
                _ffn_keys = jax.random.split(_k3, _T)
                _ffn_out = jax.vmap(self.ffn)(_norm_x2, _ffn_keys)
            else:
                _ffn_out = jax.vmap(self.ffn, in_axes=(0, None))(_norm_x2, None)
            x = x + _ffn_out * _scale_mlp[None, :]
            return x

    return MnistAttnBlock, get_timestep_embedding, precompute_frequencies


@app.cell(hide_code=True)
def _(
    MnistAttnBlock,
    eqx,
    get_timestep_embedding,
    jax,
    jnp,
    mnist_config,
    precompute_frequencies,
):
    class JiTTiny(eqx.Module):
        freq_bands: jax.Array
        time_emb_dim: int
        Patch: int
        emb_dim: int
        head_size: int
        n_heads: int
        class_tokens: int
        bottleneck_dim: int
        in_channels: int
        image_size: int

        patch_proj1: eqx.nn.Linear
        patch_proj2: eqx.nn.Linear
        class_token_proj: eqx.nn.Linear

        pos_emb: jax.Array
        cond_proj: eqx.nn.Linear
        blocks: list
        out_proj: eqx.nn.Linear
        class_emb: eqx.nn.Embedding

        def __init__(self, key=None):
            if key is None: key = jax.random.PRNGKey(0)
            _k1, _k2, _k3, _k4, _k5, _k6, _k7 = jax.random.split(key, 7)

            self.in_channels = mnist_config["in_channels"]
            self.image_size = mnist_config["image_size"]
            self.time_emb_dim = mnist_config["emb_dim"]
            self.Patch = mnist_config["patch_size"]
            self.emb_dim = mnist_config["emb_dim"]
            self.class_tokens = mnist_config["class_tokens"]
            self.bottleneck_dim = mnist_config["bottleneck_dim"]
            self.n_heads = mnist_config["n_heads"]
            self.head_size = self.emb_dim // self.n_heads

            _half_dim = self.time_emb_dim // 2
            self.freq_bands = jnp.exp(-jnp.log(10000.0) * jnp.arange(_half_dim, dtype=jnp.float32) / _half_dim)

            _patch_dim = self.in_channels * self.Patch * self.Patch
            self.patch_proj1 = eqx.nn.Linear(_patch_dim, self.bottleneck_dim, key=_k1)
            self.patch_proj2 = eqx.nn.Linear(self.bottleneck_dim, self.emb_dim, key=_k2)

            _num_patches = (self.image_size // self.Patch) * (self.image_size // self.Patch)
            self.pos_emb = jax.random.normal(_k3, (_num_patches, self.emb_dim)) * 0.02

            if self.class_tokens > 0:
                self.class_token_proj = eqx.nn.Linear(self.time_emb_dim, self.emb_dim, key=_k4)
            else:
                self.class_token_proj = None

            self.cond_proj = eqx.nn.Linear(self.time_emb_dim, self.emb_dim, key=_k5)

            _block_keys = jax.random.split(_k6, mnist_config["n_layers"])
            self.blocks = [
                MnistAttnBlock(
                    n_heads=self.n_heads,
                    emb_dim=self.emb_dim,
                    head_dim=self.head_size,
                    cond_dim=self.time_emb_dim,
                    dropout_rate=mnist_config["dropout_rate"],
                    key=_block_keys[i]
                )
                for i in range(mnist_config["n_layers"])
            ]

            self.out_proj = eqx.nn.Linear(self.emb_dim, _patch_dim, key=_k7)
            self.class_emb = eqx.nn.Embedding(num_embeddings=10, embedding_size=self.time_emb_dim, key=_k7)

        def __call__(self, x: jax.Array, t: jax.Array, y: jax.Array) -> jax.Array:
            _t_emb = get_timestep_embedding(t * 1000.0, self.freq_bands)
            _class_emb = self.class_emb(y)
            _cond_emb = _t_emb + _class_emb

            C, H, W = x.shape
            _num_patches_h = H // self.Patch
            _num_patches_w = W // self.Patch
            _T = _num_patches_h * _num_patches_w

            _patches = x.reshape(C, _num_patches_h, self.Patch, _num_patches_w, self.Patch)
            _patches = jnp.transpose(_patches, (1, 3, 0, 2, 4))
            _patches = _patches.reshape(_T, -1)

            _h = jax.vmap(self.patch_proj1)(_patches)
            _h = jax.vmap(self.patch_proj2)(_h)
            _h = _h + self.pos_emb

            if self.class_tokens > 0:
                _c_token = self.class_token_proj(_class_emb)
                _c_tokens = jnp.repeat(_c_token[None, :], self.class_tokens, axis=0)
                _h = jnp.concatenate([_c_tokens, _h], axis=0)

            _T_total = _h.shape[0]
            _cos_freq, _sin_freq = precompute_frequencies(self.head_size, _T_total)

            for _block in self.blocks:
                _h = _block(_h, _cos_freq, _sin_freq, cond_emb=_cond_emb)

            if self.class_tokens > 0:
                _h = _h[self.class_tokens:]

            _out_patches = jax.vmap(self.out_proj)(_h)
            _reconstructed = _out_patches.reshape(_num_patches_h, _num_patches_w, C, self.Patch, self.Patch)
            _reconstructed = jnp.transpose(_reconstructed, (2, 0, 3, 1, 4))
            return _reconstructed.reshape(C, H, W)

    return (JiTTiny,)


@app.cell(hide_code=True)
def _(eqx, jax, jnp):
    @eqx.filter_jit  
    def loss_fn_mnist(model, x_0, y, key):  
        B = x_0.shape[0]  
        _k1, _k2 = jax.random.split(key)  

        _logit_t = jax.random.normal(_k1, (B, 1, 1, 1)) * 0.8 - 0.8
        _t = jax.nn.sigmoid(_logit_t)
        _noise = jax.random.normal(_k2, x_0.shape)  

        _z_t = _t * x_0 + (1.0 - _t) * _noise  
        _v = x_0 - _noise

        _model_vmap = eqx.filter_vmap(model)  
        _pred_x_0 = _model_vmap(_z_t, _t, y)

        _pred_v = (_pred_x_0 - _z_t) / jnp.clip(1.0 - _t, 0.05) 
        return jnp.mean((_v - _pred_v) ** 2)  

    @eqx.filter_jit
    def train_step_mnist(model, optimizer, opt_state, x_0, y, key):
        _loss, _grads = eqx.filter_value_and_grad(loss_fn_mnist)(model, x_0, y, key)
        _updates, _new_opt_state = optimizer.update(_grads, opt_state, params=model)
        _new_model = eqx.apply_updates(model, _updates)
        return _new_model, _new_opt_state, _loss

    return (train_step_mnist,)


@app.cell(hide_code=True)
def _(eqx, jax, jnp, mnist_config):
    @eqx.filter_jit
    def generate_images_heun(model, num_samples, y, key, num_steps=None, return_trajectory=False):
        _init_key, _ = jax.random.split(key)
        _z = jax.random.normal(_init_key, (num_samples, mnist_config["in_channels"], mnist_config["image_size"], mnist_config["image_size"]))

        steps = mnist_config["sampling_steps"] if num_steps is None else num_steps
        _ts = jnp.linspace(0.0, 1.0, steps + 1)
        _model_vmap = eqx.filter_vmap(model)

        def step_fn(carry_z, i):
            _curr_z = carry_z
            _t = _ts[i]
            _t_next = _ts[i + 1]
            _dt = _t_next - _t

            _t_batch = jnp.full((num_samples, 1, 1, 1), _t)
            _pred_x0_1 = _model_vmap(_curr_z, _t_batch, y)
            _v1 = (_pred_x0_1 - _curr_z) / jnp.clip(1.0 - _t, 0.05)

            _z_target = _curr_z + _dt * _v1

            _t_next_batch = jnp.full((num_samples, 1, 1, 1), _t_next)
            _pred_x0_2 = _model_vmap(_z_target, _t_next_batch, y)
            _v2 = (_pred_x0_2 - _z_target) / jnp.clip(1.0 - _t_next, 0.05)

            next_z = _curr_z + _dt * 0.5 * (_v1 + _v2)
            return next_z, _curr_z

        final_z, trajectory = jax.lax.scan(step_fn, _z, jnp.arange(steps))
        if return_trajectory:
            return jnp.concatenate([trajectory, final_z[None, ...]], axis=0)
        return final_z

    return (generate_images_heun,)


@app.cell(hide_code=True)
def _(MNIST_X_RAW, MNIST_Y_RAW, jax, jnp, mnist_form, np):
    if mnist_form.value is not None:
        # Training has actually been kicked off — only now do we pay the
        # cost of padding the images and moving them onto the GPU.
        _x_train_padded = np.pad(MNIST_X_RAW, ((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=-1.0)
        _x_train_padded = _x_train_padded[:, None, :, :]

        MNIST_X_GPU = jax.device_put(jnp.array(_x_train_padded, dtype=jnp.bfloat16))
        MNIST_Y_GPU = jax.device_put(MNIST_Y_RAW)
    else:
        MNIST_X_GPU = None
        MNIST_Y_GPU = None

    def create_jax_stream(x_data, y_data, batch_size, start_key):
        _key = start_key
        _num_samples = x_data.shape[0]
        while True:
            _key, _subkey = jax.random.split(_key)
            _indices = jax.random.permutation(_subkey, _num_samples)[:batch_size]
            yield (x_data[_indices], y_data[_indices], _key)

    return MNIST_X_GPU, MNIST_Y_GPU, create_jax_stream


@app.cell(hide_code=True)
def _(
    JiTTiny,
    MNIST_X_GPU,
    MNIST_Y_GPU,
    create_jax_stream,
    eqx,
    jax,
    mnist_config,
    mnist_form,
    mo,
    optax,
    train_step_mnist,
):
    if mnist_form.value is not None:
        def _run_training():
            _key = jax.random.PRNGKey(42)
            _key, _model_key, _stream_key = jax.random.split(_key, 3)
            _stream = create_jax_stream(MNIST_X_GPU, MNIST_Y_GPU, mnist_config["batch_size"], _stream_key)

            _current_model = JiTTiny(key=_model_key)

            _steps_per_epoch = len(MNIST_X_GPU) // mnist_config["batch_size"]
            _warmup_steps = mnist_config["warmup_epochs"] * _steps_per_epoch

            _warmup_schedule = optax.linear_schedule(init_value=0.0, end_value=mnist_config["learning_rate"], transition_steps=_warmup_steps)
            _constant_schedule = optax.constant_schedule(mnist_config["learning_rate"])
            _lr_schedule = optax.join_schedules([_warmup_schedule, _constant_schedule], boundaries=[_warmup_steps])

            _optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate=_lr_schedule, b1=0.9, b2=0.95, weight_decay=0.001)
            )
            _opt_state = _optimizer.init(eqx.filter(_current_model, eqx.is_inexact_array))

            _losses = []
            for _epoch in mo.status.progress_bar(
                range(mnist_config["num_epochs"]),
                title="JiT-Tiny MNIST Training",
                subtitle="Default Setting make the training time about 2-3 minutes",
                show_eta=True,
                show_rate=True,
            ):
                _epoch_loss = 0.0
                for _step in range(_steps_per_epoch):
                    _x, _y, _ = next(_stream)
                    _key, _step_key = jax.random.split(_key)
                    _current_model, _opt_state, _loss_val = train_step_mnist(_current_model, _optimizer, _opt_state, _x, _y, _step_key)
                    _epoch_loss += _loss_val
                _losses.append(float(_epoch_loss) / _steps_per_epoch)
            return _current_model, _losses

        MNIST_TRAINED_MODEL, MNIST_LOSS_HISTORY = _run_training()
        jax.clear_caches()
    else:
        MNIST_TRAINED_MODEL = None
        MNIST_LOSS_HISTORY = None
    return MNIST_LOSS_HISTORY, MNIST_TRAINED_MODEL


@app.cell(hide_code=True)
def _(MNIST_TRAINED_MODEL, eqx, generate_images_heun, jax, jnp, np):
    if MNIST_TRAINED_MODEL is not None:
        _all_digits_pack = {}
        _inference_key = jax.random.PRNGKey(1337)

        # 1. Precompute 8 sample images for every single digit 0-9
        for _digit in range(10):
            _y_target = jnp.full((8,), _digit, dtype=jnp.int32)
            _gen_padded = generate_images_heun(MNIST_TRAINED_MODEL, 8, _y_target, _inference_key)
            _cropped = _gen_padded[:, :, 2:-2, 2:-2]
            _scaled = jnp.clip((_cropped + 1.0) / 2.0, 0.0, 1.0)
            _all_digits_pack[str(_digit)] = np.array(_scaled)[:, 0].tolist()

        # 2. Precompute dynamic trajectories and inner visions for Digit 7
        _init_noise = jax.random.normal(jax.random.PRNGKey(80085), (1, 1, 32, 32))
        _ts = jnp.linspace(0.0, 1.0, 25)
        _model_vmap = eqx.filter_vmap(MNIST_TRAINED_MODEL)
        _y_label = jnp.array([7], dtype=jnp.int32)

        _z_frames, _x0_frames = [], []
        _curr_z = _init_noise

        for _idx in range(24):
            _t, _t_next = _ts[_idx], _ts[_idx + 1]
            _dt = _t_next - _t

            _z_frames.append(np.array(jnp.clip((_curr_z[0, 0, 2:-2, 2:-2] + 1.0) / 2.0, 0.0, 1.0)).tolist())

            _t_batch = jnp.full((1, 1, 1, 1), _t)
            _pred_x0 = _model_vmap(_curr_z, _t_batch, _y_label)
            _x0_frames.append(np.array(jnp.clip((_pred_x0[0, 0, 2:-2, 2:-2] + 1.0) / 2.0, 0.0, 1.0)).tolist())

            _v1 = (_pred_x0 - _curr_z) / jnp.clip(1.0 - _t, 0.05)
            _z_euler = _curr_z + _dt * _v1

            _pred_x0_next = _model_vmap(_z_euler, jnp.full((1, 1, 1, 1), _t_next), _y_label)
            _v2 = (_pred_x0_next - _z_euler) / jnp.clip(1.0 - _t_next, 0.05)
            _curr_z = _curr_z + _dt * 0.5 * (_v1 + _v2)

        _z_frames.append(np.array(jnp.clip((_curr_z[0, 0, 2:-2, 2:-2] + 1.0) / 2.0, 0.0, 1.0)).tolist())
        _final_pred = _model_vmap(_curr_z, jnp.full((1, 1, 1, 1), 1.0), _y_label)
        _x0_frames.append(np.array(jnp.clip((_final_pred[0, 0, 2:-2, 2:-2] + 1.0) / 2.0, 0.0, 1.0)).tolist())

        MNIST_UI_PACK = {
            "digits": _all_digits_pack,
            "z_buffer": _z_frames,
            "x0_buffer": _x0_frames
        }
        # Free generation caches after packing UI data
        jax.clear_caches()
    else:
        MNIST_UI_PACK = None
    return (MNIST_UI_PACK,)


@app.cell(hide_code=True)
def _(MNIST_UI_PACK, anywidget, json, traitlets):
    class MnistGenerationGridWidget(anywidget.AnyWidget):
        _esm = """
        export default {
            render({ model, el }) {
                let digits_all = {};

                el.innerHTML = `
                <div style="display: flex; flex-direction: column; gap: 16px; font-family: system-ui, sans-serif; background: #f8fafc; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; max-width: 900px; margin: 0 auto; box-shadow: 0 4px 10px rgba(0,0,0,0.05); box-sizing: border-box;">
                    <div style="text-align: center; width: 100%; border-bottom: 1px solid #e2e8f0; padding-bottom: 10px;">
                        <div style="font-size: 15px; font-weight: 700; letter-spacing: 0.5px;">Let's Check Our Generations 👀</div>
                    </div>
                    <div>
                        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 14px; background: #f8fafc; padding: 10px 18px; border-radius: 8px; border: 1px solid #e2e8f0;">
                            <span style="font-size: 13px; font-weight: 600; color: #475569;">Target Digit Profile:</span>
                            <input type="range" id="digitClassSlider" min="0" max="9" step="1" value="0" style="flex-grow: 1; accent-color: #2563eb; cursor: pointer; margin: 0;">
                            <span id="digitClassLabel" style="font-family: monospace; font-size: 14px; color: #2563eb; font-weight: 700; min-width: 30px; text-align: right;">0</span>
                        </div>
                        <div style="display: flex; gap: 8px; justify-content: space-between; background: #f1f5f9; padding: 16px; border-radius: 10px; border: 1px solid #cbd5e1;" id="digitGridContainer">
                        </div>
                    </div>
                </div>
                `;

                let grid_box = el.querySelector("#digitGridContainer");
                let canvases = [];
                for (let idx = 0; idx < 8; idx++) {
                    let canv = document.createElement("canvas");
                    canv.width = 56;
                    canv.height = 56;
                    canv.style.background = "#000";
                    canv.style.borderRadius = "4px";
                    grid_box.appendChild(canv);
                    canvases.push(canv);
                }

                let slider_class = el.querySelector("#digitClassSlider");
                let lbl_class = el.querySelector("#digitClassLabel");

                function update_digit_row(digit_val) {
                    lbl_class.textContent = String(digit_val);
                    let imgs = digits_all[String(digit_val)];
                    if (!imgs) return;
                    for (let i = 0; i < 8; i++) {
                        let img_data = imgs[i];
                        let ctx = canvases[i].getContext("2d");
                        ctx.clearRect(0,0,56,56);
                        for (let r = 0; r < 28; r++) {
                            for (let c = 0; c < 28; c++) {
                                let val = Math.floor(img_data[r][c] * 255);
                                ctx.fillStyle = `rgb(${val},${val},${val})`;
                                ctx.fillRect(c*2, r*2, 2, 2);
                            }
                        }
                    }
                }

                slider_class.addEventListener("input", () => {
                    update_digit_row(parseInt(slider_class.value));
                });

                function update() {
                    let config = JSON.parse(model.get("digits_json"));
                    if (config && config.digits) {
                        digits_all = config.digits;
                        update_digit_row(parseInt(slider_class.value));
                    }
                }

                model.on("change:digits_json", update);
                update();
            }
        }
        """
        digits_json = traitlets.Unicode("{}").tag(sync=True)

    mnist_generation_widget = MnistGenerationGridWidget()
    mnist_generation_widget.digits_json = (
        json.dumps({"digits": MNIST_UI_PACK["digits"]}) if MNIST_UI_PACK is not None else "{}"
    )
    return (mnist_generation_widget,)


@app.cell(hide_code=True)
def _(MNIST_UI_PACK, anywidget, json, traitlets):
    class MnistModelSeesWidget(anywidget.AnyWidget):
        _esm = """
        export default {
            render({ model, el }) {
                let z_buf = [], x0_buf = [];
                let anim_state = { playing: false, interval_id: null };

                el.innerHTML = `
                <div style="display: flex; flex-direction: column; gap: 20px; font-family: system-ui, sans-serif; background: #f8fafc; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; max-width: 900px; margin: 0 auto; box-shadow: 0 4px 10px rgba(0,0,0,0.05); box-sizing: border-box;">
                    <div style="text-align: center; width: 100%; border-bottom: 1px solid #e2e8f0; padding-bottom: 10px;">
                        <div style="font-size: 15px; font-weight: 700; margin-bottom: 4px; letter-spacing: 0.5px;">The Denoising Process</div>
                        <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Compare the noisy trajectory state x<sub>t</sub> alongside the model's clean image projection x-pred at each step</div>
                    </div>

                    <div style="display: flex; flex-direction: row; gap: 24px; justify-content: center; align-items: center; width: 100%; flex-wrap: wrap;">
                        <!-- Canvas Card 1: Noisy State -->
                        <div style="display: flex; flex-direction: column; align-items: center; gap: 8px; background: #f8fafc; padding: 14px; border-radius: 10px; border: 1px solid #e2e8f0; width: 220px; box-sizing: border-box;">
                            <span style="font-size: 11px; font-weight: 700; color: #475569;">Noisy State (x<sub>t</sub>)</span>
                            <canvas id="animCanvasZ" width="196" height="196" style="background: #000; border-radius: 6px; border: 1px solid #cbd5e1; display: block;"></canvas>
                        </div>

                        <!-- Canvas Card 2: Reconstructed Clean -->
                        <div style="display: flex; flex-direction: column; align-items: center; gap: 8px; background: #f8fafc; padding: 14px; border-radius: 10px; border: 1px solid #e2e8f0; width: 220px; box-sizing: border-box;">
                            <span style="font-size: 11px; font-weight: 700; color: #475569;">Clean Prediction (x-pred)</span>
                            <canvas id="animCanvasX0" width="196" height="196" style="background: #000; border-radius: 6px; border: 1px solid #cbd5e1; display: block;"></canvas>
                        </div>
                    </div>

                    <div style="display: flex; align-items: center; gap: 14px; background: #f8fafc; padding: 10px 20px; border-radius: 10px; border: 1px solid #e2e8f0; box-sizing: border-box;">
                        <button id="animPlayBtn" style="background: #f1f5f9; border: 1px solid #2563eb; border-radius: 6px; padding: 6px 14px; font-size: 13px; font-weight: 500; color: #2563eb; cursor: pointer; min-width: 70px; transition: all 0.2s; display: flex; align-items: center; justify-content: center; gap: 4px;">▶ play</button>
                        <input type="range" id="animSlider" min="0" max="24" step="1" value="0" style="flex-grow: 1; accent-color: #2563eb; cursor: pointer; margin: 0;">
                        <span id="animLabel" style="font-family: monospace; font-size: 13px; color: #2563eb; font-weight: 700; min-width: 130px; text-align: right;">Step: 1/25 (t=0.00)</span>
                    </div>
                </div>
                `;

                let c_z = el.querySelector("#animCanvasZ");
                let c_x0 = el.querySelector("#animCanvasX0");
                let ctx_z = c_z.getContext("2d");
                let ctx_x0 = c_x0.getContext("2d");

                let slider_anim = el.querySelector("#animSlider");
                let btn_play = el.querySelector("#animPlayBtn");
                let lbl_anim = el.querySelector("#animLabel");

                function draw_anim_frame(frame_idx) {
                    let t_val = frame_idx / 24.0;
                    lbl_anim.textContent = `Step: ${frame_idx+1}/25 (t=${t_val.toFixed(2)})`;

                    if (z_buf.length === 0) return;

                    ctx_z.clearRect(0,0,200,200);
                    let mat_z = z_buf[frame_idx];
                    for (let r = 0; r < 28; r++) {
                        for (let c = 0; c < 28; c++) {
                            let v = Math.floor(mat_z[r][c] * 255);
                            ctx_z.fillStyle = `rgb(${v},${v},${v})`;
                            ctx_z.fillRect(c*7, r*7, 7, 7);
                        }
                    }

                    ctx_x0.clearRect(0,0,200,200);
                    let mat_x0 = x0_buf[frame_idx];
                    for (let r = 0; r < 28; r++) {
                        for (let c = 0; c < 28; c++) {
                            let v = Math.floor(mat_x0[r][c] * 255);
                            ctx_x0.fillStyle = `rgb(${v},${v},${v})`;
                            ctx_x0.fillRect(c*7, r*7, 7, 7);
                        }
                    }
                }

                function tick_anim() {
                    let step = parseInt(slider_anim.value) + 1;
                    if (step > 24) step = 0;
                    slider_anim.value = String(step);
                    draw_anim_frame(step);
                }

                btn_play.addEventListener("click", () => {
                    if (anim_state.playing) {
                        clearInterval(anim_state.interval_id);
                        anim_state.playing = false;
                        btn_play.textContent = "▶ play";
                    } else {
                        anim_state.interval_id = window.setInterval(tick_anim, 250);
                        anim_state.playing = true;
                        btn_play.textContent = "⏸ pause";
                    }
                });

                slider_anim.addEventListener("input", () => {
                    if (anim_state.playing) {
                        clearInterval(anim_state.interval_id);
                        anim_state.playing = false;
                        btn_play.textContent = "▶ play";
                    }
                    draw_anim_frame(parseInt(slider_anim.value));
                });

                function update() {
                    let config = JSON.parse(model.get("trajectory_json"));
                    if (config && config.z_buffer) {
                        z_buf = config.z_buffer;
                        x0_buf = config.x0_buffer;
                        draw_anim_frame(parseInt(slider_anim.value));
                    }
                }

                model.on("change:trajectory_json", update);
                update();
            }
        }
        """
        trajectory_json = traitlets.Unicode("{}").tag(sync=True)

    mnist_model_sees_widget = MnistModelSeesWidget()
    mnist_model_sees_widget.trajectory_json = (
        json.dumps({"z_buffer": MNIST_UI_PACK["z_buffer"], "x0_buffer": MNIST_UI_PACK["x0_buffer"]})
        if MNIST_UI_PACK is not None
        else "{}"
    )
    return (mnist_model_sees_widget,)


@app.cell(hide_code=True)
def _(MNIST_LOSS_HISTORY, MNIST_TRAINED_MODEL, mo, plt):
    if MNIST_TRAINED_MODEL is not None and MNIST_LOSS_HISTORY is not None:
        _fig, _ax = plt.subplots(figsize=(6, 3.5))
        _ax.plot(
            range(1, len(MNIST_LOSS_HISTORY) + 1),
            MNIST_LOSS_HISTORY,
            marker='o',
            color='#2563eb',
            linewidth=2.0,
        )
        _ax.set_title("JiT-Tiny MNIST Training Loss", fontsize=11, fontweight='bold', pad=10)
        _ax.set_xlabel("Epoch", fontsize=9)
        _ax.set_ylabel("Average Loss", fontsize=9)
        _ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        _ax.set_xticks(range(1, len(MNIST_LOSS_HISTORY) + 1))
        _fig.tight_layout()
        _o = mo.center(_fig)
    else:
        _o = None
    _o
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###We can verify the training process by looking at the samples
    """)
    return


@app.cell(hide_code=True)
def _(MNIST_TRAINED_MODEL, mnist_generation_widget):
    if MNIST_TRAINED_MODEL is not None:
        _o = mnist_generation_widget
    else:
        _o = None
    _o
    return


@app.cell(hide_code=True)
def _(
    MNIST_TRAINED_MODEL,
    MNIST_X_RAW,
    MNIST_Y_RAW,
    PCA_MEAN,
    PCA_TOP2,
    generate_images_heun,
    jax,
    jnp,
    np,
):
    if MNIST_TRAINED_MODEL is not None:
        _N_PER_DIGIT = 80
        _N_STEPS = 40

        _key = jax.random.PRNGKey(777)
        _k6, _k7 = jax.random.split(_key)

        _traj_6 = generate_images_heun(MNIST_TRAINED_MODEL, _N_PER_DIGIT, jnp.full((_N_PER_DIGIT,), 6, dtype=jnp.int32), _k6, num_steps=_N_STEPS, return_trajectory=True)
        _traj_7 = generate_images_heun(MNIST_TRAINED_MODEL, _N_PER_DIGIT, jnp.full((_N_PER_DIGIT,), 7, dtype=jnp.int32), _k7, num_steps=_N_STEPS, return_trajectory=True)

        _t6_flat = _traj_6[:, :, 0, 2:-2, 2:-2].reshape(-1, 784)
        _t7_flat = _traj_7[:, :, 0, 2:-2, 2:-2].reshape(-1, 784)

        _t6_pca = (np.array(_t6_flat) - PCA_MEAN) @ PCA_TOP2
        _t7_pca = (np.array(_t7_flat) - PCA_MEAN) @ PCA_TOP2

        _n_frames = _N_STEPS + 1
        _t6_pca = _t6_pca.reshape(_n_frames, _N_PER_DIGIT, 2)
        _t7_pca = _t7_pca.reshape(_n_frames, _N_PER_DIGIT, 2)

        _n_total = _N_PER_DIGIT * 2
        _traj = np.zeros((_n_total, _n_frames, 2), dtype=np.float32)
        _traj[:_N_PER_DIGIT] = _t6_pca.transpose(1, 0, 2)
        _traj[_N_PER_DIGIT:] = _t7_pca.transpose(1, 0, 2)

        _labels_rev = np.zeros(_n_total, dtype=np.int32)
        _labels_rev[:_N_PER_DIGIT] = 6
        _labels_rev[_N_PER_DIGIT:] = 7

        _mask = (MNIST_Y_RAW == 6) | (MNIST_Y_RAW == 7)
        _X_flat = MNIST_X_RAW[_mask].reshape(-1, 784).astype(np.float64)
        _proj_all = (_X_flat - PCA_MEAN) @ PCA_TOP2
        _rng = np.random.RandomState(99)
        _gt_idx = _rng.choice(len(_proj_all), size=min(1600, len(_proj_all)), replace=False)

        PCA_REVERSE = {
            "mode": "reverse",
            "title": "Reverse Process",
            "subtitle": "The trained model steers noise samples toward the data distribution (PCA view 6 & 7)",
            "theme": "light",
            "accent_color": "#2563eb",
            "legend": [
                {"color": "rgba(148,163,184,0.35)", "label": "digit 6 (target)"},
                {"color": "rgba(148,163,184,0.35)", "label": "digit 7 (target)"},
                {"color": "#2563eb", "label": "sampled 6"},
                {"color": "#8b5cf6", "label": "sampled 7"}
            ],
            "traj": _traj.tolist(),
            "labels_rev": _labels_rev.tolist(),
            "n_steps": int(_n_frames),
            "gt_proj": _proj_all[_gt_idx].tolist(),
            "gt_labels": MNIST_Y_RAW[_mask][_gt_idx].tolist()
        }
        jax.clear_caches()
    else:
        PCA_REVERSE = None
    return (PCA_REVERSE,)


@app.cell(hide_code=True)
def _(MNIST_TRAINED_MODEL, PCA_REVERSE, PcaWidget, json):
    if MNIST_TRAINED_MODEL is not None and PCA_REVERSE is not None:
        pca_rev_widget = PcaWidget()
        pca_rev_widget.config_json = json.dumps(PCA_REVERSE)
    else:
        pca_rev_widget = None
    return (pca_rev_widget,)


@app.cell(hide_code=True)
def _(MNIST_TRAINED_MODEL, pca_rev_widget):
    if MNIST_TRAINED_MODEL is not None:
        _o = pca_rev_widget
    else:
        _o = None
    _o
    return


@app.cell(hide_code=True)
def _(MNIST_TRAINED_MODEL, mnist_model_sees_widget):
    if MNIST_TRAINED_MODEL is not None:
        _o = mnist_model_sees_widget
    else:
        _o = None
    _o
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    More Stuff:

    1.  The loss function of $x$-prediction can be viewed as a special case of **[Mean Flows for One-step Generative Modeling](https://www.alphaxiv.org/abs/2505.13447)** where the $t_{\text{end}}$ is fixed to the data manifold.

    2.  Similar principles apply to language models. In **[Flow Map Language Models](https://www.alphaxiv.org/abs/2602.16813)** , similar results were obtained for $x$-pred, $v$-pred, and $\epsilon$-pred. There they used **token-prediction** as $x$-pred.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Thank you for interacting with the notebook, The visuals and explanation were done by the author of the notebook. while, most of the code was generated on Antigravity by Opus 4.6 and Gemini 3.5 Flash
    """)
    return


if __name__ == "__main__":
    app.run()
