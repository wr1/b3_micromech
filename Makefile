# Run examples with the b3-tex env (override if needed):
#   make solve RUN="micromamba run -n b3-micromech"
#   make solve RUN="uv run b3-micromech"
RUN ?= micromamba run -n b3-tex
OUT ?= results

SQUARE_YAML = examples/ud_transverse.yaml
HEX_YAML = examples/ud_transverse_hex.yaml
HEX_AMR_YAML = examples/ud_transverse_hex_amr.yaml
AMR_YAML = examples/ud_transverse_amr.yaml
SWEEP_YAML = examples/sweep_default.yaml
SWEEP_HEX_YAML = examples/sweep_hex_hypercube.yaml
SWEEP_HEX_3D_YAML = examples/sweep_hex_3d_response.yaml
SWEEP_HEX_HIGH_VF_YAML = examples/sweep_hex_high_vf.yaml
SURROGATE_OUT ?= results/surrogate_demo
SURROGATE_3D_OUT ?= results/surrogate_3d_demo
SURROGATE_HIGH_VF_OUT ?= results/surrogate_high_vf

.PHONY: help install test examples validate \
	solve solve-square solve-hex solve-amr solve-hex-amr \
	plot plot-square plot-hex plot-amr plot-hex-amr \
	sweep sweep-hex-hypercube sweep-hex-high-vf sweep-hex-3d-response \
	demo-surrogate demo-surrogate-3d demo-mesomech-batch \
	viz viz-square viz-hex viz-amr viz-hex-amr plot-loadcases

help: ## list targets
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## editable install into the active env
	$(RUN) pip install -e ".[viz,sweep,surrogate,test]"

test: ## run pytest
	$(RUN) pytest

examples: solve-square solve-hex sweep viz-square viz-hex ## run all bundled examples

validate: ## check YAML against Mori–Tanaka reference
	$(RUN) b3-micromech validate $(SQUARE_YAML)
	$(RUN) b3-micromech validate $(HEX_YAML)

solve: solve-square ## homogenize square RVE (default example)

solve-square: ## homogenize square transverse RVE
	$(RUN) b3-micromech solve $(SQUARE_YAML) --out $(OUT)/square

solve-hex: ## homogenize hexagonal transverse RVE
	$(RUN) b3-micromech solve $(HEX_YAML) --out $(OUT)/hex

solve-hex-amr: ## homogenize hex RVE with local_cloud + stiffness-jump AMR
	$(RUN) b3-micromech solve $(HEX_AMR_YAML) --out $(OUT)/hex_amr

solve-amr: ## homogenize with local_cloud sampling + stiffness-jump AMR
	$(RUN) b3-micromech solve $(AMR_YAML) --out $(OUT)/amr

plot: plot-square ## deformation plots for square RVE

plot-square: ## plot bundle for square RVE
	$(RUN) b3-micromech plot $(SQUARE_YAML) --out $(OUT)/square/plots

plot-hex: ## plot bundle for hexagonal RVE
	$(RUN) b3-micromech plot $(HEX_YAML) --out $(OUT)/hex/plots

plot-hex-amr: ## plot bundle for hex AMR RVE (includes amr_refinement.png)
	$(RUN) b3-micromech plot $(HEX_AMR_YAML) --out $(OUT)/hex_amr/plots

plot-amr: ## plot bundle for AMR RVE (includes amr_refinement.png)
	$(RUN) b3-micromech plot $(AMR_YAML) --out $(OUT)/amr/plots

sweep: ## hypercube sweep for surrogate training
	$(RUN) b3-micromech sweep $(SWEEP_YAML) --out $(OUT)/sweep

sweep-hex-hypercube: ## hex hypercube sweep (84 solves, mid-Vf + packing cluster)
	$(RUN) b3-micromech sweep $(SWEEP_HEX_YAML) --out $(SURROGATE_OUT)

sweep-hex-high-vf: ## high-Vf hex sweep (48 solves, packing-limit cluster only)
	$(RUN) b3-micromech sweep $(SWEEP_HEX_HIGH_VF_YAML) --out $(SURROGATE_HIGH_VF_OUT)

demo-surrogate: ## full chain: hex sweep → train MLP → 1000 preds → plots
	$(RUN) python examples/demo_surrogate_chain.py --out $(SURROGATE_OUT) --jobs 1

sweep-hex-3d-response: ## hex 3D-response sweep (Vf×E_m×E_Lf, 117 solves)
	$(RUN) b3-micromech sweep $(SWEEP_HEX_3D_YAML) --out $(SURROGATE_3D_OUT)

demo-surrogate-3d: ## 3D response surrogate: Vf×E_m×E_Lf → MLP → response surfaces
	$(RUN) python examples/demo_surrogate_3d_response.py --out $(SURROGATE_3D_OUT) --jobs 1

demo-mesomech-batch: ## vectorized FEA surrogate batch + b3_tex LUT registration
	$(RUN) python examples/demo_mesomech_batch.py --model $(SURROGATE_OUT)/surrogate_model.joblib

viz: viz-square ## solve + plot square RVE

viz-square: solve-square plot-square ## solve then plot square RVE

viz-hex: solve-hex plot-hex ## solve then plot hexagonal RVE

viz-hex-amr: solve-hex-amr plot-hex-amr ## solve then plot hexagonal AMR RVE

viz-amr: solve-amr plot-amr ## solve then plot AMR RVE

plot-loadcases: ## example script: six loadcase deformation figures
	$(RUN) python examples/plot_loadcases.py $(SQUARE_YAML) $(OUT)/square/plots