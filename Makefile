# Reproduction pipeline for the manuscript
# Magnetic-field-inhomogeneity bias in indirect Zeeman measurements
# of the positronium hyperfine interval (PRA AR12783)
#
# Usage:
#   make all           # run everything: self-tests + main + fase2 + figures
#   make reproduce     # regenerate Tables I, II and the Fase 2 numerical checks
#   make selftests     # run module-level verification suites
#   make main          # main pipeline → corrected_results.json
#   make fase2         # Fase 2 investigations (TM110, non-Hermitian, pressure)
#   make figures       # regenerate manuscript figures 1-3
#   make clean         # remove generated artifacts (JSON, figures, caches)
#
# Rationale: this Makefile is the single source of truth for what "reproduce the
# paper" means. Running `make reproduce` should regenerate every numerical value
# quoted in the manuscript. If the paper and the code drift apart, this target
# will surface the discrepancy instead of hiding it in manual runs.

PYTHON  ?= python3
REPLDIR := replication

SELFTESTS := $(REPLDIR)/breit_rabi.py \
             $(REPLDIR)/magnetic_field_models.py \
             $(REPLDIR)/ps_distributions.py

GENERATED := $(REPLDIR)/corrected_results.json \
             $(REPLDIR)/fig1_zeeman_eigenvalues.png \
             $(REPLDIR)/resonance_curves_corrected.png \
             $(REPLDIR)/ritter_c2_scan.png

.PHONY: all reproduce selftests main fase2 figures clean help

all: selftests reproduce figures

reproduce: main fase2

selftests:
	@echo "=== Module self-tests ==="
	@for f in $(SELFTESTS); do \
	    echo "--- $$f ---"; \
	    (cd $(REPLDIR) && $(PYTHON) $$(basename $$f)) || exit 1; \
	done

main: $(REPLDIR)/corrected_results.json

$(REPLDIR)/corrected_results.json: $(REPLDIR)/run_corrected_simulation.py \
                                   $(REPLDIR)/breit_rabi.py \
                                   $(REPLDIR)/magnetic_field_models.py \
                                   $(REPLDIR)/ps_distributions.py \
                                   $(REPLDIR)/zeeman_resonance.py
	@echo "=== Main simulation (Mills + Ritter + Ishida) ==="
	cd $(REPLDIR) && $(PYTHON) run_corrected_simulation.py

fase2:
	@echo "=== Fase 2: TM110, non-Hermitian, pressure curve ==="
	cd $(REPLDIR) && $(PYTHON) fase2_investigations.py

figures:
	@echo "=== Generate manuscript figures ==="
	cd $(REPLDIR) && $(PYTHON) generate_figures.py

clean:
	rm -f $(GENERATED)
	rm -rf $(REPLDIR)/__pycache__

help:
	@sed -n '/^# Usage:/,/^$$/p' Makefile | sed 's/^# \{0,1\}//'
