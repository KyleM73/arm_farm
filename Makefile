VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python

.PHONY: setup patch_venv

setup: patch_venv
	@echo "Activate venv with: source $(VENV_DIR)/bin/activate"

$(VENV_DIR)/bin/activate:
	@uv venv $(VENV_DIR)

patch_venv: $(VENV_DIR)/bin/activate
	@uv pip install -e .
	@mkdir -p $(VENV_DIR)/bin/postactivate.d
	@cp scripts/libpython_patch.sh $(VENV_DIR)/bin/postactivate.d/
	@chmod +x $(VENV_DIR)/bin/postactivate.d/libpython_patch.sh
	@bash scripts/venv_patch.sh $(VENV_DIR)/bin/activate