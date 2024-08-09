install:
	@pip install -v .

install-all:
	@pip install -v ".[all]"

uninstall:
	@pip -v uninstall mos4d

editable:
	@pip install scikit-build-core pyproject_metadata pathspec pybind11
	@pip install --no-build-isolation -ve .
