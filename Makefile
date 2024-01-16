export USER_ID:=$(shell id -u)
export GROUP_ID:=$(shell id -g)

build:
	@echo Build docker image...
	@docker-compose build project

test: check-env
	@echo NVIDIA and CUDA setup
	@docker-compose run project nvidia-smi
	@echo Pytorch CUDA setup installed?
	@docker-compose run project python3 -c "import torch; print(torch.cuda.is_available())"
	@echo MinkowskiEngine installed?
	@docker-compose run project python3 -c "import MinkowskiEngine as ME; print(ME.__version__)"

run: check-env
	@docker-compose run project

clean:
	@echo Removing docker image...
	@docker-compose rm project


check-env:
ifndef DATA
	$(error Please specify where your data is located, export DATA=<path>)
endif
