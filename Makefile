CODE_PATH?=src
DATA_PATH?=data
NOTEBOOKS_PATH?=notebooks
REQUIREMENTS_PATH?=requirements
RESULTS_PATH?=logs
PROJECT_PATH_STORAGE?=storage:kern_segmentation
CODE_PATH_STORAGE?=$(PROJECT_PATH_STORAGE)/$(CODE_PATH)
DATA_PATH_STORAGE?=$(PROJECT_PATH_STORAGE)/$(DATA_PATH)

NOTEBOOKS_PATH_STORAGE?=$(PROJECT_PATH_STORAGE)/$(NOTEBOOKS_PATH)
REQUIREMENTS_PATH_STORAGE?=$(PROJECT_PATH_STORAGE)/$(REQUIREMENTS_PATH)
RESULTS_PATH_STORAGE?=$(PROJECT_PATH_STORAGE)/$(RESULTS_PATH)

PROJECT_PATH_ENV?=/project
CODE_PATH_ENV?=$(PROJECT_PATH_ENV)/$(CODE_PATH)
DATA_PATH_ENV?=$(PROJECT_PATH_ENV)/$(DATA_PATH)
NOTEBOOKS_PATH_ENV?=$(PROJECT_PATH_ENV)/$(NOTEBOOKS_PATH)
REQUIREMENTS_PATH_ENV?=$(PROJECT_PATH_ENV)/$(REQUIREMENTS_PATH)
RESULTS_PATH_ENV?=$(PROJECT_PATH_ENV)/$(RESULTS_PATH)

SETUP_NAME?=kern
TRAINING_NAME?=training
JUPYTER_NAME?=jupyter
JUPYTER2_NAME?=jupyter2
JUPYTER3_NAME?=jupyter3
TENSORBOARD_NAME?=tensorboard
FILEBROWSER_NAME?=filebrowser

BASE_ENV_NAME?=neuromation/base
CUSTOM_ENV_NAME?=image:neuro/custom

##### SETUP #####

.PHONY: setup
setup:
	neuro kill $(SETUP_NAME)
	neuro run \
		--name $(SETUP_NAME) \
		--preset gpu-small \
		--detach \
		--volume $(PROJECT_PATH_STORAGE):$(PROJECT_PATH_ENV):rw \
		$(BASE_ENV_NAME) \
		'tail -f /dev/null'
	neuro cp -r $(REQUIREMENTS_PATH) $(REQUIREMENTS_PATH_STORAGE)
	neuro exec $(SETUP_NAME) 'pip install -r $(REQUIREMENTS_PATH_ENV)/pip.txt'
	neuro job save $(SETUP_NAME) $(CUSTOM_ENV_NAME)
	neuro kill $(SETUP_NAME)

##### STORAGE #####

.PHONY: upload_code
upload_code:
	neuro cp -r -T -u $(CODE_PATH) $(CODE_PATH_STORAGE)

.PHONY: clean_code
clean_code:
	neuro rm -r $(CODE_PATH_STORAGE)


.PHONY: upload_data
upload_data:
	( neuro mkdir -p $(DATA_PATH_STORAGE); \
	neuro cp -r -T -u $(DATA_PATH) $(DATA_PATH_STORAGE); )

.PHONY: clean_data
clean_data:
	neuro rm -r $(DATA_PATH_STORAGE)

.PHONY: upload_notebooks
upload_notebooks:
	( neuro mkdir -p $(PROJECT_PATH_STORAGE); \
	neuro mkdir -p $(NOTEBOOKS_PATH_STORAGE); \
	neuro cp -r -T -u $(NOTEBOOKS_PATH) $(NOTEBOOKS_PATH_STORAGE); )

.PHONY: download_notebooks
download_notebooks:
	neuro cp -r $(NOTEBOOKS_PATH_STORAGE) $(NOTEBOOKS_PATH)

.PHONY: download_code
download_code:
	neuro cp -r $(CODE_PATH_STORAGE) $(CODE_PATH)


.PHONY: clean_notebooks
clean_notebooks:
	neuro rm -r $(NOTEBOOKS_PATH_STORAGE)

.PHONY: upload
upload: upload_notebooks upload_code upload_data  upload_config upload_tests

.PHONY: clean
clean: clean_code clean_data clean_notebooks

##### JOBS #####

.PHONY: training
training:
	neuro run \
		--name $(TRAINING_NAME) \
		--preset сpu-large \
		--volume $(CODE_PATH_STORAGE):$(CODE_PATH_ENV):ro \
		--volume $(PROJECT_PATH_STORAGE):$(PROJECT_PATH_ENV):ro \
		--volume $(RESULTS_PATH_STORAGE):$(RESULTS_PATH_ENV):rw \
		$(CUSTOM_ENV_NAME) \
		'python $(CODE_PATH_ENV)/train.py --path $(PROJECT_PATH_ENV) --name $(TRAINING_NAME)'

.PHONY: kill_training
kill_training:
	neuro kill $(TRAINING_NAME)

.PHONY: connect_training
connect_training:
	neuro exec $(TRAINING_NAME) bash

.PHONY: jupyter
jupyter: 
	neuro run \
		--name $(JUPYTER_NAME) \
		--preset gpu-small \
		--http 8888 --no-http-auth --detach \
		--volume $(CODE_PATH_STORAGE):$(CODE_PATH_ENV):rw \
		--volume $(NOTEBOOKS_PATH_STORAGE):$(NOTEBOOKS_PATH_ENV):rw \
		--volume $(PROJECT_PATH_STORAGE):$(PROJECT_PATH_ENV):rw \
		--volume $(RESULTS_PATH_STORAGE):$(RESULTS_PATH_ENV):rw \
		$(CUSTOM_ENV_NAME) \
		'jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir=$(PROJECT_PATH_ENV)'
	neuro job browse $(JUPYTER_NAME)


.PHONY: jupyter2
jupyter2: 
	neuro run \
		--name $(JUPYTER2_NAME) \
		--preset gpu-small \
		--http 8888 --no-http-auth --detach \
		--volume $(CODE_PATH_STORAGE):$(CODE_PATH_ENV):rw \
		--volume $(NOTEBOOKS_PATH_STORAGE):$(NOTEBOOKS_PATH_ENV):rw \
		--volume $(PROJECT_PATH_STORAGE):$(PROJECT_PATH_ENV):rw \
		--volume $(RESULTS_PATH_STORAGE):$(RESULTS_PATH_ENV):rw \
		$(CUSTOM_ENV_NAME) \
		'jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir=$(PROJECT_PATH_ENV)'
	neuro job browse $(JUPYTER2_NAME)

.PHONY: jupyter3
jupyter3: 
	neuro run \
		--name $(JUPYTER3_NAME) \
		--preset gpu-small \
		--http 8888 --no-http-auth --detach \
		--volume $(CODE_PATH_STORAGE):$(CODE_PATH_ENV):rw \
		--volume $(NOTEBOOKS_PATH_STORAGE):$(NOTEBOOKS_PATH_ENV):rw \
		--volume $(PROJECT_PATH_STORAGE):$(PROJECT_PATH_ENV):rw \
		--volume $(RESULTS_PATH_STORAGE):$(RESULTS_PATH_ENV):rw \
		$(CUSTOM_ENV_NAME) \
		'jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir=$(PROJECT_PATH_ENV)'
	neuro job browse $(JUPYTER3_NAME)

.PHONY: kill_jupyter
kill_jupyter:
	neuro kill $(JUPYTER_NAME)

.PHONY: tensorboard
tensorboard:
	neuro run \
		--name $(TENSORBOARD_NAME) \
		--preset cpu-small \
		--http 6006 --no-http-auth --detach \
		-e GCS_READ_CACHE_MAX_SIZE_MB=0 \
		--volume $(RESULTS_PATH_STORAGE):$(RESULTS_PATH_ENV):ro \
		tensorflow/tensorflow \
		'tensorboard --logdir=$(RESULTS_PATH_ENV)'
	neuro job browse $(TENSORBOARD_NAME)

.PHONY: kill_tensorboard
kill_tensorboard:
	neuro kill $(TENSORBOARD_NAME)

.PHONY: filebrowser
filebrowser:
	neuro run \
		--name $(FILEBROWSER_NAME) \
		--preset cpu-small \
		--http 80 --no-http-auth --detach \
		--volume $(PROJECT_PATH_STORAGE):/srv:rw \
		filebrowser/filebrowser
	neuro job browse $(FILEBROWSER_NAME)

.PHONY: kill_filebrowser
kill_filebrowser:
	neuro kill $(FILEBROWSER_NAME)

.PHONY: kill
kill: kill_training kill_jupyter kill_tensorboard kill_filebrowser

##### LOCAL #####

.PHONY: setup_local
setup_local:
	pip install -r requirements/pip.txt

.PHONY: lint
lint:
	flake8 .
	mypy .

.PHONY: install
install:
	python setup.py install --user

##### MISC #####

.PHONY: ps
ps:
	neuro ps
