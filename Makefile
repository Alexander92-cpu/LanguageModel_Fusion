IMAGE_NAME=languagemodelfusion:1.0.0
CONTAINER_NAME=lm_fusion_container

.PHONY: create-container-k2complex
create-container-k2complex:
	@echo Create Docker container
	docker create --name $(CONTAINER_NAME) \
	--memory 32G \
	--shm-size 8G \
	--gpus '"device=0"' \
	-it \
	-v ${PWD}/conf:/app/conf:ro \
	$(IMAGE_NAME) bash

.PHONY: run-k2complex
run-k2complex:
	@echo Run Docker container
	docker cp main.py $(CONTAINER_NAME):/app && \
	docker start $(CONTAINER_NAME) && \
	docker attach $(CONTAINER_NAME)
