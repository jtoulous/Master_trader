RED := \033[31m
GREEN := \033[32m
RESET := \033[0m

all: venv install

venv:
	@echo "$(GREEN) ===> creating virtual environnement...$(RESET)"
	@watch -n 0.1 echo -ne "\\033[1G\\033[32m[   ] \\033[31m▉▉▉▉\\033[0m" &
	@python -m venv virtualEnv
	@kill %1
	@echo "$(GREEN) ===> Done.$(RESET)"

install:
	@echo "$(GREEN) ===> installing dependencies...$(RESET)"
	@. virtualEnv/bin/activate && pip install -r requirements.txt
	@echo "$(GREEN) ===> Done.$(RESET)"

reset:
	@echo "$(RED) ===> resetting models...$(RESET)"
	@rm -f srcs/models/*.pkl
	@echo "$(RED) ===> Done.$(RESET)"
	@echo "$(RED) ===> removing preprocessed files...$(RESET)"
	@rm -f srcs/data/EURUSD/EURUSD_preprocessed.csv
	@echo "$(RED) ===> Done.$(RESET)"

clean:
	@echo "$(RED) ===> removing virtual environnement...$(RESET)"
	@rm -rf virtualEnv
	@echo "$(RED) ===> Done.$(RESET)"
	@echo "$(RED) ===> resetting models...$(RESET)"
	@rm -f srcs/models/*.pkl
	@echo "$(RED) ===> Done.$(RESET)"
	@echo "$(RED) ===> removing preprocessed files...$(RESET)"
	@rm -f srcs/data/EURUSD/EURUSD_preprocessed.csv
	@echo "$(RED) ===> Done.$(RESET)"

.PHONY: all venv install clean
