# SPECTER Tracer Build
#
# Build both NCCL and cuBLAS tracers as shared libraries
# for use with LD_PRELOAD

CC = gcc
CFLAGS = -Wall -Wextra -O2 -fPIC -g
LDFLAGS = -shared -ldl -lpthread

# Output directory
BUILD_DIR = ../../build

# Targets
NCCL_LIB = $(BUILD_DIR)/libspecter_nccl.so
CUBLAS_LIB = $(BUILD_DIR)/libspecter_cublas.so
COMBINED_LIB = $(BUILD_DIR)/libspecter.so

.PHONY: all clean nccl cublas combined

all: nccl cublas combined

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

nccl: $(NCCL_LIB)

$(NCCL_LIB): nccl_tracer.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

cublas: $(CUBLAS_LIB)

$(CUBLAS_LIB): cublas_tracer.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

# Combined tracer (both NCCL and cuBLAS in one library)
combined: $(COMBINED_LIB)

$(COMBINED_LIB): nccl_tracer.c cublas_tracer.c combined_tracer.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ combined_tracer.c $(LDFLAGS)
	@echo "Built: $@"

clean:
	rm -f $(NCCL_LIB) $(CUBLAS_LIB) $(COMBINED_LIB)

# Installation (optional)
PREFIX ?= /usr/local
install: all
	install -d $(PREFIX)/lib
	install -m 755 $(NCCL_LIB) $(PREFIX)/lib/
	install -m 755 $(CUBLAS_LIB) $(PREFIX)/lib/
	install -m 755 $(COMBINED_LIB) $(PREFIX)/lib/

# Help
help:
	@echo "SPECTER Tracer Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build all tracers (default)"
	@echo "  nccl     - Build NCCL tracer only"
	@echo "  cublas   - Build cuBLAS tracer only"
	@echo "  combined - Build combined tracer"
	@echo "  clean    - Remove built files"
	@echo "  install  - Install to PREFIX (default: /usr/local)"
	@echo ""
	@echo "Usage:"
	@echo "  LD_PRELOAD=./build/libspecter_nccl.so torchrun ... train.py"
	@echo "  LD_PRELOAD=./build/libspecter.so torchrun ... train.py"
	@echo ""
	@echo "Environment variables:"
	@echo "  SPECTER_OUTPUT - Output directory for traces (default: .)"
	@echo "  SPECTER_SYNC   - Set to 1 for synchronous timing (more accurate but slower)"
