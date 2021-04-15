PROJECT_ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJ_DIR := .
OUTPUT_DIRECTORY := _build
NNOM_DIR := ../../..
SDK_ROOT := ../../../../targets/nrf52840/nRF5_SDK_17.0.2_d674dde

SRC_FILES += \
  $(NNOM_DIR)/port/nnom_port.c \
  $(NNOM_DIR)/src/backends/nnom_local.c \
  $(NNOM_DIR)/src/backends/nnom_local_q15.c \
  $(NNOM_DIR)/src/core/nnom.c \
  $(NNOM_DIR)/src/core/nnom_layers.c \
  $(NNOM_DIR)/src/core/nnom_tensor.c \
  $(NNOM_DIR)/src/core/nnom_utils.c \
  $(NNOM_DIR)/src/layers/nnom_activation.c \
  $(NNOM_DIR)/src/layers/nnom_baselayer.c \
  $(NNOM_DIR)/src/layers/nnom_conv2d.c \
  $(NNOM_DIR)/src/layers/nnom_cropping.c \
  $(NNOM_DIR)/src/layers/nnom_dw_conv2d.c \
  $(NNOM_DIR)/src/layers/nnom_global_pool.c \
  $(NNOM_DIR)/src/layers/nnom_input.c \
  $(NNOM_DIR)/src/layers/nnom_lstm_cell.c \
  $(NNOM_DIR)/src/layers/nnom_maxpool.c \
  $(NNOM_DIR)/src/layers/nnom_rnn.c \
  $(NNOM_DIR)/src/layers/nnom_softmax.c \
  $(NNOM_DIR)/src/layers/nnom_upsample.c \
  $(NNOM_DIR)/src/layers/nnom_avgpool.c \
  $(NNOM_DIR)/src/layers/nnom_concat.c \
  $(NNOM_DIR)/src/layers/nnom_conv2d_trans.c \
  $(NNOM_DIR)/src/layers/nnom_dense.c \
  $(NNOM_DIR)/src/layers/nnom_flatten.c \
  $(NNOM_DIR)/src/layers/nnom_gru_cell.c \
  $(NNOM_DIR)/src/layers/nnom_lambda.c \
  $(NNOM_DIR)/src/layers/nnom_l2_normalize.c \
  $(NNOM_DIR)/src/layers/nnom_matrix.c \
  $(NNOM_DIR)/src/layers/nnom_output.c \
  $(NNOM_DIR)/src/layers/nnom_simple_cell.c \
  $(NNOM_DIR)/src/layers/nnom_sumpool.c \
  $(NNOM_DIR)/src/layers/nnom_zero_padding.c \
  \
  $(PROJ_DIR)/test_main.c \
  
INC_FOLDERS += \
  $(PROJ_DIR) \
  $(NNOM_DIR)/inc \
  $(NNOM_DIR)/inc/layers \
  $(NNOM_DIR)/port \
  \
  $(SDK_ROOT)/components/toolchain/cmsis/include \
  $(SDK_ROOT)/components/toolchain/cmsis/dsp/Include \
  
INCLUDES = $(addprefix -I,$(INC_FOLDERS))

ODIR = $(OUTPUT_DIRECTORY)
_OBJS = \
	nnom_port.o \
	nnom_local.o \
	nnom_local_q15.o \
	nnom.o \
	nnom_layers.o \
	nnom_tensor.o \
	nnom_utils.o \
	nnom_activation.o \
	nnom_baselayer.o \
	nnom_conv2d.o \
	nnom_cropping.o \
	nnom_dw_conv2d.o \
	nnom_global_pool.o \
	nnom_input.o \
	nnom_lstm_cell.o \
	nnom_maxpool.o \
	nnom_rnn.o \
	nnom_softmax.o \
	nnom_upsample.o \
	nnom_avgpool.o \
	nnom_concat.o \
	nnom_conv2d_trans.o \
	nnom_dense.o \
	nnom_flatten.o \
	nnom_gru_cell.o \
	nnom_lambda.o \
	nnom_l2_normalize.o \
	nnom_matrix.o \
	nnom_output.o \
	nnom_simple_cell.o \
	nnom_sumpool.o \
	nnom_zero_padding.o \
	test_main.o
OBJS = $(patsubst %,$(ODIR)/%,$(_OBJS))

# Libraries common to all targets
LIB_FILES += \
	$(SDK_ROOT)/components/toolchain/cmsis/dsp/GCC/libarm_cortexM4lf_math.a \


ifeq ($(BUILD_MODE),debug)
	CFLAGS += -g
else ifeq ($(BUILD_MODE),run)
	CFLAGS += -O2
else ifeq ($(BUILD_MODE),linuxtools)
	CFLAGS += -g -pg -fprofile-arcs -ftest-coverage
	LDFLAGS += -pg -fprofile-arcs -ftest-coverage
else
    $(error Build mode $(BUILD_MODE) not supported by this Makefile)
endif

# ARM Compiler
CC := arm-none-eabi-gcc

# C flags common to all targets
CFLAGS += $(INCLUDES)
CFLAGS += -std=gnu11
CFLAGS += -mcpu=cortex-m4
CFLAGS += -mthumb -mabi=aapcs
CFLAGS += -Wall -Werror
CFLAGS += -mfloat-abi=hard -mfpu=fpv4-sp-d16

# Linker flags
LDFLAGS += -mcpu=cortex-m4
LDFLAGS += -mfloat-abi=hard -mfpu=fpv4-sp-d16
LDFLAGS += --specs=nosys.specs

# Add standard libraries at the very end of the linker input, after all objects
# that may need symbols provided by these libraries.
LIB_FILES += -lm


.PHONY: clean all

all:	test_main

test_main:	$(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIB_FILES)

$(OUTPUT_DIRECTORY)/%.o:	%.c $(INC_FOLDERS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<

vpath %.c $(sort $(dir $(SRC_FILES)))

print:
	echo $(LIBS)

clean:
	rm -fr $(OUTPUT_DIRECTORY)/*.o $(OBJS) test_main