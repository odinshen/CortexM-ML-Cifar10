##########################################################################################################################
# File automatically-generated by tool: [projectgenerator] version: [2.26.0] date: [Mon Feb 19 09:10:28 GMT 2018]
##########################################################################################################################

# ------------------------------------------------
# Generic Makefile (based on gcc)
#
# ChangeLog :
#	2017-02-10 - Several enhancements + project update mode
#   2015-07-22 - first version
# ------------------------------------------------

######################################
# target
######################################
TARGET = CNN_Camera


######################################
# building variables
######################################
# debug build?
# make DEBUG=0 all > build with optimization
DEBUG ?= 1

#######################################
# paths
#######################################
# source path
SOURCES_DIR =  \
Drivers/CMSIS \
Application/User/Src/main.c \
Application/User/Src/stm32f7xx_hal_msp.c \
Drivers/STM32F7xx_HAL_Driver \
Application \
Drivers \
Application/User/Src \
Application/User/Src/stm32f7xx_it.c \
Application/User \
Application/MAKEFILE

# firmware library path
PERIFLIB_PATH = 

# Build path
BUILD_DIR = build

######################################
# source
######################################
# C sources
C_SOURCES =  \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_tim.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_i2c_ex.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_pwr_ex.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_dma_ex.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_cortex.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_dma.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_rcc_ex.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_pwr.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_gpio.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_flash.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_i2c.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_flash_ex.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_tim_ex.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_dma2d.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_rcc.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_ltdc_ex.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_ltdc.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_sdram.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_uart.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_ll_fmc.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_dcmi.c \
Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_dcmi_ex.c \
Drivers/CMSIS/DSP_Lib/Source/SupportFunctions/arm_fill_q15.c \
Drivers/CMSIS/DSP_Lib/Source/SupportFunctions/arm_copy_q7.c \
Drivers/BSP/STM32746G-Discovery/stm32746g_discovery_lcd.c \
Drivers/BSP/STM32746G-Discovery/stm32746g_discovery_camera.c \
Drivers/BSP/STM32746G-Discovery/stm32746g_discovery_sdram.c \
Drivers/BSP/STM32746G-Discovery/stm32746g_discovery.c \
Drivers/BSP/Components/ov9655/ov9655.c \
App/NNFunctions/convolve_CHW_q15_basic.c \
App/NNFunctions/convolve_cmsis_CHW_q7.c \
App/NNFunctions/convolve_HWC_q15_basic.c \
App/NNFunctions/convolve_HWC_q15_full.c \
App/NNFunctions/convolve_HWC_q7_basic.c \
App/NNFunctions/convolve_HWC_q7_full.c \
App/NNFunctions/convolve_HWC_q7_RGB.c\
App/NNFunctions/fully_connected_q15_x4.c \
App/NNFunctions/fully_connected_q15.c \
App/NNFunctions/fully_connected_q7_x2.c \
App/NNFunctions/fully_connected_q7_x4.c \
App/NNFunctions/fully_connected_q7.c \
App/NNFunctions/mat_mult_kernel_q7_q15.c \
App/NNFunctions/mat_mult_RELU_kernel_q7_q15.c \
App/NNFunctions/norm_q7_HWC.c \
App/NNFunctions/pool_q7_HWC.c \
App/NNFunctions/relu_q15.c \
App/NNFunctions/relu_q7.c \
App/NNFunctions/separable_conv_HWC_q7.c \
App/NNFunctions/sigmoid.c \
App/NNFunctions/softmax.c \
App/NNFunctions/tanh.c \
App/NNSupportFunctions/arm_expand_q7_to_q15_no_shift_shuffle.c \
App/NNSupportFunctions/arm_q7_to_q15_no_shift_shuffle.c \
App/NNSupportFunctions/CHW_to_HWC_q15.c \
App/NNSupportFunctions/CHW_to_HWC_q7.c \
App/NNSupportFunctions/HWC_to_CHW_q15.c \
App/NNSupportFunctions/HWC_to_CHW_q7.c \
App/cmsis-nn/Source/arm_convolve_HWC_q15_basic.c \
App/cmsis-nn/Source/arm_convolve_HWC_q15_fast.c \
App/cmsis-nn/Source/arm_convolve_HWC_q7_basic.c \
App/cmsis-nn/Source/arm_convolve_HWC_q7_fast.c \
App/cmsis-nn/Source/arm_convolve_HWC_q7_RGB.c \
App/cmsis-nn/Source/arm_nn_mat_mult_kernel_q7_q15.c \
App/cmsis-nn/Source/arm_depthwise_separable_conv_HWC_q7.c \
App/cmsis-nn/Source/arm_fully_connected_q7_opt.c \
App/cmsis-nn/Source/arm_fully_connected_q15.c \
App/cmsis-nn/Source/arm_fully_connected_q7.c \
App/cmsis-nn/Source/arm_pool_q7_HWC.c \
App/cmsis-nn/Source/arm_relu_q15.c \
App/cmsis-nn/Source/arm_relu_q7.c \
App/cmsis-nn/Source/arm_softmax_q7.c \
App/cmsis-nn/Source/arm_q7_to_q15_reordered_no_shift.c \
App/cmsis-nn/Source/arm_q7_to_q15_no_shift.c \
App/cmsis-nn/Source/arm_nntables.c \
Src/system_stm32f7xx.c \
Src/stm32f7xx_it.c \
Src/stm32f7xx_hal_msp.c \
App/D2DResize/D2DResize.c

CPP_SOURCES = \
App/cnn/cnn.cpp \
App/GUI/GUI.cpp \
Src/main.cpp

# ASM sources
ASM_SOURCES =  \
startup_stm32f746xx.s


######################################
# firmware library
######################################
PERIFLIB_SOURCES = 


#######################################
# binaries
#######################################

#ARM_GCC_PATH = ~/arm_tools/gcc-arm-none-eabi-7-2017-q4-major/bin
BINPATH = $(ARM_GCC_PATH)
PREFIX = arm-none-eabi-
CC = $(PREFIX)gcc
CPP = $(PREFIX)g++
AS = $(PREFIX)gcc -x assembler-with-cpp
CP = $(PREFIX)objcopy
AR = $(PREFIX)ar
SZ = $(PREFIX)size
HEX = $(CP) -O ihex
BIN = $(CP) -O binary -S



#######################################
# CFLAGS
#######################################
# cpu
CPU = -mcpu=cortex-m7

# fpu
FPU = -mfpu=fpv5-sp-d16

# float-abi
FLOAT-ABI = -mfloat-abi=softfp
#-mfpu=fpv5-sp-d16' '-mfloat-abi=softfp'
# mcu
MCU = $(CPU) -mthumb $(FPU) $(FLOAT-ABI)

# macros for gcc
# AS defines
AS_DEFS = 

# C defines
C_DEFS =  \
-DUSE_HAL_DRIVER \
-DSTM32F746xx \
-D__FPU_PRESENT=1 \
-DARM_MATH_CM7 


# AS includes
AS_INCLUDES = 

# C includes
C_INCLUDES =  \
-IInc \
-IDrivers/STM32F7xx_HAL_Driver/Inc \
-IDrivers/STM32F7xx_HAL_Driver/Inc/Legacy \
-IDrivers/BSP/STM32746G-Discovery \
-IDrivers/BSP/Components/Common \
-IDrivers/BSP/Components/ov9655 \
-IDrivers/BSP/Components/rk043fn48h \
-IDrivers/CMSIS/Device/ST/STM32F7xx/Include \
-IDrivers/CMSIS/Include \
-IApp/cnn \
-IApp/cmsis-nn/Include \
-IApp/GUI \
-IApp/Bitmaps \
-IApp/D2DResize \
-IApp/NNFunctions \
-IApp/NNSupportFunctions \
-IUtilities/Fonts

ifeq ($(DEBUG), 1)
OPT = -Og -gdwarf-2
else
OPT = -O3
endif

# compile gcc flags
ASFLAGS = $(MCU) $(AS_DEFS) $(AS_INCLUDES) $(OPT) -Wall -fdata-sections -ffunction-sections 

CFLAGS = $(MCU) '-std=gnu99' $(C_DEFS) $(C_INCLUDES) $(OPT) -Wall -fdata-sections -ffunction-sections '-fno-exceptions' '-fno-builtin' '-ffunction-sections' '-fdata-sections' '-funsigned-char' '-fno-delete-null-pointer-checks' '-fomit-frame-pointer'

CPPFLAGS = $(MCU) '-std=gnu++11' '-fno-builtin' $(C_DEFS) $(C_INCLUDES) $(OPT) -Wall '-fno-exceptions' '-fno-builtin' '-ffunction-sections' '-fdata-sections' '-funsigned-char' '-fno-delete-null-pointer-checks' '-fomit-frame-pointer' -fdata-sections -ffunction-sections

# Generate dependency information
CFLAGS += -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)"
CPPFLAGS += -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)"



#######################################
# LDFLAGS
#######################################
# link script
LDSCRIPT = STM32F746NGHx_FLASH.ld

# libraries
LIBS = -lc -lm -lnosys -lstdc++ -lsupc++ -lgcc 
LIBDIR =
LDFLAGS = $(MCU) -specs=nano.specs -T$(LDSCRIPT) $(LIBDIR) $(LIBS) -Wl,-Map=$(BUILD_DIR)/$(TARGET).map,--cref -Wl,--gc-sections -u _printf_float

# default action: build all
all: $(BUILD_DIR)/$(TARGET).elf $(BUILD_DIR)/$(TARGET).hex $(BUILD_DIR)/$(TARGET).bin


#######################################
# build the application
#######################################
# list of objects
OBJECTS = $(addprefix $(BUILD_DIR)/,$(notdir $(C_SOURCES:.c=.o)))
vpath %.c $(sort $(dir $(C_SOURCES)))
OBJECTS += $(addprefix $(BUILD_DIR)/,$(notdir $(CPP_SOURCES:.cpp=.o)))
vpath %.cpp $(sort $(dir $(CPP_SOURCES)))
# list of ASM program objects
OBJECTS += $(addprefix $(BUILD_DIR)/,$(notdir $(ASM_SOURCES:.s=.o)))
vpath %.s $(sort $(dir $(ASM_SOURCES)))

$(BUILD_DIR)/%.o: %.cpp Makefile | $(BUILD_DIR) 
	$(CPP) -c $(CPPFLAGS) -Wa,-a,-ad,-alms=$(BUILD_DIR)/$(notdir $(<:.cpp=.lst)) $< -o $@

$(BUILD_DIR)/%.o: %.c Makefile | $(BUILD_DIR) 
	$(CC) -c $(CFLAGS) -Wa,-a,-ad,-alms=$(BUILD_DIR)/$(notdir $(<:.c=.lst)) $< -o $@

$(BUILD_DIR)/%.o: %.s Makefile | $(BUILD_DIR)
	$(AS) -c $(CFLAGS) $< -o $@

$(BUILD_DIR)/$(TARGET).elf: $(OBJECTS) Makefile
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@
	$(SZ) $@

$(BUILD_DIR)/%.hex: $(BUILD_DIR)/%.elf | $(BUILD_DIR)
	$(HEX) $< $@
	
$(BUILD_DIR)/%.bin: $(BUILD_DIR)/%.elf | $(BUILD_DIR)
	$(BIN) $< $@	
	
$(BUILD_DIR):
	mkdir $@		

#######################################
# clean up
#######################################
clean:
	-rm -fR .dep $(BUILD_DIR)
  
#######################################
# dependencies
#######################################
-include $(shell mkdir .dep 2>/dev/null) $(wildcard .dep/*)

# *** EOF ***
