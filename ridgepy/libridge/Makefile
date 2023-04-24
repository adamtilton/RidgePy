# Variables for file and directory locations
SRC_DIR = src/
INC_DIR = inc/
OBJ_DIR = build/objs/
LIB_DIR = build/libs/

# C compiler and flags
CC = gcc
CFLAGS = -c -g -Wall -fPIC -MMD -std=c99 -I $(INC_DIR) -march=armv8-a+crc -arch arm64

# Linker flags and library location
LDFLAGS = -shared -lm -L $(LIB_DIR) -march=armv8-a+crc -arch arm64
LDFLAGS += -Wl,-rpath,$(CUDIR)$(LIB_DIR)

# Target library file and object files
TARGET_MODULE = $(LIB_DIR)libridge.so
OBJS = $(patsubst $(SRC_DIR)%.c,$(OBJ_DIR)%.o,$(SRC_FILES))

# Dependency files
DEPS = $(OBJS:.o=.d)

# Default target: build the library
all: $(TARGET_MODULE)

# Source files
SRC_FILES = \
	$(SRC_DIR)kalman_filter_mode.c \
	$(SRC_DIR)kalman_filter_network.c \
	$(SRC_DIR)math_approximations.c

# Include dependency files in the build process
-include $(DEPS)

# Rule for building object files from source files
$(OBJ_DIR)%.o: $(SRC_DIR)%.c
	# Create necessary directories if they do not exist
	mkdir -p $(@D)
	# Compile source file into object file
	$(CC) $(CFLAGS) $< -o $@

# Rule for building library from object files
$(TARGET_MODULE): $(OBJS)
	# Create necessary directories if they do not exist
	mkdir -p $(@D)
	# Build library file
	$(CC) -o $@ $^ $(LDFLAGS)

# Phony target for cleaning build directories
.PHONY: clean
clean:
	# Remove library file and object files from build directories
	rm -f $(TARGET_MODULE) $(OBJS) $(DEPS)
