SRC_DIR = src/
INC_DIR = inc/
OBJ_DIR = build/objs/
LIB_DIR = build/libs/

CC = gcc
CFLAGS = -c -g -Wall -fPIC -MMD -std=c99 -I $(INC_DIR)
LDFLAGS = -shared -lm -L $(LIB_DIR)
LDFLAGS += -Wl,-rpath,$(CUDIR)$(LIB_DIR)

TARGET_MODULE = $(LIB_DIR)libridge.so
OBJS = $(patsubst $(SRC_DIR)%.c,$(OBJ_DIR)%.o,$(SRC_FILES))
DEP = $(OBJS:.o=.d)

all: $(TARGET_MODULE)

SRC_FILES = \
	$(SRC_DIR)kalman_filter_mode.c \
	$(SRC_DIR)kalman_filter_network.c \
	$(SRC_DIR)math_approximations.c

-include $(DEPS)
$(OBJ_DIR)%.o: $(SRC_DIR)%.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $< -o $@

$(TARGET_MODULE): $(OBJS)
	mkdir -p $(@D)
	$(CC) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET_MODULE) $(OBJS) $(DEPS)