CC ?= gcc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -Iinclude
LDFLAGS ?= -lm

SRC = src/runtime.c src/ops.c src/quant.c src/tensor.c
OBJ = $(SRC:.c=.o)

.PHONY: all test clean

all: test

test: tests/test_pipeline
	./tests/test_pipeline

tests/test_pipeline: tests/test_pipeline.c $(SRC) include/runtime.h include/ops.h include/quant.h include/tensor.h
	$(CC) $(CFLAGS) -o $@ tests/test_pipeline.c $(SRC) $(LDFLAGS)

clean:
	rm -f $(OBJ) tests/test_pipeline
