CC      = gcc
CFLAGS  = -O3 -march=native -flto -Wall -Wextra -std=gnu11
LDFLAGS = -lncursesw -ldl -lpthread
TARGET  = nv-monitor

all: $(TARGET)

$(TARGET): nv-monitor.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/

.PHONY: all clean install
