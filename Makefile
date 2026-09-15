CXX ?= g++
CXXFLAGS ?= -std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
SDL_CFLAGS ?= $(shell pkg-config --cflags sdl3 2>/dev/null)
SDL_LIBS ?= $(shell pkg-config --libs sdl3 2>/dev/null)

TARGET := build/agent-ari
SOURCES := $(wildcard src/*.cpp)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCES)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(SDL_CFLAGS) -Iinclude $(SOURCES) $(SDL_LIBS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf build
