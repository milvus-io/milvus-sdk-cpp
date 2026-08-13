BUILD_DIR ?= cmake_build
JOBS ?= 4
build:
	conan install . --output-folder=$(BUILD_DIR) --build=missing -s build_type=Release -s compiler.cppstd=14 -s:b compiler.cppstd=17
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$(BUILD_DIR)/conan_toolchain.cmake
	cmake --build $(BUILD_DIR) -j$(JOBS)
run:
	@bash -c '. $(BUILD_DIR)/conanrun.sh 2>/dev/null || true; ./$(BUILD_DIR)/tutorial_index'
clean:
	rm -rf $(BUILD_DIR)
.PHONY: build run clean
