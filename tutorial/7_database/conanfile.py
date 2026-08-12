import os
from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain

class Tutorial(ConanFile):
    settings = "os", "arch", "compiler", "build_type"
    def requirements(self):
        version = os.getenv("MILVUS_SDK_VERSION", "3.0.2")
        user = os.getenv("MILVUS_SDK_USER", "milvus")
        channel = os.getenv("MILVUS_SDK_CHANNEL", "dev")
        self.requires(f"milvus-sdk-cpp/{version}@{user}/{channel}")
    def generate(self):
        CMakeDeps(self).generate()
        CMakeToolchain(self).generate()
