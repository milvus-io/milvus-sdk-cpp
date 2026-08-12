import os
from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain
class Tutorial(ConanFile):
    settings = "os", "arch", "compiler", "build_type"
    def requirements(self): self.requires(f"milvus-sdk-cpp/{os.getenv('MILVUS_SDK_VERSION','3.0.2')}@{os.getenv('MILVUS_SDK_USER','milvus')}/{os.getenv('MILVUS_SDK_CHANNEL','dev')}")
    def generate(self): CMakeDeps(self).generate(); CMakeToolchain(self).generate()
