from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps, cmake_layout
from conan.tools.build import check_min_cppstd


class MilvusSdkCppConan(ConanFile):
    name = "milvus-sdk-cpp"
    # The upstream project defines its own (git-based) version; keep Conan's
    # version flexible.
    version = "0.0"

    license = "Apache-2.0"
    url = "https://github.com/milvus-io/milvus-sdk-cpp"
    description = "Milvus C++ SDK"

    settings = "os", "arch", "compiler", "build_type"

    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_tests": [True, False],  # Explicit option for tests
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "with_tests": False,  # Explicit default for tests

        # gRPC pulls a lot of deps; keep it lean enough to be usable.
        "grpc/*:shared": False,
        "protobuf/*:shared": False,

        # Ensure OpenSSL backend for TLS.
        "grpc/*:secure": True,
    }

    exports_sources = (
        "CMakeLists.txt",
        "cmake/*",
        "src/*",
        "examples/*",
        "test/*",
        "thirdparty/*",
        "LICENSE",
        "README.md",
        "DEVELOPMENT.md",
        "CHANGELOG.md",
    )

    def requirements(self):
        # ConanCenter currently provides these gRPC versions; pick one close to
        # what upstream uses while still available.
        self.requires("grpc/1.65.0")
        # grpc/1.65.0 expects protobuf 5.x and abseil 20240116.x.
        self.requires("protobuf/5.27.0")
        self.requires("abseil/20240116.2")
        self.requires("nlohmann_json/3.11.3")

        if self.options.with_tests:
            self.requires("gtest/1.12.1")

    def layout(self):
        cmake_layout(self)

    def validate(self):
        # The project currently builds with C++14.
        check_min_cppstd(self, 14)

    def configure(self):
        # Ensure dependency graph variants match the project's C++ standard.
        # (Without this, Conan may resolve packages for gnu17 and then fail to
        # find compatible binaries.)
        self.settings.compiler.cppstd = "14"

    def generate(self):
        tc = CMakeToolchain(self)

        # Map upstream options
        tc.variables["MILVUS_BUILD_TEST"] = bool(self.options.with_tests)

        # Disable the legacy thirdparty switches. We'll use Conan targets.
        tc.variables["MILVUS_WITH_GRPC"] = "package"
        tc.variables["MILVUS_WITH_GTEST"] = "package"

        # GRPC_PATH is only relevant for their custom prebuilt tree; keep empty.
        tc.variables["GRPC_PATH"] = ""

        # Typical Conan toolchain knobs
        tc.variables["CMAKE_POSITION_INDEPENDENT_CODE"] = bool(self.options.fPIC)
        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        # Consumers should normally use the project's exported CMake targets.
        # Keep empty for now.
        pass
