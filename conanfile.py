from conan import ConanFile

class MilvusSdkConanFile(ConanFile):
    name = "milvus-sdk-cpp"
    version = "2.2.0"
    description = 'Milvus C++ SDK'
    license = 'Apache'
    homepage = 'https://github.com/milvus-io/milvus-sdk-cpp'

    settings = 'os', 'compiler', 'build_type', 'arch'

    requires = 'grpc/1.50.1', 'nlohmann_json/3.11.2', 'protobuf/[>3.21.0]'
    test_requires = 'gtest/1.13.0'

    generators = 'CMakeDeps', 'CMakeToolchain'
