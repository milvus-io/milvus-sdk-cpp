# use pip install clang-tidy & clang-format

if (CMAKE_HOST_APPLE)
    execute_process(
            COMMAND brew --prefix llvm
            OUTPUT_VARIABLE USER_LLVM_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY
    )
    set(USER_CLANG_TOOLS_PATH ${USER_LLVM_PATH}/bin)
    message(STATUS ${USER_CLANG_TOOLS_PATH})
endif ()

find_program(CLANG_TIDY_BIN
        NAMES
        clang-tidy
        PATHS
        ${USER_CLANG_TOOLS_PATH}
        ${ClangTools_PATH}
        $ENV{CLANG_TOOLS_PATH}
        $ENV{HOME}/.local/bin
        /usr/local/bin
        /usr/bin
        NO_DEFAULT_PATH
)

if ("${CLANG_TIDY_BIN}" STREQUAL "CLANG_TIDY_BIN-NOTFOUND")
    set(CLANG_TIDY_FOUND 0)
    message("clang-tidy not found")
else ()
    set(CLANG_TIDY_FOUND 1)
    message("clang-tidy found at ${CLANG_TIDY_BIN}")
endif ()

find_program(CLANG_FORMAT_BIN
        NAMES
        clang-format
        PATHS
        clang-tidy
        PATHS
        ${USER_CLANG_TOOLS_PATH}
        ${ClangTools_PATH}
        $ENV{CLANG_TOOLS_PATH}
        $ENV{HOME}/.local/bin
        /usr/local/bin
        /usr/bin
        NO_DEFAULT_PATH
)

if ("${CLANG_FORMAT_BIN}" STREQUAL "CLANG_FORMAT_BIN-NOTFOUND")
    set(CLANG_FORMAT_FOUND 0)
    message("clang-format not found")
else ()
    set(CLANG_FORMAT_FOUND 1)
    message("clang-format found at ${CLANG_FORMAT_BIN}")
endif ()

