cmake_minimum_required(VERSION 3.20)

project(AgentAri VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FIRSTAGENT_BUILD_APP "Build the SDL realtime application" ON)
option(FIRSTAGENT_ENABLE_VULKAN "Build the Vulkan compute backend when the SDK is available" ON)

set(FIRSTAGENT_VULKAN_SDK_ROOT "${CMAKE_SOURCE_DIR}/third_party/vulkan-sdk/usr")
find_path(FIRSTAGENT_VULKAN_INCLUDE_DIR NAMES vulkan/vulkan.h
    PATHS "${FIRSTAGENT_VULKAN_SDK_ROOT}/include" /usr/include)
find_library(FIRSTAGENT_VULKAN_LIBRARY NAMES vulkan libvulkan.so.1
    PATHS "${FIRSTAGENT_VULKAN_SDK_ROOT}/lib64" /usr/lib64 /usr/lib)
find_program(FIRSTAGENT_GLSLC_EXECUTABLE NAMES glslc
    PATHS "${FIRSTAGENT_VULKAN_SDK_ROOT}/bin")

set(FIRSTAGENT_VULKAN_READY OFF)
if(FIRSTAGENT_ENABLE_VULKAN AND FIRSTAGENT_VULKAN_INCLUDE_DIR AND
   FIRSTAGENT_VULKAN_LIBRARY AND FIRSTAGENT_GLSLC_EXECUTABLE)
    set(FIRSTAGENT_VULKAN_READY ON)
    set(FIRSTAGENT_VULKAN_SHADER "${CMAKE_CURRENT_BINARY_DIR}/firstagent-matmul.comp.spv")
    set(FIRSTAGENT_VULKAN_ELEMENTWISE_SHADER
        "${CMAKE_CURRENT_BINARY_DIR}/firstagent-elementwise.comp.spv")
    add_custom_command(
        OUTPUT "${FIRSTAGENT_VULKAN_SHADER}"
        COMMAND "${FIRSTAGENT_GLSLC_EXECUTABLE}" -o "${FIRSTAGENT_VULKAN_SHADER}"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul.comp"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul.comp"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${FIRSTAGENT_VULKAN_ELEMENTWISE_SHADER}"
        COMMAND "${FIRSTAGENT_GLSLC_EXECUTABLE}" -o "${FIRSTAGENT_VULKAN_ELEMENTWISE_SHADER}"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/elementwise.comp"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/elementwise.comp"
        VERBATIM
    )
    add_custom_target(firstagent-vulkan-shader
        DEPENDS "${FIRSTAGENT_VULKAN_SHADER}" "${FIRSTAGENT_VULKAN_ELEMENTWISE_SHADER}")
endif()

if(FIRSTAGENT_BUILD_APP)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SDL3 REQUIRED IMPORTED_TARGET sdl3)
endif()

# Backend targets are separate so applications can link only the compute
# implementation they need. The public kernel policies remain header-only.
add_library(firstagent-kernels-cpu INTERFACE)
target_include_directories(firstagent-kernels-cpu INTERFACE include)

add_library(firstagent-kernels-vulkan STATIC src/VulkanBackend.cpp)
target_include_directories(firstagent-kernels-vulkan PUBLIC include)
if(FIRSTAGENT_VULKAN_READY)
    add_dependencies(firstagent-kernels-vulkan firstagent-vulkan-shader)
    target_compile_definitions(firstagent-kernels-vulkan PRIVATE
        FIRSTAGENT_HAS_VULKAN=1
        FIRSTAGENT_VULKAN_SHADER_PATH="${FIRSTAGENT_VULKAN_SHADER}"
        FIRSTAGENT_VULKAN_ELEMENTWISE_SHADER_PATH="${FIRSTAGENT_VULKAN_ELEMENTWISE_SHADER}"
    )
    target_include_directories(firstagent-kernels-vulkan PRIVATE "${FIRSTAGENT_VULKAN_INCLUDE_DIR}")
    target_link_libraries(firstagent-kernels-vulkan PUBLIC "${FIRSTAGENT_VULKAN_LIBRARY}")
else()
    target_compile_definitions(firstagent-kernels-vulkan PRIVATE FIRSTAGENT_HAS_VULKAN=0)
endif()
target_compile_options(firstagent-kernels-vulkan PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)

add_library(firstagent-nn STATIC
    src/Hardware.cpp
    src/NeuralNetwork.cpp
    src/Tensor.cpp
    src/Tokenizer.cpp
    src/TokenizerNeural.cpp
)

target_include_directories(firstagent-nn PUBLIC include)
target_link_libraries(firstagent-nn PUBLIC firstagent-kernels-cpu firstagent-kernels-vulkan)
target_compile_options(firstagent-nn PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)

if(FIRSTAGENT_BUILD_APP)
    add_executable(first-agent
        src/Agent.cpp
        src/AgentWorker.cpp
        src/Memory.cpp
        src/SdlWindow.cpp
        src/Tool.cpp
        src/main.cpp
    )

    target_include_directories(first-agent PRIVATE include)
    target_compile_definitions(first-agent PRIVATE
        FIRSTAGENT_PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_link_libraries(first-agent PRIVATE firstagent-nn PkgConfig::SDL3)

    target_compile_options(first-agent PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
    )
endif()

enable_testing()

add_executable(firstagent-tests
    tests/TensorTests.cpp
)
target_link_libraries(firstagent-tests PRIVATE firstagent-nn)
target_compile_options(firstagent-tests PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)
add_test(NAME firstagent-tensor-tests COMMAND firstagent-tests)

add_executable(firstagent-tokenizer-tests
    tests/TokenizerTests.cpp
)
target_link_libraries(firstagent-tokenizer-tests PRIVATE firstagent-nn)
target_compile_options(firstagent-tokenizer-tests PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)
add_test(NAME firstagent-tokenizer-tests COMMAND firstagent-tokenizer-tests)

add_executable(firstagent-system-tests
    tests/SystemTests.cpp
)
target_link_libraries(firstagent-system-tests PRIVATE firstagent-nn)
target_compile_options(firstagent-system-tests PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)
add_test(NAME firstagent-system-tests COMMAND firstagent-system-tests)

add_executable(firstagent-agent-worker-tests
    tests/AgentWorkerTests.cpp
    src/Agent.cpp
    src/AgentWorker.cpp
    src/Memory.cpp
    src/Tool.cpp
)
target_include_directories(firstagent-agent-worker-tests PRIVATE include)
target_link_libraries(firstagent-agent-worker-tests PRIVATE firstagent-nn)
target_compile_options(firstagent-agent-worker-tests PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)
add_test(NAME firstagent-agent-worker-tests COMMAND firstagent-agent-worker-tests)

add_executable(firstagent-word-demo
    tools/WordPredictorDemo.cpp
)
target_compile_definitions(firstagent-word-demo PRIVATE
    FIRSTAGENT_PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(firstagent-word-demo PRIVATE firstagent-nn)
target_compile_options(firstagent-word-demo PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)

add_executable(firstagent-toy
    tools/FirstAgentToy.cpp
)
target_compile_definitions(firstagent-toy PRIVATE
    FIRSTAGENT_PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(firstagent-toy PRIVATE firstagent-nn)
target_compile_options(firstagent-toy PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)

include(GNUInstallDirs)
if(FIRSTAGENT_BUILD_APP)
    install(TARGETS first-agent RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
