cmake_minimum_required(VERSION 3.20)

project(AgentAri VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(AGENTARI_BUILD_APP "Build the SDL realtime application" ON)
option(AGENTARI_ENABLE_VULKAN "Build the Vulkan compute backend when the SDK is available" ON)

set(AGENTARI_VULKAN_SDK_ROOT "${CMAKE_SOURCE_DIR}/third_party/vulkan-sdk/usr")
find_path(AGENTARI_VULKAN_INCLUDE_DIR NAMES vulkan/vulkan.h
    PATHS "${AGENTARI_VULKAN_SDK_ROOT}/include" /usr/include)
find_library(AGENTARI_VULKAN_LIBRARY NAMES vulkan libvulkan.so.1
    PATHS "${AGENTARI_VULKAN_SDK_ROOT}/lib64" /usr/lib64 /usr/lib)
find_program(AGENTARI_GLSLC_EXECUTABLE NAMES glslc
    PATHS "${AGENTARI_VULKAN_SDK_ROOT}/bin")

set(AGENTARI_VULKAN_READY OFF)
if(AGENTARI_ENABLE_VULKAN AND AGENTARI_VULKAN_INCLUDE_DIR AND
   AGENTARI_VULKAN_LIBRARY AND AGENTARI_GLSLC_EXECUTABLE)
    set(AGENTARI_VULKAN_READY ON)
    set(AGENTARI_VULKAN_SHADER "${CMAKE_CURRENT_BINARY_DIR}/agentari-matmul.comp.spv")
    set(AGENTARI_VULKAN_ELEMENTWISE_SHADER
        "${CMAKE_CURRENT_BINARY_DIR}/agentari-elementwise.comp.spv")
    add_custom_command(
        OUTPUT "${AGENTARI_VULKAN_SHADER}"
        COMMAND "${AGENTARI_GLSLC_EXECUTABLE}" -o "${AGENTARI_VULKAN_SHADER}"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul.comp"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul.comp"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${AGENTARI_VULKAN_ELEMENTWISE_SHADER}"
        COMMAND "${AGENTARI_GLSLC_EXECUTABLE}" -o "${AGENTARI_VULKAN_ELEMENTWISE_SHADER}"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/elementwise.comp"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/elementwise.comp"
        VERBATIM
    )
    add_custom_target(agentari-vulkan-shader
        DEPENDS "${AGENTARI_VULKAN_SHADER}" "${AGENTARI_VULKAN_ELEMENTWISE_SHADER}")
endif()

if(AGENTARI_BUILD_APP)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SDL3 REQUIRED IMPORTED_TARGET sdl3)
endif()

# Backend targets are separate so applications can link only the compute
# implementation they need. The public kernel policies remain header-only.
add_library(agentari-kernels-cpu INTERFACE)
target_include_directories(agentari-kernels-cpu INTERFACE include)

add_library(agentari-kernels-vulkan STATIC src/VulkanBackend.cpp)
target_include_directories(agentari-kernels-vulkan PUBLIC include)
if(AGENTARI_VULKAN_READY)
    add_dependencies(agentari-kernels-vulkan agentari-vulkan-shader)
    target_compile_definitions(agentari-kernels-vulkan PRIVATE
        AGENTARI_HAS_VULKAN=1
        AGENTARI_VULKAN_SHADER_PATH="${AGENTARI_VULKAN_SHADER}"
        AGENTARI_VULKAN_ELEMENTWISE_SHADER_PATH="${AGENTARI_VULKAN_ELEMENTWISE_SHADER}"
    )
    target_include_directories(agentari-kernels-vulkan PRIVATE "${AGENTARI_VULKAN_INCLUDE_DIR}")
    target_link_libraries(agentari-kernels-vulkan PUBLIC "${AGENTARI_VULKAN_LIBRARY}")
else()
    target_compile_definitions(agentari-kernels-vulkan PRIVATE AGENTARI_HAS_VULKAN=0)
endif()
target_compile_options(agentari-kernels-vulkan PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)

add_library(agentari-nn STATIC
    src/AtomicTokenizer.cpp
    src/ContextTokenizer.cpp
    src/HierarchicalPrediction.cpp
    src/HierarchicalTokenizer.cpp
    src/SemanticPartLearner.cpp
    src/SpatialNeuronMap.cpp
    src/TeacherFeedback.cpp
    src/WordVocabulary.cpp
    src/Hardware.cpp
    src/NeuralNetwork.cpp
    src/RunLog.cpp
    src/Tensor.cpp
    src/Tokenizer.cpp
    src/TokenizerNeural.cpp
)

target_include_directories(agentari-nn PUBLIC include)
target_link_libraries(agentari-nn PUBLIC agentari-kernels-cpu agentari-kernels-vulkan)
target_compile_options(agentari-nn PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)

if(AGENTARI_BUILD_APP)
    add_executable(agent-ari
        src/Agent.cpp
        src/AgentWorker.cpp
        src/Memory.cpp
        src/SdlWindow.cpp
        src/Tool.cpp
        src/main.cpp
    )

    target_include_directories(agent-ari PRIVATE include)
    target_compile_definitions(agent-ari PRIVATE
        AGENTARI_PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_link_libraries(agent-ari PRIVATE agentari-nn PkgConfig::SDL3)

    target_compile_options(agent-ari PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
    )
endif()

enable_testing()

add_executable(agentari-tests
    tests/TensorTests.cpp
)
target_link_libraries(agentari-tests PRIVATE agentari-nn)
target_compile_options(agentari-tests PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)
add_test(NAME agentari-tensor-tests COMMAND agentari-tests)

add_executable(agentari-tokenizer-tests
    tests/TokenizerTests.cpp
)
target_link_libraries(agentari-tokenizer-tests PRIVATE agentari-nn)
target_compile_options(agentari-tokenizer-tests PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)
add_test(NAME agentari-tokenizer-tests COMMAND agentari-tokenizer-tests)

add_executable(agentari-hierarchical-tokenizer-tests
    tests/HierarchicalTokenizerTests.cpp
)
target_link_libraries(agentari-hierarchical-tokenizer-tests PRIVATE agentari-nn)
target_compile_options(agentari-hierarchical-tokenizer-tests PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)
add_test(NAME agentari-hierarchical-tokenizer-tests
         COMMAND agentari-hierarchical-tokenizer-tests)

add_executable(agentari-hierarchical-prediction-tests
    tests/HierarchicalPredictionTests.cpp
)
target_link_libraries(agentari-hierarchical-prediction-tests PRIVATE agentari-nn)
target_compile_options(agentari-hierarchical-prediction-tests PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)
add_test(NAME agentari-hierarchical-prediction-tests
         COMMAND agentari-hierarchical-prediction-tests)

add_executable(agentari-system-tests
    tests/SystemTests.cpp
)
target_link_libraries(agentari-system-tests PRIVATE agentari-nn)
target_compile_options(agentari-system-tests PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)
add_test(NAME agentari-system-tests COMMAND agentari-system-tests)

add_executable(agentari-agent-worker-tests
    tests/AgentWorkerTests.cpp
    src/Agent.cpp
    src/AgentWorker.cpp
    src/Memory.cpp
    src/Tool.cpp
)
target_include_directories(agentari-agent-worker-tests PRIVATE include)
target_link_libraries(agentari-agent-worker-tests PRIVATE agentari-nn)
target_compile_options(agentari-agent-worker-tests PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)
add_test(NAME agentari-agent-worker-tests COMMAND agentari-agent-worker-tests)

add_executable(agentari-word-demo
    tools/WordPredictorDemo.cpp
)
target_compile_definitions(agentari-word-demo PRIVATE
    AGENTARI_PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(agentari-word-demo PRIVATE agentari-nn)
target_compile_options(agentari-word-demo PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)

add_executable(agentari-hierarchical-demo
    tools/HierarchicalPredictionDemo.cpp
)
target_link_libraries(agentari-hierarchical-demo PRIVATE agentari-nn)
target_compile_options(agentari-hierarchical-demo PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)

add_executable(agentari-console
    tools/AgentAriConsole.cpp
)
target_link_libraries(agentari-console PRIVATE agentari-nn)
target_compile_options(agentari-console PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)

add_executable(agentari-teacher-demo
    tools/TeacherFeedbackDemo.cpp
)
target_link_libraries(agentari-teacher-demo PRIVATE agentari-nn)
target_compile_options(agentari-teacher-demo PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)

add_executable(agentari-toy
    tools/AgentAriToy.cpp
)
target_compile_definitions(agentari-toy PRIVATE
    AGENTARI_PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(agentari-toy PRIVATE agentari-nn)
target_compile_options(agentari-toy PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)

include(GNUInstallDirs)
if(AGENTARI_BUILD_APP)
    install(TARGETS agent-ari RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(TARGETS agentari-console RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
