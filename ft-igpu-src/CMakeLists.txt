cmake_minimum_required(VERSION 3.16)
project(freetoken-igpu CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(FT_WITH_VULKAN "Enable Vulkan iGPU backend" OFF)

add_library(ftcore STATIC
    src/gguf.cpp
    src/graph.cpp
    src/tokenizer.cpp
    src/cpu_backend.cpp
    src/vulkan_backend.cpp
    src/engine.cpp
    src/spec.cpp)
target_include_directories(ftcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(Threads REQUIRED)
target_link_libraries(ftcore PUBLIC Threads::Threads)

if(WIN32)
    target_link_libraries(ftcore PUBLIC pdh)
endif()

if(FT_WITH_VULKAN)
    find_package(Vulkan REQUIRED)
    target_compile_definitions(ftcore PUBLIC FT_HAVE_VULKAN)
    target_link_libraries(ftcore PUBLIC Vulkan::Vulkan)
endif()

if(MSVC)
    target_compile_options(ftcore PUBLIC /EHsc /utf-8 /W3)
else()
    target_compile_options(ftcore PUBLIC -O2 -Wall -Wextra)
endif()

add_executable(ft-cli src/main.cpp)
target_link_libraries(ft-cli PRIVATE ftcore)

add_executable(ft-tests src/tests/test_core.cpp)
target_link_libraries(ft-tests PRIVATE ftcore)

enable_testing()
add_test(NAME core COMMAND ft-tests)
