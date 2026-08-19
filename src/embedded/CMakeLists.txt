cmake_minimum_required(VERSION 3.10)
project(EdgeVisionEmbeddedTest)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

# 定义包含目录
include_directories(core/include)

# 模拟 Helium 头文件 (在非 ARM 平台编译时)
if(NOT DEFINED ENV{ARM_TOOLCHAIN})
    add_definitions(-DMOCK_HELIUM)
    include_directories(mock/include)
endif()

# 核心源文件
set(SOURCES
    core/ev_operators.c
    ra8p1_main_app.c
    mock/fsp_mock.c
)

add_executable(ev_test_bench ${SOURCES} mock/test_main.c)

# 编译选项
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(ev_test_bench PRIVATE -Wall -Wextra -O2)
endif()
