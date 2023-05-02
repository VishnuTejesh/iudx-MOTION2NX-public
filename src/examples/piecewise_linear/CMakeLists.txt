add_executable(piecewise_linear piecewise_linear_function.cpp)

find_package(Boost COMPONENTS json log program_options REQUIRED)

target_compile_features(piecewise_linear PRIVATE cxx_std_20)
target_link_libraries(piecewise_linear
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)