add_executable(millionaires_problem millionaires_problem.cpp)
add_executable(millionaires_problem_constMul millionaires_problem_constMul.cpp)

find_package(Boost COMPONENTS json log program_options REQUIRED)

target_compile_features(millionaires_problem PRIVATE cxx_std_20)
target_link_libraries(millionaires_problem
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)

target_compile_features(millionaires_problem_constMul PRIVATE cxx_std_20)
target_link_libraries(millionaires_problem_constMul
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)