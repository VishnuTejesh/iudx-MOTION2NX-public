// MIT License
//
// Copyright (c) 2021 Lennart Braun
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <string.h>
#include <sys/resource.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <regex>
#include <stdexcept>

#include <boost/algorithm/string.hpp>
#include <boost/json/serialize.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include "algorithm/circuit_loader.h"
#include "base/gate_factory.h"
#include "base/two_party_backend.h"
#include "communication/communication_layer.h"
#include "communication/tcp_transport.h"
#include "statistics/analysis.h"
#include "utility/logger.h"
#include "utility/new_fixed_point.h"


void testMemoryOccupied() {
  int tSize = 0, resident = 0, share = 0;
  std::ifstream buffer("/proc/self/statm");
  buffer >> tSize >> resident >> share;
  buffer.close();

  long page_size_kb =
      sysconf(_SC_PAGE_SIZE) / 1024;  // in case x86-64 is configured to use 2MB pages
  double rss = resident * page_size_kb;
  std::cout << "RSS - " << rss << " kB\n";

  double shared_mem = share * page_size_kb;
  std::cout << "Shared Memory - " << shared_mem << " kB\n";

  std::cout << "Private Memory - " << rss - shared_mem << "kB\n";
  std::cout << std::endl;
}

namespace po = boost::program_options;

struct Options {
  std::size_t threads;
  bool json;
  std::size_t num_repetitions;
  std::size_t num_simd;
  bool sync_between_setup_and_online;
  MOTION::MPCProtocol arithmetic_protocol;
  MOTION::MPCProtocol boolean_protocol;
  std::size_t num_functions;
  std::uint64_t input_value_x;
  std::vector<std::uint64_t> input_value_a;
  std::vector<std::uint64_t> constant_m, constant_c;
  std::size_t my_id;
  std::size_t fractional_bits;
  MOTION::Communication::tcp_parties_config tcp_config;
  bool no_run = false;
};
std::vector<std::uint64_t> encode_double_to_int(std::vector<std::double_t> input, Options& options)
{
  std::vector<std::uint64_t> temp;    
  for(auto inp: input)
    {
    temp.push_back(MOTION::new_fixed_point::encode<uint64_t, double>(inp,options.fractional_bits));
    }
  return temp;
}

std::optional<Options> parse_program_options(int argc, char* argv[]) {
  Options options;
  boost::program_options::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help,h", po::bool_switch()->default_value(false),"produce help message")
    ("config-file", po::value<std::string>(), "config file containing options")
    ("my-id", po::value<std::size_t>()->required(), "my party id")
    ("party", po::value<std::vector<std::string>>()->multitoken(),
     "(party id, IP, port), e.g., --party 1,127.0.0.1,7777")
    ("threads", po::value<std::size_t>()->default_value(0), "number of threads to use for gate evaluation")
    ("json", po::bool_switch()->default_value(false), "output data in JSON format")
    ("arithmetic-protocol", po::value<std::string>()->required(), "2PC protocol (GMW or BEAVY)")
    ("boolean-protocol", po::value<std::string>()->required(), "2PC protocol (Yao, GMW or BEAVY)")
    ("fractional-bits", po::value<std::size_t>()->default_value(13),"number of fractional bits for fixed-point arithmetic")
    ("num-functions", po::value<std::size_t>(), "Number of Linear Functions")
    ("input-value-x", po::value<std::double_t>(), "input value ('x') for Piecewise Linear Functions")
    ("input-value-a", po::value<std::string>()->multitoken(), "input value ('a')for Linear Function Problem. Eg. --input-value-a -2,10,3,5.5 ") // Number of 'a's = Num. of 'm'-1 = Num of 'c'-1
    ("constant-m", po::value<std::string>()->required(), "input slope 'm' for Linear Function Problem,  Eg. --constant-m -2,10,3,5.5 6")
    ("constant-c", po::value<std::string>()->required(), "Y intercept 'c' for Linear Function Problem, Eg. --constant-m -2,10,3,5.5 3")
    ("repetitions", po::value<std::size_t>()->default_value(1), "number of repetitions")
    ("num-simd", po::value<std::size_t>()->default_value(1), "number of SIMD values")
    ("sync-between-setup-and-online", po::bool_switch()->default_value(false),
     "run a synchronization protocol before the online phase starts")
    ("no-run", po::bool_switch()->default_value(false), "just build the circuit, but not execute it")
    ;
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  bool help = vm["help"].as<bool>();
  if (help) {
    std::cerr << desc << "\n";
    return std::nullopt;
  }
  if (vm.count("config-file")) {
    std::ifstream ifs(vm["config-file"].as<std::string>().c_str());
    po::store(po::parse_config_file(ifs, desc), vm);
  }
  try {
    po::notify(vm);
  } catch (std::exception& e) {
    std::cerr << "Config file error:" << e.what() << "\n\n";
    std::cerr << desc << "\n";
    return std::nullopt;
  }

  options.my_id = vm["my-id"].as<std::size_t>();
  options.threads = vm["threads"].as<std::size_t>();
  options.json = vm["json"].as<bool>();
  options.num_repetitions = vm["repetitions"].as<std::size_t>();
  options.num_simd = vm["num-simd"].as<std::size_t>();
  options.sync_between_setup_and_online = vm["sync-between-setup-and-online"].as<bool>();
  options.no_run = vm["no-run"].as<bool>();
  if (options.my_id > 1) {
    std::cerr << "my-id must be one of 0 and 1\n";
    return std::nullopt;
  }

  auto arithmetic_protocol = vm["arithmetic-protocol"].as<std::string>();
  boost::algorithm::to_lower(arithmetic_protocol);
  if (arithmetic_protocol == "gmw") {
    options.arithmetic_protocol = MOTION::MPCProtocol::ArithmeticGMW;
  } else if (arithmetic_protocol == "beavy") {
    options.arithmetic_protocol = MOTION::MPCProtocol::ArithmeticBEAVY;
  } else {
    std::cerr << "invalid protocol: " << arithmetic_protocol << "\n";
    return std::nullopt;
  }
  auto boolean_protocol = vm["boolean-protocol"].as<std::string>();
  boost::algorithm::to_lower(boolean_protocol);
  if (boolean_protocol == "yao") {
    options.boolean_protocol = MOTION::MPCProtocol::Yao;
  } else if (boolean_protocol == "gmw") {
    options.boolean_protocol = MOTION::MPCProtocol::BooleanGMW;
  } else if (boolean_protocol == "beavy") {
    options.boolean_protocol = MOTION::MPCProtocol::BooleanBEAVY;
  } else {
    std::cerr << "invalid protocol: " << boolean_protocol << "\n";
    return std::nullopt;
  }
   options.fractional_bits = vm["fractional-bits"].as<std::size_t>();
  //---------------------------------parse input arguments---------------------------------------------
  // 1. Function to parse the string input and convert it to a vector of doubles
  const auto parse_input_argument = ([](std::string const& v) {
            std::vector<double> double_values;
            auto it = boost::make_split_iterator(v, boost::token_finder(boost::algorithm::is_any_of(" ,")));
            std::transform(it, {}, std::back_inserter(double_values), [](auto& s) {
                      return boost::lexical_cast<std::double_t>(s);
                    });
            return double_values;
        });
  // 2. Parse the num-functions argument. Validate the input. 
  options.num_functions = vm["num-functions"].as<std::size_t>();
  if(options.num_functions<1)
    {
      std::cerr << "num-functions must be greater than or equal to 1. "<<std::endl;
    return std::nullopt;
    }
  // 3. Parse the publicly known constants - constant_m and constant_c. Validate the inputs and encode it.
  std::vector<double> constant_m_double = parse_input_argument(vm["constant-m"].as<std::string>()); 
  std::vector<double> constant_c_double = parse_input_argument(vm["constant-c"].as<std::string>()); 
  auto num_m = constant_m_double.size(), num_c = constant_c_double.size();
  if((num_m!=num_c) || (num_m!=options.num_functions)|| num_m<1)
    {
      std::cerr<<"Number of values in constant-m, constant-c should match with the num-functions. There should be atleast one function.";
      return std::nullopt;
    }
  std::cout<<"Num of m values:"<<num_m<<"\nNum of c values: "<<num_c<<std::endl;
  options.constant_m = encode_double_to_int(constant_m_double,options);
  options.constant_c = encode_double_to_int(constant_c_double,options);
  if((options.constant_m.size()!=options.constant_c.size()) || (options.constant_m.size()<1))
    {
      std::cerr<<"Options.constant_m, Options.constant_c, and options.input_value_a are of incorrect sizes after encoding.";
      return std::nullopt;
    }
  
  for(auto i=0;i<options.num_functions;i++)
    {
      std::cout<<"m"<<i<<"= "<<options.constant_m[i]<<std::endl;
      std::cout<<"c"<<i<<"= "<<options.constant_c[i]<<std::endl;
    }
  // 4. Parse the party specific input. Party0 - 'x', Party1 - 'a' vector.
  if(options.my_id==1){
      // 4a. Validate the input-value-x and encode it.
      if(vm.count("input-value-x")!=1){
          std::cerr<<"Party 1 must give one value for input-value-x\n";
          return std::nullopt;
        }
      std::cout<<"input-value-x before encoding is "<<vm["input-value-x"].as<std::double_t>()<<std::endl;
      options.input_value_x = MOTION::new_fixed_point::encode<uint64_t, double>(vm["input-value-x"].as<std::double_t>(),options.fractional_bits);
      std::cout<<"input-value-x after encoding is "<<options.input_value_x<<std::endl;
    }
  else{
      // 4b. Parse the input-value-a string into a vector of doubles, encode it and validate.
      std::vector<double> input_value_a_double = parse_input_argument(vm["input-value-a"].as<std::string>());
      auto num_a=input_value_a_double.size();
      if(num_a<1)
        {
          std::cerr<<"Party 0 must give values for input-value-a\n";
          return std::nullopt;
        }
      else if(num_m!=(num_a+1))
        {
          std::cerr<<"In Party 0 input-value-a be of size (num-functions-1)\n";
          return std::nullopt;
        }
      else
        { 
          options.input_value_a = encode_double_to_int(input_value_a_double,options);
          //Checking whether options is correctly populated. 
          if(options.num_functions!=(options.input_value_a.size()+1))
            {
              std::cerr<<"Options.input_value_a is of incorrect size after encoding.";
              return std::nullopt;
            }
          std::cout<<"After encoding input-value-a:\n";
          for(auto i=0;i<options.num_functions-1;i++)
            {std::cout<<"a"<<i<<"= "<<options.input_value_a[i]<<std::endl;}
          }
    }
  //----------------------------------------------------------------------------------------------
  const auto parse_party_argument =
      [](const auto& s) -> std::pair<std::size_t, MOTION::Communication::tcp_connection_config> {
    const static std::regex party_argument_re("([01]),([^,]+),(\\d{1,5})");
    std::smatch match;
    if (!std::regex_match(s, match, party_argument_re)) {
      throw std::invalid_argument("invalid party argument");
    }
    auto id = boost::lexical_cast<std::size_t>(match[1]);
    auto host = match[2];
    auto port = boost::lexical_cast<std::uint16_t>(match[3]);
    return {id, {host, port}};
  };

  const std::vector<std::string> party_infos = vm["party"].as<std::vector<std::string>>();
  if (party_infos.size() != 2) {
    std::cerr << "expecting two --party options\n";
    return std::nullopt;
  }

  options.tcp_config.resize(2);
  std::size_t other_id = 2;

  const auto [id0, conn_info0] = parse_party_argument(party_infos[0]);
  const auto [id1, conn_info1] = parse_party_argument(party_infos[1]);
  if (id0 == id1) {
    std::cerr << "need party arguments for party 0 and 1\n";
    return std::nullopt;
  }
  options.tcp_config[id0] = conn_info0;
  options.tcp_config[id1] = conn_info1;

  return options;
}

std::unique_ptr<MOTION::Communication::CommunicationLayer> setup_communication(
    const Options& options) {
  MOTION::Communication::TCPSetupHelper helper(options.my_id, options.tcp_config);
  return std::make_unique<MOTION::Communication::CommunicationLayer>(options.my_id,
                                                                     helper.setup_connections());
}

auto create_circuit(const Options& options, MOTION::TwoPartyBackend& backend) {
  // retrieve the gate factories for the chosen protocols
  auto& gate_factory_arith = backend.get_gate_factory(options.arithmetic_protocol);
  auto& gate_factory_bool = backend.get_gate_factory(options.boolean_protocol);

  // share the inputs using the arithmetic protocol
  // here we first specify the input of party 0, then that of party 1
  std::vector<ENCRYPTO::ReusableFiberPromise<MOTION::IntegerValues<uint64_t>>> input_promises;
  std::vector<MOTION::WireVector> input_0_arith, input_0_bool; // all the 'a's
  MOTION::WireVector input_1_arith,input_1_bool; //the 'x' value

  if (options.my_id == 0) 
  { // 1. Has a vector of wirevectors for a (my)
    // 2. Has a single wirevector for x (other)
    for(auto i=0;i<options.num_functions-1;i++)
      {
      auto pair = gate_factory_arith.make_arithmetic_64_input_gate_my(options.my_id, 1);
      auto input_promise = std::move(pair.first); 
      auto input_0_arith_tmp = std::move(pair.second); 
      input_promises.push_back(std::move(input_promise));
      input_0_arith.push_back(std::move(input_0_arith_tmp));
      }
    input_1_arith = gate_factory_arith.make_arithmetic_64_input_gate_other(1 - options.my_id, 1); // Has the wire for 'x'
  } 
  else 
  { 
     // 1. Has a vector of wirevectors for a (other)
     // 2. Has a single wirevector for x (my)
    for(auto i=0;i<options.num_functions-1;i++)
      {
      auto input_0_arith_tmp = gate_factory_arith.make_arithmetic_64_input_gate_other(1 - options.my_id, 1);
      input_0_arith.push_back(std::move(input_0_arith_tmp)); //Has wires for 'a'
      }
    auto pair = gate_factory_arith.make_arithmetic_64_input_gate_my(options.my_id, 1);
    auto input_promise=std::move(pair.first);
    input_promises.push_back(std::move(input_promise));
    input_1_arith = std::move(pair.second); //Has wires for 'x' value
  }

  //---------------------- convert the arithmetic shares into Boolean shares----------------------
  //1. Converting x arithmetic to boolean

  input_1_bool = backend.convert(options.boolean_protocol, std::move(input_1_arith));
  //2. Converting all the arithmetic 'a' shares to boolean shares
  for(auto inp : input_0_arith)
    {
      auto input_0_bool_temp = backend.convert(options.boolean_protocol, inp);
      input_0_bool.push_back(std::move(input_0_bool_temp));
    }
  //----------------------------------------------------------------------------------------------
  
  MOTION::CircuitLoader circuit_loader;
  std::vector<MOTION::WireVector> I_outputs, Z_outputs,J_outputs, final_outputs;

  // ENCRYPTO::ReusableFiberFuture<MOTION::IntegerValues<std::uint64_t>>* output_futures =
  //     new ENCRYPTO::ReusableFiberFuture<MOTION::IntegerValues<std::uint64_t>>[options.num_functions];
  // std::array<ENCRYPTO::ReusableFiberFuture<MOTION::IntegerValues<std::uint64_t>>, 64>* output_futures = new std::array<ENCRYPTO::ReusableFiberFuture<MOTION::IntegerValues<std::uint64_t>>, 64>();

  auto output_futures = std::make_unique<ENCRYPTO::ReusableFiberFuture<MOTION::IntegerValues<std::uint64_t>>[]>(options.num_functions);

  //1.  Compute the I-function = -----------------------------------------
  //                            |x<a[0] | x<a[1] | x<a[2]|......|x<a[n-1]|
  //                           -------------------------------------------
  for(auto i=0;i<options.num_functions-1;i++)
      {
      // 1a. Load the gt circuit.
      auto& gt_circuit =
          circuit_loader.load_gt_circuit(64, options.boolean_protocol != MOTION::MPCProtocol::Yao); 

      // 1b. Compute the I-function. Apply 'the greater than' circuit to the Boolean shares
     
      auto gt_output = backend.make_circuit(gt_circuit, input_0_bool.at(i), input_1_bool); //Boolean Output

      // 1c. Convert boolean bit gt output to arithmetic. 
      
      auto I_arithmetic_output = gate_factory_arith.make_convert_bit_to_arithmetic_beavy_gate(std::move(gt_output));
      I_outputs.push_back(I_arithmetic_output); 
      }

  // 2. Compute J-function = --------------------------------------------------------------------
  //                         |I0 | (1-I0).I1 | (1-I1).I2 |(1-I2).I3|(1-In-2).In-1 | (1-In-1)|
  //                        --------------------------------------------------------------------

  // 2a.  Adding I0 directly to J-function
  J_outputs.push_back(I_outputs.at(0));
  for(auto i=1;i<options.num_functions-1;i++)
    {
    // 2b.  Compute -(Ii-1)
    auto neg_I = gate_factory_arith.make_unary_gate(ENCRYPTO::PrimitiveOperationType::NEG,I_outputs[i-1]);
    // 2c. Compute 1 + (-(Ii-1))
    auto negIplusOne_output = gate_factory_arith.make_constADD_gate(neg_I,1,options.fractional_bits);
    // 2d. Compute (1-(Ii-1)) . Ii
    auto mult_output = gate_factory_arith.make_binary_gate(ENCRYPTO::PrimitiveOperationType::MUL,negIplusOne_output,I_outputs[i]);
    J_outputs.push_back(std::move(mult_output));
    }
  // 2e. Compute -(I[n-1])  
  auto neg_I = gate_factory_arith.make_unary_gate(ENCRYPTO::PrimitiveOperationType::NEG,I_outputs[options.num_functions-2]);
  // 2f. Compute 1 - (I[n-1])
  auto negIplusOne_output = gate_factory_arith.make_constADD_gate(neg_I,1,options.fractional_bits);
  J_outputs.push_back(std::move(negIplusOne_output));
  

  // 3. Compute Z-function = --------------------------------------------------------------------
  //                         |m0.x+c0 | m1.x+c1 | m2.x+c2 | m3.x+c3|..... | mn.x+cn|
  //                        --------------------------------------------------------------------
  for(auto i=0;i<options.num_functions;i++)
    {
    
    //3a.  Calculating m.x
    auto mx_mult_output = gate_factory_arith.make_constMul_gate(input_1_arith,options.constant_m[i],options.fractional_bits);
    //3b. Calculating m.x + c
    auto plusC_output = gate_factory_arith.make_constADD_gate(mx_mult_output,options.constant_c[i],options.fractional_bits);
    Z_outputs.push_back(plusC_output);
    }
  // 4. Compute y-function = --------------------------------------------------------------------
  //                         |Z0.J0 | Z1.J1 | Z2.J2 | Z3.J3 |..... | Zn.Jn |
  //                        --------------------------------------------------------------------  
  for(auto i=0;i<options.num_functions;i++)
    {
      //4a. Compute Z[i].J[i] 
      auto final_out = gate_factory_arith.make_binary_gate(ENCRYPTO::PrimitiveOperationType::MUL,Z_outputs[i],J_outputs[i]);
      //4b. Make the arithmetic output gate
      auto output_future = gate_factory_arith.make_arithmetic_64_output_gate_my(MOTION::ALL_PARTIES,final_out);
      final_outputs.push_back(std::move(final_out));
    // (*output_futures)[i] = std::move(output_future);  
      output_futures[i] = std::move(output_future);  
    }  

  // 5. Return promises and futures
  return std::make_pair(std::move(input_promises), std::move(output_futures)); 
}

void run_circuit(const Options& options, MOTION::TwoPartyBackend& backend) {
  // 1. build the circuit and gets promise/future for the input/output
  auto [input_promises, output_futures] = create_circuit(options, backend);
  
  if (options.no_run) {
    return;
  }

  // 2. Set the promise with our input values based on the party ID
  if(options.my_id==0)
    {
      //2a.  Set all the 'a' values at party 0
      for(auto i=0;i<options.num_functions-1;i++)
        {
          input_promises[i].set_value({options.input_value_a[i]});
        }
    }
  else
    {
      // 2b.  Set the 'x' value at party 1. 
        input_promises[0].set_value({options.input_value_x});
    }

  // 3. Execute the protocol
  backend.run();
  // 4. Retrieve the results from output future
try
   {
      for(auto i=0;i<options.num_functions;i++)
        {
        auto bvs = output_futures[i].get();
        auto result = bvs.at(0);
        // std::cout<<"Result before final decode: "<<result<<std::endl;
        double temp =
              MOTION::new_fixed_point::decode<uint64_t, double>(result, options.fractional_bits); 
        std::cout<<"Function "<<i<<" : "<<temp<<std::endl;
        }  
   }  
catch(std::exception& e)
  {
    std::cerr<<"Error during reconstruction"<<e.what()<<std::endl;
  }
// delete[] output_futures;
}
void print_stats(const Options& options,
                 const MOTION::Statistics::AccumulatedRunTimeStats& run_time_stats,
                 const MOTION::Statistics::AccumulatedCommunicationStats& comm_stats) {
  if (options.json) {
    auto obj = MOTION::Statistics::to_json("millionaires_problem", run_time_stats, comm_stats);
    obj.emplace("party_id", options.my_id);
    obj.emplace("arithmetic_protocol", MOTION::ToString(options.arithmetic_protocol));
    obj.emplace("boolean_protocol", MOTION::ToString(options.boolean_protocol));
    obj.emplace("simd", options.num_simd);
    obj.emplace("threads", options.threads);
    obj.emplace("sync_between_setup_and_online", options.sync_between_setup_and_online);
    std::cout << obj << "\n";
  } else {
    std::cout << MOTION::Statistics::print_stats("millionaires_problem", run_time_stats,
                                                 comm_stats);
  }
}

int main(int argc, char* argv[]) {
  testMemoryOccupied();

  auto options = parse_program_options(argc, argv);
  if (!options.has_value()) {
    return EXIT_FAILURE;
  }

  try {
    auto comm_layer = setup_communication(*options);
    auto logger = std::make_shared<MOTION::Logger>(options->my_id,
                                                   boost::log::trivial::severity_level::trace);
    comm_layer->set_logger(logger);
    MOTION::Statistics::AccumulatedRunTimeStats run_time_stats;
    MOTION::Statistics::AccumulatedCommunicationStats comm_stats;
    for (std::size_t i = 0; i < options->num_repetitions; ++i) {
      MOTION::TwoPartyBackend backend(*comm_layer, options->threads,
                                      options->sync_between_setup_and_online, logger);
      run_circuit(*options, backend);

      // run_mult_circuit(*options, backend);

      comm_layer->sync();
      comm_stats.add(comm_layer->get_transport_statistics());
      comm_layer->reset_transport_statistics();
      run_time_stats.add(backend.get_run_time_stats());
    }
    comm_layer->shutdown();
    print_stats(*options, run_time_stats, comm_stats);
  } catch (std::runtime_error& e) {
    std::cerr << "ERROR OCCURRED: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
  testMemoryOccupied();
  return EXIT_SUCCESS;
}
