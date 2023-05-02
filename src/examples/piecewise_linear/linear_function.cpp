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

#include "algorithm/circuit_loader.h"
#include "base/gate_factory.h"
#include "base/two_party_backend.h"
#include "communication/communication_layer.h"
#include "communication/tcp_transport.h"
#include "statistics/analysis.h"
#include "utility/logger.h"

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
  std::uint64_t input_m,input_c,input_a,input_x;
  std::size_t my_id;
  MOTION::Communication::tcp_parties_config tcp_config;
  bool no_run = false;
};

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
    ("input-m", po::value<std::uint64_t>(), "m value")
    ("input-c", po::value<std::uint64_t>(), "c value")
    ("input-a", po::value<std::uint64_t>(), "a value")
    ("input-x", po::value<std::uint64_t>(), "x value")
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
    std::cerr << "error:" << e.what() << "\n\n";
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

//   options.input_value = vm["input-value"].as<std::uint64_t>();
    //Sanity check for the input values
  if((options.my_id ==0) && !vm["input-m"].empty() && !vm["input-c"].empty() && !vm["input-a"].empty())
    {  
    options.input_m = vm["input-m"].as<std::uint64_t>();
    options.input_c = vm["input-c"].as<std::uint64_t>();
    options.input_a = vm["input-a"].as<std::uint64_t>();
    }
 else if((options.my_id==1) && !vm["input-x"].empty())
    {
    options.input_x = vm["input-x"].as<std::uint64_t>();
    }
 else
    {
        std::cerr<<"If party 0- Enter input values for m,c, and a.\nElse, enter input value for x"<<"\n";
        return std::nullopt;
    }

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
  // NB: the inputs need to always be specified in the same order:
  // here we first specify the input of party 0, then that of party 1
  std::vector<ENCRYPTO::ReusableFiberPromise<MOTION::IntegerValues<uint64_t>>> input_promises;
  // std::vector<MOTION::WireVector> input_0_arith, input_1_arith;

  ENCRYPTO::ReusableFiberPromise<MOTION::IntegerValues<uint64_t>> input_promise_m,input_promise_a, input_promise_x,input_promise_c;
  MOTION::WireVector input_0_arith_m,input_0_arith_c,input_0_arith_a, input_1_arith_x;
  if (options.my_id == 0) 
  {
    auto pairm = gate_factory_arith.make_arithmetic_64_input_gate_my(options.my_id, 1);
    input_promise_m = std::move(pairm.first); //has the input promise
    input_0_arith_m = std::move(pairm.second); // has the input m
    input_promises.push_back(std::move(input_promise_m));


    auto pairc = gate_factory_arith.make_arithmetic_64_input_gate_my(options.my_id, 1);
    input_promise_c = std::move(pairc.first); //has the input promise
    input_0_arith_c = std::move(pairc.second); // has the input c
    input_promises.push_back(std::move(input_promise_c));

    auto paira = gate_factory_arith.make_arithmetic_64_input_gate_my(options.my_id, 1);
    input_promise_a = std::move(paira.first); //has the input promise
    input_0_arith_a = std::move(paira.second); // has the input a
    input_promises.push_back(std::move(input_promise_a));

    input_1_arith_x = gate_factory_arith.make_arithmetic_64_input_gate_other(1 - options.my_id, 1); //has input x of party 1
  } 
  else
   {
    input_0_arith_m = gate_factory_arith.make_arithmetic_64_input_gate_other(1 - options.my_id, 1);
    input_0_arith_c = gate_factory_arith.make_arithmetic_64_input_gate_other(1 - options.my_id, 1);
    input_0_arith_a = gate_factory_arith.make_arithmetic_64_input_gate_other(1 - options.my_id, 1);
  
    auto pair = gate_factory_arith.make_arithmetic_64_input_gate_my(options.my_id, 1);
    input_promise_x = std::move(pair.first);
    input_1_arith_x = std::move(pair.second);
    input_promises.push_back(std::move(input_promise_x));
  }

  // converting the arithmetic input shares into Boolean shares for a>=x ie x<=a   
  auto input_0_bool_a = backend.convert(options.boolean_protocol, input_0_arith_a);
  auto input_1_bool_x = backend.convert(options.boolean_protocol, input_1_arith_x);

  // load a boolean circuit to compute 'greater-than'
  MOTION::CircuitLoader circuit_loader;
  auto& gt_circuit =
      circuit_loader.load_gt_circuit(64, options.boolean_protocol != MOTION::MPCProtocol::Yao); //Definition at MOTION2NX/src/motioncore/algorithm/circuit_loader.cpp
  // apply the circuit to the Boolean sahres
  auto output = backend.make_circuit(gt_circuit, input_0_bool_a, input_1_bool_x); //Boolean Output for a>=x
  std::cout<<"Size of gt circuit output wires: "<<output.size()<<std::endl;
  

  // --------------------------- output gates --------------------
  //Arithmetic_output has the arithmetic form of the gt boolean output. (eg. 1 for 1, 0 for 0)
  auto arithmetic_output = gate_factory_arith.make_convert_bit_to_arithmetic_beavy_gate(std::move(output));
  std::cout<<"Size of arithmetic beavy gate: "<<arithmetic_output.size()<<std::endl;
  
  //Calculating m.x
  auto mx_mult_output = gate_factory_arith.make_binary_gate(ENCRYPTO::PrimitiveOperationType::MUL, input_0_arith_m, input_1_arith_x);

  //Calculating m.x + c
  auto plusC_output = gate_factory_arith.make_binary_gate(ENCRYPTO::PrimitiveOperationType::ADD,mx_mult_output,input_0_arith_c);

  //Calculating (m.x+c)(x<=a)
  auto final_output = gate_factory_arith.make_binary_gate(ENCRYPTO::PrimitiveOperationType::MUL, plusC_output,arithmetic_output);

  auto output_future = gate_factory_arith.make_arithmetic_64_output_gate_my(MOTION::ALL_PARTIES,final_output);

  // return promise and future to allow setting inputs and retrieving outputs
//   if(options.my_id == 0)
//     {return std::make_pair(std::move(input_promise_a), std::move(output_future)); }
//   else
//     {return std::make_pair(std::move(input_promise_x), std::move(output_future)); }
    return std::make_pair(std::move(input_promises), std::move(output_future));

}

void run_circuit(const Options& options, MOTION::TwoPartyBackend& backend) {
  // build the circuit and gets promise/future for the input/output
  auto [input_promises, output_future] = create_circuit(options, backend);
  std::cout<<"Created the circuit"<<std::endl;
  if (options.no_run) {
    return;
  }

  // set the promise with our input value
  if(options.my_id==0)
    {
    std::cout<<"Setting value of 'm'"<<std::endl;
    input_promises[0].set_value({options.input_m});
    std::cout<<"Setting value of 'c'"<<std::endl;
    input_promises[1].set_value({options.input_c});
    std::cout<<"Setting value of 'a'"<<std::endl;
    input_promises[2].set_value({options.input_a});
    }
  else
    {
    std::cout<<"Setting value of 'x'"<<std::endl;
    input_promises[0].set_value({options.input_x});
    }
  // execute the protocol
  backend.run();
  std::cout<<"Backend run is done"<<std::endl;
  // retrieve the result from the future
//   bool gt_result = bvs;//.at(0).Get(0);
try
   {
    auto bvs = output_future.get();
    // std::cout<<"Output_Future BVS: "<<bvs<<std::endl;
    auto function_result = bvs.at(0);
    // assert(function_result);
    std::cout<<"\n--------------\nLinear Function output is "<<function_result<<std::endl;
   }  
catch(std::exception& e)
  {
    std::cerr<<"Error during reconstruction"<<e.what()<<std::endl;
  }
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

// auto create_mult_circuit(const Options& options, MOTION::TwoPartyBackend& backend) {
//   // retrieve the gate factories for the chosen protocols
//   auto& gate_factory_arith = backend.get_gate_factory(options.arithmetic_protocol);  // gmw
//   auto& gate_factory_bool = backend.get_gate_factory(options.boolean_protocol);      // beavy

//   // share the inputs using the arithmetic protocol
//   // NB: the inputs need to always be specified in the same order:
//   // here we first specify the input of party 0, then that of party 1
//   ENCRYPTO::ReusableFiberPromise<MOTION::IntegerValues<uint64_t>> input_promise;
//   MOTION::WireVector input_0_arith, input_1_arith;
//   if (options.my_id == 0) {
//     auto pair = gate_factory_arith.make_arithmetic_64_input_gate_my(options.my_id, 1);
//     input_promise = std::move(pair.first);
//     input_0_arith = std::move(pair.second);
//     input_1_arith = gate_factory_arith.make_arithmetic_64_input_gate_other(1 - options.my_id, 1);
//   } else {
//     input_0_arith = gate_factory_arith.make_arithmetic_64_input_gate_other(1 - options.my_id, 1);
//     auto pair = gate_factory_arith.make_arithmetic_64_input_gate_my(options.my_id, 1);
//     input_promise = std::move(pair.first);
//     input_1_arith = std::move(pair.second);
//   }

//   // convert the arithmetic shares into Boolean shares
//   auto input_0_bool = backend.convert(options.boolean_protocol, input_0_arith);
//   auto input_1_bool = backend.convert(options.boolean_protocol, input_1_arith);

//   // load a boolean circuit for to compute 'greater-than'
//   MOTION::CircuitLoader circuit_loader;

//   auto& gt_circuit = circuit_loader.load_circuit(fmt::format("int_mul{}_{}.bristol", 64, "depth"),
//                                                  MOTION::CircuitFormat::Bristol);
//   // apply the circuit to the Boolean sahres
//   auto output = backend.make_circuit(gt_circuit, input_0_bool, input_1_bool);

//   // create an output gates of the result
//   auto output_future = gate_factory_bool.make_boolean_output_gate_my(MOTION::ALL_PARTIES, output);

//   // return promise and future to allow setting inputs and retrieving outputs
//   return std::make_pair(std::move(input_promise), std::move(output_future));
// }

// void run_mult_circuit(const Options& options, MOTION::TwoPartyBackend& backend) {
//   // build the circuit and gets promise/future for the input/output
//   auto [input_promise, output_future] = create_mult_circuit(options, backend);

//   if (options.no_run) {
//     return;
//   }

//   // set the promise with our input value
//   input_promise.set_value({options.input_value});

//   // execute the protocol
//   backend.run();

//   // retrieve the result from the future
//   auto bvs = output_future.get();
//   auto mult_result = 0;

//   // Conversion of result to readable arithmetic format
//   for (int i = 63; i >= 0; i--) {
//     mult_result = mult_result + bvs.at(i).Get(0);
//     // left shift is easier to implement
//     mult_result = mult_result << 1;
//   }
//   mult_result = mult_result >> 1;

//   if (!options.json) {
//     std::cout << "The multiplication result is:- " << mult_result << std::endl;
//   }
// }

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
