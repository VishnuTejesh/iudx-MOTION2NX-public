// MIT License
//
// Copyright (c) 2020 Lennart Braun
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

#include "gate.h"

#include "base/gate_factory.h"
#include "crypto/motion_base_provider.h"
#include "crypto/multiplication_triple/mt_provider.h"
#include "crypto/sharing_randomness_generator.h"
#include "gmw_provider.h"
#include "utility/helpers.h"
#include "utility/logger.h"
#include "wire.h"

namespace MOTION::proto::gmw {

namespace detail {

template <typename WireType>
BasicGMWBinaryGate<WireType>::BasicGMWBinaryGate(std::size_t gate_id, GMWWireVector&& in_b,
                                                 GMWWireVector&& in_a)
    : NewGate(gate_id),
      num_wires_(in_a.size()),
      inputs_a_(std::move(in_a)),
      inputs_b_(std::move(in_b)) {
  if (num_wires_ == 0) {
    throw std::logic_error("number of wires need to be positive");
  }
  if (num_wires_ != inputs_b_.size()) {
    throw std::logic_error("number of wires need to be the same for both inputs");
  }
  auto num_simd = inputs_a_[0]->get_num_simd();
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    if (inputs_a_[wire_i]->get_num_simd() != num_simd ||
        inputs_b_[wire_i]->get_num_simd() != num_simd) {
      throw std::logic_error("number of SIMD values need to be the same for all wires");
    }
  }
  outputs_.reserve(num_wires_);
  std::generate_n(std::back_inserter(outputs_), num_wires_,
                  [num_simd] { return std::make_shared<WireType>(num_simd); });
}

template class BasicGMWBinaryGate<BooleanGMWWire>;
template class BasicGMWBinaryGate<ArithmeticGMWWire<std::uint8_t>>;
template class BasicGMWBinaryGate<ArithmeticGMWWire<std::uint16_t>>;
template class BasicGMWBinaryGate<ArithmeticGMWWire<std::uint32_t>>;
template class BasicGMWBinaryGate<ArithmeticGMWWire<std::uint64_t>>;

template <typename WireType>
BasicGMWUnaryGate<WireType>::BasicGMWUnaryGate(std::size_t gate_id, GMWWireVector&& in,
                                               bool forward)
    : NewGate(gate_id), num_wires_(in.size()), inputs_(std::move(in)) {
  if (num_wires_ == 0) {
    throw std::logic_error("number of wires need to be positive");
  }
  auto num_simd = inputs_[0]->get_num_simd();
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    if (inputs_[wire_i]->get_num_simd() != num_simd) {
      throw std::logic_error("number of SIMD values need to be the same for all wires");
    }
  }
  if (forward) {
    outputs_ = inputs_;
  } else {
    outputs_.reserve(num_wires_);
    std::generate_n(std::back_inserter(outputs_), num_wires_,
                    [num_simd] { return std::make_shared<WireType>(num_simd); });
  }
}

template class BasicGMWUnaryGate<BooleanGMWWire>;
template class BasicGMWUnaryGate<ArithmeticGMWWire<std::uint8_t>>;
template class BasicGMWUnaryGate<ArithmeticGMWWire<std::uint16_t>>;
template class BasicGMWUnaryGate<ArithmeticGMWWire<std::uint32_t>>;
template class BasicGMWUnaryGate<ArithmeticGMWWire<std::uint64_t>>;

}  // namespace detail

BooleanGMWInputGateSender::BooleanGMWInputGateSender(
    std::size_t gate_id, GMWProvider& gmw_provider, std::size_t num_wires, std::size_t num_simd,
    ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>&& input_future)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      num_wires_(num_wires),
      num_simd_(num_simd),
      input_id_(gmw_provider.get_next_input_id(num_wires)),
      input_future_(std::move(input_future)) {
  outputs_.reserve(num_wires_);
  std::generate_n(std::back_inserter(outputs_), num_wires_,
                  [num_simd] { return std::make_shared<BooleanGMWWire>(num_simd); });
}

void BooleanGMWInputGateSender::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWInputGateSender::evaluate_setup start", gate_id_));
    }
  }

  auto my_id = gmw_provider_.get_my_id();
  auto num_parties = gmw_provider_.get_num_parties();
  auto& mbp = gmw_provider_.get_motion_base_provider();
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    auto& wire = outputs_[wire_i];
    wire->get_share().Resize(num_simd_);
    for (std::size_t party_id = 0; party_id < num_parties; ++party_id) {
      if (party_id == my_id) {
        continue;
      }
      auto& rng = mbp.get_my_randomness_generator(party_id);
      wire->get_share() ^= rng.GetBits(input_id_ + wire_i, num_simd_);
    }
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWInputGateSender::evaluate_setup end", gate_id_));
    }
  }
}

void BooleanGMWInputGateSender::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWInputGateSender::evaluate_online start", gate_id_));
    }
  }

  // wait for input value
  const auto inputs = input_future_.get();

  // compute my share
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    auto& w_o = outputs_[wire_i];
    auto& share = w_o->get_share();
    const auto& input_bits = inputs.at(wire_i);
    if (input_bits.GetSize() != num_simd_) {
      throw std::runtime_error("size of input bit vector != num_simd_");
    }
    share ^= input_bits;
    w_o->set_online_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWInputGateSender::evaluate_online end", gate_id_));
    }
  }
}

BooleanGMWInputGateReceiver::BooleanGMWInputGateReceiver(std::size_t gate_id,
                                                         GMWProvider& gmw_provider,
                                                         std::size_t num_wires,
                                                         std::size_t num_simd,
                                                         std::size_t input_owner)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      num_wires_(num_wires),
      num_simd_(num_simd),
      input_owner_(input_owner),
      input_id_(gmw_provider.get_next_input_id(num_wires)) {
  outputs_.reserve(num_wires_);
  std::generate_n(std::back_inserter(outputs_), num_wires_,
                  [num_simd] { return std::make_shared<BooleanGMWWire>(num_simd); });
}

void BooleanGMWInputGateReceiver::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWInputGateReceiver::evaluate_setup start", gate_id_));
    }
  }

  auto& mbp = gmw_provider_.get_motion_base_provider();
  auto& rng = mbp.get_their_randomness_generator(input_owner_);
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    auto& wire = outputs_[wire_i];
    wire->get_share() = rng.GetBits(input_id_ + wire_i, num_simd_);
    wire->set_online_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWInputGateReceiver::evaluate_setup end", gate_id_));
    }
  }
}

void BooleanGMWInputGateReceiver::evaluate_online() {
  // nothing to do
}

// Determine the total number of bits in a collection of wires.
static std::size_t count_bits(const BooleanGMWWireVector& wires) {
  return std::transform_reduce(std::begin(wires), std::end(wires), 0, std::plus<>(),
                               [](const auto& a) { return a->get_num_simd(); });
}

BooleanGMWOutputGate::BooleanGMWOutputGate(std::size_t gate_id, GMWProvider& gmw_provider,
                                           BooleanGMWWireVector&& inputs, std::size_t output_owner)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      num_wires_(inputs.size()),
      output_owner_(output_owner),
      inputs_(std::move(inputs)) {
  std::size_t my_id = gmw_provider_.get_my_id();
  if (output_owner_ == ALL_PARTIES || output_owner_ == my_id) {
    share_futures_ = gmw_provider_.register_for_bits_messages(gate_id_, count_bits(inputs_));
  }
}

ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>
BooleanGMWOutputGate::get_output_future() {
  std::size_t my_id = gmw_provider_.get_my_id();
  if (output_owner_ == ALL_PARTIES || output_owner_ == my_id) {
    return output_promise_.get_future();
  } else {
    throw std::logic_error("not this parties output");
  }
}

void BooleanGMWOutputGate::evaluate_setup() {
  // nothing to do
}

void BooleanGMWOutputGate::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWOutputGate::evaluate_online start", gate_id_));
    }
  }

  std::size_t my_id = gmw_provider_.get_my_id();
  ENCRYPTO::BitVector<> my_share;
  // auto num_bits = count_bits(inputs_);  // TODO: reserve
  for (const auto& wire : inputs_) {
    my_share.Append(wire->get_share());
  }
  if (output_owner_ != my_id) {
    if (output_owner_ == ALL_PARTIES) {
      gmw_provider_.broadcast_bits_message(gate_id_, my_share);
    } else {
      gmw_provider_.send_bits_message(output_owner_, gate_id_, std::move(my_share));
    }
  }
  if (output_owner_ == ALL_PARTIES || output_owner_ == my_id) {
    std::size_t num_parties = gmw_provider_.get_num_parties();
    for (std::size_t party_id = 0; party_id < num_parties; ++party_id) {
      if (party_id == my_id) {
        continue;
      }
      const auto other_share = share_futures_[party_id].get();
      my_share ^= other_share;
    }
    // TODO: set_value of output_promise_
    std::vector<ENCRYPTO::BitVector<>> outputs;
    outputs.reserve(num_wires_);
    std::size_t bit_offset = 0;
    for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
      auto num_simd = inputs_[wire_i]->get_num_simd();
      outputs.push_back(my_share.Subset(bit_offset, bit_offset + num_simd));
      bit_offset += num_simd;
    }
    output_promise_.set_value(std::move(outputs));
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: BooleanGMWOutputGate::evaluate_online end", gate_id_));
    }
  }
}

BooleanGMWINVGate::BooleanGMWINVGate(std::size_t gate_id, const GMWProvider& gmw_provider,
                                     GMWWireVector&& in)
    : detail::BasicGMWUnaryGate<BooleanGMWWire>(gate_id, std::move(in),
                                                !gmw_provider.is_my_job(gate_id)),
      is_my_job_(gmw_provider.is_my_job(gate_id)) {}

void BooleanGMWINVGate::evaluate_setup() {
  // nothing to do
}

void BooleanGMWINVGate::evaluate_online() {
  if (!is_my_job_) {
    return;
  }

  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& w_in = inputs_[wire_i];
    w_in->wait_online();
    auto& w_o = outputs_[wire_i];
    w_o->get_share() = ~w_in->get_share();
    w_o->set_online_ready();
  }
}

void BooleanGMWXORGate::evaluate_setup() {
  // nothing to do
}

void BooleanGMWXORGate::evaluate_online() {
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& w_a = inputs_a_[wire_i];
    const auto& w_b = inputs_b_[wire_i];
    w_a->wait_online();
    w_b->wait_online();
    auto& w_o = outputs_[wire_i];
    w_o->get_share() = w_a->get_share() ^ w_b->get_share();
    w_o->set_online_ready();
  }
}

BooleanGMWANDGate::BooleanGMWANDGate(std::size_t gate_id, GMWProvider& gmw_provider,
                                     GMWWireVector&& in_a, GMWWireVector&& in_b)
    : detail::BasicGMWBinaryGate<BooleanGMWWire>(gate_id, std::move(in_a), std::move(in_b)),
      gmw_provider_(gmw_provider) {
  auto num_bits = count_bits(inputs_a_);
  mt_offset_ = gmw_provider_.get_mt_provider().RequestBinaryMTs(num_bits);
  share_futures_ = gmw_provider_.register_for_bits_messages(gate_id_, 2 * count_bits(inputs_a_));
}

void BooleanGMWANDGate::evaluate_setup() {
  // nothing to do
}

void BooleanGMWANDGate::evaluate_online() {
  auto num_bits = count_bits(inputs_a_);
  auto num_simd = inputs_a_[0]->get_num_simd();
  const auto& mtp = gmw_provider_.get_mt_provider();
  auto mts = mtp.GetBinary(mt_offset_, num_bits);
  ENCRYPTO::BitVector<> x;
  ENCRYPTO::BitVector<> y;
  x.Reserve(Helpers::Convert::BitsToBytes(num_bits));
  y.Reserve(Helpers::Convert::BitsToBytes(num_bits));

  // collect all shares into a single buffer
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& wire_x = inputs_a_[wire_i];
    x.Append(wire_x->get_share());
    const auto& wire_y = inputs_b_[wire_i];
    y.Append(wire_y->get_share());
  }
  // mask values with a, b
  auto de = x ^ mts.a;
  de.Append(y ^ mts.b);
  gmw_provider_.broadcast_bits_message(gate_id_, de);
  // compute d, e
  auto num_parties = gmw_provider_.get_num_parties();
  auto my_id = gmw_provider_.get_my_id();
  for (std::size_t party_id = 0; party_id < num_parties; ++party_id) {
    if (party_id == my_id) {
      continue;
    }
    de ^= share_futures_[party_id].get();
  }
  auto e = de.Subset(num_bits, 2 * num_bits);
  auto d = std::move(de);
  d.Resize(num_bits);
  x &= e;  // x & e
  y &= d;  // y & d
  d &= e;  // d & e
  auto result = std::move(mts.c);
  result ^= x;
  result ^= y;
  if (gmw_provider_.is_my_job(gate_id_)) {
    result ^= d;
  }

  // distribute data among wires
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    auto& wire_o = outputs_[wire_i];
    wire_o->get_share() = result.Subset(wire_i * num_simd, (wire_i + 1) * num_simd);
    wire_o->set_online_ready();
  }
}

template <typename T>
ArithmeticGMWInputGateSender<T>::ArithmeticGMWInputGateSender(
    std::size_t gate_id, GMWProvider& gmw_provider, std::size_t num_simd,
    ENCRYPTO::ReusableFiberFuture<std::vector<T>>&& input_future)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      num_simd_(num_simd),
      input_id_(gmw_provider.get_next_input_id(1)),
      input_future_(std::move(input_future)),
      output_(std::make_shared<ArithmeticGMWWire<T>>(num_simd)) {
  output_->get_share().resize(num_simd, 0);
}

template <typename T>
void ArithmeticGMWInputGateSender<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWInputGateSender<T>::evaluate_setup start", gate_id_));
    }
  }

  auto my_id = gmw_provider_.get_my_id();
  auto num_parties = gmw_provider_.get_num_parties();
  auto& mbp = gmw_provider_.get_motion_base_provider();
  auto& share = output_->get_share();
  for (std::size_t party_id = 0; party_id < num_parties; ++party_id) {
    if (party_id == my_id) {
      continue;
    }
    auto& rng = mbp.get_my_randomness_generator(party_id);
    std::transform(std::begin(share), std::end(share),
                   std::begin(rng.GetUnsigned<T>(input_id_, num_simd_)), std::begin(share),
                   std::minus{});
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWInputGateSender<T>::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticGMWInputGateSender<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWInputGateSender<T>::evaluate_online start", gate_id_));
    }
  }

  // wait for input value
  const auto input = input_future_.get();
  if (input.size() != num_simd_) {
    throw std::runtime_error("size of input bit vector != num_simd_");
  }

  // compute my share
  auto& share = output_->get_share();
  std::transform(std::begin(share), std::end(share), std::begin(input), std::begin(share),
                 std::plus{});
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWInputGateSender::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticGMWInputGateSender<std::uint8_t>;
template class ArithmeticGMWInputGateSender<std::uint16_t>;
template class ArithmeticGMWInputGateSender<std::uint32_t>;
template class ArithmeticGMWInputGateSender<std::uint64_t>;

template <typename T>
ArithmeticGMWInputGateReceiver<T>::ArithmeticGMWInputGateReceiver(std::size_t gate_id,
                                                                  GMWProvider& gmw_provider,
                                                                  std::size_t num_simd,
                                                                  std::size_t input_owner)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      num_simd_(num_simd),
      input_owner_(input_owner),
      input_id_(gmw_provider.get_next_input_id(1)),
      output_(std::make_shared<ArithmeticGMWWire<T>>(num_simd)) {}

template <typename T>
void ArithmeticGMWInputGateReceiver<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWInputGateReceiver::evaluate_setup start", gate_id_));
    }
  }

  auto& mbp = gmw_provider_.get_motion_base_provider();
  auto& rng = mbp.get_their_randomness_generator(input_owner_);
  output_->get_share() = rng.GetUnsigned<T>(input_id_, num_simd_);
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWInputGateReceiver::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticGMWInputGateReceiver<T>::evaluate_online() {
  // nothing to do
}

template class ArithmeticGMWInputGateReceiver<std::uint8_t>;
template class ArithmeticGMWInputGateReceiver<std::uint16_t>;
template class ArithmeticGMWInputGateReceiver<std::uint32_t>;
template class ArithmeticGMWInputGateReceiver<std::uint64_t>;

template <typename T>
ArithmeticGMWOutputGate<T>::ArithmeticGMWOutputGate(std::size_t gate_id, GMWProvider& gmw_provider,
                                                    ArithmeticGMWWireP<T>&& input,
                                                    std::size_t output_owner)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      output_owner_(output_owner),
      input_(std::move(input)) {
  std::size_t my_id = gmw_provider_.get_my_id();
  // if (output_owner_ == ALL_PARTIES || output_owner_ == my_id) {
  //   share_futures_ = gmw_provider_.register_for_integer_messages<T>(gate_id_, num_simd_);
  // }
}

template <typename T>
ENCRYPTO::ReusableFiberFuture<std::vector<T>> ArithmeticGMWOutputGate<T>::get_output_future() {
  std::size_t my_id = gmw_provider_.get_my_id();
  if (output_owner_ == ALL_PARTIES || output_owner_ == my_id) {
    return output_promise_.get_future();
  } else {
    throw std::logic_error("not this parties output");
  }
}

template <typename T>
void ArithmeticGMWOutputGate<T>::evaluate_setup() {
  // nothing to do
}

template <typename T>
void ArithmeticGMWOutputGate<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWOutputGate::evaluate_online start", gate_id_));
    }
  }

  // std::size_t my_id = gmw_provider_.get_my_id();
  // ENCRYPTO::BitVector<> my_share;
  // // auto num_bits = count_bits(inputs_);  // TODO: reserve
  // for (const auto& wire : inputs_) {
  //   my_share.Append(wire->get_share());
  // }
  // if (output_owner_ != my_id) {
  //   if (output_owner_ == ALL_PARTIES) {
  //     gmw_provider_.broadcast_bits_message(gate_id_, my_share);
  //   } else {
  //     gmw_provider_.send_bits_message(output_owner_, gate_id_, std::move(my_share));
  //   }
  // }
  // if (output_owner_ == ALL_PARTIES || output_owner_ == my_id) {
  //   std::size_t num_parties = gmw_provider_.get_num_parties();
  //   for (std::size_t party_id = 0; party_id < num_parties; ++party_id) {
  //     if (party_id == my_id) {
  //       continue;
  //     }
  //     const auto other_share = share_futures_[party_id].get();
  //     my_share ^= other_share;
  //   }
  //   // TODO: set_value of output_promise_
  //   std::vector<ENCRYPTO::BitVector<>> outputs;
  //   outputs.reserve(num_wires_);
  //   std::size_t bit_offset = 0;
  //   for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
  //     auto num_simd = inputs_[wire_i]->get_num_simd();
  //     outputs.push_back(my_share.Subset(bit_offset, bit_offset + num_simd));
  //     bit_offset += num_simd;
  //   }
  //   output_promise_.set_value(std::move(outputs));
  // }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: BooleanGMWOutputGate::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticGMWOutputGate<std::uint8_t>;
template class ArithmeticGMWOutputGate<std::uint16_t>;
template class ArithmeticGMWOutputGate<std::uint32_t>;
template class ArithmeticGMWOutputGate<std::uint64_t>;

template <typename T>
ArithmeticGMWNEGGate<T>::ArithmeticGMWNEGGate(std::size_t gate_id, const GMWProvider&,
                                              GMWWireVector&& in)
    : detail::BasicGMWUnaryGate<ArithmeticGMWWire<T>>(gate_id, std::move(in), false) {}

template <typename T>
void ArithmeticGMWNEGGate<T>::evaluate_setup() {
  // nothing to do
}

template <typename T>
void ArithmeticGMWNEGGate<T>::evaluate_online() {
  for (std::size_t wire_i = 0; wire_i < this->num_wires_; ++wire_i) {
    const auto& w_in = this->inputs_[wire_i];
    w_in->wait_online();
    auto& w_o = this->outputs_[wire_i];
    assert(w_o->get_share().size() == w_in->get_num_simd());
    std::transform(std::begin(w_in->get_share()), std::end(w_in->get_share()),
                   std::begin(w_o->get_share()), std::negate{});
    w_o->set_online_ready();
  }
}

template class ArithmeticGMWNEGGate<std::uint8_t>;
template class ArithmeticGMWNEGGate<std::uint16_t>;
template class ArithmeticGMWNEGGate<std::uint32_t>;
template class ArithmeticGMWNEGGate<std::uint64_t>;

template <typename T>
void ArithmeticGMWADDGate<T>::evaluate_setup() {
  // nothing to do
}

template <typename T>
void ArithmeticGMWADDGate<T>::evaluate_online() {
  for (std::size_t wire_i = 0; wire_i < this->num_wires_; ++wire_i) {
    const auto& w_a = this->inputs_a_[wire_i];
    const auto& w_b = this->inputs_b_[wire_i];
    w_a->wait_online();
    w_b->wait_online();
    auto& w_o = this->outputs_[wire_i];
    std::transform(std::begin(w_a->get_share()), std::end(w_a->get_share()),
                   std::begin(w_b->get_share()), std::begin(w_o->get_share()), std::plus{});
    w_o->set_online_ready();
  }
}

template class ArithmeticGMWADDGate<std::uint8_t>;
template class ArithmeticGMWADDGate<std::uint16_t>;
template class ArithmeticGMWADDGate<std::uint32_t>;
template class ArithmeticGMWADDGate<std::uint64_t>;

template <typename T>
ArithmeticGMWMULGate<T>::ArithmeticGMWMULGate(std::size_t gate_id, GMWProvider& gmw_provider,
                                              GMWWireVector&& in_a, GMWWireVector&& in_b)
    : detail::BasicGMWBinaryGate<ArithmeticGMWWire<T>>(gate_id, std::move(in_a), std::move(in_b)),
      gmw_provider_(gmw_provider) {
  // TODO: register MTs
}

template <typename T>
void ArithmeticGMWMULGate<T>::evaluate_setup() {
  // TODO: wait for MTs
}

template <typename T>
void ArithmeticGMWMULGate<T>::evaluate_online() {
  // TODO: compute MUL
}

template class ArithmeticGMWMULGate<std::uint8_t>;
template class ArithmeticGMWMULGate<std::uint16_t>;
template class ArithmeticGMWMULGate<std::uint32_t>;
template class ArithmeticGMWMULGate<std::uint64_t>;

template <typename T>
ArithmeticGMWSQRGate<T>::ArithmeticGMWSQRGate(std::size_t gate_id, GMWProvider& gmw_provider,
                                              GMWWireVector&& in_a, GMWWireVector&& in_b)
    : detail::BasicGMWBinaryGate<ArithmeticGMWWire<T>>(gate_id, std::move(in_a), std::move(in_b)),
      gmw_provider_(gmw_provider) {
  // TODO: register MTs
}

template <typename T>
void ArithmeticGMWSQRGate<T>::evaluate_setup() {
  // TODO: wait for MTs
}

template <typename T>
void ArithmeticGMWSQRGate<T>::evaluate_online() {
  // TODO: compute SQR
}

template class ArithmeticGMWSQRGate<std::uint8_t>;
template class ArithmeticGMWSQRGate<std::uint16_t>;
template class ArithmeticGMWSQRGate<std::uint32_t>;
template class ArithmeticGMWSQRGate<std::uint64_t>;

}  // namespace MOTION::proto::gmw
