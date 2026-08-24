// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

#include "milvus/ClientRequestContext.h"

#include <algorithm>
#include <iomanip>
#include <random>
#include <sstream>

namespace milvus {
namespace {
thread_local std::string request_id;
}

void
ClientRequestContext::Set(const std::string& value) {
    request_id = value;
}

const std::string&
ClientRequestContext::Get() {
    return request_id;
}

void
ClientRequestContext::Clear() {
    request_id.clear();
}

std::string
ClientRequestContext::NewRequestId() {
    std::random_device device;
    std::mt19937_64 generator(device());
    std::uniform_int_distribution<uint64_t> distribution;
    uint64_t high = 0;
    uint64_t low = 0;
    do {
        high = distribution(generator);
        low = distribution(generator);
    } while (high == 0 && low == 0);
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << high << std::setw(16) << low;
    return stream.str();
}

bool
ClientRequestContext::IsValid(const std::string& value) {
    return value.size() == 32 && value != std::string(32, '0') &&
           std::all_of(value.begin(), value.end(), [](char character) {
               return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
           });
}

ScopedClientRequestId::ScopedClientRequestId(const std::string& value) : previous_(ClientRequestContext::Get()) {
    ClientRequestContext::Set(value);
}

ScopedClientRequestId::~ScopedClientRequestId() {
    ClientRequestContext::Set(previous_);
}

}  // namespace milvus
