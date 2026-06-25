// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "milvus/response/cdc/DumpMessagesTypes.h"

#include <utility>

namespace milvus {

const ReplicateMessageID&
DumpedMessage::MessageID() const {
    return message_id_;
}

void
DumpedMessage::SetMessageID(ReplicateMessageID&& message_id) {
    message_id_ = std::move(message_id);
}

DumpedMessage&
DumpedMessage::WithMessageID(ReplicateMessageID&& message_id) {
    SetMessageID(std::move(message_id));
    return *this;
}

const std::string&
DumpedMessage::Payload() const {
    return payload_;
}

void
DumpedMessage::SetPayload(const std::string& payload) {
    payload_ = payload;
}

DumpedMessage&
DumpedMessage::WithPayload(const std::string& payload) {
    SetPayload(payload);
    return *this;
}

const std::unordered_map<std::string, std::string>&
DumpedMessage::Properties() const {
    return properties_;
}

void
DumpedMessage::SetProperties(std::unordered_map<std::string, std::string>&& properties) {
    properties_ = std::move(properties);
}

DumpedMessage&
DumpedMessage::WithProperties(std::unordered_map<std::string, std::string>&& properties) {
    SetProperties(std::move(properties));
    return *this;
}

DumpedMessage&
DumpedMessage::AddProperty(const std::string& key, const std::string& value) {
    properties_[key] = value;
    return *this;
}

}  // namespace milvus
