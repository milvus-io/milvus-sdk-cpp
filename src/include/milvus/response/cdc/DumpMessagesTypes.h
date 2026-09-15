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

#pragma once

#include <string>
#include <unordered_map>

#include "milvus/Export.h"
#include "milvus/types/ReplicateConfiguration.h"

namespace milvus {

/**
 * @brief A message dumped from a Pulsar channel by MilvusClientV2::DumpMessages().
 */
class MILVUS_SDK_API DumpedMessage {
    /**
     * @brief Get the message identifier.
     * @return the message ID.
     */
 public:
    const ReplicateMessageID&
    MessageID() const;

    /**
     * @brief Set the message identifier.
     *
     * @param [in] message_id
     */
    void
    SetMessageID(ReplicateMessageID&& message_id);

    /**
     * @brief Set the message identifier.
     *
     * @param [in] message_id
     */
    DumpedMessage&
    WithMessageID(ReplicateMessageID&& message_id);

    /**
     * @brief Get the message payload.
     * @return the payload.
     */
    const std::string&
    Payload() const;

    /**
     * @brief Set the message payload.
     *
     * @param [in] payload
     */
    void
    SetPayload(const std::string& payload);

    /**
     * @brief Set the message payload.
     *
     * @param [in] payload
     */
    DumpedMessage&
    WithPayload(const std::string& payload);

    /**
     * @brief Get the message properties.
     * @return the properties.
     */
    const std::unordered_map<std::string, std::string>&
    Properties() const;

    /**
     * @brief Set the message properties.
     *
     * @param [in] string, properties
     */
    void
    SetProperties(std::unordered_map<std::string, std::string>&& properties);

    /**
     * @brief Set the message properties.
     *
     * @param [in] string, properties
     */
    DumpedMessage&
    WithProperties(std::unordered_map<std::string, std::string>&& properties);

    /**
     * @brief Add a message property.
     *
     * @param [in] key, value
     */
    DumpedMessage&
    AddProperty(const std::string& key, const std::string& value);

 private:
    ReplicateMessageID message_id_;
    std::string payload_;
    std::unordered_map<std::string, std::string> properties_;
};

}  // namespace milvus
