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

#include <cstdint>
#include <string>

#include "milvus/Export.h"
#include "milvus/types/ReplicateConfiguration.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::DumpMessages() to dump messages from a Pulsar channel.
 */
class MILVUS_SDK_API DumpMessagesRequest {
 public:
    /**
     * @brief Get the Pulsar channel name.
     * @return the p channel.
     */
    const std::string&
    PChannel() const;

    /**
     * @brief Set the Pulsar channel name.
     * @param [in] pchannel the pchannel.
     */
    void
    SetPChannel(const std::string& pchannel);

    /**
     * @brief Set the Pulsar channel name.
     * @param [in] pchannel the pchannel.
     */
    DumpMessagesRequest&
    WithPChannel(const std::string& pchannel);

    /**
     * @brief Get the start message identifier.
     * @return the start message ID.
     */
    const ReplicateMessageID&
    StartMessageID() const;

    /**
     * @brief Set the start message identifier.
     * @param [in] start_message_id the start message ID.
     */
    void
    SetStartMessageID(ReplicateMessageID&& start_message_id);

    /**
     * @brief Set the start message identifier.
     * @param [in] start_message_id the start message ID.
     */
    DumpMessagesRequest&
    WithStartMessageID(ReplicateMessageID&& start_message_id);

    /**
     * @brief Get the start time tick.
     * @return the start time tick.
     */
    uint64_t
    StartTimeTick() const;

    /**
     * @brief Set the start time tick.
     * @param [in] start_timetick the start timetick.
     */
    void
    SetStartTimeTick(uint64_t start_timetick);

    /**
     * @brief Set the start time tick.
     * @param [in] start_timetick the start timetick.
     */
    DumpMessagesRequest&
    WithStartTimeTick(uint64_t start_timetick);

    /**
     * @brief Get the end time tick.
     * @return the end time tick.
     */
    uint64_t
    EndTimeTick() const;

    /**
     * @brief Set the end time tick.
     * @param [in] end_timetick the end timetick.
     */
    void
    SetEndTimeTick(uint64_t end_timetick);

    /**
     * @brief Set the end time tick.
     * @param [in] end_timetick the end timetick.
     */
    DumpMessagesRequest&
    WithEndTimeTick(uint64_t end_timetick);

 private:
    std::string pchannel_;
    ReplicateMessageID start_message_id_;
    uint64_t start_timetick_{0};
    uint64_t end_timetick_{0};
};

}  // namespace milvus
