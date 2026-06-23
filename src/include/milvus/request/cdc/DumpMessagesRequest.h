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

class MILVUS_SDK_API DumpMessagesRequest {
 public:
    const std::string&
    PChannel() const;

    void
    SetPChannel(const std::string& pchannel);

    DumpMessagesRequest&
    WithPChannel(const std::string& pchannel);

    const ReplicateMessageID&
    StartMessageID() const;

    void
    SetStartMessageID(ReplicateMessageID&& start_message_id);

    DumpMessagesRequest&
    WithStartMessageID(ReplicateMessageID&& start_message_id);

    uint64_t
    StartTimeTick() const;

    void
    SetStartTimeTick(uint64_t start_timetick);

    DumpMessagesRequest&
    WithStartTimeTick(uint64_t start_timetick);

    uint64_t
    EndTimeTick() const;

    void
    SetEndTimeTick(uint64_t end_timetick);

    DumpMessagesRequest&
    WithEndTimeTick(uint64_t end_timetick);

 private:
    std::string pchannel_;
    ReplicateMessageID start_message_id_;
    uint64_t start_timetick_{0};
    uint64_t end_timetick_{0};
};

}  // namespace milvus
