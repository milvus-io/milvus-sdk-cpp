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

#include "milvus/request/cdc/DumpMessagesRequest.h"

namespace milvus {

const std::string&
DumpMessagesRequest::PChannel() const {
    return pchannel_;
}

void
DumpMessagesRequest::SetPChannel(const std::string& pchannel) {
    pchannel_ = pchannel;
}

DumpMessagesRequest&
DumpMessagesRequest::WithPChannel(const std::string& pchannel) {
    SetPChannel(pchannel);
    return *this;
}

const ReplicateMessageID&
DumpMessagesRequest::StartMessageID() const {
    return start_message_id_;
}

void
DumpMessagesRequest::SetStartMessageID(ReplicateMessageID&& start_message_id) {
    start_message_id_ = std::move(start_message_id);
}

DumpMessagesRequest&
DumpMessagesRequest::WithStartMessageID(ReplicateMessageID&& start_message_id) {
    SetStartMessageID(std::move(start_message_id));
    return *this;
}

uint64_t
DumpMessagesRequest::StartTimeTick() const {
    return start_timetick_;
}

void
DumpMessagesRequest::SetStartTimeTick(uint64_t start_timetick) {
    start_timetick_ = start_timetick;
}

DumpMessagesRequest&
DumpMessagesRequest::WithStartTimeTick(uint64_t start_timetick) {
    SetStartTimeTick(start_timetick);
    return *this;
}

uint64_t
DumpMessagesRequest::EndTimeTick() const {
    return end_timetick_;
}

void
DumpMessagesRequest::SetEndTimeTick(uint64_t end_timetick) {
    end_timetick_ = end_timetick;
}

DumpMessagesRequest&
DumpMessagesRequest::WithEndTimeTick(uint64_t end_timetick) {
    SetEndTimeTick(end_timetick);
    return *this;
}

}  // namespace milvus
