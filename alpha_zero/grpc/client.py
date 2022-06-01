# Copyright 2019 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An example of multiprocessing concurrency with gRPC."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import atexit
import logging
import multiprocessing
import operator
import sys

import grpc
import service_pb2
import service_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:53445') as channel:
        stub = service_pb2_grpc.CentralServiceStub(channel)
        response = stub.inference(service_pb2.InferenceResponse(client_id='123'))
    print(response)


if __name__ == '__main__':
    logging.basicConfig()
    run()
