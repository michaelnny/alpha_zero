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
"""An example of multiprocess concurrency with gRPC."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from concurrent import futures
import contextlib
import datetime
import logging
import multiprocessing
import socket
import sys
import time

import grpc
import service_pb2
import service_pb2_grpc

_LOGGER = logging.getLogger(__name__)

_ONE_DAY = datetime.timedelta(days=1)
_PROCESS_COUNT = 2  # multiprocessing.cpu_count()
_THREAD_CONCURRENCY = _PROCESS_COUNT


class Server(service_pb2_grpc.CentralServiceServicer):
    def inference(self, request, context):
        print(f'Client id: {request.client_id}')
        # context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        # context.set_details('Method not implemented!')
        return service_pb2.InferenceResponse(client_id=request.client_id)

    def update_weights(self, request, context):
        print(f'Client id: {request.client_id}')
        # context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        # context.set_details('Method not implemented!')
        return service_pb2.UpdateWeightsResponse(client_id=request.client_id)


def _wait_forever(server):
    try:
        while True:
            time.sleep(_ONE_DAY.total_seconds() * 30)
    except KeyboardInterrupt:
        server.stop(None)


def _run_server(bind_address):
    """Start a server in a subprocess."""
    _LOGGER.info('Starting new server.')
    options = (('grpc.so_reuseport', 1),)

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=_THREAD_CONCURRENCY,
        ),
        options=options,
    )
    service_pb2_grpc.add_CentralServiceServicer_to_server(Server(), server)
    server.add_insecure_port(bind_address)
    server.start()
    _wait_forever(server)


@contextlib.contextmanager
def _reserve_port():
    """Find and reserve a port for all subprocesses to use."""
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0:
        raise RuntimeError('Failed to set SO_REUSEPORT.')
    sock.bind(('', 0))
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def main():
    with _reserve_port() as port:
        bind_address = 'localhost:{}'.format(port)
        _LOGGER.info("Binding to '%s'", bind_address)
        sys.stdout.flush()
        workers = []
        for _ in range(_PROCESS_COUNT):
            # NOTE: It is imperative that the worker subprocesses be forked before
            # any gRPC servers start up. See
            # https://github.com/grpc/grpc/issues/16001 for more details.
            worker = multiprocessing.Process(target=_run_server, args=(bind_address,))
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()


if __name__ == '__main__':
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[PID %(process)d] %(message)s')
    handler.setFormatter(formatter)
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    main()
