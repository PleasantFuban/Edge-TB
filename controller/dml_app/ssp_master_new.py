import time
from flask import request
import numpy as np
import dml_utils
import worker_utils
from role_base import Role


class SspMaster(Role):
    initial_weights: list
    workers: list

    current_clocks: dict
    """ Current clock of the worker nodes. """

    server_clock: int
    """ The minimum worker clock. """

    stopped_nodes: set

    def _hook_handle_structure(self) -> None:
        self.workers = self.conf['child_node']      # just for convenience

    def load_actions(self) -> None:
        super().load_actions()

        @self.app.route('/start', methods=['GET'])
        def route_start():
            worker_utils.send_print(self.ctl_addr, 'Master: Testing on initial model weights')
            _, initial_acc = dml_utils.test(self.nn.model, self.test_data, self.test_labels)
            msg = dml_utils.log_acc(initial_acc, 0)
            worker_utils.send_print(self.ctl_addr, self.node_name + ': ' + msg)
            self.executor.submit(self.on_route_start)
            return ''

        # This interface need not be changed if the mechanism is to be changed
        @self.app.route('/clock', methods=['POST'])
        def route_clock_post():
            weights = dml_utils.parse_arrays(request.files.get('weights'))
            self.executor.submit(self.on_route_clock, weights, request.args.get('node'))
            return ''

        @self.app.route('/clock', methods=['GET'])
        def route_clock_get():
            return self.server_clock

        @self.app.route('/data', methods=['GET'])
        def route_data_get():
            node = request.args.get('node')
            self.executor.submit(self.on_route_data_get, node)
            return ''

        @self.app.route('/stop', methods=['POST'])
        def route_stop():
            node = request.args.get('node')
            self.stopped_nodes.add(node)
            if len(self.stopped_nodes) == len(self.workers):
                worker_utils.send_data('GET', '/finish', self.ctl_addr)
            return ''

    def on_route_start(self):
        self.current_clocks = {}
        self.server_clock = 0
        self.initial_weights = self.nn.model.get_weights()
        for node in self.workers:
            self.current_clocks[node] = 0
            dml_utils.send_arrays(self.initial_weights, '/start', self.workers, self.conf['connect'])
        worker_utils.send_print(self.ctl_addr, 'start SSP')

    def on_route_clock(self, received_weights, node):
        self.weights_lock.acquire()

        # for custom data transfer and processing >>>>>
        # simply average with the current weights
        new_weights = np.add(self.nn.model.get_weights(), received_weights) / 2
        dml_utils.assign_weights(self.nn.model, new_weights)
        # <<<<< for custom data transfer and processing

        self.current_clocks[node] += 1

        msg = f'Clock from {node} ({self.current_clocks[node]}) at {time.asctime(time.localtime(time.time()))}'
        worker_utils.log(msg)

        if min(list(self.current_clocks.values())) > self.server_clock:
            self.server_clock += 1
            _, acc = dml_utils.test(self.nn.model, self.test_data, self.test_labels)
            msg = dml_utils.log_acc(acc, self.server_clock)
            worker_utils.send_print(self.ctl_addr, self.node_name + ': ' + msg)

            if self.server_clock >= self.conf['duration']:
                self.stopped_nodes = set()
                # send a stop signal to all nodes; then wait for all nodes to finish the current iteration
                for node in self.workers:
                    worker_utils.send_data('POST', '/stop', self.conf['connect'][node])

        self.weights_lock.release()

    def on_route_data_get(self, node):
        self.weights_lock.acquire()
        weights = self.nn.model.get_weights()
        self.weights_lock.release()

        dml_utils.send_arrays_single(weights, '/latest', node, self.conf['connect'])


if __name__ == '__main__':
    import argparse
    import os
    from nns.nn_fashion_mnist import nn

    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--host', type=str, help='App host', default='0.0.0.0')
    parser.add_argument('-p', '--port', type=int, help='DML port', default=3333)
    args = parser.parse_args()

    dirname = os.path.abspath(os.path.dirname(__file__))
    role = SspMaster(dirname, nn,
                     os.path.join(dirname, '../dataset/FASHION_MNIST/train_data'),
                     os.path.join(dirname, '../dataset/FASHION_MNIST/test_data'),
                     os.path.abspath(os.path.join(dirname, '../dml_file/log/')))
    role.run(host=args.host, port=args.port)
