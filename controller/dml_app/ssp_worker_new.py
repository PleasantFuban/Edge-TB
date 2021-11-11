import time
from typing import Any
from flask import request
import dml_utils
import worker_utils
from role_base import Role


class SspWorker(Role):

    current_clock: int
    stop: bool
    local_cache: Any
    server_ok: bool
    server_weights: Any

    def _hook_handle_structure(self) -> None:
        self.current_clock = 0
        self.stop = False

    def load_actions(self) -> None:
        super().load_actions()

        @self.app.route('/start', methods=['POST'])
        def route_start():
            initial_weights = dml_utils.parse_arrays(request.files.get('weights'))
            self.executor.submit(self.on_route_start, initial_weights).add_done_callback(self.on_route_start_cb)
            return ''

        @self.app.route('/latest', methods=['POST'])
        def route_latest():
            self.server_weights = dml_utils.parse_arrays(request.files.get('weights'))
            self.server_ok = True
            return ''

        @self.app.route('/stop', methods=['POST'])
        def route_stop():
            # finish the current iteration and stop
            self.stop = True
            return ''

    def on_route_start(self, initial_weights):
        dml_utils.assign_weights(self.nn.model, initial_weights)

    def on_route_start_cb(self, _):
        self.executor.submit(self.train_loop)

    def read_weights_data(self, force_server_access: bool):
        staleness = self.conf['staleness']
        if force_server_access or (self.current_clock % staleness == 0):
            # access the master
            address = self.conf['connect'][self.conf['master_node']]
            path = '/data?node=' + self.node_name
            self.server_ok = False
            worker_utils.send_data('GET', path, address)
            while not self.server_ok:
                time.sleep(0.1)
            return self.server_weights
        else:
            # consider the local cache
            return self.local_cache

    def update(self, update, push_now=True):
        new_data = self.do_update(update)
        # update local cache
        self.local_cache = new_data
        if push_now:
            # push the update to the server
            dml_utils.send_arrays_single(new_data, '/clock', self.conf['master_node'], self.conf['connect'],
                                         dtype='weights', source=self.node_name)

    # noinspection PyMethodMayBeStatic
    def do_update(self, update):
        return update

    def train_loop(self):
        address = self.conf['connect'][self.conf['master_node']]
        print(f'Start train loop, master address is {address}')
        while not self.stop:
            read_from_server: bool = False

            # read the server clock first
            clock = worker_utils.send_data('GET', '/clock', address)
            clock = int(clock)
            if (self.current_clock - clock) > self.conf['staleness']:
                read_from_server = True

            latest_weights = self.read_weights_data(read_from_server)
            while (self.current_clock - clock) > self.conf['staleness']:
                time.sleep(0.1)     # TODO can it be dynamic?
                clock = int(worker_utils.send_data('GET', '/clock', address))
            dml_utils.assign_weights(self.nn.model, latest_weights)

            # now meet the conditions, do the training job
            _, update_data = self.do_train()
            self.update(update_data)
            worker_utils.send_data('POST', f'/stop?node={self.node_name}', address)
        return True

    def do_train(self, clock_inc: bool = True):
        begin_time = time.time()

        loss_list = dml_utils.train(self.nn.model, self.train_data, self.train_labels, 1, self.conf['batch_size'],
                                    self.conf['train_len'])
        last_epoch_loss = loss_list[-1]
        dml_utils.log_loss(last_epoch_loss, self.current_clock)

        _, acc = dml_utils.test(self.nn.model, self.test_data, self.test_labels)
        dml_utils.log_acc(acc, self.current_clock)

        if clock_inc:
            self.current_clock += 1
        epoch_time = time.time() - begin_time
        worker_utils.log('Last epoch time: %s' % epoch_time)
        return epoch_time, self.nn.model.get_weights()


if __name__ == '__main__':
    import argparse
    import os
    from nns.nn_fashion_mnist import nn

    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--host', type=str, help='App host', default='0.0.0.0')
    parser.add_argument('-p', '--port', type=int, help='DML port', default=3333)
    args = parser.parse_args()

    dirname = os.path.abspath(os.path.dirname(__file__))
    role = SspWorker(dirname, nn,
                     os.path.join(dirname, '../dataset/FASHION_MNIST/train_data'),
                     os.path.join(dirname, '../dataset/FASHION_MNIST/test_data'),
                     os.path.abspath(os.path.join(dirname, '../dml_file/log/')))
    role.run(host=args.host, port=args.port)
