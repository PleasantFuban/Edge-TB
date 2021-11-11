from typing import Any
from flask import request
import dml_utils
import worker_utils
from role_base import Role


class PrSgdMaster(Role):

    epoch_ready_nodes: set
    received_weights: Any
    current_epoch: int
    epoch_finished_nodes: set

    def _hook_handle_structure(self) -> None:
        self.workers = self.conf['child_node']      # just for convenience

    def load_actions(self) -> None:
        super(PrSgdMaster, self).load_actions()

        @self.app.route('/start', methods=['GET'])
        def route_start():
            self.executor.submit(self.on_route_start)
            return ''

        # inter-node communication is only needed to calculate the initial point at the beginning of each epoch
        @self.app.route('/epoch', methods=['POST'])
        def route_epoch():
            weights = dml_utils.parse_arrays(request.files.get('weights'))
            node_name = request.form.get('source')
            self.executor.submit(self.on_route_epoch, weights, node_name)
            return ''

        @self.app.route('/finish', methods=['POST'])
        def route_finish():
            node_name = request.args.get('node')
            self.executor.submit(self.on_route_finish_by_node, node_name)
            return ''

    def on_route_start(self):
        self.epoch_ready_nodes = set()
        self.epoch_finished_nodes = set()
        self.received_weights = []
        self.current_epoch = 1

        dml_utils.send_arrays(self.nn.model.get_weights(), '/epoch', self.workers, self.conf['connect'])
        worker_utils.send_print(self.ctl_addr, 'start PR-SGD')

    def start_new_epoch(self):
        self.current_epoch += 1
        for node in self.workers:
            worker_utils.send_data('GET', '/epoch', self.conf['connect'][node])

    def on_route_epoch(self, weights, node_name):
        try:
            self.weights_lock.acquire()
            self.epoch_ready_nodes.add(node_name)
            dml_utils.store_weights(self.received_weights, weights, len(self.epoch_ready_nodes))
            self.weights_lock.release()

            if len(self.epoch_ready_nodes) == len(self.workers):
                averaged_weights = dml_utils.avg_weights(self.received_weights, len(self.workers))
                dml_utils.assign_weights(self.nn.model, averaged_weights)

                _, acc = dml_utils.test(self.nn.model, self.test_data, self.test_labels)
                msg = dml_utils.log_acc(acc, self.current_epoch)
                worker_utils.send_print(self.ctl_addr, self.node_name + ': ' + msg)

                self.epoch_ready_nodes = set()
                self.received_weights.clear()

                # send the new averaged weights back to the workers; continue training
                dml_utils.send_arrays(averaged_weights, '/epoch', self.workers, self.conf['connect'])
        except Exception as e:
            print(e)

    def on_route_finish_by_node(self, node_name):
        self.epoch_finished_nodes.add(node_name)
        if len(self.epoch_finished_nodes) == len(self.workers):
            if self.current_epoch == self.conf['epochs']:
                worker_utils.log('>>>>> training ended <<<<<')
                worker_utils.send_data('GET', '/finish', self.ctl_addr)
            else:
                self.epoch_finished_nodes.clear()
                self.start_new_epoch()


if __name__ == '__main__':
    import argparse
    import os
    from nns.nn_fashion_mnist import nn

    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--host', type=str, help='App host', default='0.0.0.0')
    parser.add_argument('-p', '--port', type=int, help='DML port', default=3333)
    args = parser.parse_args()

    dirname = os.path.abspath(os.path.dirname(__file__))
    role = PrSgdMaster(dirname, nn,
                       os.path.join(dirname, '../dataset/FASHION_MNIST/train_data'),
                       os.path.join(dirname, '../dataset/FASHION_MNIST/test_data'),
                       os.path.abspath(os.path.join(dirname, '../dml_file/log/')))
    role.run(host=args.host, port=args.port)
