from typing import Any
from flask import request
import dml_utils
import worker_utils
from role_base import Role


class PrSgdWorker(Role):

    current_epoch: int
    averaged_weights: Any
    averaged_weights_ready: bool

    def _hook_handle_structure(self) -> None:
        self.current_epoch = 0

    def load_actions(self) -> None:
        super(PrSgdWorker, self).load_actions()

        @self.app.route('/epoch', methods=['GET'])
        def route_epoch_get():
            self.executor.submit(self.on_route_epoch_get)
            return ''

        @self.app.route('/epoch', methods=['POST'])
        def route_epoch_post():
            epoch_initial_weights = dml_utils.parse_arrays(request.files.get('weights'))
            self.on_route_epoch_post(epoch_initial_weights)
            return ''

    def on_route_epoch_get(self):
        dml_utils.send_arrays_single(self.nn.model.get_weights(), '/epoch',
                                     self.conf['master_node'], self.conf['connect'], source=self.node_name)

    def on_route_epoch_post(self, weights):
        self.current_epoch += 1
        dml_utils.assign_weights(self.nn.model, weights)
        self.executor.submit(self.train_epoch)

    def train_epoch(self):
        try:
            for j in range(0, self.conf['iterations']):
                # just perform local SGD; we split it into two steps for demonstration
                gradients = dml_utils.compute_gradients(self.nn.model, self.train_data, self.train_labels)
                dml_utils.apply_gradients(self.nn.model, gradients)

            _, acc = dml_utils.test(self.nn.model, self.test_data, self.test_labels)
            msg = dml_utils.log_acc(acc, self.current_epoch)
            worker_utils.send_print(self.ctl_addr, self.node_name + ': ' + msg)

            worker_utils.send_data('POST', f'/finish?node={self.node_name}', self.conf['connect'][self.conf['master_node']])
        except Exception as e:
            print(e)


if __name__ == '__main__':
    import argparse
    import os
    from nns.nn_fashion_mnist import nn

    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--host', type=str, help='App host', default='0.0.0.0')
    parser.add_argument('-p', '--port', type=int, help='DML port', default=3333)
    args = parser.parse_args()

    dirname = os.path.abspath(os.path.dirname(__file__))
    role = PrSgdWorker(dirname, nn,
                       os.path.join(dirname, '../dataset/FASHION_MNIST/train_data'),
                       os.path.join(dirname, '../dataset/FASHION_MNIST/test_data'),
                       os.path.abspath(os.path.join(dirname, '../dml_file/log/')))
    role.run(host=args.host, port=args.port)
