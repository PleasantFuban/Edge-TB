import numpy as np
from flask import request
import dml_utils
import worker_utils
from role_base import Role


class RaPeer(Role):
    current_round: int
    current_step: int
    next_partition: int
    left_node_name: str
    temp_weights: list
    temp_weights_flat: list     # a flattened view of temp_weights

    def _hook_handle_structure(self) -> None:
        self.current_round = 0
        self.current_step = 0
        conns = list(self.conf['connect'].keys())
        self.left_node_name = conns[0]

    def load_actions(self) -> None:
        super(RaPeer, self).load_actions()

        @self.app.route('/train', methods=['GET'])
        def route_train():
            self.current_round += 1
            print(f'GET at /train, round {self.current_round}')
            self.executor.submit(self.on_route_train)
            return ''

        @self.app.route('/send', methods=['GET'])
        def route_send_weights():
            stage = request.args.get('stage')
            step = request.args.get('step')
            path = '/add' if stage == 'reduce' else '/update'
            self.executor.submit(self.send_weights, path, int(step))
            return ''

        @self.app.route('/add', methods=['POST'])
        def route_add():
            partition = request.args.get('partition')
            print(f'POST at /add, round {self.current_round}, partition {partition}')
            received_weights = dml_utils.parse_arrays(request.files.get('weights'))
            self.executor.submit(self.on_route_add, int(partition), received_weights)
            return ''

        @self.app.route('/update', methods=['POST'])
        def route_update():
            partition = request.args.get('partition')
            print(f'POST at /update, round {self.current_round}, partition {partition}')
            received_weights = dml_utils.parse_arrays(request.files.get('weights'))
            self.executor.submit(self.on_route_update, int(partition), received_weights)
            return ''

    def on_route_train(self):
        self.weights_lock.acquire()

        loss_list = dml_utils.train(self.nn.model, self.train_data, self.train_labels,
                                    self.conf['epoch'], self.conf['batch_size'], self.conf['train_len'])
        last_epoch_loss = loss_list[-1]
        msg = dml_utils.log_loss(last_epoch_loss, self.current_round)
        worker_utils.send_print(self.ctl_addr, self.node_name + ': ' + msg)

        worker_utils.send_data('POST', '/ok?name=' + self.node_name, self.ctl_addr)

        self.temp_weights = self.nn.model.get_weights()
        self.temp_weights_flat = [layer_w.ravel() for layer_w in self.temp_weights]
        self.next_partition = self.conf['pos']
        if self.conf['pos'] == 0:
            print(self.temp_weights_flat[0][0])
        self.weights_lock.release()

    def send_weights(self, path: str, step: int):
        # pos starts from 0
        partition = self.next_partition
        print(f'Node {self.node_name} - sending weights in round {self.current_round} to {path}, '
              f'step {step}, pos {self.conf["pos"]}, partition {partition}')

        weights_to_send = list()
        for i in range(len(self.temp_weights_flat)):
            range_start = len(self.temp_weights_flat[i]) // self.conf['ring_size'] * partition
            range_end = min(range_start + len(self.temp_weights_flat[i]) // self.conf['ring_size'],
                            len(self.temp_weights_flat[i]))
            weights_to_send.append(self.temp_weights_flat[i][range_start:range_end])

        path = f'{path}?partition={partition}'
        dml_utils.send_arrays(weights_to_send, path, [self.left_node_name], self.conf['connect'])
        self.next_partition = (self.next_partition + 1) % self.conf['ring_size']

    def on_route_add(self, partition, weights):
        for i in range(len(self.temp_weights_flat)):
            range_start = len(self.temp_weights_flat[i]) // self.conf['ring_size'] * partition
            range_end = min(range_start + len(self.temp_weights_flat[i]) // self.conf['ring_size'],
                            len(self.temp_weights_flat[i]))
            new_weights = np.add(self.temp_weights_flat[i][range_start:range_end], weights[i])
            self.temp_weights_flat[i][range_start:range_end] = new_weights

        worker_utils.send_data('POST', '/ok?name=' + self.node_name, self.ctl_addr)

    def on_route_update(self, partition, weights):
        self.current_step += 1

        for i in range(len(self.temp_weights_flat)):
            range_start = len(self.temp_weights_flat[i]) // self.conf['ring_size'] * partition
            range_end = min(range_start + len(self.temp_weights_flat[i]) // self.conf['ring_size'],
                            len(self.temp_weights_flat[i]))
            self.temp_weights_flat[i][range_start:range_end] = weights[i]

        if self.current_step == self.conf['ring_size'] - 1:
            self.current_step = 0
            self.weights_lock.acquire()
            weights = dml_utils.avg_weights([np.array(self.temp_weights)], self.conf['ring_size'])
            dml_utils.assign_weights(self.nn.model, weights)

            _, acc = dml_utils.test_on_batch(self.nn.model, self.test_data, self.test_labels, self.conf['batch_size'])
            msg = dml_utils.log_acc(acc, self.current_round)
            worker_utils.send_print(self.ctl_addr, self.node_name + ': ' + msg)
            self.weights_lock.release()
            print('Assign weights, round: %s, sync: %s' % (self.current_round, self.conf['sync']))
            if self.current_round == self.conf['sync']:
                worker_utils.send_data('POST', 'finish?name=' + self.node_name, self.ctl_addr)
                return

        worker_utils.send_data('POST', '/ok?name=' + self.node_name, self.ctl_addr)


if __name__ == '__main__':
    import argparse
    import os
    from nns.nn_fashion_mnist import nn

    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--host', type=str, help='App host', default='0.0.0.0')
    parser.add_argument('-p', '--port', type=int, help='DML port', default=3333)
    args = parser.parse_args()

    dirname = os.path.abspath(os.path.dirname(__file__))
    role = RaPeer(dirname, nn,
                  os.path.join(dirname, '../dataset/FASHION_MNIST/train_data'),
                  os.path.join(dirname, '../dataset/FASHION_MNIST/test_data'),
                  os.path.abspath(os.path.join(dirname, '../dml_file/log/')))
    role.run(host=args.host, port=args.port)
