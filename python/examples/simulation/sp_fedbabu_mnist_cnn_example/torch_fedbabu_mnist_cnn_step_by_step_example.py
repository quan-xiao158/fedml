# import fedml
# from fedml import FedMLRunner
import os
import sys
path=os.path.abspath('../../..')
sys.path.append(path)
import fedml1 
from fedml1 import FedMLRunner
if __name__ == "__main__":
    # init FedML framework
    args = fedml1.init()

    # init device
    device = fedml1.device.get_device(args)

    # load data
    dataset, output_dim = fedml1.data.load(args)

    # load model
    model = fedml1.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
