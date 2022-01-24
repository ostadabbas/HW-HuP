from utils import TrainOptions
from train import Trainer
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    trainer = Trainer(options)
    trainer.train()
    trainer.summary_writer.close()