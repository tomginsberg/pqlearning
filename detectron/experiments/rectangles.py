from datasets.pqmodule import PQModule
from datasets.rectangles import Rectangles
from detectron.utils import rejectron_trainer, train_rejectors
from modelling import pretrained

if __name__ == '__main__':
    # define h
    h = pretrained.mlp_large_trained_on_rectangles()

    # define a PQDataModule
    pq = PQModule(p=Rectangles(split='train'),
                  q=Rectangles(split='test'),
                  p_prime=Rectangles(split='val'),
                  batch_size=200, num_workers=0)

    # with open('synthetic_data/rectangles/test_tmp.json', 'w') as f:
    #     json.dump([x[0].tolist() for x in pq.q], f)

    # get a default rejectron trainer
    trainer = rejectron_trainer(save_directory='rectangles/rejectron', max_epochs=50, dryrun=True, run_name='')

    # train !
    train_rejectors(pq=pq, h=h, trainer=trainer, num_rejectors=16)
