"""Run a training job on Cloud ML Engine for a given use case.
Usage:
  trainer.task [-v]

Options:
  -h --help     Show this screen.
  -v            Debug output
"""
from docopt import docopt

# import trainer.example  as model # Your example.py file.
import trainer.model as model
from trainer.utils import interactive_console

if __name__ == '__main__':
    arguments = docopt(__doc__)
    if arguments['-v']:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    # # Assign model variables to commandline arguments
    # model.DATA_FILE = arguments['<data_file>']
    # model.WORD_EMBEDDINGS_FILE = arguments['<embeddings_file>']
    # model.OUTPUT_DIR = arguments['<outdir>']
    # # Run the training job
    # model.train_and_evaluate()
    model.train()

