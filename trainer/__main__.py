"""
Usage:
    trainer prepare_word2vec [options]
    trainer download_mentions [options]
    trainer train [--epochs=<epochs> --maxlen=<maxlen> --batch-size=<batch_size> --train-on-all --val --save] [options]
    trainer evaluate [options]
    trainer predict [--all] [options]
    trainer publish [--all] [options]
    trainer show_prediction <mention_id> [options]
    trainer run [--train]
    trainer inspect

Options:
    -v                      Debug output.
    -h --help               Show this screen.
    --backend-host=<value>  Backend host (default: "http://vps-3c1c3381.vps.ovh.net:8080")
    --cache-dir=<value>     Cache and data dir
    --models-dir=<value>    Models dir
"""
from docopt import docopt
from trainer.worker import Worker
import readline
import code
import rlcompleter


def int_or_none(value):
    return int(value) if value is not None else None


if __name__ == '__main__':
    arguments = docopt(__doc__)

    if arguments['-v']:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    worker = Worker(
        backend_host=arguments['--backend-host'],
        cache_dir=arguments['--cache-dir'],
        models_dir=arguments['--models-dir']
    )

    if arguments['prepare_word2vec']:
        worker.prepare_word2vec()

    elif arguments['download_mentions']:
        worker.download_mentions()

    elif arguments['train']:
        worker.train(
            epochs=int_or_none(arguments['--epochs']),
            val=arguments['--val'],
            train_on_all=arguments['--train-on-all'],
            save=arguments['--save'],
            maxlen=int_or_none(arguments['--maxlen']),
            batch_size=int_or_none(arguments['--batch-size'])
        )
    elif arguments['evaluate']:
        worker.evaluate()

    elif arguments['predict']:
        worker.predict(
            only_not_checked=not arguments['--all']
        )

    elif arguments['publish']:
        worker.publish(
            only_unpublished=not arguments['--all']
        )
    elif arguments['show_prediction']:
        worker.show_prediction(int_or_none(arguments['<mention_id>']))

    elif arguments['run']:
        worker.run(
            train=arguments['--train']
        )
    elif arguments['inspect']:
        context = globals().copy()
        context.update(locals())
        readline.set_completer(rlcompleter.Completer(context).complete)
        readline.parse_and_bind("tab: complete")
        shell = code.InteractiveConsole(context)
        shell.interact()
