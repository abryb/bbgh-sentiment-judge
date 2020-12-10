import code
import readline
import rlcompleter


def interactive_console(_globals, _locals):
    """
    Opens interactive console with current execution state.
    Call it with: `console.open(globals(), locals())`
    """
    context = _globals.copy()
    context.update(_locals)
    readline.set_completer(rlcompleter.Completer(context).complete)
    readline.parse_and_bind("tab: complete")
    shell = code.InteractiveConsole(context)
    shell.interact()
