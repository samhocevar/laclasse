import click

from . import arith
from . import dumps
from . import render
from . import sources

class MyGroup(click.Group):
    ''' Override the --help flag everywhere to allow -h as a shortcut. '''
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('context_settings', {})
        kwargs['context_settings'].setdefault('help_option_names', ['-h', '--help'])
        super().__init__(*args, **kwargs)

    def add_command(self, cmd, name=None):
        cmd.context_settings.setdefault('help_option_names', ['-h', '--help'])
        super().add_command(cmd, name=name)

    ''' Override argument parsing if arithmetic mode is detected. '''
    def parse_args(self, ctx, args):
        if arith.try_parse(args):
            ctx.exit(0)
        return super().parse_args(ctx, args)

@click.group(cls=MyGroup)
def cli():
    pass

cli.add_command(render.render)
cli.add_command(sources.info)
cli.add_command(sources.dump)
cli.add_command(dumps.sequence)

if __name__ == '__main__':
    cli()
