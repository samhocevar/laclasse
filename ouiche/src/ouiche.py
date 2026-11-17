import click

import render

class MyGroup(click.Group):
    ''' Override the --help flag everywhere to allow -h as a shortcut. '''
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('context_settings', {})
        kwargs['context_settings'].setdefault('help_option_names', ['-h', '--help'])
        super().__init__(*args, **kwargs)

    def add_command(self, cmd, name=None):
        cmd.context_settings.setdefault('help_option_names', ['-h', '--help'])
        super().add_command(cmd, name=name)

@click.group(cls=MyGroup)
def cli():
    pass

cli.add_command(render.render)

if __name__ == '__main__':
    cli()
