"""Compile the paper, with a private fallback for this minimal TeX runtime.

On a standard TeX Live installation this runs the usual pdflatex/bibtex
sequence. The fallback creates search data, font maps and a format only in
the project's regenerable tmp directory; it never changes system TeX files.
"""
from pathlib import Path
import json
import os
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def run(command,log,env):
    with log.open('w') as out:
        subprocess.run(command,cwd=HERE,env=env,stdout=out,
                       stderr=subprocess.STDOUT,check=True)


def main():
    env=os.environ.copy()
    normal=subprocess.run(['kpsewhich','-engine=pdftex','-format=fmt','pdflatex.fmt'],capture_output=True,
                          text=True).stdout.strip()
    engine=['pdflatex']
    if not normal:
        distribution=Path('/usr/share/texlive/texmf-dist')
        if not (distribution/'tex/latex/base/latex.ltx').exists():
            raise RuntimeError('A complete TeX Live installation is needed to compile this manuscript.')
        private=ROOT/'tmp/ei_manuscript_tex'
        private.mkdir(parents=True,exist_ok=True)
        tree=private/'texmf'
        tree.mkdir(exist_ok=True)
        for name in ['tex','fonts','bibtex','web2c','dvips']:
            target=distribution/name
            local=tree/name
            if target.exists() and not local.exists():
                local.symlink_to(target,target_is_directory=True)
        subprocess.run(['mktexlsr',str(tree)],check=True)
        env['TEXMF']=str(tree)
        env['TEXFONTMAPS']=str(private)+os.pathsep
        mapdir=distribution/'fonts/map/dvips/amsfonts'
        (private/'pdftex.map').write_text('\n'.join((mapdir/name).read_text()
          for name in ['cm.map','cmextra.map','symbols.map','latxfont.map']))
        fmt=private/'pdflatex.fmt'
        if not fmt.exists():
            with (private/'format_build.log').open('w') as out:
                subprocess.run(['pdftex','-ini','-etex','-jobname=pdflatex',
                                '-interaction=nonstopmode','-halt-on-error',
                                str(distribution/'tex/latex/tex-ini-files/pdflatex.ini')],
                               cwd=private,env=env,stdout=out,
                               stderr=subprocess.STDOUT,check=True)
        engine+=['-fmt='+str(fmt.with_suffix(''))]
    engine+=['-interaction=nonstopmode','-halt-on-error','main.tex']
    run(engine,HERE/'build_first.log',env)
    run(['bibtex','main'],HERE/'bibtex.log',env)
    run(engine,HERE/'build_second.log',env)
    run(engine,HERE/'build_final.log',env)
    log=(HERE/'main.log').read_text(errors='replace')
    validation={'pdf':'main.pdf',
                'bytes':(HERE/'main.pdf').stat().st_size,
                'undefined_references':'There were undefined references' in log,
                'undefined_citations':'undefined on input' in log,
                'overfull_lines':[line for line in log.splitlines() if 'Overfull' in line],
                'private_TeX_fallback':not bool(normal)}
    (HERE/'build_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation,indent=2))


if __name__=='__main__':
    main()
