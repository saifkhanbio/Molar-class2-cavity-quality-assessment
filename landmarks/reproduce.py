"""Regenerate fitted camera and landmark estimates from the original project."""
import argparse
from pathlib import Path
import derive_landmarks as d
import export_landmarks as e

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True,help='06APR_CL2 project directory')
parser.add_argument('--output',type=Path,required=True,help='A separate output directory')
parser.add_argument('--cases',type=int,nargs='*',default=list(range(1,19))+list(range(35,46)))
args=parser.parse_args()
d.BASE=args.source.resolve();e.ROOT=str(d.BASE)
e.OUT=args.output.resolve();d.CACHE=e.OUT/'registration'
if e.OUT==d.BASE or e.OUT==d.BASE/'STLFILES':
    parser.error('Choose a separate output directory to preserve the source data.')
for sub in ['previews','picked_points','per_view','scripts','registration']:
    (e.OUT/sub).mkdir(parents=True,exist_ok=True)
for num in sorted(set(n-10 if 11<=n<=18 else n for n in args.cases)):
    for view in ['O','P']:
        d.fit(num,view)
        d.refine_existing(num,view)
e.main(args.cases)
