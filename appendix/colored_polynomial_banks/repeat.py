"""Start one previously fixed temporal repeat; retains an unavailable outcome on device failure."""
import argparse,datetime,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('slot',choices=('0905','0942'));p.add_argument('archive',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();here=Path(__file__).resolve().parent;dest=a.output/f'repeat_{a.slot}.json'
assert not dest.exists()
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
cmd=[sys.executable,'-Bu',str(here/'probe.py'),str(a.archive),str(a.output),'--repeat',a.slot]
log=a.output/f'repeat_{a.slot}.log'
with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
if r.returncode:
    note=log.read_text(errors='replace').lower()
    if not any(x in note for x in ('out of memory','device is busy','devices are busy','driver not initialized','devices unavailable')):
        raise SystemExit(r.returncode)
    dest.write_text(json.dumps(dict(mode=a.slot,unavailable=True,began_utc=started,
                    reason='CUDA device allocation or availability failure',returncode=r.returncode),indent=2)+'\n')
print(dest.name,flush=True)
