"""Extract OpenROAD runtime from the organizer's immutable public image layer.

No Docker daemon/root access is needed. Verify the full downloaded layer digest
before extracting regular files and safe relative links inside this workspace.
"""
import hashlib,json,os,shutil,tarfile,time,urllib.request
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parent
DIGEST='26d51041d2ba93d400b6f0fb34b22846e87ff1183b7aa1f1d9e77065ae5c020f'
SIZE=3454564731

def urlopen(request):
    for attempt in range(3):
        try:return urllib.request.urlopen(request,timeout=60)
        except (urllib.error.URLError,TimeoutError):
            if attempt==2:raise
            time.sleep(2)

def main():
    archive=ROOT/'packages'/('ispd27-openroad-'+DIGEST+'.tar.gz')
    archive.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    if not archive.exists():
        with urlopen('https://auth.docker.io/token?service=registry.docker.io&scope=repository:udxs/ispd27:pull') as f:token=json.load(f)['token']
        request=urllib.request.Request('https://registry-1.docker.io/v2/udxs/ispd27/blobs/sha256:'+DIGEST,headers={'Authorization':'Bearer '+token})
        temp=archive.with_suffix('.partial');total=0;last=start
        with urlopen(request) as response,temp.open('wb') as target:
            while True:
                chunk=response.read(8<<20)
                if not chunk:break
                target.write(chunk);total+=len(chunk)
                now=time.perf_counter()
                if now-last>15:print('download',total,'/',SIZE,'seconds',round(now-start,1),flush=True);last=now
        if total!=SIZE:raise ValueError('Incomplete image layer')
        temp.rename(archive)
    digest=hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):digest.update(block)
    if digest.hexdigest()!=DIGEST:raise ValueError('Container layer SHA256 mismatch')
    destination=ROOT/'contest-runtime';destination.mkdir(exist_ok=True);count=0;links=[]
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            path=PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe image path')
            name=str(path)
            selected=name.startswith(('install/OpenROAD/','usr/local/bin/','usr/local/lib/','usr/lib/','lib/','lib64/','opt/')) or name=='OpenROAD/build/bin/openroad'
            if not selected or '/.wh.' in name:continue
            target=destination.joinpath(*path.parts)
            if member.isdir():target.mkdir(parents=True,exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as source,target.open('wb') as output:shutil.copyfileobj(source,output,1<<20)
                target.chmod(member.mode & 0o777);count+=1
            elif member.issym():links.append((target,member.linkname))
    for target,link in links:
        candidate=(destination/link.lstrip('/')) if link.startswith('/') else target.parent/link
        if not candidate.resolve().is_relative_to(destination.resolve()):raise ValueError('Unsafe image link')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() or target.is_symlink():target.unlink()
        target.symlink_to(os.path.relpath(candidate,target.parent))
    (destination/'provenance.json').write_text(json.dumps(dict(image='udxs/ispd27:v1',manifest='sha256:c6f14196a4cab8d667ae8ed951ad2e3ee2bfa5760fe5d91dd4171046e0226b43',layer='sha256:'+DIGEST,files=count,seconds=time.perf_counter()-start),indent=2)+'\n')
    print('EXTRACTED',count,'seconds',round(time.perf_counter()-start,1),flush=True)

if __name__=='__main__':main()
