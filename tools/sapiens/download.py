"""下载并校验 Meta 官方 Sapiens-Lite 0.3B 人体分割权重，不修改 Python 环境。
根目录运行：python tools/sapiens/download.py。约1.36GB，支持断点续传。
校验官方 Hugging Face LFS SHA256；不下载其他任务或更大的模型。
"""
from pathlib import Path
import hashlib,json,time,urllib.request
HERE=Path(__file__).resolve().parent
REPO='facebook/sapiens-seg-0.3b-torchscript'
FILE='sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2'


def metadata(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url,timeout=40) as response:return json.load(response)
        except Exception:
            if attempt==4:raise
            time.sleep(2)


def main():
    meta=metadata('https://huggingface.co/api/models/'+REPO)
    sha=meta['sha'];tree=metadata(f'https://huggingface.co/api/models/{REPO}/tree/{sha}')
    entry=next(x for x in tree if x['path']==FILE)
    folder=HERE/'checkpoints';folder.mkdir(parents=True,exist_ok=True);path=folder/FILE
    expected=entry['lfs']['oid'];size=entry['size']
    for attempt in range(8):
        start=path.stat().st_size if path.exists() else 0
        if start==size:break
        req=urllib.request.Request(f'https://huggingface.co/{REPO}/resolve/{sha}/{FILE}?download=true',headers={'Range':f'bytes={start}-','User-Agent':'local-sapiens-evaluation'})
        try:
            with urllib.request.urlopen(req,timeout=90) as response:
                append=start>0 and response.status==206
                completed=start if append else 0;mark=completed//(128*1024*1024)
                with path.open('ab' if append else 'wb') as stream:
                    while True:
                        block=response.read(4*1024*1024)
                        if not block:break
                        stream.write(block);completed+=len(block)
                        if completed//(128*1024*1024)>mark:
                            mark=completed//(128*1024*1024);print(f'{completed/size:.1%} ({completed/1e6:.0f}/{size/1e6:.0f} MB)',flush=True)
        except Exception as error:
            print(f'Attempt {attempt+1}: {type(error).__name__}: {error}',flush=True);time.sleep(2)
    if not path.exists() or path.stat().st_size!=size:raise RuntimeError('Incomplete weight download')
    digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    if digest!=expected:raise RuntimeError('Official SHA256 mismatch; do not load this file')
    (HERE/'checkpoint_manifest.json').write_text(json.dumps({'repository':REPO,'revision':sha,'file':FILE,'size':size,'sha256':digest},indent=2),encoding='utf-8')
    print('Official SHA256 verified: '+digest,flush=True)

if __name__=='__main__':main()
