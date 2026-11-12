"""Run a finite experiment with durable logs across an SSH disconnect."""
import os,sys,json,argparse,subprocess,hashlib,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=[0,1],required=True);parser.add_argument('--name',required=True);parser.add_argument('--program',required=True);parser.add_argument('arguments',nargs=argparse.REMAINDER);args=parser.parse_args()
    if not all(c.isalnum() or c in '-_' for c in args.name):raise ValueError('Simple experiment name required')
    program=(ROOT/args.program).resolve();assert program.is_relative_to(ROOT) and program.is_file() and program.suffix=='.py'
    out=ROOT/'experiments/run_registry'/args.name;out.mkdir(parents=True,exist_ok=False);arguments=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    command=[sys.executable,str(program),*arguments]
    with (out/'output.log').open('xb') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=process.pid,command=command,gpu=args.gpu,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),program_sha256=hashlib.sha256(program.read_bytes()).hexdigest(),
        scope='Finite user-authorized experiment,detached only for connection resilience;does not restart automatically')
    (out/'job.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
