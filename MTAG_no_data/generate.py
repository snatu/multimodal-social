SG_PATH = '/home/shounak_rtml/11777/Standard-Grid'
import sys
sys.path.append(SG_PATH)
import standard_grid
from hp import hp
from os.path import join

if __name__=="__main__":
    hash_len = 10 # the number of characters in each hash.  if running lots of tests, may have collision if too few chars.  elif running few tests, can be nice to have smaller identifying strings
    # email_args= {
    #     'subject': 'Hello there',
    #     'text': '',
    #     'to_addr': 'your-email@gmail.com',
    #     'secrets_path': '/home/shounak_rtml/11777/MTAG/mailgun_secrets.json',
    # }
    grid = standard_grid.Grid('/home/shounak_rtml/11777/MTAG/main.py','/home/shounak_rtml/11777/MTAG/results/', hash_len=hash_len, email_args=None)

    for k,v in hp.items():
        grid.register(k,v)

    grid.generate_grid()
    grid.shuffle_grid()
    grid.generate_shell_instances(prefix='python ',postfix='')
    
    # Breaks the work across num_gpus GPUs, num_parallel jobs on each gpu
    num_gpus = 1
    num_parallel = 1
    hash_out = grid.create_runner(num_runners=num_gpus,runners_prefix=['CUDA_VISIBLE_DEVICES=%d sh'%i for i in range(num_gpus)],parallel=num_parallel)

    print(f'''

hash='{hash_out}'

root=/home/shounak_rtml/11777/MTAG/
attempt='0'
cd $root/results/${{hash}}/central/attempt_${{attempt}}/
chmod +x main.sh
./main.sh
cd $root
p status.py ${{hash}}
p interpret.py ${{hash}}

    ''')


