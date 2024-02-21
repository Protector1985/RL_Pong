
from classes.ActorCritic import ActorCritic
import multiprocessing as mp
from classes.Training import Training
mp.set_start_method('spawn', force=True)


training = Training()

def main():
    
    MasterNode = ActorCritic()
    MasterNode.share_memory()
    processes = []

    params = {
        'epochs': 1000,
        'n_workers': 14
    }

    counter = mp.Value('i', 0)


    for i in range(params['n_workers']):
        process = mp.Process(target=training.worker, args=(i, MasterNode, counter, params))
        process.start()
        processes.append(process)
        
    for process in processes:
        process.join()
        
    for process in processes:
        process.terminate()
        
if __name__ == '__main__':
    main()
        
    


