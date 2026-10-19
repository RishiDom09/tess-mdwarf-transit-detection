"""Original balanced sampler with overlapping training batches."""
import numpy as np

class Sampler:
    def __init__(self,cache,sampler_seed,augmentation_seed,batch_size=1000,shift=100):
        self.cache=cache;self.sample_rng=np.random.RandomState(sampler_seed);self.augmentation_rng=np.random.RandomState(augmentation_seed)
        self.batch_size=batch_size;self.shift=shift;self.buffer=[];self.draws=0;self.exposures={}
        if not 0<shift<=batch_size:raise ValueError('Invalid window')

    def draw(self):
        group='active' if self.sample_rng.random_sample()<.5 else 'inactive'
        stars=self.cache.groups[group];tic=stars[self.sample_rng.randint(len(stars))]
        positive=self.sample_rng.random_sample()<.5
        options=self.cache.positive[tic]
        index=options[self.sample_rng.randint(len(options))] if positive else self.cache.control[tic]
        x,y,row=self.cache.example(index,training=True,rng=self.augmentation_rng)
        self.draws+=1;self.exposures[row['example_id']]=self.exposures.get(row['example_id'],0)+1
        return x,y,index,self.draws

    def batch(self):
        if self.buffer:self.buffer=self.buffer[self.shift:]
        while len(self.buffer)<self.batch_size:self.buffer.append(self.draw())
        return np.stack([v[0] for v in self.buffer]),np.array([v[1] for v in self.buffer],np.float32)[:,None]
